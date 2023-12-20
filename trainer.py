import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from network import ContactNet
from utils import load_network


class Trainer:
    def __init__(self, train_loader, test_loader, embedding_size, device,
                 dataset, lr, lat_grid_num, lng_grid_num, contact_factor, score):

        self.model = ContactNet(embedding_size, device, contact_factor).to(device)
        self.network = load_network(dataset, lat_grid_num, lng_grid_num).to(device)

        self.crit = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.steplr = StepLR(self.optimizer, step_size=3, gamma=0.9)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model_path = f"models_pth/model_{dataset}_{contact_factor}_{score}.pth"
        self.train_log_path = f"logs/train_log_{dataset}_{contact_factor}_{score}.log"
        self.test_log_path = f"logs/test_log_{dataset}_{contact_factor}_{score}.log"

        self.device = device

    def train(self):
        self.model.train()
        self.model.mode = 'train'
        epo_loss = 0

        for batch in self.train_loader:
            trajs_a, trajs_b, labels, indices_x, indices_y = [tensor.to(self.device) for tensor in batch]

            self.optimizer.zero_grad()

            indices_x = F.one_hot(indices_x, num_classes=trajs_a.size(1)).float()
            indices_y = F.one_hot(indices_y, num_classes=trajs_b.size(1)).float()

            att_ab, att_ba, indices_ab, indices_ba = self.model(self.network, trajs_a, trajs_b)

            norm = torch.norm(att_ab - att_ba, dim=-1, p=2)
            score_loss = self.crit(torch.exp(-norm), labels.to(self.device))

            location_loss = self.crit(indices_ab, indices_x) + self.crit(indices_ba, indices_y)

            loss = score_loss + location_loss / 2

            loss.backward()
            epo_loss += loss.item()
            self.optimizer.step()

        self.steplr.step()
        return epo_loss / len(self.train_loader)

    def test(self):
        self.model.eval()
        self.model.mode = 'test'

        all_dist = []
        with torch.no_grad():
            for batch in self.test_loader:
                idx, j, trajs_a, trajs_b = batch
                att_ab, att_ba = self.model(self.network, trajs_a.to(self.device), trajs_b.to(self.device))

                norm = torch.norm(att_ab - att_ba, dim=-1, p=2)

                id_dist = np.stack((idx, j, torch.exp(-norm).cpu().numpy())).T
                all_dist.append(id_dist)

        all_dist = np.concatenate(all_dist)
        return all_dist

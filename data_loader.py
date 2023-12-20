import numpy as np
import torch
from torch.utils.data import Dataset


def padding(batch):
    max_len = max(map(len, batch))
    padded_batch = [x + [[0] * 11] * (max_len - len(x)) for x in batch]
    return torch.tensor(padded_batch)[:, :, 4:]


class collater_train:
    def __init__(self, trajs, top_k_num):
        self.trajs = trajs
        self.top_k_num = top_k_num

    def __call__(self, batch):
        batch = np.array(batch)
        batch = torch.tensor(batch, dtype=torch.float32)

        samples_a, samples_b, labels, indices = batch[:, 0].long(), batch[:, 1].long(), batch[:, 2], batch[:, 3:]
        indices_a, indices_b = indices[:, :self.top_k_num].long(), indices[:, self.top_k_num:].long()

        trajs_a = padding(self.trajs[samples_a])
        trajs_b = padding(self.trajs[samples_b])

        return trajs_a, trajs_b, labels, indices_a, indices_b


class collater_test:
    def __init__(self, trajs):
        self.trajs = trajs

    def __call__(self, batch):
        batch = np.array(batch)
        idxs, js = batch[:, 0], batch[:, 1]

        trajs_a = padding(self.trajs[idxs])
        trajs_b = padding(self.trajs[js])

        return idxs, js, trajs_a, trajs_b


class MyDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        data_seqs = self.seqs[index]
        return data_seqs

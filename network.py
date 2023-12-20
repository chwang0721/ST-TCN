import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from d2v import Date2VecConvert


class GAT(nn.Module):
    def __init__(self, feature_size, embedding_size):
        super(GAT, self).__init__()
        self.conv = GATConv(feature_size, embedding_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x)
        return x


class SoftSort_p2(nn.Module):
    def __init__(self, tau=1e-6):
        super(SoftSort_p2, self).__init__()
        self.tau = tau

    def forward(self, scores):
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = ((scores.transpose(1, 2) - sorted) ** 2).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        return P_hat


class ContactATT(nn.Module):
    def __init__(self, dim):
        super(ContactATT, self).__init__()
        self.queries = nn.Linear(dim, dim, bias=False)
        self.keys = nn.Linear(dim, dim, bias=False)
        self.values = nn.Linear(dim, dim, bias=False)

        self.FFN = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        nn.init.eye_(self.queries.weight)
        nn.init.eye_(self.keys.weight)
        nn.init.eye_(self.values.weight)

    def forward(self, x, y, mask):
        queries = self.queries(x)
        keys = self.keys(y)
        values = self.values(y)

        mask = (~mask).int().unsqueeze(1)
        scores = torch.exp(-torch.cdist(queries, keys))
        scores = scores.masked_fill(mask == 0, -1e10)
        scores = F.softmax(scores, dim=-1)
        att_out = torch.matmul(scores, values)
        return att_out, scores


class Co_Att(nn.Module):
    def __init__(self, dim):
        super(Co_Att, self).__init__()

        self.Wq_s = nn.Linear(dim, dim, bias=False)
        self.Wk_s = nn.Linear(dim, dim, bias=False)
        self.Wv_s = nn.Linear(dim, dim, bias=False)

        self.Wq_t = nn.Linear(dim, dim, bias=False)
        self.Wk_t = nn.Linear(dim, dim, bias=False)
        self.Wv_t = nn.Linear(dim, dim, bias=False)

        self.dim_k = dim ** 0.5

        self.FFN_s = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        self.FFN_t = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, seq_s, seq_t, mask):
        seq_t = seq_t.unsqueeze(2)
        seq_s = seq_s.unsqueeze(2)
        mask = (~mask).int().unsqueeze(-1).unsqueeze(-1)

        q_s, k_s, v_s = self.Wq_s(seq_t), self.Wk_s(seq_s), self.Wv_s(seq_s)
        q_t, k_t, v_t = self.Wq_t(seq_s), self.Wk_t(seq_t), self.Wv_t(seq_t)

        scores_s = torch.matmul(q_s / self.dim_k, k_s.transpose(2, 3))
        scores_t = torch.matmul(q_t / self.dim_k, k_t.transpose(2, 3))

        scores_s = scores_s.masked_fill(mask == 0, -1e10)
        scores_t = scores_t.masked_fill(mask == 0, -1e10)

        coatt_s = F.softmax(scores_s, dim=-1)
        coatt_t = F.softmax(scores_t, dim=-1)

        attn_s = torch.matmul(coatt_s, v_s)
        attn_t = torch.matmul(coatt_t, v_t)

        att_s = self.layer_norm(self.FFN_s(attn_s) + attn_s)
        att_t = self.layer_norm(self.FFN_t(attn_t) + attn_t)

        return torch.concat((att_s.squeeze(2), att_t.squeeze(2)), dim=-1)


class ContactNet(nn.Module):
    def __init__(self, embedding_size, device, contact_factor):
        super(ContactNet, self).__init__()

        self.co_attention = Co_Att(embedding_size)
        self.gat = GAT(embedding_size, embedding_size)
        self.d2v = Date2VecConvert(device)
        self.soft_sort = SoftSort_p2()
        self.contact_att = ContactATT(embedding_size * 2)

        self.FFN = nn.Sequential(
            nn.Linear(embedding_size * 4, embedding_size * 2),
            nn.ReLU(),
            nn.Linear(embedding_size * 2, embedding_size * 4),
            nn.Dropout(0.1)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.e_size = embedding_size
        self.contact_factor = contact_factor
        self.mode = 'test'

    def padding_mask(self, inp):
        return inp == 0

    def contact(self, x, y, mask):
        mask = (~mask).int().unsqueeze(1)
        scores = torch.exp(-torch.cdist(x, y))
        scores = scores.masked_fill(mask == 0, -1e10)
        scores = self.softmax(scores)
        return scores

    def forward(self, network, trajs_a, trajs_b):
        batch_size = len(trajs_a)
        trajs_a, times_a = trajs_a[:, :, 0].long(), trajs_a[:, :, 1:]
        trajs_b, times_b = trajs_b[:, :, 0].long(), trajs_b[:, :, 1:]

        nodes = self.gat(network)

        mask_a = self.padding_mask(trajs_a)
        mask_b = self.padding_mask(trajs_b)

        embed_mask_a = (~mask_a).float().unsqueeze(-1)
        embed_mask_b = (~mask_b).float().unsqueeze(-1)

        emb_a_s = torch.index_select(nodes, 0, trajs_a.flatten()).reshape(batch_size, -1, self.e_size)
        emb_b_s = torch.index_select(nodes, 0, trajs_b.flatten()).reshape(batch_size, -1, self.e_size)

        emb_a_t = self.d2v(times_a)
        emb_b_t = self.d2v(times_b)

        emb_a = self.co_attention(emb_a_s, emb_a_t, mask_a)
        emb_b = self.co_attention(emb_b_s, emb_b_t, mask_b)

        att_emb_a, scores_ab = self.contact_att(emb_a, emb_b, mask_b)
        att_emb_b, scores_ba = self.contact_att(emb_b, emb_a, mask_a)
        scores_ab_max = torch.max(scores_ab, dim=-1).values
        scores_ba_max = torch.max(scores_ba, dim=-1).values

        emb_a = torch.concat((emb_a, emb_a - att_emb_a), dim=-1) * embed_mask_a
        emb_b = torch.concat((emb_b, emb_b - att_emb_b), dim=-1) * embed_mask_b

        traj_arange = torch.arange(batch_size).unsqueeze(-1)

        if self.mode == 'train':
            indices_a = self.soft_sort(scores_ab_max)[:, :self.contact_factor]
            indices_b = self.soft_sort(scores_ba_max)[:, :self.contact_factor]
            att_ab = emb_a[traj_arange, torch.argmax(indices_a, dim=-1)]
            att_ba = emb_b[traj_arange, torch.argmax(indices_b, dim=-1)]

            return self.FFN(att_ab.mean(1)), self.FFN(att_ba.mean(1)), indices_a, indices_b

        elif self.mode == 'test':
            indices_a = torch.argsort(-scores_ab_max, dim=-1)[:, :self.contact_factor]
            indices_b = torch.argsort(-scores_ba_max, dim=-1)[:, :self.contact_factor]
            att_ab = emb_a[traj_arange, indices_a]
            att_ba = emb_b[traj_arange, indices_b]

            return self.FFN(att_ab.mean(1)), self.FFN(att_ba.mean(1))

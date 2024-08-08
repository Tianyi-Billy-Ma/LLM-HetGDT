import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import GCNConv


class HeCo(nn.Module):
    def __init__(self, model_config, num_node_types=3):
        super().__init__()

        self.target_node_type = "user"

        ModelConfig = model_config.ModelConfig

        self.dropout = nn.Dropout(p=ModelConfig.dropout)

        self.input_dim = ModelConfig.input_dim
        self.hidden_dim = ModelConfig.hidden_dim
        self.num_metapaths = ModelConfig.num_metapaths
        self.sample_rate = ModelConfig.sample_rate
        self.num_neighbors = ModelConfig.num_neighbors
        self.attention = ModelConfig.attention
        self.tau = ModelConfig.tau
        self.lam = ModelConfig.lam

        self.mappings = nn.ModuleList()
        for _ in range(num_node_types):
            self.mappings.append(Linear(self.input_dim, self.hidden_dim))
        self.mp = MPEncoder(
            num_metapaths=self.num_metapaths,
            hidden_dim=self.hidden_dim,
            attention_dropout=self.attention,
        )
        self.sc = SCEncoder(
            self.hidden_dim, self.sample_rate, self.num_neighbors, self.attention
        )
        self.contrast = Contrast(self.hidden_dim, self.tau, self.lam)

    def forward(self, batch):
        h_all = {}
        mp_edge_index = [batch[mp_type].edge_index for mp_type in batch.metapath_dict]

        nei_index = [
            batch[self.target_node_type].nei_index[node_type]
            for node_type in batch.x_dict
            if node_type != self.target_node_type
        ]
        pos = batch[self.target_node_type].pos
        for idx, (node_type, node_feats) in enumerate(batch.x_dict.items()):
            h = self.mappings[idx](node_feats)
            h = self.dropout(h)
            h = F.elu(h)
            h_all[node_type] = h
        z_mp = self.mp(h_all[self.target_node_type], mp_edge_index)
        z_sc = self.sc([h_all[node_type] for node_type in batch.x_dict], nei_index)
        loss = self.contrast(z_mp, z_sc, pos)
        return loss

    def get_embeds(self, batch=None, **kwargs):
        if batch:
            feats = {node_type: batch[node_type].x for node_type in batch.x_dict}
            mp_edge_index = [
                batch[mp_type].edge_index for mp_type in batch.metapath_dict
            ]
        else:
            feats = kwargs["feats"]
            mp_edge_index = kwargs["mp_edge_index"]
        zs = {
            node_type: F.elu(self.mappings[idx](feat))
            for idx, (node_type, feat) in enumerate(feats.items())
        }
        # z_mp = F.elu(self.mappings[0](feat))
        z_mp = self.mp(zs[self.target_node_type], mp_edge_index)
        zs[self.target_node_type] = z_mp
        return zs


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2sc = matrix_mp2sc / (
            torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8
        )
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (
            torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8
        )
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.layer = GCNConv(in_ft, out_ft)
        self.act = nn.PReLU()

    def forward(self, seq, adj):
        out = self.layer(seq, adj)
        return self.act(out)


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        return z_mp


class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("sc ", beta.data.cpu().numpy())  # type-level attention
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(
            torch.empty(size=(1, 2 * hidden_dim)), requires_grad=True
        )
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):
        nei_emb = F.embedding(nei, h)
        h_refer = torch.unsqueeze(h_refer, 1)
        h_refer = h_refer.expand_as(nei_emb)
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)
        nei_emb = (att * nei_emb).sum(dim=1)
        return nei_emb


class SCEncoder(nn.Module):
    def __init__(self, hidden_dim, sample_rate, num_neighbors, attention_dropout):
        super(SCEncoder, self).__init__()
        self.intra = nn.ModuleList(
            [intra_att(hidden_dim, attention_dropout) for _ in range(num_neighbors)]
        )
        self.inter = inter_att(hidden_dim, attention_dropout)
        self.sample_rate = sample_rate
        self.num_neighbors = num_neighbors

    def forward(self, nei_h, nei_index):
        embeds = []
        for i in range(self.num_neighbors):
            sele_nei = []
            sample_num = self.sample_rate[i]
            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(
                        np.random.choice(per_node_nei.cpu(), sample_num, replace=False)
                    )[np.newaxis]
                else:
                    if per_node_nei.shape[0] != 0:
                        select_one = torch.tensor(
                            np.random.choice(
                                per_node_nei.cpu(), sample_num, replace=True
                            )
                        )[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = torch.cat(sele_nei, dim=0).cuda()
            one_type_emb = F.elu(self.intra[i](sele_nei, nei_h[i + 1], nei_h[0]))
            embeds.append(one_type_emb)
        z_mc = self.inter(embeds)
        return z_mc


class MPEncoder(nn.Module):
    def __init__(self, num_metapaths, hidden_dim, attention_dropout):
        super(MPEncoder, self).__init__()
        self.P = num_metapaths
        self.node_level = nn.ModuleList(
            [GCN(hidden_dim, hidden_dim) for _ in range(num_metapaths)]
        )
        self.att = Attention(hidden_dim, attention_dropout)

    def forward(self, h, mps):
        embeds = []
        for i in range(self.P):
            embeds.append(self.node_level[i](h, mps[i]))
        z_mp = self.att(embeds)
        return z_mp

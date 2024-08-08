import torch
import torch.nn as nn
from easydict import EasyDict

import math
import numpy as np

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter

from models.MLP import MLP


class iHGT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.node_types = self.config.node_types

        self.num_node_types = len(self.node_types)
        self.target_node_type = self.config.node_types[0]
        self.num_metapath_types = self.config.num_metapath_types

        self.num_node_type_tokens = self.config.num_node_type_tokens
        self.node_type_token_dim = self.config.node_type_token_dim
        self.class_token_dim = self.config.class_token_dim

        self.batch_size = self.config.batch_size

        self.num_classes = self.config.num_classes

        self.node_type_tokens = nn.ParameterDict(
            {
                node_type: nn.Parameter(
                    torch.randn(self.num_node_type_tokens, self.node_type_token_dim)
                )
                for node_type in self.node_types
            }
        )

        self.class_tokens = nn.Parameter(
            torch.randn((self.num_classes, self.class_token_dim))
        )

        self.nei_PMAs = nn.ModuleDict(
            {
                node_type: PMA(
                    in_dim=self.config.MLP_output_dim,
                    hid_dim=self.config.MLP_hidden_dim,
                    out_dim=self.config.MLP_output_dim,
                    num_layers=self.config.MLP_num_layers,
                    dropout=self.config.dropout,
                    Normalization=self.config.MLP_norm,
                    heads=self.config.heads,
                )
                for node_type in self.node_types
                if node_type != self.target_node_type
            }
        )

        self.mp_PMAs = nn.ModuleList(
            [
                PMA(
                    in_dim=self.config.MLP_output_dim,
                    hid_dim=self.config.MLP_hidden_dim,
                    out_dim=self.config.MLP_output_dim,
                    num_layers=self.config.MLP_num_layers,
                    dropout=self.config.dropout,
                    Normalization=self.config.MLP_norm,
                    heads=self.config.heads,
                )
                for _ in range(self.num_metapath_types)
            ]
        )

        self.na_semantic_layer = SemanticAttention(
            input_dim=self.config.MLP_hidden_dim,
            hidden_size=self.config.MLP_hidden_dim,
        )

        self.mp_semantic_layer = SemanticAttention(
            input_dim=self.config.MLP_hidden_dim,
            hidden_size=self.config.MLP_hidden_dim,
        )
        self.linear = nn.Linear(
            self.config.MLP_hidden_dim * 2, self.config.MLP_hidden_dim
        )

        self.contrast_head = ContrastiveLoss(
            self.config.MLP_hidden_dim, self.config.MLP_hidden_dim, self.config.tau
        )

        self.over_sampling = OverSampling()

    def reset_parameters(self, batch, pretrain_model):
        mask = batch[self.target_node_type].mask

        x = pretrain_model.get_embeds(batch)[self.target_node_type]
        x = x[mask]
        y = batch[self.target_node_type].y[mask]

        for label in range(self.num_classes):
            self.class_tokens.data[label] = torch.mean(x[y == label], dim=0)

        for _, emb_layer in self.node_type_tokens.items():
            emb_layer.data.uniform_(-1, 1)

        for _, layer in self.nei_PMAs.items():
            layer.reset_parameters()

        for layer in self.mp_PMAs:
            layer.reset_parameters()
        self.na_semantic_layer.reset_parameters()
        self.mp_semantic_layer.reset_parameters()
        self.linear.reset_parameters()
        self.contrast_head.reset_parameters()

    def forward(self, batch, model):
        mask = batch[self.target_node_type].mask

        # x = batch[self.target_node_type].x
        y = batch[self.target_node_type].y

        prompt_node_embs = {}
        for node_type in self.node_type_tokens.keys():
            curr_node_type_x = batch[node_type].x
            curr_node_type_token = self.node_type_tokens[node_type]

            curr_node_type_weight = torch.matmul(
                curr_node_type_x,
                curr_node_type_token.T,
            )
            curr_node_type_weight = torch.exp(F.leaky_relu(curr_node_type_weight))
            curr_node_type_weight = curr_node_type_weight / curr_node_type_weight.sum(
                dim=1, keepdim=True
            )
            curr_node_type_emb = curr_node_type_x + torch.matmul(
                curr_node_type_weight, curr_node_type_token
            )
            prompt_node_embs[node_type] = curr_node_type_emb

        mp_edge_index = [batch[mp_type].edge_index for mp_type in batch.metapath_dict]
        pretrained_node_embs = model.get_embeds(
            feats=prompt_node_embs, mp_edge_index=mp_edge_index
        )

        x_mp = self.metapath_aggregation(pretrained_node_embs, mp_edge_index)        
        x_out = F.leaky_relu(x_mp)

        y_train = y[mask]
        x_train = x_out[mask]
        # x_train = x_out
        # self.smote
        # x_train, y_train = self.over_sampling(x_train, y_train)

        loss, logits = self.contrast_head(x_train, self.class_tokens, y_train)

        logits = logits[: x_out.shape[0]]

        data_to_return = EasyDict(x=x_out, loss=loss, logits=logits)

        return data_to_return

    def neighbor_aggregation(self, pretrained_node_embs, nei_indices):
        # Neighbor aggregation:
        x_na = {}
        target_node_embs = pretrained_node_embs[self.target_node_type]

        for node_type in self.node_types:
            if node_type == self.target_node_type:
                continue
                # TODO: Implement target node level neighbor aggregation
            curr_x_na = []
            # nei_index = [
            #     batch[self.target_node_type].nei_index[node_type][idx]
            #     for idx in mask.nonzero().squeeze()
            # ]
            nei_index = nei_indices[node_type]

            for batch_idx in range(0, len(nei_index), self.batch_size):
                batch_nei_index = nei_index[
                    batch_idx : batch_idx + self.config.batch_size
                ]

                source_indices = torch.cat(batch_nei_index)
                target_indices = torch.cat(
                    [
                        torch.full((neighbors.shape[0],), i, dtype=torch.long)
                        for i, neighbors in enumerate(batch_nei_index)
                    ]
                ).to(source_indices.device)

                batch_edge_index = torch.stack([source_indices, target_indices], dim=0)

                nei_x = pretrained_node_embs[node_type]

                pma = self.nei_PMAs[node_type]

                batch_x_na = pma(
                    nei_x,
                    target_node_embs[batch_idx : batch_idx + self.config.batch_size],
                    batch_edge_index,
                    num_nodes=target_node_embs[
                        batch_idx : batch_idx + self.config.batch_size
                    ].shape[0],
                )
                curr_x_na.append(batch_x_na)

            curr_x_na = torch.cat(curr_x_na, dim=0)
            curr_x_na = F.relu(curr_x_na)
            curr_x_na = F.dropout(
                curr_x_na, p=self.config.dropout, training=self.training
            )
            x_na[node_type] = curr_x_na

        x_na = torch.stack(
            [curr_x_na for neighbor_type, curr_x_na in x_na.items()], dim=1
        )
        x_na, _ = self.na_semantic_layer(x_na)
        return x_na

    def metapath_aggregation(self, pretrained_node_embs, mp_edge_indices):
        # Metapath aggregation:

        x_mp = []
        num_nodes = pretrained_node_embs[self.target_node_type].shape[0]
        for metapath_idx, mp_edge_index in enumerate(mp_edge_indices):
            pma = self.mp_PMAs[metapath_idx]
            curr_x_mp = pma(
                pretrained_node_embs[self.target_node_type],
                pretrained_node_embs[self.target_node_type],
                mp_edge_index,
                num_nodes=num_nodes,
            )
            curr_x_mp = F.relu(curr_x_mp)
            curr_x_mp = F.dropout(
                curr_x_mp, p=self.config.dropout, training=self.training
            )
            x_mp.append(curr_x_mp)

        x_mp = torch.stack(x_mp, dim=1)
        x_mp, _ = self.mp_semantic_layer(x_mp)
        return x_mp


class OverSampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, y):
        occ = torch.eye(int(y.max() + 1), int(y.max() + 1)).to(y.device)[y].sum(axis=0)
        dominant_class = torch.argmax(occ)
        n_occ = int(occ[dominant_class].item())
        xs, ys = [], []
        for i in range(len(occ)):
            if i != dominant_class:
                # calculate the amount of synthetic data to generate
                # N = (n_occ - occ[i]) * 100 / occ[i]
                N = int((n_occ - occ[i]) * 0.5)
                candidates = X[y == i]
                selection = torch.randint(
                    0, candidates.shape[0], (int(N),), device=X.device
                )
                xs.append(candidates[selection])
                ys.append(torch.ones(int(N)) * i)
        xs = torch.cat(xs).to(X.device)
        ys = torch.cat(ys).to(y.device)
        X_return = torch.cat((X, xs))
        y_return = torch.cat((y, ys))
        return X_return, y_return.long()


class ContrastiveLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim, temperature):
        super().__init__()
        self.tau = temperature
        self.lin = nn.Linear(input_dim, hidden_dim)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def sim(self, z1, z2):
        # z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        # z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        # sim_matrix = torch.log(
        #     torch.mm(z1_norm, mat2=z2_norm.T) / torch.mm(z1_norm, z2_norm.T) / self.tau
        # )
        # return sim_matrix
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        sim_matrix = torch.mm(z1, z2.T) / self.tau
        return sim_matrix

    def forward(self, z1, z2, y):
        # num_of_classes = y.unique().shape[0]
        # num_samples = y.shape[0]
        # samples_per_cls = torch.bincount(y)
        # weight_per_cls = num_samples / samples_per_cls
        # weights = weight_per_cls[y]

        z1 = self.lin(z1)
        z2 = self.lin(z2)
        z_sim1 = self.sim(z1, z2)
        z_sim2 = self.sim(z2, z1)

        positive_sim1 = torch.exp(z_sim1[torch.arange(y.shape[0]), y])
        full_sim1 = torch.sum(torch.exp(z_sim1), dim=-1)
        l1 = -torch.log(positive_sim1 / full_sim1).sum()

        return l1, z_sim1


class PMA(MessagePassing):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers,
        dropout,
        Normalization="None",
        InputNorm=False,
        heads=1,
        negative_slope=0.2,
    ):
        super(PMA, self).__init__(node_dim=0)

        self.f_enc = nn.Linear(in_dim * 2, hid_dim)
        self.f_enc_k = nn.Linear(in_dim * 2, hid_dim)
        # self.f_enc = nn.Identity()
        # self.f_enc_k = nn.Identity()
        self.f_dec = MLP(
            in_dim, hid_dim, out_dim, num_layers, dropout, Normalization, InputNorm
        )

        self.ln0 = nn.LayerNorm(hid_dim)
        self.ln1 = nn.LayerNorm(hid_dim)
        self.att_r = nn.Parameter(torch.Tensor(1, heads, hid_dim // heads))

        self.hid_dim = hid_dim
        self.heads = heads
        self.attention_hidden = hid_dim // heads
        self.negative_slope = negative_slope
        self.dropout = dropout

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def reset_parameters(self):
        if isinstance(self.f_enc, nn.Linear):
            self.glorot(self.f_enc.weight)
        if isinstance(self.f_enc_k, nn.Linear):
            self.glorot(self.f_enc_k.weight)

        self.f_dec.reset_parameters()
        nn.init.xavier_uniform_(self.att_r)

        self.ln0.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, src, dst, edge_index, num_nodes, aggr="mean"):
        self.num_nodes = num_nodes
        out = self.propagate(edge_index, x=src, dst=dst, aggregate=aggr)
        out += self.att_r
        out = out.view(-1, self.hid_dim)
        out = self.ln0(out)
        out = self.ln1(out + F.relu(self.f_dec(out)))
        return out

    def message(self, x_j, dst_i, index, ptr):
        num_nodes = self.num_nodes
        H, C = self.heads, self.attention_hidden
        x_j = torch.cat([x_j, dst_i], dim=-1)
        x_K = self.f_enc_k(x_j).view(-1, H, C)
        x_V = self.f_enc(x_j).view(-1, H, C)
        alpha = (x_K * self.att_r).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_V * alpha.unsqueeze(-1)
        return out

    def aggregate(self, inputs, index, dim_size=None, aggregate=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        if aggregate is None:
            raise ValueError("aggr was not passed!")
        # dim = self.node_dim if not self.attention else 0
        # dim = self.node_dim
        return scatter(
            inputs, index, dim=self.node_dim, dim_size=self.num_nodes, reduce=aggregate
        )


class SemanticAttention(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def reset_parameters(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.project.apply(weight_init)

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        out_emb = (beta * z).sum(1)  # (N, D * K)
        att_mp = beta.mean(0).squeeze()

        return out_emb, att_mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from easydict import EasyDict
from functools import partial
from sklearn.model_selection import train_test_split


def create_split_masks(y, train=None, valid=None, test=None):
    assert train + valid + test == 1.0, "Ratios must sum to 1.0"

    train_valid = train + valid
    train_valid_indices, test_indices = train_test_split(
        range(len(y)), test_size=test, stratify=y
    )
    valid_adjusted = valid / train_valid
    train_indices, valid_indices = train_test_split(
        train_valid_indices,
        test_size=valid_adjusted,
        stratify=y[train_valid_indices],
    )
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    valid_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train_indices] = True
    valid_mask[valid_indices] = True
    test_mask[test_indices] = True

    return EasyDict(train=train_mask, valid=valid_mask, test=test_mask)


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor
        else:
            batch_list = graph.batch_num_nodes
            batch_size = len(batch_list)
            batch_list = torch.Tensor(batch_list).long().to(tensor.device)
            batch_index = (
                torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
            )
            batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(
                tensor
            )
            mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
            mean = mean.scatter_add_(0, batch_index, tensor)
            mean = (mean.T / batch_list).T
            mean = mean.repeat_interleave(batch_list, dim=0)

            sub = tensor - mean * self.mean_scale

            std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
            std = std.scatter_add_(0, batch_index, sub.pow(2))
            std = ((std.T / batch_list).T + 1e-6).sqrt()
            std = std.repeat_interleave(batch_list, dim=0)
            return self.weight * sub / std + self.bias


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


def sparse_sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    y_indices = y.nonzero(as_tuple=True)

    x_value = x[y_indices]
    y_value = y[y_indices]
    dot_product = x_value * y_value

    y_edge_index = torch.stack(y_indices, dim=0)

    sum_dot_product = torch.zeros(x.shape).to(x.device)
    sum_dot_product[y_indices] = dot_product
    sum_dot_product = sum_dot_product.sum(-1)
    # sum_dot_product = torch.sparse_coo_tensor(y_edge_index, dot_product, x.size()).sum(
    #     -1
    # )
    loss = (torch.ones_like(sum_dot_product) - sum_dot_product).pow_(alpha)
    # loss = (1 - sum_dot_product.values()).pow_(alpha)

    # Compute the loss
    # loss = (1 - sum_dot_product).pow_(alpha)
    loss = loss.mean()

    return loss


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def normalize(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    rowsum = x.sum(1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    x = r_mat_inv.dot(x)
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.todense()


def sim(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class BGRL_EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = (
            1
            - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        )
        self.step += 1
        return old * beta + (1 - beta) * new

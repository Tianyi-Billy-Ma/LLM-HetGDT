import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        dropout=0.5,
        Normalization="bn",
        InputNorm=False,
    ):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ["bn", "ln", "None"]
        if Normalization == "bn":
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(input_dim))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(input_dim, output_dim))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(input_dim))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(input_dim, hidden_dim))
                self.normalizations.append(nn.BatchNorm1d(hidden_dim))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_dim, hidden_dim))
                    self.normalizations.append(nn.BatchNorm1d(hidden_dim))
                self.lins.append(nn.Linear(hidden_dim, output_dim))
        elif Normalization == "ln":
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(input_dim))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(input_dim, output_dim))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(input_dim))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(input_dim, hidden_dim))
                self.normalizations.append(nn.LayerNorm(hidden_dim))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_dim, hidden_dim))
                    self.normalizations.append(nn.LayerNorm(hidden_dim))
                self.lins.append(nn.Linear(hidden_dim, output_dim))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(input_dim, output_dim))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(input_dim, hidden_dim))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_dim, hidden_dim))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_dim, output_dim))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is "Identity"):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

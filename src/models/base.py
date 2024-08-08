import torch
import pytorch_lightning as pl
import torch.nn as nn
from models.MLP import MLP

from models.layers import HANLayer
from utils.functions import create_activation


class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        EncoderModelClass = config.model_config.EncoderModelClass
        EncoderModelConfig = config.model_config.EncoderModelConfig

        DecoderModelClass = config.model_config.DecoderModelClass
        DecoderModelConfig = config.model_config.DecoderModelConfig

        self.encoder = globals()[EncoderModelClass](**EncoderModelConfig)
        self.decoder = globals()[DecoderModelClass](**DecoderModelConfig)

        self.target_node_type = config.train.additional.target_node_type

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, batch):
        output = self.encoder(batch)
        output = self.decoder(output)
        return output


class HAN(nn.Module):
    def __init__(
        self,
        num_metapath,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_heads,
        num_output_heads,
        activation,
        dropout,
        norm,
        encoding=False,
    ):
        super(HAN, self).__init__()
        self.out_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.han_layers = nn.ModuleList()
        self.activation = create_activation(activation)

        last_activation = (
            create_activation(activation) if encoding else create_activation(None)
        )
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.han_layers.append(
                HANLayer(
                    num_metapath,
                    input_dim,
                    output_dim,
                    num_output_heads,
                    dropout,
                    last_activation,
                    last_norm,
                )
            )
        else:
            # input projection (no residual)
            self.han_layers.append(
                HANLayer(
                    num_metapath,
                    input_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    self.activation,
                    norm,
                )
            )
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.han_layers.append(
                    HANLayer(
                        num_metapath,
                        hidden_dim * num_heads,
                        hidden_dim,
                        num_heads,
                        dropout,
                        self.activation,
                        norm,
                    )
                )
            # output projection
            self.han_layers.append(
                HANLayer(
                    num_metapath,
                    hidden_dim * num_heads,
                    output_dim,
                    num_output_heads,
                    dropout,
                    last_activation,
                    last_norm,
                )
            )

    def forward(self, h, gs):
        for gnn in self.han_layers:
            h, att_mp = gnn(h, gs)
        return h, att_mp


class LogReg(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

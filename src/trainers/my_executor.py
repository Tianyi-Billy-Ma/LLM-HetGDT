import logging

import pytorch_lightning as pl
import torch
from easydict import EasyDict

logger = logging.getLogger(__name__)

import numpy as np

from models.heco import HeCo
from models.hgmae import HGMAE
from src.models.model import iHGT
from trainers.base_executor import BaseExecutor


class iHGTExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.dataname = list(data_loader.data.keys())[0]
        self.target_node_type = self.config.train.additional.target_node_type

        PretrainModelClass = config.model_config.PretrainModelClass
        PretrainModelConfig = config.model_config.PretrainModelConfig
        PretrainModelCkptPath = config.model_config.PretrainModelCkptPath
        PretrainModelName = config.model_config.PretrainModelName
        PretrainModelCkpt = torch.load(PretrainModelCkptPath)
        PretrainModelWeights = {
            k[len(PretrainModelName) + 1 :]: v
            for k, v in PretrainModelCkpt["state_dict"].items()
            if k.startswith(f"{PretrainModelName}.")
        }

        self.pretrain_model = globals()[PretrainModelClass](PretrainModelConfig)
        self.pretrain_model.load_state_dict(PretrainModelWeights)

        ModelClass = config.model_config.ModelClass
        ModelClassConfig = config.model_config.ModelClassConfig

        self.model = globals()[ModelClass](ModelClassConfig)

        self.model.reset_parameters(
            next(iter(self.train_dataloaders[0])), self.pretrain_model
        )

    def configure_optimizers(self):
        model_optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.config.train.lr,
        )
        return model_optimizer

    def train_dataloader(self):
        self.train_dataloader_names = list(
            self.data_loader.data_loaders["train"].keys()
        )

        # TODO: we only allow one train data loader at the moment
        return self.train_dataloaders[0]

    def val_dataloader(self):
        self.val_dataloader_names = list(self.data_loader.data_loaders["valid"].keys())

        return self.valid_dataloaders

    def test_dataloader(self):
        self.test_dataloader_names = list(self.data_loader.data_loaders["test"].keys())

        return self.test_dataloaders

    def training_step(self, batch, batch_idx):
        # pretrained_emb = self.pretrain_model.get_embeds(batch)
        loss_dict = self.model(batch, self.pretrain_model)
        data_to_return = EasyDict(loss=loss_dict.loss)
        self.log(
            "train/loss",
            loss_dict.loss.item(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return data_to_return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._compute_logit(batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._compute_logit(batch, batch_idx, dataloader_idx)

    def _compute_logit(self, batch, batch_idx, dataloader_idx):
        loss_dict = self.model(batch, self.pretrain_model)

        data_to_return = EasyDict()

        logits = loss_dict.logits
        y_pred = torch.argmax(logits, dim=1)
        y_true = batch[self.target_node_type].y[batch[self.target_node_type].mask]

        data_to_return["y_true"] = y_true.detach().cpu().numpy()
        data_to_return["y_pred"] = y_pred.detach().cpu().numpy()
        data_to_return["loss"] = loss_dict.loss.item()

        return data_to_return

    def evaluate_outputs(self, step_outputs, current_data_loader, dataset_name):
        data_used_for_metrics = EasyDict(
            y_true=step_outputs.y_true,
            y_pred=step_outputs.y_pred,
        )
        log_dict = self.compute_metrics(data_used_for_metrics)

        for key, val in step_outputs.items():
            if key.endswith("loss"):
                log_dict[key] = val

        return log_dict

    def logging_results(self, log_dict, prefix):
        metrics_to_log = EasyDict()

        for metric, value in log_dict.metrics.items():
            metrics_to_log[f"{prefix}/{metric}"] = value
        metrics_to_log[f"{prefix}/epoch"] = self.current_epoch

        logger.info(
            f"Evaluation results [{self.trainer.state.stage}]: {metrics_to_log}"
        )
        if self.trainer.state.stage in ["sanity_check"]:
            logging.warning("Sanity check mode, not saving to loggers.")
            return
        for metric, value in metrics_to_log.items():
            if type(value) in [float, int, np.float64]:
                self.log(
                    metric,
                    float(value),
                    # on_step=False,
                    # on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
            else:
                logger.info(f"{metric} is not a type that can be logged, skippped.")

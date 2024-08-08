import torch
import pytorch_lightning as pl


import logging

logger = logging.getLogger(__name__)
import torch.nn.functional as F
from easydict import EasyDict
import numpy as np

from trainers.base_executor import BaseExecutor
from models.heco import HeCo
from models.base import LogReg


class HeCoExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.dataname = list(data_loader.data.keys())[0]
        self.target_node_type = self.config.train.additional.target_node_type

        ModelClass = globals()[self.config.model_config.ModelClass]

        model_config = self.config.model_config
        self.model = ModelClass(model_config)

        ClassifierModelClass = self.config.model_config.ClassifierModelClass
        ClassifierModelConfig = self.config.model_config.ClassifierModelConfig
        self.classifier = globals()[ClassifierModelClass](**ClassifierModelConfig)

        self.automatic_optimization = False
        self.loss_fn = F.nll_loss

    def configure_optimizers(self):
        model_optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.config.train.lr,
        )
        classifier_optimizer = torch.optim.Adam(
            list(self.classifier.parameters()),
            lr=self.config.train.lr,
            weight_decay=self.config.train.wd,
        )
        return [model_optimizer, classifier_optimizer]

    def training_step(self, batch, batch_idx):
        model_optimizer, classifier_optimizer = self.optimizers()
        data_to_log = EasyDict()
        loss = self.model(batch)
        data_to_log["train/loss"] = loss

        model_optimizer.zero_grad()
        self.manual_backward(loss)
        model_optimizer.step()

        mask = batch[self.target_node_type].mask
        y_true = batch[self.target_node_type].y
        embs = self.model.get_embeds(batch)

        embs = embs[self.target_node_type]

        output = self.classifier(embs)
        logits = F.log_softmax(output, dim=1)
        loss = self.loss_fn(logits[mask], y_true[mask])
        data_to_log["train/classifier_loss"] = loss.item()

        classifier_optimizer.zero_grad()
        self.manual_backward(loss)
        classifier_optimizer.step()

        self.log_dict(data_to_log, prog_bar=True, logger=True, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._compute_logit(batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._compute_logit(batch, batch_idx, dataloader_idx)

    def _compute_logit(self, batch, batch_idx, dataloader_idx):
        train_loss = self.model(batch)
        embs = self.model.get_embeds(batch)
        embs = embs[self.target_node_type]
        embs = self.classifier(embs)

        data_to_return = EasyDict()

        data_to_return["val_loss"] = train_loss.item()

        mask = batch[self.target_node_type].mask
        y_true = batch[self.target_node_type].y

        logits = F.log_softmax(embs, dim=1)
        pred_loss = self.loss_fn(logits[mask], y_true[mask])
        y_pred = torch.argmax(logits, dim=1)

        data_to_return["pred_loss"] = pred_loss.item()
        data_to_return["y_true"] = y_true[mask].detach().cpu().numpy()
        data_to_return["y_pred"] = y_pred[mask].detach().cpu().numpy()

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
                    logger=True,
                    sync_dist=True,
                )
            else:
                logger.info(f"{metric} is not a type that can be logged, skippped.")

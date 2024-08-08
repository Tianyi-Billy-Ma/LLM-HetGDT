# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import math
import time
import os
import os.path as osp
import sys
import scipy
import datetime
import numpy as np
import json
import operator
import wandb
import glob
import tarfile
import logging

logger = logging.getLogger(__name__)

from pprint import pprint
from tqdm import tqdm
from easydict import EasyDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from utils.metrics_log_callback import MetricsHistoryLogger

from trainers.metric_processors import MetricsProcessor
from utils.dirs import *


class BaseExecutor(pl.LightningModule, MetricsProcessor):
    additional_plugins = []

    def __init__(self, config, data_loader):
        super().__init__()
        self.config = config
        self.data_loader = data_loader
        self.train_dataloaders = list(self.data_loader.data_loaders["train"].values())
        self.valid_dataloaders = list(self.data_loader.data_loaders["valid"].values())
        self.test_dataloaders = list(self.data_loader.data_loaders["test"].values())

        logger.info(f"Initializing {self.__class__.__name__}...")

        # label smoother imported from huggingface transformers
        label_smoothing_factor = self.config.train.additional.get(
            "label_smoothing_factor", 0
        )
        if label_smoothing_factor != 0:
            from transformers.trainer_pt_utils import LabelSmoother

            self.label_smoother = LabelSmoother(epsilon=label_smoothing_factor)
        else:
            self.label_smoother = None

    def setup(self, stage):
        """
        set loggers as class attributes for easy access
        """
        for trainer_logger in self.trainer.loggers:
            if type(trainer_logger) == TensorBoardLogger:
                self.tb_logger = trainer_logger
            elif type(trainer_logger) == WandbLogger:
                self.wandb_logger = trainer_logger
                self.wandb_logger.watch(self.model, log_freq=500, log_graph=False)
            elif type(trainer_logger) == MetricsHistoryLogger:
                self.metrics_history_logger = trainer_logger
            else:
                logger.warning(f"Unsupported logger type: {type(trainer_logger)}")

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.train.lr)

        if self.config.train.scheduler == "liner":
            raise NotImplementedError
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.train.additional.scheduler_step_size,
                gamma=self.config.train.additional.scheduler_gamma,
            )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def log_max_and_min_metrics(self):
        """
        Log the max and min metrics for each metric automatically
        """
        for metric_name, metric_values in self.metrics_history_logger.history.items():
            if metric_name in ["epoch", "loss_step", "loss_epoch", "loss"]:
                continue
            if len(metric_values) > 0 and type(metric_values[0]) in [
                float,
                int,
                np.float64,
            ]:
                self.log(
                    f"{metric_name}_auto_max",
                    float(max(metric_values)),
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"{metric_name}_auto_min",
                    float(min(metric_values)),
                    on_step=False,
                    on_epoch=True,
                )

    def on_train_epoch_end(self):
        if self.global_rank == 0:
            self.log_max_and_min_metrics()

    def on_fit_end(self):
        if self.global_rank == 0:
            pass

    def on_test_end(self):
        if self.global_rank == 0:
            pass

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

    def on_exception(self, trainer, pl_module, exception):
        # handle exception

        if self.wandb_logger and trainer.is_global_zero:
            if self.wandb_logger.experiment is not None:
                logger.error(
                    f"Attempting to stop the wandb run {self.wandb_logger.experiment}"
                )
                self.wandb_logger.experiment.finish()

    def on_validation_epoch_start(self):
        self.validation_step_outputs = [None for _ in range(len(self.val_dataloader()))]

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_step_outputs[dataloader_idx] = outputs

    def on_validation_epoch_end(self, validation_step_outputs=None):
        validation_step_outputs = self.validation_step_outputs
        for i in range(len(self.val_dataloader())):
            if len(validation_step_outputs) == 1:
                validation_step_output = validation_step_outputs[i]
            log_dict = self.evaluate_outputs(
                validation_step_output,
                self.val_dataloader()[i],
                self.val_dataloader_names[i],
            )
            self.logging_results(log_dict, prefix=self.val_dataloader_names[i])

    def on_test_epoch_start(self):
        self.test_step_outputs = [None for _ in range(len(self.test_dataloader()))]

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_step_outputs[dataloader_idx] = outputs

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        # self.save_HF_model()
        for i in range(len(self.test_dataloader())):
            test_step_output = test_step_outputs[i]

            log_dict = self.evaluate_outputs(
                test_step_output,
                self.test_dataloader()[i],
                self.test_dataloader_names[i],
            )
            self.logging_results(
                log_dict,
                prefix=f"{self.config.test.evaluation_name}_{self.test_dataloader_names[i]}",
            )
        return None

    def evaluate_outputs(
        self,
        step_outputs,
        current_data_loader,
        dataset_name,
    ):
        raise NotImplementedError

    def logging_results(self, log_dict, prefix="test"):
        raise NotImplementedError

    def save_HF_model(self):
        if self.global_rank != 0:
            logger.info("global rank is not 0, skip saving models")
            return
        logger.info("Saving model in the Huggingface format...")
        path_save_model = osp.join(
            self.config.saved_model_path, f"epoch_{self.current_epoch}.pth"
        )
        torch.save(self.model.state_dict(), path_save_model)

        logger.info("Model has been saved to {}".format(path_save_model))

    def forward(self, **kwargs):
        return self.model(**kwargs)

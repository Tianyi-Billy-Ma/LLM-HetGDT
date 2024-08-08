import glob
import json
import logging
import os
import sys
from pprint import pprint

import torch
import torch.distributed as dist
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import wandb
from argument import parse_args
from data_loader_manager import *
from trainers import *
from utils.initialization import get_checkpoint_model_path, initialization
from utils.metrics_log_callback import MetricsHistoryLogger
from utils.seed import set_seed

logger = logging.getLogger(__name__)
sys.dont_write_bytecode = True

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main(args):
    # print("==" * 30 + "\n\n" + "ARGUMENTS:\n\n" + f"{args}\n\n")
    config = initialization(args)
    if config is None:
        raise ValueError("No config file is obtained, exiting...")

    args = config.args

    # pprint(config)

    if config.seed:
        # set_seed(config.seed)
        seed_everything(config.seed, workers=True)
        logger.info(f"All seeds have been set to {config.seed}")

    DataLoaderWrapper = globals()[config.data_loader.type]

    assert (
        DataLoaderWrapper is not None
    ), f"Data Loader {config.data_loader.type} not found"

    data_loader_manager = DataLoaderWrapper(config)

    if config.mode == "create_data":
        data_loader_manager.build_dataset()
        logger.info("Finished building data, exiting main program...")
        return

    tb_logger = TensorBoardLogger(
        save_dir=config.tensorboard_path, name=config.experiment_name
    )

    callback_list = []
    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.saved_model_path,
        # every_n_train_steps=config.train.save_interval,
        save_top_k=config.train.additional.save_top_k,
        monitor=(
            config.train.additional.save_top_k_metric
            if "save_top_k_metric" in config.train.additional.keys()
            else None
        ),
        mode=config.train.additional.save_top_k_mode,
        filename="best",
        save_last=True,
        verbose=False,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )
    callback_list.append(checkpoint_callback)

    # Early Stopping Callback
    if (
        "save_top_k_metric" in config.train.additional.keys()
        and config.train.additional.get("early_stop_patience", 0) > 0
    ):
        early_stop_callback = EarlyStopping(
            monitor=config.train.additional.save_top_k_metric,
            patience=config.train.additional.early_stop_patience,
            verbose=True,
            mode=config.train.additional.save_top_k_mode,
            check_on_train_epoch_end=True,
        )
        callback_list.append(early_stop_callback)

    metrics_history_logger = MetricsHistoryLogger()

    # Get plugins
    plugin_names = config.train.additional.plugins
    plugins = [globals()[plugin_name]() for plugin_name in plugin_names]

    all_loggers = [tb_logger, metrics_history_logger]
    if config.args.disable_wandb_logging:
        # Disable logging wandb tables
        config.args.log_prediction_tables = False
    else:
        # Wandb logger
        logger.info(
            "init wandb logger with the following settings: {}".format(config.WANDB)
        )
        wandb_logger = WandbLogger(config=config, **config.WANDB)
        all_loggers.append(wandb_logger)

    additional_args = {
        "accumulate_grad_batches": config.train.additional.gradient_accumulation_steps,
        "default_root_dir": config.saved_model_path,
        "max_epochs": config.train.epochs,
        "limit_train_batches": (
            2
            if args["limit_train_batches"] is None
            and config.data_loader.dummy_dataloader
            else args["limit_train_batches"]
        ),
        "limit_val_batches": (
            2
            if args["limit_val_batches"] is None and config.data_loader.dummy_dataloader
            else args["limit_val_batches"]
        ),
        "limit_test_batches": (
            2
            if args["limit_test_batches"] is None
            and config.data_loader.dummy_dataloader
            else args["limit_test_batches"]
        ),
        "logger": all_loggers,
        "callbacks": callback_list,
        "plugins": plugins,
        "log_every_n_steps": 1,
        "check_val_every_n_epoch": config.valid.epoch_size,
        "deterministic": False
        # "val_check_interval": config.valid.step_size
        * config.train.additional.gradient_accumulation_steps,  # this is to use global_step as the interval number: global_step * grad_accumulation = batch_idx (val_check_interval is based on batch_idx)
    }

    if args.strategy == "ddp":
        from pytorch_lightning.strategies import DDPStrategy

        pass
        # additional_args["strategy"] = DDPStrategy(find_unused_parameters=True)

    trainer_args = EasyDict(
        accelerator=args.accelerator,
        devices=args.devices,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )
    trainer_args.update(additional_args)

    trainer = Trainer(**vars(trainer_args))
    logger.info(f"arguments passed to trainer: {str(args)}")
    logger.info(f"additional arguments passed to trainer: {str(additional_args)}")

    # Find checkpoints in saved_model_path
    if config.mode == "train":
        checkpoint_to_load = get_checkpoint_model_path(
            saved_model_path=config.saved_model_path,
            load_model_path=config.train.load_model_path,
            load_epoch=config.train.load_epoch,
            load_best_model=config.train.load_best_model,
        )
        if not checkpoint_to_load:
            logger.warning("No checkpoint found. Starting from scratch.")
            checkpoint_to_load = None
    else:
        checkpoint_to_load = get_checkpoint_model_path(
            saved_model_path=config.saved_model_path,
            load_model_path=config.test.load_model_path,
            load_epoch=config.test.load_epoch,
            load_best_model=config.test.load_best_model,
        )
        if not checkpoint_to_load:
            logger.error("No checkpoint found. Please check your config file.")

    # init data loader manager
    data_loader_manager.build_dataset()

    # init train excecutor
    Train_Executor = globals()[config.train.type]
    executor = Train_Executor(config, data_loader_manager)

    if config.mode == "train":
        # After Initialization, save config files
        with open(
            os.path.join(config.experiment_path, "config.jsonnet"), "w"
        ) as config_file:
            save_config = config.copy()
            # save_config.pop('device') # Not serialisable
            json.dump(save_config, config_file, indent=4)
            logger.info(
                f"config file was successfully saved to {config.experiment_path} for future use."
            )
        # Start training
        trainer.fit(
            executor,
            ckpt_path=checkpoint_to_load,
        )
    else:
        # Start testing
        trainer.test(
            executor,
            ckpt_path=checkpoint_to_load if checkpoint_to_load else None,
        )

    if not config.args.disable_wandb_logging:
        logger.info("task finished. finishing wandb process...")
        wandb.finish()


def run(arg_list=None):
    args = parse_args(arg_list)
    experiment_mode = args.mode
    for current_run in range(args.num_runs):
        args.current_run = current_run
        if experiment_mode == "run":
            for mode in ["train", "test"]:
                args.mode = mode
                main(args)
        else:
            main(args)


if __name__ == "__main__":
    arg_list = [
        # "configs/twitter/HGMAE_twitter_split_118.jsonnet",
        # "configs/twitter/MP2Vec_twitter_split_118.jsonnet",
        "configs/twitter/iHGT_twitter_split_118.jsonnet",
        # "configs/twitter/iHGT_twitter_split_217.jsonnet",
        # "configs/twitter/iHGT_twitter_split_415.jsonnet",
        # "configs/twitter/iHGT_twitter_split_226.jsonnet",
        # "configs/twitter/HeCo_twitter_split_118.jsonnet",
        # "configs/twitter/ReWeight_twitter_split_118.jsonnet",
        # "configs/twitter/Smote_twitter_split_118.jsonnet",
        # "configs/twitter/Smote_twitter_split_226.jsonnet",
        # "configs/twitter/OverSampling_twitter_split_118.jsonnet",
        # "configs/twitter/OverSampling_twitter_split_226.jsonnet",
        "--mode",
        "run",
        "--override",
        "--num_runs",
        "5",
        # "--log_prediction_tables",
        # "--disable_wandb_logging",
        "--opts",
        "reset=1",
        "train.additional.early_stop_patience=50",
        # "train.additional.save_top_k_metric=valid/verSampling_twitter.valid/f1_macro",
    ]

    run(arg_list)

import sys

sys.dont_write_bytecode = True

import os
import glob
import wandb

from utils.config_system import process_config
from utils.dirs import reset_folders, create_dirs, reset_wandb_runs


import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter

logger = logging.getLogger(__name__)


def reset_logging():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


def get_checkpoint_model_path(
    saved_model_path, load_epoch=-1, load_best_model=False, load_model_path=""
):
    if load_model_path:
        path_save_model = load_model_path
        if not os.path.exists(path_save_model):
            raise FileNotFoundError("Model file not found: {}".format(path_save_model))
    else:
        if load_best_model:
            file_name = "best.ckpt"
        else:
            if load_epoch == -1:
                file_name = "last.ckpt"
            else:
                file_name = "model_step_{}.ckpt".format(load_epoch)

        path_save_model = os.path.join(saved_model_path, file_name)

        file_names = glob.glob(f"{saved_model_path}/*.ckpt", recursive=True)
        logger.info(f"available checkpoints: {file_names}")

        if not os.path.exists(path_save_model):
            logger.warning(
                "No checkpoint exists from '{}'. Skipping...".format(path_save_model)
            )
            logger.info("**First time to train**")
            return ""  # return empty string to indicate that no model is loaded
        else:
            logger.info("Loading checkpoint from '{}'".format(path_save_model))
    return path_save_model


def initialization(args):
    # Check if the mode is valid
    assert args.mode in ["create_data", "train", "test"]

    # ======================= Process Config =======================
    config = process_config(args)

    # print("==" * 30 + "\n\n" + "CONFIGURATION:\n\n" + f"{config}\n\n")
    if config is None:
        return None
    dirs = [config.log_path]

    dirs += (
        [config.saved_model_path, config.imgs_path, config.tensorboard_path]
        if config.mode == "train"
        else [config.imgs_path, config.results_path]
    )

    delete_confirm = "n"
    if config.reset and config.mode == "train":
        # Reset all the folders
        print("You are deleting following dirs: ", dirs, "input y to continue")
        if config.args.override:
            delete_confirm = "y"
        else:
            delete_confirm = input()
        if delete_confirm == "y":
            reset_folders(dirs)
            # Reset load epoch after reset
            config.train.load_epoch = 0
        else:
            print("reset cancelled.")

    create_dirs(dirs)
    # print("==" * 30 + "\n\n" + "CREATED DIRS:\n\n" + f"{dirs}\n\n")

    # ======================= Setup Logger =======================
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s : %(message)s (in %(pathname)s:%(lineno)d)"
    log_console_format = "[%(levelname)s] - %(name)s : %(message)s"

    # Main logger
    reset_logging()

    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))
    from utils.color_logging import CustomFormatter

    custom_output_formatter = CustomFormatter(custom_format=log_console_format)
    console_handler.setFormatter(custom_output_formatter)

    info_file_handler = RotatingFileHandler(
        os.path.join(config.log_path, "info.log"), maxBytes=10**6, backupCount=5
    )
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(Formatter(log_file_format))

    exp_file_handler = RotatingFileHandler(
        os.path.join(config.log_path, "debug.log"), maxBytes=10**6, backupCount=5
    )
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        os.path.join(config.log_path, "error.log"), maxBytes=10**6, backupCount=5
    )
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(info_file_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)

    # setup a hook to log unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            if wandb.run is not None:
                logger.error(f"Attempting to stop the wandb run {wandb.run}")
                wandb.finish()  # stop wandb if keyboard interrupt is raised
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        logger.error(
            f"Uncaught exception: {exc_type} --> {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        if not config.args.disable_wandb_logging and wandb.run is not None:
            wandb.finish()
            # subprocess.run(["wandb", "sync", "--sync-all"])
            logger.info("Force sync wandb files")

    sys.excepthook = handle_exception

    if not config.args.disable_wandb_logging:
        # setup wandb
        WANDB_CACHE_DIR = config.WANDB.pop("CACHE_DIR")
        if WANDB_CACHE_DIR:
            os.environ["WANDB_CACHE_DIR"] = WANDB_CACHE_DIR
        else:
            os.environ["WANDB_CACHE_DIR"] = ""

        WANDB_DIR = config.WANDB.pop("DIR")
        if WANDB_DIR:
            os.environ["WANDB_DIR"] = WANDB_DIR
        else:
            os.environ["WANDB_DIR"] = ""

        config.WANDB.dir = os.environ["WANDB_DIR"]

        # add base_model as a tag
        config.WANDB.tags.append(config.model_config.base_model)
        # add modules as tags
        config.WANDB.tags.extend(config.model_config.modules)

        all_runs = wandb.Api(timeout=19).runs(
            path=f"{config.WANDB.entity}/{config.WANDB.project}",
            filters={"config.experiment_name": config.experiment_name},
        )

        if config.reset and config.mode == "train" and delete_confirm == "y":
            reset_wandb_runs(all_runs)
            config.WANDB.name = config.experiment_name
        else:
            if len(all_runs) > 0:
                config.WANDB.id = all_runs[0].id
                config.WANDB.resume = "must"
                config.WANDB.name = config.experiment_name
            else:
                config.WANDB.name = config.experiment_name
        if config.experiment_group:
            config.WANDB.group = config.experiment_group
    logger.info(f"Initialization done with the config: {str(config)}")
    return config

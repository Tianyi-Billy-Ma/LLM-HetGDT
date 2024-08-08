import sys

sys.dont_write_bytecode = True


from easydict import EasyDict
import logging

logger = logging.getLogger(__name__)


class DataLoaderWrapper:
    def __init__(self, config):
        self.config = config

        self.data_loaders = EasyDict(
            {
                "train": {},
                "valid": {},
                "test": {},
            }
        )

    def set_io(self, io):
        self.io = io

    def build_dataset(self):
        self.data = EasyDict()
        self.splits = EasyDict()

        dataset_modules = self.config.data_loader.dataset_modules.module_list
        for dataset_module in dataset_modules:
            module_config = self.config.data_loader.dataset_modules.module_dict[
                dataset_module
            ]
            logger.info("Loading dataset module: {}".format(module_config))
            loading_func = getattr(self, dataset_module)
            loading_func(module_config)
            print("data columns: {}".format(self.data.keys()))

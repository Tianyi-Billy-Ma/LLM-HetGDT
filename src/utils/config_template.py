from dataclasses import dataclass
from transformers import PretrainedConfig


@dataclass
class ModelArguments(PretrainedConfig):
    def __init__(self, config, **kwargs):
        for key, value in config.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

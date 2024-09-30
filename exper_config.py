# coding : utf-8
# Author : yuxiang Zeng

from default_config import *
from dataclasses import dataclass


@dataclass
class TestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 100
    device: str = 'cpu'

# coding : utf-8
# Author : yuxiang Zeng
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    logger: str = 'None'


@dataclass
class ExperimentConfig:
    seed: int = 0
    rounds: int = 5
    epochs: int = 1000
    patience: int = 100

    verbose: int = 0
    device: str = 'cuda'
    debug: bool = False
    experiment: bool = False
    program_test: bool = False
    record: bool = True

    train_device: str = 'desktop-cpu-core-i7-7820x-fp32'
    device_name: str = 'core-i7-7820x'

@dataclass
class BaseModelConfig:
    model: str = 'ours'
    rank: int = 40
    retrain: bool = False


@dataclass
class DatasetInfo:
    path: str = './datasets/'
    dataset: str = 'cpu'
    train_size: int = 500
    density: float = 0.80




@dataclass
class TrainingConfig:
    bs: int = 32
    lr: float = 0.001
    decay: float = 0.0001
    loss_func: str = 'L1Loss'
    optim: str = 'AdamW'


@dataclass
class OtherConfig:
    classification: bool = False
    visualize: bool = True
    inductive: bool = False

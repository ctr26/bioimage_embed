from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from hydra import compose, initialize
from omegaconf import OmegaConf
from types import SimpleNamespace
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import albumentations
from dataclasses import dataclass, field
from bioimage_embed.augmentations import (
    DEFAULT_AUGMENTATION_LIST,
    DEFAULT_AUGMENTATION,
    DEFAULT_AUGMENTATION_DICT,
)
import albumentations as A
import os
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# A.compose(DEFAULT_AUGMENTATION_LIST).to_dict()


@dataclass
class Recipe:
    _target_: str = "types.SimpleNamespace"
    opt: str = "adamw"
    weight_decay: float = 0.001
    momentum: float = 0.9
    sched: str = "cosine"
    epochs: int = 50
    lr: float = 1e-4
    min_lr: float = 1e-6
    t_initial: int = 10
    t_mul: int = 2
    lr_min: Optional[float] = None
    decay_rate: float = 0.1
    warmup_lr: float = 1e-6
    warmup_lr_init: float = 1e-6
    warmup_epochs: int = 5
    cycle_limit: Optional[int] = None
    t_in_epochs: bool = False
    noisy: bool = False
    noise_std: float = 0.1
    noise_pct: float = 0.67
    noise_seed: Optional[int] = None
    cooldown_epochs: int = 5
    warmup_t: int = 0


@dataclass
class Transform:
    _target_: str = "albumentations.Compose"
    transforms: dict = field(default_factory=lambda: DEFAULT_AUGMENTATION_DICT)


# @dataclass
# class AlbumentationsTransform:
#     _target_: str = "albumentations.from_dict"
#     transform_dict: dict = field(default_factory=A.from_dict)
#     transform = A.from_dict(OmegaConf.to_container(cfg.albumentations, resolve=True))


@dataclass
class ImageDataset:
    _target_: str = "torchvision.datasets.ImageFolder"
    transform: Transform = field(default_factory=Transform)


@dataclass
class Dataset:
    pass


@dataclass
class DataLoader:
    _target_: str = "bioimage_embed.lightning.dataloader.DataModule"
    dataset: str = field(default_factory=ImageDataset)


@dataclass
class Config:
    recipe: Recipe = field(default_factory=Recipe)
    transform: Transform = field(default_factory=Transform)
    # dataloader: DataLoader = field(default_factory=DataLoader)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def train():
    main(job_name="test_app")


def write_default_config_file(config_path):
    cfg = get_default_config()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as file:
        file.write(OmegaConf.to_yaml(cfg))


# @hydra.main(config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     print(cfg)


def main(config_dir="conf", config_file="config.yaml", job_name="test_app"):
    hydra.initialize(version_base=None, config_path=config_dir, job_name=job_name)
    cfg = hydra.compose(config_name=config_file)
    return cfg


def get_default_config(config_name="config"):
    with initialize(config_path=None, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg

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
import torch

from pydantic import BaseModel, conint, validator
from pydantic.dataclasses import dataclass

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
    root: str = "data"
    transform: Transform = field(default_factory=Transform)
    
    @validator("path")
    def validate_path(cls, root: str) -> Path:
        if Path(root).exists():
            print("exist")
        return Path(root)
    # TODO check if files exist

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



def filter_dataset(dataset: torch.Tensor):
  valid_indices = []
  # Iterate through the dataset and apply the transform to each image
  for idx in range(len(dataset)):
    try:
      image, label = dataset[idx]
      # If the transform works without errors, add the index to the list of valid indices
      valid_indices.append(idx)
    except Exception as e:
      # A better way to do with would be with batch collation
      print(f"Error occurred for image {idx}: {e}")
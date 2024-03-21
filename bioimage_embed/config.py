from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from hydra import compose, initialize
from omegaconf import OmegaConf
from types import SimpleNamespace
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import albumentations
from dataclasses import dataclass, field
from bioimage_embed.augmentations import (
    DEFAULT_AUGMENTATION,
    DEFAULT_ALBUMENTATION,
)
import albumentations as A
import os
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from pydantic import BaseModel, conint, validator
from pydantic.dataclasses import dataclass
from bioimage_embed.lightning import DataModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from .lightning.torch import LitAutoEncoderTorch
from pytorch_lightning.callbacks import EarlyStopping, Callback
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator, root_validator
import numpy as np


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
    data: str = "data"



@dataclass
class ATransform:
    _target_: str = "albumentations.from_dict"
    _convert_: str = "object"
    # _convert_: str = "all"
    transform_dict: Dict = Field(default_factory=lambda: DEFAULT_ALBUMENTATION.to_dict())

@dataclass
class Transform:
    _target_: str = "bioimage_embed.augmentations.VisionWrapper"
    _convert_: str = "object"
    # transform: ATransform = field(default_factory=ATransform)
    transform_dict: Dict = Field(default_factory=lambda: DEFAULT_ALBUMENTATION.to_dict())

   
@dataclass
class Dataset:
    _target_: str = "torch.utils.data.Dataset"
    transform: Transform = Field(default_factory=Transform)
    # root: str = ""
    # @validator("path")
    # def validate_path(cls, root: str) -> Path:
    #     if Path(root).exists():
    #         print("exist")
    #     return Path(root)


@dataclass
class ImageFolderDataset(Dataset):
    _target_: str = "torchvision.datasets.ImageFolder"
    # transform: Transform = Field(default_factory=Transform)
    root: str = "data"

    @validator("path")
    def validate_path(cls, root: str) -> Path:
        if Path(root).exists():
            print("exist")
        return Path(root)


@dataclass
class DataLoader:
    _target_: str = "bioimage_embed.lightning.dataloader.DataModule"
    dataset: Dataset = Field(default_factory=Dataset)
    num_workers: int = 1


@dataclass
class Model:
    _target_: str = "bioimage_embed.models.create_model"
    model: str = "resnet18_vae"
    input_dim: List[int] = Field(default_factory=lambda: [3, 224, 224])
    latent_dim: int = 64
    pretrained: bool = True


@dataclass
class LightningModel:
    _target_: str = "bioimage_embed.lightning.torch.LitAutoEncoderTorch"
    # This should be pythae base autoencoder?
    model: Model = Field(default_factory=Model)
    args: Recipe = Field(default_factory=Recipe)


# ModelCheckpoint(dirpath=f"{self.model_dir}/", save_last=True)
@dataclass
class Trainer:
    _target_: str = "pytorch_lightning.Trainer"
    # logger: Optional[any]
    gradient_clip_val: float = 0.5
    enable_checkpointing: bool = True
    devices: str = "auto"
    accelerator: str = "auto"
    accumulate_grad_batches: int = 16
    # callbacks: List[Callback] = field(default_factory=list)
    min_epochs: int = 50
    max_epochs: int = 50  # Set a default or use a constructor to dynamically set this
    log_every_n_steps: int = 1
    # precision: int = 32

    # callbacks = [EarlyStopping(monitor="loss/val", mode="min")]


# @dataclass
# class Callbacks:
#     _target_: Optional[List[Callback]]
#     early_stopping: EarlyStopping = field(default_factory=EarlyStopping)
#     model_checkpoint: ModelCheckpoint = field(default_factory=ModelCheckpoint)


@dataclass
class EarlyStopping:
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "loss/val"
    mode: str = "min"
    patient: int = 3


@dataclass
class ModelCheckpoint:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    save_last = True
    save_top_k = 1
    monitor = "loss/val"
    mode = "min"


@dataclass
class Paths:
    model: str = "models"
    logs: str = "logs"
    tensorboard: str = "tensorboard"
    wandb: str = "wandb"

    @root_validator(
        pre=False, skip_on_failure=True
    )  # Ensures this runs after all other validations
    @classmethod
    def create_dirs(cls, values):
        # The `values` dict contains all the validated field values
        for path in values.values():
            os.makedirs(path, exist_ok=True)
        return values


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    recipe: Recipe = field(default_factory=Recipe)
    transform: Transform = field(default_factory=Transform)
    dataset: ImageFolderDataset = field(default_factory=ImageFolderDataset)
    dataloader: DataLoader = field(default_factory=DataLoader)
    trainer: Trainer = field(default_factory=Trainer)
    model: Model = field(default_factory=Model)
    lit_model: LightningModel = field(default_factory=LightningModel)
    # # callbacks: Callbacks = field(default_factory=Callbacks)


__schemas__ = {
    "receipe": Recipe,
    "transform": Transform,
    "dataset": Dataset,
    "dataloader": DataLoader,
    "trainer": Trainer,
    "model": Model,
    "lit_model": LightningModel,
}

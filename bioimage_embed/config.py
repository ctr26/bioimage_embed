"""
Configuration for bie works by using pydantic dataclasses to define the
configuration schema. This is then fed into hydra to generate the
configuration files. The configuration schema is defined in the `Config`
class. The `Config` class is a dataclass that contains other dataclasses
as fields. Each of these dataclasses define a specific part of the
configuration schema. This master Config is then wrapped by the BioImageEmbed
Class to run model training and inferece. Config is given sane defaults for
autoencoding Pydantic (in future) will be further used to validate the
configuration schema, so that the configuration files generated by hydra
are valid.

"""

# TODO need a way to copy signatures from original classes for validation
from omegaconf import OmegaConf
from bioimage_embed import augmentations as augs
import os
from dataclasses import field
from pydantic.dataclasses import dataclass

# from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from pydantic import Field
from omegaconf import II
from . import utils


@dataclass(config=dict(extra="allow"))
class Recipe:
    _target_: str = "types.SimpleNamespace"
    model: str = "resnet18_vae"
    batch_size: int = 16
    data: str = "data"
    opt: str = "adamw"
    latent_dim: int = 64
    batch_size: int = 16
    max_epochs: int = 125
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
    seed: int = 42


# Use the ALbumentations .to_dict() method to get the dictionary
# that pydantic can use
@dataclass(config=dict(extra="allow"))
class ATransform:
    _target_: Any = "albumentations.from_dict"
    _convert_: str = "object"
    # _convert_: str = "all"
    transform_dict: Dict = Field(
        default_factory=lambda: augs.DEFAULT_ALBUMENTATION.to_dict()
    )


# VisionWrapper is a helper class for applying albumentations pipelines for image augmentations in autoencoding


@dataclass(config=dict(extra="allow"))
class Transform:
    _target_: Any = "bioimage_embed.augmentations.VisionWrapper"
    _convert_: str = "object"
    # transform: ATransform = field(default_factory=ATransform)
    transform_dict: Dict = Field(
        default_factory=lambda: augs.DEFAULT_ALBUMENTATION.to_dict()
    )


@dataclass(config=dict(extra="allow"))
class Dataset:
    _target_: str = "torch.utils.data.Dataset"
    transform: Any = Field(default_factory=Transform)

    # TODO add validation for transform to be floats
    # @model_validator(mode="after")
    # def validate(self):
    #     dataset = instantiate(self)
    #     return self


@dataclass(config=dict(extra="allow"))
class FakeDataset(Dataset):
    _target_: str = "torchvision.datasets.FakeData"


@dataclass(config=dict(extra="allow"))
class ImageFolderDataset(Dataset):
    _target_: str = "torchvision.datasets.ImageFolder"
    # transform: Transform = Field(default_factory=Transform)
    root: str = II("recipe.data")


@dataclass(config=dict(extra="allow"))
class NdDataset(ImageFolderDataset):
    transform: Transform = Field(default_factory=Transform)


@dataclass(config=dict(extra="allow"))
class TiffDataset(NdDataset):
    _target_: str = "bioimage_embed.datasets.TiffDataset"


class NgffDataset(NdDataset):
    _target_: str = "bioimage_embed.datasets.NgffDataset"


@dataclass(config=dict(extra="allow"))
class DataLoader:
    _target_: str = "bioimage_embed.lightning.dataloader.DataModule"
    dataset: Any = Field(default_factory=FakeDataset)
    num_workers: int = 1
    batch_size: int = II("recipe.batch_size")


@dataclass(config=dict(extra="allow"))
class Model:
    _target_: Any = "bioimage_embed.models.create_model"
    model: str = II("recipe.model")
    input_dim: List[int] = Field(default_factory=lambda: [3, 224, 224])
    latent_dim: int = II("recipe.latent_dim")
    pretrained: bool = True


@dataclass(config=dict(extra="allow"))
class Callback:
    pass


@dataclass(config=dict(extra="allow"))
class EarlyStopping(Callback):
    _target_: Any = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "loss/val"
    mode: str = "min"
    patience: int = 3


@dataclass(config=dict(extra="allow"))
class ModelCheckpoint(Callback):
    _target_: Any = "pytorch_lightning.callbacks.ModelCheckpoint"
    save_last = True
    save_top_k = 1
    monitor = "loss/val"
    mode = "min"
    # dirpath: str = Field(default_factory=lambda: utils.hashing_fn(Recipe()))
    dirpath: str = f"{II('paths.model')}/{II('uuid')}"


@dataclass(config=dict(extra="allow"))
class LightningModel:
    _target_: str = "bioimage_embed.lightning.torch.AEUnsupervised"
    # This should be pythae base autoencoder?
    model: Any = Field(default_factory=Model)
    args: Any = Field(default_factory=lambda: II("recipe"))


class LightningModelSupervised(LightningModel):
    _target_: str = "bioimage_embed.lightning.torch.AESupervised"


@dataclass(config=dict(extra="allow"))
class Callbacks:
    # _target_: str = "collections.OrderedDict"
    model_checkpoint: Any = Field(default_factory=ModelCheckpoint)
    # early_stopping: Any = Field(default_factory=EarlyStopping)



@dataclass(config=dict(extra="allow"))
class Trainer:
# class Trainer(pytorch_lightning.Trainer):
    _target_: Any = "pytorch_lightning.Trainer"
    logger: Any = None
    gradient_clip_val: float = 0.5
    enable_checkpointing: bool = True
    devices: Any = "auto"
    accelerator: str = "auto"
    accumulate_grad_batches: int = 16
    min_epochs: int = 1
    max_epochs: int = II("recipe.max_epochs")
    num_nodes: int = 1
    log_every_n_steps: int = 1
    # This is not a clean implementation but I am not sure how to do it better
    callbacks: Any = Field(
        default_factory=lambda: list(vars(Callbacks()).values()), frozen=True
    )
    # TODO idea here would be to use pydantic to validate omegaconf

# TODO add argument caching for checkpointing


@dataclass(config=dict(extra="allow"))
class Paths:
    model: str = "models"
    logs: str = "logs"
    tensorboard: str = "tensorboard"
    wandb: str = "wandb"

    def __post_init__(self):
        for path in self.__dict__.values():
            os.makedirs(path, exist_ok=True)


@dataclass(config=dict(extra="allow"))
class Config:
    # This has to be dataclass.field instead of pydantic Field for somereason
    paths: Any = field(default_factory=Paths)
    recipe: Any = field(default_factory=Recipe)
    dataloader: Any = field(default_factory=DataLoader)
    trainer: Any = field(default_factory=Trainer)
    lit_model: Any = field(default_factory=LightningModel)
    callbacks: Any = field(default_factory=Callbacks)
    uuid: str = field(default_factory=lambda: utils.hashing_fn(Recipe()))


@dataclass(config=dict(extra="allow"))
class SupervisedConfig(Config):
    lit_model: LightningModel = field(default_factory=LightningModel)


__schemas__ = {
    "recipe": Recipe,
    "transform": Transform,
    "dataset": FakeDataset,
    "dataloader": DataLoader,
    "trainer": Trainer,
    "model": Model,
    "lit_model": LightningModel,
}


def resolve_schema(schema):
    cfg = OmegaConf.structured(Config())
    schema = OmegaConf.structured(schema, parent=cfg)
    return schema


def resolve_config(cfg):
    """
    Resolves the config using omegaconf,
    without the flag this will crash with mixed types
    """
    ocfg = OmegaConf.structured(cfg, flags={"allow_objects": True})
    OmegaConf.resolve(ocfg)
    return ocfg

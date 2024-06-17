from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from omegaconf import OmegaConf
import hydra
from dataclasses import field
from bioimage_embed.augmentations import DEFAULT_AUGMENTATION_LIST
import albumentations as A
import os


@dataclass
class Receipe:
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
    lr_min: float = None
    decay_rate: float = 0.1
    warmup_lr: float = 1e-6
    warmup_lr_init: float = 1e-6
    warmup_epochs: int = 5
    cycle_limit: int = None
    t_in_epochs: bool = False
    noisy: bool = False
    noise_std: float = 0.1
    noise_pct: float = 0.67
    noise_seed: int = None
    cooldown_epochs: int = 5
    warmup_t: int = 0


@dataclass
class Transform:
    _target_: str = "albumentations.Compose"
    transforms: A.Compose = field(default_factory=A.Compose(DEFAULT_AUGMENTATION_LIST))


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


# def cs_generator():
cs = ConfigStore.instance()
cs.store(name="receipe", node=Receipe)
cs.store(name="dataloader", node=DataLoader)


# return cs
def train():
    main(job_name="test_app")


def write_default_config_file(config_path, config_filename, config):
    os.makedirs(config_path, exist_ok=True)
    with open(os.path.join(config_path, config_filename), "w") as file:
        file.write(OmegaConf.to_yaml(config))


def main(config_path="conf", job_name="test_app"):
    config_file = os.path.join(config_path, "config.yaml")

    # Check if the configuration directory exists, if not, create it
    if not os.path.exists(config_path):
        os.makedirs(config_path)
        # Initialize Hydra with a basic configuration
        hydra.initialize(version_base=None, config_path=config_path, job_name=job_name)
        cfg = hydra.compose(config_name="config")
        # Save the default configuration
        with open(config_file, "w") as file:
            file.write(OmegaConf.to_yaml(cfg))
    else:
        # Initialize Hydra normally if the configuration directory exists
        hydra.initialize(version_base=None, config_path=config_path, job_name=job_name)
        cfg = hydra.compose(config_name="config")

    print(OmegaConf.to_yaml(cfg))

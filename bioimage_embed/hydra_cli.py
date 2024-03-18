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

from .data_model import Config

# A.compose(DEFAULT_AUGMENTATION_LIST).to_dict()

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

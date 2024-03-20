# from .hydra_cli import train, infer
# from typer import Typer

# app = Typer()
# app.command()(train)
# app.command()(infer)

# def main():
#     app()
    
# if __name__ == "__main__":
#     main()

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
import bioimage_embed
from pydantic import BaseModel, conint, validator
from pydantic.dataclasses import dataclass
from bioimage_embed.lightning import DataModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from .lightning.torch import LitAutoEncoderTorch
from pytorch_lightning.callbacks import EarlyStopping, Callback
from typing import List, Optional
from . import config


cs = ConfigStore.instance()
cs.store(name="config", node=config.Config)

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
import bioimage_embed
from pydantic import BaseModel, conint, validator
from pydantic.dataclasses import dataclass
from bioimage_embed.lightning import DataModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from .lightning.torch import LitAutoEncoderTorch
from pytorch_lightning.callbacks import EarlyStopping, Callback
from typing import List, Optional
from .config import Config, Recipe, Transform
from torch.autograd import Variable
from hydra.utils import instantiate



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
        return torch.utils.data.Subset(dataset, valid_indices)

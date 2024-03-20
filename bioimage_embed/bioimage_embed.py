import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.model_selection import train_test_split
import logging
from .config import Config
from bioimage_embed.lightning import DataModule
from .lightning.torch import LitAutoEncoderTorch
from hydra.utils import instantiate
from torch.autograd import Variable

# Environment Setup
# torch.multiprocessing.set_sharing_strategy("file_system")
logging.basicConfig(level=logging.INFO)


class BioImageEmbed:
    cfg: Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        self.recipe = instantiate(cfg.recipe)
        # self.model_dir = "models/"
        self.lit_model = instantiate(cfg.lit_model, model=cfg.model, args=cfg.recipe)
        # self.logger = initialize(cfg.logger)
        self.dataset = instantiate(
            cfg.dataset,
            root=cfg.recipe.data,
            transform=cfg.transform,
        )
        self.dataloader = instantiate(cfg.dataloader, dataset=cfg.dataset)
        # self.callbacks = initialize(cfg.callbacks)
        self.callbacks = None
        # self.trainer = instantiate(cfg.trainer, callbacks=[self.callbacks])
        self.trainer = instantiate(cfg.trainer)

    # TODO fix resume

    def train(self, resume: bool = True):
        self.model.eval()
        try:
            self.trainer.fit(
                self.lit_model,
                datamodule=self.dataloader,
                ckpt_path=f"{self.cfg.paths.model}/last.ckpt",
            )
        except:
            self.trainer.fit(self.lit_model, datamodule=self.dataloader)

    def validate(self):
        validation = self.trainer.validate(self.lit_model, datamodule=self.dataloader)
        # testing = trainer.test(lit_model, datamodule=dataloader)

    def __call__(self, x):
        return self.lit_model(x)

    def infer(self):
        dataloader = DataModule(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.receipe.num_workers,
            # TODO: Add transform here if batch_size > 1 assuming averaging?
            # Transform is commented here to avoid augmentations in real data
            # HOWEVER, applying a the transform multiple times and averaging the results might produce better latent embeddings
            # transform=transform,
        )
        dataloader.setup()
        predictions = self.trainer.predict(self.lit_model, datamodule=dataloader)
        return predictions

    def export(self):
        example_input = Variable(torch.rand(1, *self.cfg.recipe.input_dim))


# class BioImageEmbed:
#     def __init__(self, cfg: Config):
#         self.cfg = cfg

#     def load_data(self):
#         self.dataloader.setup()

#     def train(self):
#         lit_model = LitAutoEncoderTorch(self.model, self.recipe)
#         self.model.eval()
#         checkpoint_callback = ModelCheckpoint(dirpath=f"{model_dir}/", save_last=True)

#     # def get_model_path(self):
#     #       model_dir = f"my_models/{dataset_path}_{model._get_name()}_{lit_model._get_name()}"

#     # pass

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

        # self.logger = initialize(cfg.logger)
        # a_wrapper = lambda x: self.albumentation(image=x)["image"]
        cfg.dataset.root = cfg.recipe.data
        cfg.dataset.transform = cfg.transform

        cfg.lit_model.model = cfg.model
        cfg.lit_model.args = cfg.recipe
        # self.albumentation = instantiate(cfg.transform)
        # TODO this maybe a hack to get albumentations to work
        # I think this can be done in the original dataclass
        # self.transform_dict = instantiate(cfg.transform.transform_dict, _convert_="object")

        self.recipe = instantiate(cfg.recipe)
        self.dataloader = instantiate(cfg.dataloader, dataset=cfg.dataset)
        # self.callbacks = initialize(cfg.callbacks)
        self.callbacks = None
        self.trainer = instantiate(cfg.trainer)
        self.paths = instantiate(cfg.paths)
        self.lit_model = instantiate(cfg.lit_model)
        self.setup()

    def setup(self):
        self.make_dirs()
        self.dataloader.setup()
        self.lit_model.model.eval()

    def model_check(self):
        data = self.dataloader.dataset[0]
        dataloader_0 = next(iter(self.dataloader.train_dataloader()))
        output = self.lit_model(dataloader_0)
        logging.info("Model Check Passed")

    def trainer_check(self):
        trainer = instantiate(self.cfg.trainer, fast_dev_run=True)
        trainer.fit(self.lit_model, self.dataloader)
        logging.info("Trainer Check Passed")

    # TODO fix resume
    def make_dirs(self):
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def train(self, resume: bool = True):
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

import torch

torch.cuda.empty_cache()
# from . import models, lightning, cli, export, config
from .lightning import AESupervised, AEUnsupervised, AE,  AutoEncoderSupervised, AutoEncoderUnsupervised, AutoEncoder

# TODO: Fix this import as it currently produces too many warnings
from .models import ModelFactory, create_model
from .bie import BioImageEmbed
from .config import Config
from . import augmentations

import logging
logging.captureWarnings(True)

__all__ = [
    "AESupervised",
    "AutoEncoderUnsupervised",
    "AEUnsupervised",
    "AutoEncoderSupervised",
    "AutoEncoder"
    "AE"
    "BioImageEmbed",
    "Config",
    "augmentations",
]

import bioimage_embed
import bioimage_embed.config as config
# from ray.tune.integration.pytorch_lightning import (
#     TuneReportCallback,
#     TuneReportCheckpointCallback,

# )
import albumentations as A  
from types import SimpleNamespace
from ray import tune
import numpy as np
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from hydra.utils import instantiate
import ray
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
import os
import glob
from PIL import Image
from typing import List
from torch.utils.data import Dataset
import torch
from joblib import Memory
from pydantic.dataclasses import dataclass
from pytorch_lightning import loggers as pl_loggers
params = {
        "model": "resnet50_vqvae",
        # "data": "data",
        "opt": "adamw",
        "max_epochs": 1000,
        "max_steps": -1,
        "weight_decay":0.0001,
        "momentum": 0.9,
        # "sched": "cosine",
        "epochs": 1000,
        "lr": 1e-3,
        "batch_size": 16,
    }
memory = Memory(location='.', verbose=0)

@memory.cache
def get_file_list(glob_str):
    return glob.glob(os.path.join(glob_str), recursive=True)


class GlobDataset(Dataset):
    def __init__(self, glob_str,transform=None):
        self.file_list = get_file_list(glob_str)
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.file_list[idx]
        image = Image.open(img_name)
        # breakpoint()
        image = np.array(image)
        if self.transform:
            # t = A.Compose([A.ToRGB(),transform, A.RandomCrop(224,224)]) 
            t = A.Compose([A.ToRGB(),self.transform])
            image = t(image=image)

        # breakpoint()
        # sample = {'image': image, 'path': img_name}

        return image["image"], 0

root_dir = '/nfs/ftp/public/databases/IDR/idr0093-mueller-perturbation/'
root_dir = '/nfs/research/uhlmann/ctr26/idr/idr0093-mueller-perturbation/'

if __name__ == "__main__":
    print("training")
    input_dim = [3, 224, 224]
    
    # mock_dataset = config.ImageFolderDataset(
    #     _target_="bioimage_embed.datasets.FakeImageFolder",
    #     image_size=input_dim,
    #     num_classes=1,
    # )
    # breakpoint()
    transform = instantiate(config.ATransform())
    dataset = GlobDataset(root_dir+'**/*.tif*',transform)
    dataloader = config.DataLoader(dataset=dataset,num_workers=32)

    assert instantiate(dataloader,batch_size=1)
    assert dataset[0]

    model = config.Model(input_dim=input_dim)

    lit_model = config.LightningModel(
        _target_="bioimage_embed.lightning.torch.AutoEncoderSupervised",
        model=model
    )

    wandb = pl_loggers.WandbLogger(project="idr", name="0093")
    trainer = config.Trainer(
        accelerator="auto",
        devices=1,
        num_nodes=1,
        # strategy="ddp",
        callbacks=[],
        plugin=[],
        logger=[wandb],
        )
    
    cfg = config.Config(
        dataloader=dataloader,
        lit_model=lit_model,
        trainer=trainer,
        recipe=config.Recipe(**params),
    )
    # breakpoint()
    
    bie = bioimage_embed.BioImageEmbed(cfg)
    wandb.watch(bie.icfg.lit_model, log="all")
    
    bie.train()
    wandb.finish()

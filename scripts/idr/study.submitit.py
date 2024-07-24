import bioimage_embed
import bioimage_embed.config as config
import wandb
from pytorch_lightning import LightningModule, Trainer
import albumentations as A  
from types import SimpleNamespace
from ray import tune
import numpy as np
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from hydra.utils import instantiate
import os
import glob
from PIL import Image
from typing import List
from torch.utils.data import Dataset
import torch
from joblib import Memory
from pydantic.dataclasses import dataclass
from pytorch_lightning import loggers as pl_loggers
import submitit
import os 

NUM_GPUS_PER_NODE = 1
NUM_NODES = 1


params = {
    "model": "resnet50_vqvae",
    # "data": "data",
    "opt": "lamb",
    "latent_dim": 224**2//4,
    "max_epochs": 1000,
    "max_steps": -1,
    "weight_decay": 0.0001,
    "momentum": 0.9,
    # "sched": "cosine",
    "epochs": 1000,
    "lr": 1e-3,
    "batch_size": 16,
    "sched": "cosine",
    }
memory = Memory(location='.', verbose=0)

@memory.cache
def get_file_list(glob_str):
    return glob.glob(os.path.join(glob_str), recursive=True)

def collate_fn(batch):
    # Filter out None values
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

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
        try:
            image = Image.open(img_name)
        except:
            return None,None
        # breakpoint()
        image = np.array(image)
        if self.transform:
            # t = A.Compose([A.ToRGB(),transform, A.RandomCrop(224,224)]) 
            t = A.Compose([A.ToRGB(),self.transform])
            image = t(image=image)

        # breakpoint()

        return image["image"], 0

root_dir = '/nfs/ftp/public/databases/IDR/idr0093-mueller-perturbation/'
root_dir = '/nfs/research/uhlmann/ctr26/idr/idr0093-mueller-perturbation/'



def train(num_gpus_per_node=1,num_nodes=1):

    print("training")
    input_dim = [3, 224, 224]
    
    # mock_dataset = config.ImageFolderDataset(
    #     _target_="bioimage_embed.datasets.FakeImageFolder",
    #     image_size=input_dim,
    #     num_classes=1,
    # )
    
    transform = instantiate(config.ATransform())
    dataset = GlobDataset(root_dir+'**/*.tif*',transform)
    # dataset = RandomDataset(32, 64)
    dataloader = config.DataLoader(dataset=dataset,num_workers=os.cpu_count(),collate_fn=collate_fn)

    assert instantiate(dataloader,batch_size=1)
    assert dataset[0]

    model = config.Model(input_dim=input_dim)
    
    lit_model = config.LightningModel(
        # _target_="bioimage_embed.lightning.torch.AutoEncoderSupervised",
        model=model
    )
    wandb = pl_loggers.WandbLogger(project="idr", name="0093",log_model="all")
    trainer = config.Trainer(
        accelerator="auto",
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        strategy="ddp",
        callbacks=[],
        # plugin=[],
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
    # wandb.watch(bie.icfg.lit_model, log="all")
    # wandb.run.define_metric("mse/val", summary="best")
    # wandb.run.define_metric("loss/val.loss", summary="best")
    
    bie.train()
    wandb.finish()

def main():
    logdir = "lightning_slurm/"
    os.makedirs(logdir, exist_ok=True)

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=logdir)
    executor.update_parameters(
        mem_gb=2 * 32 * 4,  # 2GB per CPU, 32 CPUs per task, 4 tasks per node
        timeout_min=1440*2,   # 48 hours
        # slurm_partition="your_partition_name",  # Replace with your partition name
        gpus_per_node=NUM_GPUS_PER_NODE,
        tasks_per_node=1,
        cpus_per_task=8,
        nodes=NUM_NODES,
        slurm_constraint="a100",
    )
    job = executor.submit(train, NUM_GPUS_PER_NODE, NUM_NODES)

if __name__ == "__main__":
    main()
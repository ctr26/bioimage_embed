# %%
from pathlib import Path

#  %%
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

# Note - you must have torchvision installed for this example
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from bioimage_embed.lightning import DatamoduleGlob

from bioimage_embed.datasets import DatasetGlob
from bioimage_embed.lightning import AutoEncoderUnsupervised
import matplotlib.pyplot as plt
from pythae.models import VAE, VAEConfig


max_epochs = 2

window_size = 128 * 2
batch_size = 128
num_training_updates = 15000

num_hiddens = 64
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3
num_workers = 8
data_samples = 128  # Set to -1 for all images
dataset = "idr0093"
data_dir = "data"
train_dataset_glob = f"{data_dir}/**/*{dataset}*/**/*tif"


train_dataset = DatasetGlob(train_dataset_glob, samples=data_samples)


transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine((0, 360)),
        transforms.RandomResizedCrop(size=window_size),
        transforms.ToTensor(),
    ]
)


train_dataset = DatasetGlob(
    train_dataset_glob, transform=transform, samples=data_samples
)


plt.imshow(train_dataset[10][0], cmap="gray")
dataloader = DatamoduleGlob(
    train_dataset_glob,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    transform=transform,
    pin_memory=True,
    persistent_workers=True,
)

model = VAE(
    model_config=VAEConfig(input_dim=(1, window_size, window_size), latent_dim=10),
)

model_name = model._get_name()
model_dir = f"models/{dataset}_{model_name}"

# %%
lit_model = AutoEncoderUnsupervised(model)

tb_logger = pl_loggers.TensorBoardLogger(f"{model_dir}/runs/")

Path(f"{model_dir}/").mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(dirpath=f"{model_dir}/", save_last=True)

trainer = pl.Trainer(
    logger=tb_logger,
    accelerator="gpu",
    devices=1,
    accumulate_grad_batches=1,
    min_epochs=50,
    max_epochs=max_epochs,
    profiler="simple",
)

trainer.fit(lit_model, datamodule=dataloader)

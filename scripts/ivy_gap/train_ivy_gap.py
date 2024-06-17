# %%
from pathlib import Path

from bioimage_embed.models import create_model
import albumentations as A

import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Note - you must have torchvision installed for this example
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from bioimage_embed.datasets import DatasetGlob
from bioimage_embed.lightning import DatamoduleGlob, AutoEncoderUnsupervised
from types import SimpleNamespace

Image.MAX_IMAGE_PIXELS = None

dataset = "ivy_gap"
data_dir = "data"
train_dataset_glob = f"{data_dir}/{dataset}/random/*png"


params = {
    "epochs": 500,
    "batch_size": 2**4,
    "num_workers": 2**4,
    "window_size": 256,
    "channels": 3,
    "latent_dim": 64,
    "num_embeddings": 512,
}

optimizer_params = {
    "opt": "LAMB",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "momentum": 0.9,
}

lr_scheduler_params = {
    "sched": "cosine",
    "min_lr": 1e-6,
    "warmup_epochs": 5,
    "warmup_lr": 1e-6,
    "cooldown_epochs": 10,
    "t_max": 50,
    "cycle_momentum": False,
}
args = SimpleNamespace(**params, **optimizer_params, **lr_scheduler_params)
channels = 3

input_dim = (args.channels, args.window_size, args.window_size)

transform = A.Compose(
    [
        # Flip the images horizontally or vertically with a 50% chance
        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=0.5,
        ),
        A.RandomCrop(
            height=args.window_size,
            width=args.window_size,
            p=1,
        ),
        # Rotate the images by a random angle within a specified range
        A.Rotate(limit=45, p=0.5),
        # Randomly scale the image intensity to adjust brightness and contrast
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        # Apply random elastic transformations to the images
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.5,
        ),
        # Shift the image channels along the intensity axis
        # Add a small amount of noise to the images
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # Crop a random part of the image and resize it back to the original size
        A.RandomCrop(
            height=224,
            width=224,
            p=1,
        ),
        A.ToFloat(max_value=1, p=1.0),
        ToTensorV2(),
    ]
)

train_dataset = DatasetGlob(train_dataset_glob, transform=transform)

# plt.imshow(train_dataset[100][0], cmap="gray")
# plt.show()
# plt.close()

# %%
dataloader = DatamoduleGlob(
    train_dataset_glob,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    transform=transform,
    pin_memory=True,
    persistent_workers=True,
    over_sampling=16,
)


model = create_model("resnet18_vqvae", input_dim=input_dim, latent_dim=args.latent_dim)
lit_model = AutoEncoderUnsupervised(model, args)

dataloader.setup()
model.eval()


model_name = model._get_name()
model_dir = f"models/{dataset}_{model_name}"

lit_model = AutoEncoderUnsupervised(model, args)

tb_logger = pl_loggers.TensorBoardLogger(f"{model_dir}/runs/")

Path(f"{model_dir}/").mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(dirpath=f"{model_dir}/", save_last=True)

trainer = pl.Trainer(
    logger=tb_logger,
    gradient_clip_val=0.5,
    enable_checkpointing=True,
    devices="auto",
    accelerator="gpu",
    accumulate_grad_batches=4,
    callbacks=[checkpoint_callback],
    min_epochs=50,
    max_epochs=args.epochs,
)

# %%

try:
    trainer.fit(lit_model, datamodule=dataloader, ckpt_path=f"{model_dir}/last.ckpt")
except:
    trainer.fit(lit_model, datamodule=dataloader)

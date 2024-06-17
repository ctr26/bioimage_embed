"""
train.py is entrypoint for docker container.
"""

import logging
import argparse
from torchvision import transforms
from google.cloud import storage
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from bioimage_embed_training.bioimage_embed.models import BioimageEmbed
from bioimage_embed_training.bioimage_embed.lightning.torch import (
    LitAutoEncoderTorch,
)
import json

# Taking configurations from python file:
logging.basicConfig(level=logging.INFO)
logging.info("Picking up config file from storage bucket")
storage_client = storage.Client()
bucket = storage_client.get_bucket("q_vertex_ai_dev")
blob = bucket.blob("config.py")
with open("cfg.py", "wb") as f:
    blob.download_to_file(f)

import cfg

# Initailizing parameters
PROJECT_ID = "prj-ext-dev-mlops-bia-363210"
LOCATION = "europe-west2"
SERIES = "01"
EXPERIMENT = "exp-dummy"
FRAMEWORK = "pytorch-lightning"
TASK = "compress-image"
MODEL_TYPE = "vq-vae"
RUN_NAME = "run-dummy"
num_residual_hiddens = 8
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
# time_stamp = 'TIMESTAMP'  # Replace with the actual value


parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--time_stamp", dest="time_stamp", required=True, type=str, help="TIMESTAMP"
# )


if cfg.mode == "hpt":
    logging.info(f"Mode of the job is {cfg.mode}")
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        required=True,
        type=float,
        help="Learning Rate",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        required=True,
        type=int,
        help="batch_size",
    )


elif cfg.mode == "scale":
    logging.info(f"Mode of the job is {cfg.mode}")
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate  # Not in use

args = parser.parse_args()
max_epochs = 15
num_hiddens = 64
commitment_cost = 0.25
decay = 0.99
num_workers = 16
data_samples = 100

model_name = "VQ_VAE"
dataset = "idr0093-mueller-perturbation"
data_dir = "ebi-bia"
train_dataset_glob = {"dataset": dataset, "data_dir": data_dir}
pin_memory = True
model_dir = f"models/{model_name}_{SERIES}"

import torch
from PIL import Image

test_img = Image.new("RGB", (1024, 1024))
transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine((0, 360)),
        transforms.RandomResizedCrop(size=512),
        # transforms.RandomCrop(size=(512,512)),
        # transforms.GaussianBlur(5),
        transforms.ToTensor(),
        # transforms.Normalize((0.485), (0.229)),
    ]
)

test_tensor = transform(test_img).unsqueeze(0)
# try:
if cfg.mode == "hpt":
    # dataloader = DatamoduleGlob(
    #     train_dataset_glob,
    #     batch_size=args.batch_size,
    #     shuffle=cfg.shuffle,
    #     num_workers=num_workers,
    #     transform=transform,
    #     pin_memory=pin_memory,
    #     samples=data_samples,
    #     timeout=cfg.timeout,
    #     persistent_workers=cfg.persistent_workers,
    # )
    ## Hyperparameterize models i.e., VQ_VAE, VAE
    model = BioimageEmbed(
        "VQ_VAE",
        channels=1,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
    )

    lit_model = LitAutoEncoderTorch(
        model, batch_size=args.batch_size, learning_rate=args.learning_rate
    )

elif cfg.mode == "scale":
    # dataloader = DatamoduleGlob(
    #     train_dataset_glob,
    #     batch_size=batch_size,
    #     shuffle=cfg.shuffle,
    #     num_workers=num_workers,
    #     transform=transform,
    #     pin_memory=pin_memory,
    #     samples=data_samples,
    #     timeout=cfg.timeout,
    #     persistent_workers=cfg.persistent_workers,
    # )
    model = BioimageEmbed(
        "VQ_VAE",
        channels=1,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
    )

    lit_model = LitAutoEncoderTorch(
        model, batch_size=batch_size, learning_rate=learning_rate
    )

# logging.info(f"length of dataloader dataset: {len(dataloader.dataset)-1}")

# gs_path = f"gs://q_vertex_ai_dev/{model_dir}/{args.time_stamp}_custom_job/"
gs_path = "models/20230222161426_custom_job"
logger_version = cfg.tb_logger
# logging.info(f"logging_path : {gs_path}{cfg.CUSTOM_JOB_NAME}/{logger_version}")

tb_logger = pl_loggers.TensorBoardLogger(
    gs_path, log_graph=True, version=logger_version, name=cfg.CUSTOM_JOB_NAME
)

# Enabling Earlystopping and ModelCheckpoint
early_stopping = EarlyStopping(
    monitor="train_loss",
    mode="min",
    patience=cfg.patience,
    verbose=True,
    log_rank_zero_only=True,
)
checkpoint_callback = ModelCheckpoint(dirpath=gs_path, verbose=True, save_last=True)

callbacks = [early_stopping, checkpoint_callback]

# trainer = pl.Trainer(
#     devices=cfg.devices,
#     accelerator=cfg.accelerator,
#     strategy=cfg.strategy,
#     num_nodes=cfg.num_nodes,
#     precision=cfg.precision,
#     logger=tb_logger,
#     enable_checkpointing=cfg.enable_checkpointing,
#     accumulate_grad_batches=cfg.accumulate_grad_batches,
#     callbacks=callbacks,
#     min_epochs=cfg.min_epochs,
#     max_epochs=max_epochs,
#     log_every_n_steps=cfg.log_every_n_steps,
#     detect_anomaly=cfg.detect_anomaly,
#     check_val_every_n_epoch=cfg.check_val_every_n_epoch,
#     gradient_clip_val=cfg.gradient_clip_val,
# )

lit_model.load_from_checkpoint("models/20230222161426_custom_job/last.ckpt")
model.eval()

import torch.jit as jit

# Trace the model using torch.jit.trace
traced_model = jit.trace(model, test_tensor)

# Save the traced model to a file
script_path = "model.pt"
torch.jit.save(traced_model, script_path)


try:
    trainer.fit(lit_model, datamodule=dataloader)
    # ckpt_path=f"gs://q_vertex_ai_dev/{model_dir}/custom_job/last.ckpt")
    Logged_Metrics = trainer.logged_metrics
    Callback_Metrics = trainer.callback_metrics
except:
    trainer.fit(lit_model, datamodule=dataloader)
    Logged_Metrics = trainer.logged_metrics
    Callback_Metrics = trainer.callback_metrics


logging.info(f"Logged metrics: {Logged_Metrics}")
logging.info(f"Callback metrics: {Callback_Metrics}")
log_metric_loss = float(Callback_Metrics["train_loss"].numpy())
ssim_metric = float(Callback_Metrics["train_ssim_score"].numpy())

logging.info(f"log_metric_loss: {log_metric_loss}")
logging.info(f"ssim_metric: {ssim_metric}")

if cfg.mode == "hpt":
    # hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="train_loss", metric_value=log_metric_loss
    )
    file_path = f"{model_dir}/{args.time_stamp}_custom_job/{cfg.CUSTOM_JOB_NAME}/loss_metric.txt"
    file_name = file_path.split("/")[-1]
    loss = {"ssim_score": ssim_metric}
    logging.info(f"Metric returned to file: gs://q_vertex_ai_dev/{file_path}")
    blob = bucket.blob(file_path)
    blob.upload_from_string(data=json.dumps(loss), content_type="application/json")
    logging.info(f"Metric is saved: {file_name}")

elif cfg.mode == "scale":
    file_path = f"{model_dir}/{args.time_stamp}_custom_job/{cfg.CUSTOM_JOB_NAME}/loss_metric.txt"
    file_name = file_path.split("/")[-1]
    loss = {"train_loss": log_metric_loss, "ssim_score": ssim_metric}
    logging.info(f"Metric returned to file: gs://q_vertex_ai_dev/{file_path}")
    blob = bucket.blob(file_path)
    blob.upload_from_string(data=json.dumps(loss), content_type="application/json")
    logging.info(f"Metric is saved: {file_name}")

logging.info("Job is completed")

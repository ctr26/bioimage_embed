import torchvision
import torch
import pytorch_lightning as pl
from timm import optim, scheduler
from types import SimpleNamespace
import argparse
from transformers.utils import ModelOutput
import torch.nn.functional as F
from monai import losses

"""
x_recon -> output of the model
z -> latent space
data -> input to the model
target -> target for supervised learning
recon_loss -> reconstruction loss
loss -> total loss
variational_loss -> loss - recon_loss
"""


class AutoEncoder(pl.LightningModule):
    args = argparse.Namespace(
        opt="adamw",
        weight_decay=0.001,
        momentum=0.9,
        sched="cosine",
        epochs=50,
        lr=1e-4,
        min_lr=1e-6,
        t_initial=10,
        t_mul=2,
        lr_min=None,
        decay_rate=0.1,
        warmup_lr=1e-6,
        warmup_lr_init=1e-6,
        warmup_epochs=5,
        cycle_limit=None,
        t_in_epochs=False,
        noisy=False,
        noise_std=0.1,
        noise_pct=0.67,
        noise_seed=None,
        cooldown_epochs=5,
        warmup_t=0,
    )

    def __init__(self, model, args=SimpleNamespace()):
        super().__init__()
        self.model = model
        self.model = self.model.to(self.device)
        # Flatten hparams
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        if args:
            self.args = SimpleNamespace(**{**vars(args), **vars(self.args)})
        self.save_hyperparameters(vars(self.args))
        # TODO update all models to use this for export to onxx
        # self.example_input_array = torch.randn(1, *self.model.input_dim)
        # self.model.train()
        # keep a handle on metrics logged by the model
        self.metrics = {}

    def forward(self, x):
        batch = ModelOutput(data=x.float())
        return self.model(batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = self.batch_to_xy(batch)
        model_output = self.forward(x)
        model_output.data = x
        model_output.target = y
        return model_output

    # Function is redundant ?
    def training_batch(self, batch, batch_idx):
        x, y = batch
        return ModelOutput(data=x.float(), target=y)

    def embedding(self, model_output: ModelOutput):
        return model_output.z.view(model_output.z.shape[0], -1)

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss, model_output = self.eval_step(batch, batch_idx)
        self.log_dict(
            {
                "loss/train": loss,
                "mse/train": F.mse_loss(model_output.recon_x, model_output.data),
            },
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            self.log_tensorboard(model_output, model_output.data)
        return loss

    def loss_function(self, model_output, batch_idx, *args, **kwargs):
        return {
            "loss": model_output.loss,
            "recon_loss": model_output.recon_loss,
            "variational_loss": model_output.loss - model_output.recon_loss,
        }

    # def logging_step(self, z, loss, x, model_output, batch_idx):
    #     self.logger.experiment.add_embedding(
    #         z,
    #         label_img=x["data"],
    #         global_step=self.current_epoch,
    #         tag="z",
    #         )

    #     self.logger.experiment.add_scalar("Loss/val", loss, batch_idx)
    #     self.logger.experiment.add_image(
    #         "val",
    #         torchvision.utils.make_grid(model_output.recon_x),
    #         batch_idx,
    #     )

    def validation_step(self, batch, batch_idx):
        # x, y = batch
        loss, model_output = self.eval_step(batch, batch_idx)
        self.log_dict(
            {
                "loss/val": loss,
                "mse/val": F.mse_loss(model_output.recon_x, model_output.data),
            }
        )
        return loss

    def test_step(self, batch, batch_idx):
        # x, y = batch
        loss, model_output = self.eval_step(batch, batch_idx)
        self.log_dict(
            {
                "loss/test": loss,
                "mse/test": F.mse_loss(model_output.recon_x, model_output.data),
            }
        )
        return loss

    # Fangless function to be overloaded later
    def batch_to_xy(self, batch):
        x, y = batch
        return x, y

    def eval_step(self, batch, batch_idx):
        model_output = self.predict_step(batch, batch_idx)
        loss = self.loss_function(model_output, batch_idx)
        return loss, model_output

    # def lr_scheduler_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    #     # Implement your own logic for updating the lr scheduler
    #     # This method will be called at each training step
    #     # Update the lr scheduler based on the provided arguments
    #     # You can access the lr scheduler using `self.lr_schedulers()`

    #     # Example:
    #     for lr_scheduler in self.lr_schedulers():
    #         lr_scheduler.step()

    def timm_optimizers(self, model):
        optimizer = optim.create_optimizer(self.args, model.parameters())
        lr_scheduler = scheduler.create_scheduler(self.args, optimizer)[0]
        return optimizer, lr_scheduler

    def timm_to_lightning(self, optimizer, lr_scheduler):
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # or 'epoch' for step vs epoch training, respectively
            },
        }

    def configure_optimizers(self):
        # optimizer = optim.create_optimizer(self.args, self.model.parameters())
        # lr_scheduler = scheduler.create_scheduler(self.args, optimizer)[0]
        optimizer, lr_scheduler = self.timm_optimizers(self.model)
        return self.timm_to_lightning(optimizer, lr_scheduler)

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch, metric=metric)

    def log_wandb(self):
        pass

    def log_tensorboard(self, model_output, x):
        # Optionally you can add more logging, for example, visualizations:
        self.logger.experiment.add_image(
            "test_input",
            torchvision.utils.make_grid(model_output.data),
            self.global_step,
        )
        self.logger.experiment.add_image(
            "test_output",
            torchvision.utils.make_grid(model_output.recon_x),
            self.global_step,
        )


class AE(AutoEncoder):
    pass


class AutoEncoderUnsupervised(AutoEncoder):
    pass


class AEUnsupervised(AutoEncoder):
    pass


def create_label_based_pairs(
    features: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create positive pairs based on labels.

    Args:
    features: Tensor of shape (b, latent_dim)
    labels: Tensor of shape (b, 1)

    Returns:
    tuple of two tensors, each of shape (n, latent_dim), where n is the number of pairs
    """
    labels = labels.squeeze()  # Convert (b, 1) to (b,)
    unique_labels = torch.unique(labels)

    if len(unique_labels) == 1:
        return torch.empty(0, features.size(1)), torch.empty(0, features.size(1))

    positive_pairs = []

    for label in unique_labels:
        mask = labels == label
        class_samples = features[mask]
        if class_samples.size(0) > 1:  # We need at least 2 samples of the same class
            # Create all possible pairs within this class
            num_samples = class_samples.size(0)
            pairs = torch.combinations(torch.arange(num_samples), r=2)
            positive_pairs.append(
                (class_samples[pairs[:, 0]], class_samples[pairs[:, 1]])
            )

    if not positive_pairs:
        return torch.empty(0, features.size(1)), torch.empty(0, features.size(1))

    input_pairs = torch.cat([pair[0] for pair in positive_pairs])
    target_pairs = torch.cat([pair[1] for pair in positive_pairs])

    return input_pairs, target_pairs


class AutoEncoderSupervised(AutoEncoder):
    criteron = losses.ContrastiveLoss()

    def loss_function(self, model_output, batch_idx):
        # x, y = batch
        loss = super().loss_function(model_output, batch_idx)
        # TODO check this
        # Scale is used as the rest of the loss functions are sums rather than means, which may mean we need to scale up the contrastive loss

        scale = torch.prod(torch.tensor(model_output.z.shape[1:]))
        pairs = create_label_based_pairs(model_output.z.squeeze(), model_output.target)
        contrastive_loss = self.criteron(*pairs)
        loss["contrastive_loss"] = scale * contrastive_loss
        loss["loss"] += loss["contrastive_loss"]
        return loss


class AESupervised(AutoEncoderSupervised):
    pass


class NDAutoEncoder(AESupervised):
    def batch_to_xy(self, batch):
        x, y = super().batch_to_xy(batch)

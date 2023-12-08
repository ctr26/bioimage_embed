import torchvision
import torch
import pytorch_lightning as pl
import pythae
from timm import optim, scheduler
from types import SimpleNamespace
import argparse
import timm
from pythae.models.base.base_utils import ModelOutput


class LitAutoEncoderTorch(pl.LightningModule):
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

    def __init__(self, model,args,**kwargs):
        super().__init__()
        self.model = model
        self.model = self.model.to(self.device)
        # Flatten hparams
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        if args:
            self.args = SimpleNamespace(**{**vars(args), **vars(self.args)})
        if kwargs:
            merged_kwargs = {k: v for d in kwargs.values() for k, v in d.items()}
            self.args = SimpleNamespace(**{**merged_kwargs, **vars(self.args)})
        self.save_hyperparameters(self.args)
        # self.model.train()

    def forward(self, batch):
        x = self.batch_to_tensor(batch)
        return ModelOutput(x=x, out=self.model(x))

    def get_results(self, batch):
        # if self.PYTHAE_FLAG:
        x = self.batch_to_tensor(batch)
        return self.model.forward(x)
        # return self.model.forward(batch)

    def batch_to_tensor(self, batch):
        return ModelOutput(data=batch)

    def embedding_from_output(self, model_output):
        return model_output.z.view(model_output.z.shape[0], -1)

    def get_model_output(self, x, batch_idx):
        model_output = self.model(x, epoch=batch_idx)
        loss = self.loss_function(model_output)
        return model_output, loss

    def training_step(self, batch, batch_idx):
        # results = self.get_results(batch)
        self.model.train()
        x = self.batch_to_tensor(batch)
        model_output, loss = self.get_model_output(
            x,
            batch_idx,
        )
        # loss = self.model.training_step(x)
        # loss = self.loss_function(model_output,optimizer_idx)

        # self.log("train_loss", self.loss)
        # self.log("train_loss", loss)
        self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)

        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(x["data"]), batch_idx
        )

        # if self.PYTHAE_FLAG:
        self.logger.experiment.add_image(
            "output",
            torchvision.utils.make_grid(model_output.recon_x),
            batch_idx,
        )

        return loss

    def loss_function(self, model_output, *args, **kwargs):
        return model_output.loss

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
        x = self.batch_to_tensor(batch)
        model_output, loss = self.get_model_output(x, batch_idx)
        z = self.embedding_from_output(model_output)
        # z_indices
        self.logger.experiment.add_embedding(
            z,
            label_img=x["data"],
            global_step=self.current_epoch,
            tag="z",
        )

        self.logger.experiment.add_scalar("Loss/val", loss, batch_idx)
        self.logger.experiment.add_image(
            "val",
            torchvision.utils.make_grid(model_output["recon_x"]),
            batch_idx,
        )

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

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer

    def test_step(self, batch, batch_idx):
        x = self.batch_to_tensor(batch)
        model_output = self.model(x)  # Forward pass with the test batch

        # Optionally compute a loss or metric if relevant
        loss = self.loss_function(model_output)

        # Log test metrics
        self.log("test_loss", loss)

        # Optionally you can add more logging, for example, visualizations:
        self.logger.experiment.add_image(
            "test_input",
            torchvision.utils.make_grid(x["data"]),
            batch_idx,
        )
        self.logger.experiment.add_image(
            "test_output",
            torchvision.utils.make_grid(model_output.recon_x),
            batch_idx,
        )

        # Return whatever data you need, for example, the loss
        return loss
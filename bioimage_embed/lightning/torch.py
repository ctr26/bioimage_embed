import torchvision
import torch
import pytorch_lightning as pl
import pythae
from timm import optim, scheduler
from types import SimpleNamespace
import argparse
import timm
from pythae.models.base.base_utils import ModelOutput
import torch.nn.functional as F

class LitAutoEncoderTorch(pl.LightningModule):
    loss_stack = []
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
        channel_aware=False,
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
        # if kwargs:
        # merged_kwargs = {k: v for d in kwargs.values() for k, v in d.items()}
        # self.args = SimpleNamespace(**{**merged_kwargs, **vars(self.args)})
        self.save_hyperparameters(vars(self.args))
        # self.model.train()

    def forward(self, batch):
        x = self.batch_to_tensor(batch)
        return ModelOutput(x=x, out=self.model(x))

    def get_results(self, batch):
        # if self.PYTHAE_FLAG:
        x = self.batch_to_tensor(batch)
        return self.model.forward(x)
        # return self.model.forward(batch)

    def batch_to_tensor(self, batch: torch.Tensor) -> ModelOutput:
        return ModelOutput(data=batch)

    def embedding_from_output(self, model_output):
        return model_output.z.view(model_output.z.shape[0], -1)

    def get_model_output(self, x, batch_idx):
        model_output = self.model(x, epoch=batch_idx)
        loss = self.loss_function(model_output)
        return model_output, loss

    def training_step(self, batch, batch_idx):
        self.model.train()
        x = self.batch_to_tensor(batch)
        model_output, loss = self.get_model_output(
            x,
            batch_idx,
        )
        self.log_dict(
            {
                "loss/train": loss,
                "mse/train": F.mse_loss(model_output.recon_x, x["data"]),
            },
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            self.log_tensorboard(model_output, x)
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
        self.log_dict(
            {
                "loss/val": loss,
                "mse/val": F.mse_loss(model_output.recon_x, x["data"]),
            }
        )
        return loss

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
        self.log_dict(
            {
                "loss/test": loss,
                "mse/test": F.mse_loss(model_output.recon_x, x["data"]),
            }
        )

        return loss
    
    def log_wandb(self):
        pass
    
    def log_tensorboard(self, model_output, x):
        # Optionally you can add more logging, for example, visualizations:
        self.logger.experiment.add_image(
            "test_input",
            torchvision.utils.make_grid(x["data"]),
            self.global_step,
        )
        self.logger.experiment.add_image(
            "test_output",
            torchvision.utils.make_grid(model_output.recon_x),
            self.global_step,
        )


class RGBLitAutoEncoderTorch(LitAutoEncoderTorch):
    def __init__(self, model, args=SimpleNamespace()):
        super().__init__(model, args)


class GrayscaleLitAutoEncoderTorch(LitAutoEncoderTorch):
    # Needs to be expanded to 3 for RGB models
    repeat = (1, 3, 1, 1)

    def __init__(self, model, args=SimpleNamespace()):
        super().__init__(model, args)

    def batch_to_tensor(self, batch: torch.Tensor) -> ModelOutput:
        return ModelOutput(data=batch.repeat(*self.repeat))


class ChannelAwareLitAutoEncoderTorch(GrayscaleLitAutoEncoderTorch):
    # Assuming the tensor that came in was say 16,5,512,512
    #  The grayscale tensor would be 16,15,512,512

    def __init__(self, model, args=SimpleNamespace()):
        super().__init__(model, args)
        # Add any additional initializations here, if necessary

    def expand_channels(self, tensor):
        b, c, *dims = tensor.shape
        tensor = tensor.unsqueeze(1)
        tensor = tensor.transpose(1, 2)
        tensor = tensor.reshape(b * c, 1, *dims)
        return tensor

    def contract_channels(self, x,c):
        b_c, dims = x.shape
        x = x.reshape(b_c // c, c, *dims)
        x = x.transpose(1, 2)
        x = x.squeeze(1)
        return x

    def batch_to_tensor(self, batch: torch.Tensor) -> ModelOutput:
        x = self.expand_channels(batch)
        # This should be the grayscale repeat
        return super().batch_to_tensor(x)

    def _(self, x: torch.Tensor):
        # TODO forgotten what this should be named
        
        
        # Mean so that RGB model is gray again,
        # Will be noisy in early epochs but should be fine
        x = x.mean(dim=1, keepdim=True)
        x = self.contract_channels(x)
        return x

    def loss_function(self, model_output, *args, **kwargs):
        # Remember its sum square losses
        # x = super().loss_function(model_output)
        return model_output.loss + self.channel_loss(model_output.z)

    def contrastive_loss(self, z_positive, z_negative):
        pass

    def channel_loss(self, x):
        model_output = self.model(x)
        b,c,dims = x.shape
        z = model_output.z.mean(dim=1, keepdim=True)
        z = self.contract_channels(z,c)
        channel_loss = euclidean_z_channel(z)
        # TODO clever mean across batches with a larger weighting from the now smaller batch
        return channel_loss.sum(dim=(1, 2)).mean(dim=0)


def js_loss_z_channel_loss(x, reduction="batchmean"):
    """Calculates the generalised js loss between a tensor and its transpose on the channel dimension

    Args:
        x (_type_): x.shape = (b, c, z)
        reduction (str, optional): _description_. Defaults to "batchmean".

    Returns:
        _type_: _description_
    """
    b, c, z = x.shape
    x = x.unsqueeze(1)
    transpose = x.permute(0, 2, 1, 3)
    return js(x, transpose, reduction=reduction)


def js(p: torch.Tensor, q: torch.Tensor, reduction="batchmean", log_target=True):
    """Calculates the js divergence between two tensors

    Args:
        p (torch.Tensor): p.shape = (b, c, z)
        q (torch.Tensor): q.shape = (b, c, z)
        reduction (str, optional): _description_. Defaults to "batchmean".
        log_target (bool, optional): _description_. Defaults to True.

    Returns:
        torch.Tensor: js divergence
    """
    # TODO checkout the log_target
    lr = F.kl_div(p, q, reduction=reduction, log_target=log_target)
    rl = F.kl_div(q, p, reduction=reduction, log_target=log_target)
    return (rl + lr) / 2


def euclidean_z_channel(x):
    return distance_z_channel(x, p=2.0)


def distance_z_channel(x, p=2.0):
    """Calculates the distance loss between a tensor and its transpose on the channel dimension

    Args:
        x (torch.Tensor):  x.shape = (b, c, z)
        p (float, optional): _description_. Defaults to 2.0 euclidean.
        reduction (str, optional): _description_. Defaults to "mean".

    Returns:
        torch.Tensor: distance
    """
    b, c, z = x.shape
    x = x.unsqueeze(1)
    # loss = F.mse_loss(x.permute(0, 2, 1, 3), x,s reduction="none")
    dist = torch.cdist(x, x.permute(0, 2, 1, 3))
    return dist


def euclidean_z_channel_loss(x, reduction="mean"):
    return distance_z_channel(x,p=2.0).mean()


def js_z_channel_loss(x, reduction="mean"):
    return js_loss_z_channel_loss(x, reduction=reduction)


_channel_aware_losses = [
    euclidean_z_channel_loss,
    js_z_channel_loss,
]

_model_classes = [
    LitAutoEncoderTorch,
    RGBLitAutoEncoderTorch,
    GrayscaleLitAutoEncoderTorch,
    ChannelAwareLitAutoEncoderTorch,
]


_3c_model_classes = [
    LitAutoEncoderTorch,
    RGBLitAutoEncoderTorch,
    ChannelAwareLitAutoEncoderTorch,
]

_1C_model_classes = [
    GrayscaleLitAutoEncoderTorch,
    ChannelAwareLitAutoEncoderTorch,
]

def autoencoder_factory(model,args=SimpleNamespace()):
    if args.channel_aware == True:
        return ChannelAwareLitAutoEncoderTorch(model, args)
    if args.input_dim[0] == 1 & args.channel_aware == False:
        return GrayscaleLitAutoEncoderTorch(model, args)
    if args.input_dim[0] == 3 & args.channel_aware == False:
        return RGBLitAutoEncoderTorch(model, args)
    return LitAutoEncoderTorch(model, args)


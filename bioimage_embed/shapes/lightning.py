import torch
import torchvision

from torch import nn
from ..lightning import AutoEncoderUnsupervised
from . import loss_functions as lf
from transformers.utils import ModelOutput
from types import SimpleNamespace


def frobenius_norm_2D_torch(tensor: torch.Tensor) -> torch.Tensor:
    return torch.norm(tensor, p="fro", dim=(-2, -1), keepdim=True)


class MaskEmbed(AutoEncoderUnsupervised):
    def __init__(self, model, args=SimpleNamespace()):
        super().__init__(model, args)

    def batch_to_tensor(self, batch):
        """
        Converts a batch of data to a tensor
        Batch is expected to be normalised to the window size which will be the same or smaller than the image size in question.
        The batch is also optionally frobenius normalised to make the loss function invariant to the size of the shape.
        """
        # x = batch[0].float()
        output = super().batch_to_tensor(batch)
        normalised_data = output["data"]
        if self.args.frobenius_norm:
            scalings = frobenius_norm_2D_torch(output["data"])
        else:
            scalings = torch.ones_like(output["data"])
        return ModelOutput(data=normalised_data / scalings, scalings=scalings)

    def loss_function(self, model_output, *args, **kwargs):
        loss_ops = lf.DistanceMatrixLoss(model_output.recon_x, norm=False)
        loss = model_output.loss

        shape_loss = torch.sum(
            torch.stack(
                [
                    loss_ops.diagonal_loss(),
                    loss_ops.symmetry_loss(),
                    loss_ops.non_negative_loss(),
                    # loss_ops.triangle_inequality(),
                    # loss_ops.clockwise_order_loss(),
                ]
            )
        )
        loss += shape_loss

        # loss += lf.diagonal_loss(model_output.recon_x)
        # loss += lf.symmetry_loss(model_output.recon_x)
        # loss += lf.triangle_inequality_loss(model_output.recon_x)
        # loss += lf.non_negative_loss(model_output.recon_x)

        variational_loss = model_output.loss - model_output.recon_loss

        # loss_dict = {
        #     "loss": loss,
        #     "shape_loss": shape_loss,
        #     "reconstruction_loss": model_output.recon_x,
        #     "variational_loss": variational_loss,
        # }
        return loss


class FixedOutput(nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def forward(self, x):
        return self.tensor


class MaskEmbedLatentAugment(MaskEmbed):
    def __init__(self, model, args_dict):
        super().__init__(model, **args_dict)

    def get_model_output(self, x, batch_idx, optimizer_idx=0):
        # Janky!
        model_output = self.model(x, epoch=batch_idx)
        # if optimizer_idx == 0:
        #     loss = self.loss_function(model_output)
        # if optimizer_idx == 1:
        #     model_encoder = self.model.encoder
        #     self.model.encoder = FixedOutput(self.guassian_sample(model_output.z))
        #     decoder_output = self.model(x, epoch=batch_idx)
        #     self.model.encoder = model_encoder
        #     loss = self.loss_function(decoder_output)

        loss = self.loss_function(model_output)
        return model_output, loss

    def guassian_sample(self, z):
        return torch.normal(0, 1, size=z.shape).to(self.device)
        # return torch.randn(*model_output.z.shape).to(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # results = self.get_results(batch)
        self.model.train()
        x = self.batch_to_tensor(batch)
        model_output, loss = self.get_model_output(
            x,
            batch_idx,
            optimizer_idx=optimizer_idx,
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

    # def configure_optimizers(self):
    #     opt_ed, lr_s_ed = self.timm_optimizers(self.model)
    #     opt_d, lr_s_d = self.timm_optimizers(self.model)
    #     return (
    #         self.timm_to_lightning(optimizer=opt_ed, lr_scheduler=lr_s_ed),
    #         self.timm_to_lightning(optimizer=opt_d, lr_scheduler=lr_s_d),
    #     )

    def configure_optimizers(self):
        opt_ed, lr_s_ed = self.timm_optimizers(self.model)
        return self.timm_to_lightning(optimizer=opt_ed, lr_scheduler=lr_s_ed)

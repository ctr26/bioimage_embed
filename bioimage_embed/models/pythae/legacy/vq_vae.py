# TODO make this a relative import

import torch
from torch import nn
from torch.nn import functional as F

from pythae import models


# from pythae.models import VQVAE, Encoder, Decoder
from pythae.models.base.base_utils import ModelOutput

# import VQVAE, Encoder, Decoder
from pythae.models.nn import BaseDecoder, BaseEncoder
from ...nets.resnet import ResnetDecoder, ResnetEncoder
from ....models.legacy import vq_vae


from pythae.models import VQVAEConfig, VAEConfig


class Encoder(BaseEncoder):
    def __init__(
        self,
        model_config,
        num_hiddens,
        num_residual_hiddens,
        num_residual_layers,
    ):
        super(Encoder, self).__init__()
        embedding_dim = model_config.latent_dim
        input_dim = model_config.input_dim[1:]

        self.model = ResnetEncoder(
            in_channels=model_config.input_dim[0],
            num_hiddens=num_hiddens,
            num_residual_hiddens=num_residual_hiddens,
            num_residual_layers=num_residual_layers,
        )


class VAEEncoder(Encoder):
    def forward(self, x):
        return ModelOutput(embedding=self.model(x["data"]))


class VQVAEEncoder(Encoder):
    def forward(self, x):
        return ModelOutput(pre_quantized=self.model(x["data"]))


class VAEDecoder(BaseDecoder):
    def __init__(
        self,
        model_config,
        num_hiddens,
        num_residual_hiddens,
        num_residual_layers,
    ):
        super(VAEDecoder, self).__init__()
        self.model = ResnetDecoder(
            in_channels=model_config.latent_dim,
            out_channels=model_config.input_dim[0],
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, x):
        reconstruction = self.model(x["embedding"])
        return ModelOutput(reconstruction=reconstruction)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class VQVAE(models.VQVAE):
    def __init__(
        self,
        model_config: VQVAEConfig,
        depth,
        encoder=None,
        decoder=None,
        strict_latent_size=True,
    ):
        super(models.BaseAE, self).__init__()
        # super(nn.Module)
        # input_dim (tuple) – The input_data dimension.

        self.model_name = "VQVAE"
        self.model_config = model_config

        if self.model_config.decay > 0.0:
            self.model_config.use_ema = True

        self.strict_latent_size = strict_latent_size
        self.model = vq_vae.VQ_VAE(
            channels=model_config.input_dim[0],
            embedding_dim=model_config.latent_dim,
            num_hiddens=model_config.latent_dim,
            num_residual_layers=depth,
        )
        self.encoder = self.model._encoder
        self.decoder = self.model._decoder
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # This isn't completely necessary for training I don't think
        # self._set_quantizer(model_config)

    def forward(self, x, epoch=None):
        # loss, x_recon, perplexity = self.model.forward(x["data"])
        z = self.model.encoder(x["data"])
        z = self.model._pre_vq_conv(z)
        proper_shape = z.shape

        if self.strict_latent_size:
            z = self.avgpool(z)
            # Features need to be in the right order for the quantizer
            z = z.permute(0, 2, 3, 1)

        loss, quantized, perplexity, encodings = self.model._vq_vae(z)
        z = quantized.flatten(1)
        if self.strict_latent_size:
            quantized = quantized.permute(0, 3, 1, 2)
            quantized = quantized.expand(-1, *proper_shape[-3:])

        x_recon = self.model._decoder(quantized)
        # return loss, x_recon, perplexity

        legacy_loss_dict = self.model.loss_function(
            loss,
            x_recon,
            perplexity,
            vq_loss=loss,
            perplexity=perplexity,
            recons=x_recon,
            input=x["data"],
        )
        # This matches how pythae returns the loss

        indices = (encodings == 1).nonzero(as_tuple=True)

        recon_loss = F.mse_loss(x_recon, x["data"], reduction="sum")
        mse_loss = F.mse_loss(x_recon, x["data"], reduction="mean")

        variational_loss = loss - mse_loss

        pythae_loss_dict = {
            "recon_loss": mse_loss,
            "vq_loss": variational_loss,
            # TODO check this proppperppply
            "loss": recon_loss + variational_loss,
            "recon_x": x_recon,
            "z": z,
            "quantized_indices": indices[0],
            "indices": indices,
        }
        return ModelOutput(**{**legacy_loss_dict, **pythae_loss_dict})


class VAE(models.VAE):
    def __init__(
        self,
        model_config: VAEConfig,
        num_hiddens=64,
        num_residual_hiddens=18,
        num_residual_layers=2,
        encoder=None,
        decoder=None,
    ):
        super(models.BaseAE, self).__init__()
        # super(nn.Module)
        # input_dim (tuple) – The input_data dimension.

        self.model_name = "VAE"
        self.model_config = model_config
        self.encoder = VAEEncoder(
            model_config,
            num_hiddens=num_hiddens,
            num_residual_hiddens=num_residual_hiddens,
            num_residual_layers=num_residual_layers,
        )
        self.decoder = VAEDecoder(
            model_config,
            num_hiddens=num_hiddens,
            num_residual_hiddens=num_residual_hiddens,
            num_residual_layers=num_residual_layers,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_hiddens, model_config.latent_dim * 2)
        # shape is (batch_size, model_config.num_hiddens, 1, 1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, epoch=None):
        h = self.encoder(x)["embedding"]
        # pre_encode_size = torch.tensor(x["data"].shape[-2:])
        # scale = torch.floor_divide(torch.tensor(x["data"].shape[-2:]),torch.tensor(h.shape[-2:]))
        pre_encode_size = torch.tensor(h.shape[-2:])
        h = self.avgpool(h)
        post_encode_size = torch.tensor(h.shape[-2:])
        scale = torch.div(pre_encode_size, post_encode_size, rounding_mode="trunc")
        h = torch.flatten(h, 1)
        h = self.fc(h)
        mu, log_var = torch.split(h, h.size(1) // 2, dim=1)
        z = self.reparameterize(mu, log_var)
        # x_recon = self.decoder(z.view(z.size(0), z.size(1), 1, 1))
        embedding = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, *scale.tolist())
        x_recon = self.decoder({"embedding": embedding})["reconstruction"]
        # return x_recon, mu, log_var

        loss_dict = self.loss_function(x_recon, x["data"], mu, log_var)
        # recon_loss = F.mse_loss(x_recon, x["data"], reduction="sum")
        return ModelOutput(recon_x=x_recon, z=z, **loss_dict)

    def loss_function(self, recons, input, mu, log_var):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recons_loss + kld_loss
        return {
            "loss": loss,
            "recon_loss": recons_loss,
            "variational_loss": kld_loss,
        }


# Resnet50_VQVAE = partial(VQVAE,num_hidden_residuals=50)

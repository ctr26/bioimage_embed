# TODO make this a relative import

import torch
from torch import nn
from torch.nn import functional as F

from torch import nn
from pythae import models


from functools import partial

# from pythae.models import VQVAE, Encoder, Decoder
from pythae.models.base.base_utils import ModelOutput

# import VQVAE, Encoder, Decoder
from pythae.models.nn import BaseDecoder, BaseEncoder
from ...nets.resnet import ResnetDecoder, ResnetEncoder
from ....models import legacy


from pythae.models import VQVAEConfig, VAEConfig


class Encoder(BaseEncoder):
    def __init__(self, model_config, **kwargs):
        super(Encoder, self).__init__()
        embedding_dim = model_config.latent_dim
        input_dim = model_config.input_dim[1:]

        self.model = ResnetEncoder(
            in_channels=model_config.input_dim[0], **{**vars(model_config), **kwargs}
        )


class VAEEncoder(Encoder):
    def forward(self, x):
        return ModelOutput(embedding=self.model(x["data"]))


class VQVAEEncoder(Encoder):
    def forward(self, x):
        return ModelOutput(pre_quantized=self.model(x["data"]))


class VAEDecoder(BaseDecoder):
    def __init__(self, model_config, **kwargs):
        super(VAEDecoder, self).__init__()
        self.model = ResnetDecoder(
            in_channels=model_config.latent_dim,
            out_channels=model_config.input_dim[0],
            **{**vars(model_config), **kwargs},
        )

    def forward(self, x):
        reconstruction = self.model(x["embedding"])
        return ModelOutput(reconstruction=reconstruction)


class VQVAE(models.VQVAE):
    def __init__(self, model_config: VQVAEConfig, **kwargs):
        super(models.BaseAE, self).__init__()
        # super(nn.Module)
        # input_dim (tuple) – The input_data dimension.

        self.model_name = "VQVAE"
        self.model_config = model_config

        if self.model_config.decay > 0.0:
            self.model_config.use_ema = True

        self.model = legacy.vq_vae.VQ_VAE(
            channels=model_config.input_dim[0],
            embedding_dim=model_config.latent_dim,
            num_hiddens=model_config.latent_dim,
            **{
                **vars(model_config),
                **kwargs,
            },
        )
        self.encoder = self.model._encoder
        self.decoder = self.model._decoder
        # This isn't completely necessary for training I don't think
        # self._set_quantizer(model_config)

    def forward(self, x, epoch=None):
        # loss, x_recon, perplexity = self.model.forward(x["data"])
        z = self.model.encoder(x["data"])
        z = self.model._pre_vq_conv(z)
        loss, quantized, perplexity, encodings = self.model._vq_vae(z)
        x_recon = self.model._decoder(quantized)
        # return loss, x_recon, perplexity
        loss_dict = self.model.loss_function(
            loss,
            x_recon,
            perplexity,
            vq_loss=loss,
            perplexity=perplexity,
            recons=x_recon,
            input=x["data"],
        )
        indices = (encodings == 1).nonzero(as_tuple=True)
        # self.model.encode()

        return ModelOutput(
            # recon_loss=recon_loss,
            vq_loss=loss,
            # loss=loss,
            recon_x=x_recon,
            z=quantized,
            quantized_indices=indices[0],
            **loss_dict,
        )
        # return ModelOutput(reconstruction=x_recon, **loss_dict)


class VAE(models.VAE):
    def __init__(
        self,
        model_config: VAEConfig,
        num_hiddens=64,
        num_residual_hiddens=18,
        num_residual_layers=2,
        **kwargs
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
            **kwargs,
        )
        self.decoder = VAEDecoder(
            model_config,
            num_hiddens=num_hiddens,
            num_residual_hiddens=num_residual_hiddens,
            num_residual_layers=num_residual_layers,
            **kwargs,
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
        return ModelOutput(recon_x=x_recon, z=z, **loss_dict)

    def loss_function(self, recons, input, mu, log_var):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recons_loss + kld_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": kld_loss}


# Resnet50_VQVAE = partial(VQVAE,num_hidden_residuals=50)

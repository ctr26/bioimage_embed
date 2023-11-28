from torch import nn
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
import torch
from pythae.models import VQVAE, VQVAEConfig, VAE, VAEConfig

from pl_bolts.models.autoencoders import (
    resnet18_encoder,
    resnet18_decoder,
    resnet50_decoder,
    resnet50_encoder,
)

class BaseResNetVQVAEEncoder(BaseEncoder):
    def __init__(
        self, model_config: VAEConfig, resnet_encoder, enc_out_dim, first_conv=False, maxpool1=False, **kwargs
    ):
        super(BaseResNetVQVAEEncoder, self).__init__()

        self.input_height = model_config.input_dim[-2]
        self.latent_dim = model_config.latent_dim
        self.enc_out_dim = enc_out_dim

        self.encoder = resnet_encoder(first_conv, maxpool1)
        self.embedding = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.prequantized = nn.Conv2d(self.enc_out_dim, self.latent_dim, 1, 1)

    def forward(self, inputs):
        x = self.encoder(inputs)
        log_covariance = self.log_var(x)
        x = x.view(-1, self.enc_out_dim, 1, 1)
        embedding = self.prequantized(x)
        return ModelOutput(embedding=embedding, log_covariance=log_covariance)


class ResNet50VQVAEEncoder(BaseResNetVQVAEEncoder):
    enc_out_dim = 2048

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet50VQVAEEncoder, self).__init__(
            model_config, resnet50_encoder, self.enc_out_dim, first_conv, maxpool1, **kwargs
        )


class ResNet18VQVAEEncoder(BaseResNetVQVAEEncoder):
    enc_out_dim = 512

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet18VQVAEEncoder, self).__init__(
            model_config, resnet18_encoder, self.enc_out_dim, first_conv, maxpool1, **kwargs
        )


class BaseResNetVQVAEDecoder(BaseDecoder):
    def __init__(
        self, model_config: VAEConfig, resnet_decoder, first_conv=False, maxpool1=False, **kwargs
    ):
        super(BaseResNetVQVAEDecoder, self).__init__()
        self.model_config = model_config
        self.latent_dim = model_config.latent_dim
        self.input_height = model_config.input_dim[-2]
        self.decoder = resnet_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        self.postquantized = nn.Conv2d(self.latent_dim, self.latent_dim, 1, 1)

    def forward(self, x):
        x = self.postquantized(x)
        x = x.view(-1, self.latent_dim)
        x = self.decoder(x)
        return ModelOutput(reconstruction=x)

class ResNet50VQVAEDecoder(BaseResNetVQVAEDecoder):
    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet50VQVAEDecoder, self).__init__(
            model_config, resnet50_decoder, first_conv, maxpool1, **kwargs
        )

class ResNet18VQVAEDecoder(BaseResNetVQVAEDecoder):
    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet18VQVAEDecoder, self).__init__(
            model_config, resnet18_decoder, first_conv, maxpool1, **kwargs
        )

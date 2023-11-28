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


class ResNet50VQVAEEncoder(BaseEncoder):

    enc_out_dim = 2048
    enc_out_dim_headless = 32

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet50VQVAEEncoder, self).__init__()

        self.input_height = model_config.input_dim[-2]
        latent_dim = model_config.latent_dim

        self.encoder = resnet50_encoder(first_conv, maxpool1)
        # Remove final layer
        # self.encoder.avgpool = nn.Identity()
        self.embedding = nn.Linear(self.enc_out_dim, latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, latent_dim)
        self.prequantized = nn.Conv2d(self.enc_out_dim, latent_dim, 1, 1)
        # self.prequantized = nn.Conv2d(self.enc_out_dim_headless, latent_dim, 1, 1)
        # self.fc = nn.Linear(401408, 256)
        self.fc = nn.Linear(self.enc_out_dim, latent_dim)

    def forward(self, inputs):
        # output = ModelOutput()
        # self.encoder(x.view(-1,32,self.input_height,self.input_height))        # y = self.encoder(x).view(-1,self.enc_out_dim,14,14)
        x = self.encoder(inputs)
        # Reshape to shape before final layer
        x = x.view(-1, self.enc_out_dim, 1, 1)
        # x = x.view(-1, self.enc_out_dim_headless, self.input_height, self.input_height)
        # self.
        # x = self.fc(x)

        # x = x.view(x.size(0), -1, 1, 1)
        # embedding = self.prequantized(x.view(-1, self.enc_out_dim, 1, 1))
        embedding = self.prequantized(x)
        # y = self.encoder(x).view(-1,2048,14,14)
        # output["embedding"] = self.prequantized(y)
        # output["embedding"] = self.embedding(x)

        # output["log_covariance"] = self.log_var(x)
        return ModelOutput(embedding=embedding)


class ResNet50VQVAEDecoder(BaseDecoder):
    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet50VQVAEDecoder, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.input_height = model_config.input_dim[-1]
        self.decoder = resnet50_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        # self.decoder.linear = nn.Identity()
        self.postquantized = nn.Conv2d(self.latent_dim, self.latent_dim, 1, 1)
        # self.fc = nn.Linear(self.enc_out_dim, latent_dim)

    def forward(self, inputs):
        # a = nn.Conv2d(16,512,1,1)
        x = self.postquantized(inputs)
        x = x.view(-1, self.latent_dim)
        # x = (latent_dim, 512, input_height, input_height)
        # decoder is expecting 
        # self.decoder(x)
        x = self.decoder(x)

        return ModelOutput(reconstruction=x)


class ResNet18VQVAEEncoder(BaseEncoder):

    enc_out_dim = 512

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet18VQVAEEncoder, self).__init__()

        input_height = model_config.input_dim[-2]
        latent_dim = model_config.latent_dim

        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.embedding = nn.Linear(self.enc_out_dim, latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, latent_dim)
        self.prequantized = nn.Conv2d(self.enc_out_dim, latent_dim, 1, 1)

    def forward(self, inputs):
        x = self.encoder(inputs)
        # x = self.fc1(x)
        embedding = self.prequantized(x.view(-1, self.enc_out_dim, 1, 1))
        log_covariance = self.log_var(x)
        return ModelOutput(embedding=embedding, log_covariance=log_covariance)


class ResNet18VQVAEDecoder(BaseDecoder):
    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet18VQVAEDecoder, self).__init__()
        self.model_config = model_config
        self.latent_dim = model_config.latent_dim
        self.input_height = model_config.input_dim[-2]
        self.decoder = resnet18_decoder(
            self.latent_dim, self.input_height, first_conv, maxpool1
        )
        self.fc1 = nn.Linear(512, 256)
        self.postquantized = nn.Conv2d(self.latent_dim, self.latent_dim, 1, 1)
        
    def forward(self, x):
        # x = self.fc1(x)
        x = self.postquantized(x)
        x = x.view(-1, self.latent_dim)
        # x = self.fc1(x)
        return ModelOutput(reconstruction=x)

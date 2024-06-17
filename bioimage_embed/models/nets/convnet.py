from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

import torch
from torch import nn


def encoder_layers(channels, hidden_dims):
    modules = []

    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    out_channels=h_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            )
        )
        channels = h_dim
    return nn.Sequential(*modules)


def decoder_layers(channels, hidden_dims):
    modules = []
    for i in range(len(hidden_dims) - 1):
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU(),
            )
        )
    final_layer = nn.Sequential(
        nn.ConvTranspose2d(
            hidden_dims[-1],
            hidden_dims[-1],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        ),
        nn.BatchNorm2d(hidden_dims[-1]),
        nn.LeakyReLU(),
        nn.Conv2d(hidden_dims[-1], out_channels=channels, kernel_size=3, padding=1),
        nn.Tanh(),
    )
    return nn.Sequential(*modules, final_layer)


class ConvNetEncoder(BaseEncoder):
    def __init__(self, model_config, hidden_dims=[32, 64, 128, 256, 512]):
        BaseEncoder.__init__(self)

        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim
        self.channels = self.input_dim[0]
        self.image_dims = self.input_dim[1:]

        self.hidden_dims = hidden_dims

        self.latent_input_dim = torch.tensor(self.image_dims) / 2 ** (len(hidden_dims))
        self.latent_input_dim_len = int(torch.prod(self.latent_input_dim.flatten(), 0))
        # Build Encoder

        self.conv_layers = encoder_layers(self.channels, hidden_dims)
        self.embedding = nn.Linear(
            hidden_dims[-1] * self.latent_input_dim_len, self.latent_dim
        )
        self.log_var = nn.Linear(
            hidden_dims[-1] * self.latent_input_dim_len, self.latent_dim
        )

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1), log_covariance=self.log_var(h1)
        )
        return output


class ConvNetDecoder(BaseDecoder):
    def __init__(self, model_config, hidden_dims=[32, 64, 128, 256, 512]):
        BaseDecoder.__init__(self)
        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim
        self.channels = self.input_dim[0]
        self.image_dims = self.input_dim[1:]
        hidden_dims = hidden_dims
        self.hidden_dims = hidden_dims

        self.latent_input_dim = torch.tensor(self.image_dims) / 2 ** (len(hidden_dims))
        self.latent_input_dim_len = int(torch.prod(self.latent_input_dim.flatten(), 0))
        self.fc = nn.Linear(self.latent_dim, self.latent_input_dim_len)

        self.decoder_input = nn.Linear(
            self.latent_dim, hidden_dims[-1] * self.latent_input_dim_len
        )
        # self.fc = self.decoder_input

        self.deconv_layers = decoder_layers(self.channels, hidden_dims[::-1])

    def forward(self, z: torch.Tensor):
        # h1 = self.fc(z)
        h1 = self.decoder_input(z).view(
            -1,
            self.hidden_dims[-1],
            int(self.latent_input_dim[-1]),
            int(self.latent_input_dim[0]),
        )
        return ModelOutput(reconstruction=self.deconv_layers(h1))

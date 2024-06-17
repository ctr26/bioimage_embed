from torch import nn
from torch.nn import functional as F
import types


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class ResnetEncoder(nn.Module):
    def __init__(
        self,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        in_channels,
        **kwargs,
    ):
        super(ResnetEncoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class ResnetDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        out_channels,
    ):
        super(ResnetDecoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


def resnet18_encoder():
    return ResnetEncoder(**resnet18_encoder_params)


def resnet_encoder_50():
    return ResnetEncoder(**resnet50_encoder_params)


def resnet18_decoder(latent_dim):
    return ResnetDecoder(**resnet18_encoder_params, out_channels=latent_dim)


def resnet50_decoder(latent_dim):
    return ResnetDecoder(**resnet50_encoder_params, out_channels=latent_dim)


# Define a Namespace for ResNet18 Encoder
resnet18_encoder_params = types.SimpleNamespace(
    num_hiddens=64,
    num_residual_layers=2,
    num_residual_hiddens=64,
)

# Define a Namespace for ResNet50 Encoder
resnet50_encoder_params = types.SimpleNamespace(
    num_hiddens=256,
    num_residual_layers=4,
    num_residual_hiddens=256,
)

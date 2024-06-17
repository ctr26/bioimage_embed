# Note - you must have torchvision installed for this example
import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        z_dim = (5, 5)
        decoder_input = (12, 12)
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 1, 3, 1)

        # self.softmax = nn.AdaptiveSoftmax()
        self.fc1 = nn.AdaptiveMaxPool2d(z_dim)
        self.fc2 = nn.AdaptiveMaxPool2d(decoder_input)

        # self.conv4 = self.contract_block(128, 256, 3, 1)

        # self.upconv4 = self.contract_block(256, 128, 3, 1)
        self.upconv3 = self.expand_block(1, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 1, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 1, out_channels, 3, 1)

        self.encoder = nn.Sequential(self.conv1, self.conv2, self.conv3, self.fc1)
        self.decoder = nn.Sequential(self.fc2, self.upconv3, self.upconv2, self.upconv1)

    # Call is essentially the same as running "forward"
    def __call__(self, x):
        # x = self.encoder(x)
        # x = self.decoder(x)
        return self.forward(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        return expand

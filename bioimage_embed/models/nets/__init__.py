from functools import partial
from .resnet import ResnetEncoder, ResnetDecoder
from .convnet import ConvNetEncoder, ConvNetDecoder

RESNETS_DEPTHS = {
    18: 512,
    34: 512,
    50: 2048,
    101: 2048,
    152: 2048,
}

RESNETS = {
    f"resnet{k}": {
        "depth": v,
        "encoder": partial(ResnetEncoder, num_residual_layers=k),
        "decoder": partial(ResnetDecoder, num_residual_layers=k),
    }
    for k, v in RESNETS_DEPTHS.items()
}

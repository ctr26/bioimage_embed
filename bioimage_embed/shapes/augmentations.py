
#  %%

import pytest
import torch
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
import bioimage_embed.models
import bioimage_embed
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch

from bioimage_embed.shapes.transforms import (
    DistogramToCoords,
    CropCentroidPipeline,
)
from bioimage_embed.shapes.transforms import (
    DistogramToCoords,
    MaskToDistogramPipeline,
    AsymmetricDistogramToMaskPipeline,
)

import albumentations as A
import numpy as np

class RepeatChannelTransform(A.ImageOnlyTransform):
    def __init__(self, repeats=3, axis=2, always_apply=True, p=1):
        super(RepeatChannelTransform, self).__init__(always_apply, p)
        self.repeats = repeats
        self.axis = axis

    def apply(self, image, **params):
        return np.repeat(image[:, :, np.newaxis], self.repeats, axis=self.axis)

class Gray2RGB(RepeatChannelTransform):
    def __init__(self, always_apply=True, p=1):
        super(Gray2RGB, self).__init__(repeats=3, axis=2, always_apply=always_apply, p=p)

def pil_to_tensor(img,window_size,interp_size):
    return T.Compose(
        [
            T.Grayscale(1),
            CropCentroidPipeline(window_size),
            MaskToDistogramPipeline(window_size, interp_size),
            # transformer_coords,
            Gray2RGB(),
            T.ToTensor(),
        ]
    )
    
class Mask2Distogram(A.Compose):
    def __init__(self, window_size, interp_size):
        super(Mask2Distogram, self).__init__(
            [
                A.Resize(window_size, window_size),
                A.Normalize(),
                A.Lambda(image=pil_to_tensor(window_size, interp_size)),
            ]
        )
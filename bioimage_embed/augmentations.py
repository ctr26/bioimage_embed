import albumentations as A
import cv2

from typing import Any

import albumentations
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import random
from albumentations.pytorch.transforms import ToTensorV2

DEFAULT_ALBUMENTATION = A.Compose(
    [
        # Flip the images horizontally or vertically with a 50% chance
        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=0.5,
        ),
        # Rotate the images by a random angle within a specified range
        A.Rotate(limit=45, p=0.5),
        # Randomly scale the image intensity to adjust brightness and contrast
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        # Apply random elastic transformations to the images
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.5,
        ),
        # Shift the image channels along the intensity axis
        A.ChannelShuffle(p=0.5),
        # Add a small amount of noise to the images
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # Crop a random part of the image and resize it back to the original size
        A.RandomResizedCrop(
            height=224, width=224, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.5
        ),
        # Adjust image intensity with a specified range for individual channels
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ToTensorV2(),
    ]
)
DEFAULT_AUGMENTATION = transforms.Compose(
    [
        transforms.RandomApply(
            [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
            ],
            p=0.5,
        ),
        transforms.RandomRotation(45),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0)],
            p=0.5,
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        # RandomGamma is not directly available in PyTorch, so you might need to implement it
        # or adjust your augmentation pipeline accordingly.
    ]
)


class AlbumentationsVisionWrapper(A.Compose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = A.from_dict(self.transforms)

    def __call__(self, image):
        # Convert PIL Image to numpy array as Albumentations works with numpy arrays
        image_np = np.array(image)
        # Apply the transformations
        transformed = super().__call__(image=image_np)
        augmented = self.transforms(image=image_np)
        # Return the transformed image
        return transformed["image"]

# class AlbumentationsTransform:
#     def __init__(self, transform):
#         self.transform = transform

#     def __call__(self, img):
#         # Convert PIL image to numpy array
#         img = np.array(img)
#         # Apply transformations
#         transformed = self.transform(image=img)
#         return transformed['image']
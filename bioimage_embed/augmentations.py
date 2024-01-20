import albumentations as A
import cv2

from typing import Any

import albumentations
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image

DEFAULT_AUGMENTATION_LIST = [
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
        height=512, width=512, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.5
    ),
    # Adjust image intensity with a specified range for individual channels
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
]

DEFAULT_AUGMENTATION = A.Compose(DEFAULT_AUGMENTATION_LIST)
DEFAULT_AUGMENTATION_DICT = DEFAULT_AUGMENTATION.to_dict()

class TransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module.

        Args:
            transforms_cfg (DictConfig): Transforms config.
        """

        self.mode = "train"

        # train augmentations
        train_aug = []
        if not transforms_cfg.train.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for aug_name in transforms_cfg.train.get("order"):
            aug = hydra.utils.instantiate(
                transforms_cfg.train.get(aug_name), _convert_="object"
            )
            train_aug.append(aug)
        self.train_aug = albumentations.Compose(train_aug)

        # valid, test and predict augmentations
        valid_test_predict_aug = []
        if not transforms_cfg.valid_test_predict.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for aug_name in transforms_cfg.valid_test_predict.get("order"):
            aug = hydra.utils.instantiate(
                transforms_cfg.valid_test_predict.get(aug_name),
                _convert_="object",
            )
            valid_test_predict_aug.append(aug)
        self.valid_test_predict_aug = albumentations.Compose(valid_test_predict_aug)

    def set_mode(self, mode: str) -> None:
        """Set `__call__` mode.

        Args:
            mode (str): Applying mode.
        """

        self.mode = mode

    def __call__(self, image: Any, **kwargs: Any) -> Any:
        """Apply TransformsWrapper module.

        That module has two modes: `train` and `valid_test_predict`.

        Args:
            image (Any): Input image.
            kwargs (Any): Additional arguments.

        Returns:
            Any: Transformation results.
        """

        if isinstance(image, Image.Image):
            image = np.asarray(image)
        if self.mode == "train":
            return self.train_aug(image=image, **kwargs)
        return self.valid_test_predict_aug(image=image, **kwargs)

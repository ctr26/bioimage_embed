from typing import Any, Tuple
from torchvision.datasets import FakeData, ImageFolder
from torchvision.transforms import ToTensor
import albumentations as A


class FakeImageFolder(FakeData):
    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        loader=None,
        is_valid_file=None,
        size: int = 1000,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 10,
    ):
        # Initialize the ImageFolder with the root directory and other parameters
        super().__init__(
            size=size,
            image_size=image_size,
            num_classes=num_classes,
            transform=transform,
        )

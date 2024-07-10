import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

DEFAULT_AUGMENTATION_LIST = [
    A.OneOf(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        p=0.5,
    ),

    A.Rotate(limit=45, p=0.5),
    # A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.ElasticTransform(
        alpha=1,
        sigma=50,
        alpha_affine=50,
        p=0.5,
    ),
    # A.ChannelShuffle(p=0.5),
    A.RandomResizedCrop(
        height=224, width=224, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=1.0
    ),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
]

DEFAULT_AUGMENTATION = A.Compose(DEFAULT_AUGMENTATION_LIST)
DEFAULT_ALBUMENTATION = A.Compose(DEFAULT_AUGMENTATION_LIST)


class VisionWrapper:
    def __init__(self, transform_dict, *args, **kwargs):
        self.transform_dict = transform_dict
        self.transform = A.from_dict(transform_dict)

    def __call__(self, image):
        try:
            img = np.array(image)
            transformed = self.transform(image=img)
            return transformed["image"]
        except:
            return None,None


class VisionWrapperSupervised:
    def __call__(self, data):
        raise NotImplementedError

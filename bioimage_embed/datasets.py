#  %%
import glob
import random

from torch.utils.data import Dataset
from PIL import Image
import numpy as np


from functools import lru_cache

from albumentations import Compose
from typing import Callable
import torch


def filter_dataset(dataset: torch.Tensor):
    valid_indices = []
    # Iterate through the dataset and apply the transform to each image
    for idx in range(len(dataset)):
        try:
            image, label = dataset[idx]
            # If the transform works without errors, add the index to the list of valid indices
            valid_indices.append(idx)
        except Exception as e:
            # A better way to do with would be with batch collation
            print(f"Error occurred for image {idx}: {e}")
        return torch.utils.data.Subset(dataset, valid_indices)


class DatasetGlob(Dataset):
    def __init__(
        self,
        path_glob,
        over_sampling=1,
        transform: Callable = Compose([]),
        samples=-1,
        shuffle=True,
        **kwargs,
    ):
        self.image_paths = glob.glob(path_glob, recursive=True)
        if shuffle:
            random.shuffle(self.image_paths)
        if samples > 0 and samples < len(self.image_paths):
            self.image_paths = self.image_paths[0:samples]
        self.transform = transform
        self.samples = samples
        self.over_sampling = over_sampling
        assert len(self.image_paths) > 0

    def __len__(self):
        return len(self.image_paths) * self.over_sampling

    @lru_cache(maxsize=None)
    def get_cached_image(self, index):
        return Image.open(self.image_paths[index])

    def getitem(self, index, cached=False):
        safe_idx = int((index) % (len(self) / self.over_sampling))

        if cached:
            x = self.get_cached_image(safe_idx)
        else:
            x = Image.open(self.image_paths[safe_idx])

        if self.transform is not None:
            augmented = self.transform(image=np.array(x))
            # x = Image.fromarray(augmented['image'])
            return augmented["image"]

    def is_image_cropped(self, image):
        if (
            np.sum(
                np.sum(image[:, 0]),
                np.sum(image[:, -1]),
                np.sum(image[0, :]),
                np.sum(image[-1, :]),
            )
            == 0
        ):
            return False
        else:
            return True

    # def __getitem__(self, index):
    #     x = self.getitem(index)
    #     if isinstance(index, slice):
    #         return [self.getitem(i) for i in range(*index.indices(len(self)))]
    #     else:
    #         return x

    def __getitem__(self, index):
        # x = self.getitem(index)
        # if self.is_image_cropped(x):
        # return index
        # if isinstance(index, slice):
        # return [self.getitem(i) for i in range(*index.indices(10))]
        # TODO implement indexing/slicing
        # return self.getitem(index)

        # return self.getitem(index)
        dummy_list = np.arange(0, self.__len__())
        loop = np.array(dummy_list[index])
        if isinstance(index, slice):
            return [self.getitem(i) for i in loop]
        return self.getitem(index)

    # else:
    #     return self.getitem(index+x)

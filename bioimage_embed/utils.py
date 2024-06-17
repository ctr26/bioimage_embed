import torch
from types import SimpleNamespace
import pickle
import base64
import hashlib


def collate_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_test_image(dataset):
    test_img = next(item for item in dataset if item is not None)
    return test_img.unsqueeze(0)


def merge_params(list_of_dicts):
    """Merge multiple dictionaries into one."""
    return SimpleNamespace({k: v for d in list_of_dicts for k, v in d.items()})


def hashing_fn(args):
    serialized_args = pickle.dumps(vars(args))
    hash_object = hashlib.sha256(serialized_args)
    hashed_string = base64.urlsafe_b64encode(hash_object.digest()).decode()
    return hashed_string

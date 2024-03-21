from .. import config
from .. import bioimage_embed
import pytest
from hydra.utils import instantiate
from unittest.mock import patch, MagicMock

schema_map = config.__schemas__
schemas = list(schema_map.values())


@pytest.mark.parametrize("Schema", schemas)
def test_schema(Schema):
    Schema()


@pytest.mark.parametrize("Schema", schemas)
def test_instantiate(Schema):
    obj = instantiate(Schema())
    assert obj is not None, "obj should not be None"


@pytest.fixture
def cfg():
    # dataset = config.ImageFolderDataset(root="data")
    input_dim = [3, 224, 224]
    mock_dataset = config.ImageFolderDataset(
        _target_="bioimage_embed.datasets.FakeImageFolder",
        image_size=input_dim,
    )
    model = config.Model(input_dim=input_dim)
    return config.Config(dataset=mock_dataset,model=model)


def test_config(cfg):
    assert cfg is not None, "Config should not be None"


def test_bioimage_embed(cfg):
    bie = bioimage_embed.BioImageEmbed(cfg)
    bie.model_check()


# def mock_dataset(root="data", *args, **kwargs):
#     return config.ImageFolderDataset(
#         _target_="torchvision.datasets.FakeData", *args, **kwargs
#     )

# def MockDataset(torchvision.datasets.FakeData)

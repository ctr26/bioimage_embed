from .. import bioimage_embed
import pytest
import torch


@pytest.fixture()
def test_bioimage_embed():
    bie = bioimage_embed.BioImageEmbed()
    bie.train()
    bie.infer()
    bie.validate()
    model_output = bie(torch.tensor([1, 2, 3, 4, 5]))
    tensor = bie.model(torch.tensor([1, 2, 3, 4, 5]))

    bie.model(torch.tensor([1, 2, 3, 4, 5]))

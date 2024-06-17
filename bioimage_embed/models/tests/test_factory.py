from bioimage_embed.models import create_model
import pytest
import torch

from bioimage_embed.models import MODELS

# MODELS = ["resnet18_vae", "resnet50_vae", "resnet18_vqvae", "resnet50_vqvae"]

# input_dim = [(3, 256, 256), (3, 224, 224),(1, 224, 224)]
image_dim = [(256, 256), (224, 224)]
channel_dim = [
    3,
]
latent_dim = [64, 16]
pretrained_options = [True, False]
progress_options = [True, False]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("ld", latent_dim)
@pytest.mark.parametrize("c", channel_dim)
@pytest.mark.parametrize("idim", image_dim)
@pytest.mark.parametrize("pretrained", pretrained_options)
@pytest.mark.parametrize("progress", progress_options)
def test_create_model(model, c, idim, ld, pretrained, progress):
    input_dim = (c, *idim)
    generated_model = create_model(model, input_dim, ld, pretrained, progress)
    data = torch.rand(1, *input_dim)
    output = generated_model({"data": data})
    assert output.z.shape[1] == ld
    assert output.recon_x.shape == data.shape
    if len(output.z.flatten()) != ld:
        pytest.skip("Not an exact latent dimension match")

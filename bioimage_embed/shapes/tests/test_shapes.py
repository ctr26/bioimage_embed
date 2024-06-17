import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt


# Note - you must have torchvision installed for this example

# from bioimage_embed.datasets import BroadDataset, DatasetGlob
from bioimage_embed.shapes.transforms import (
    MaskToDistogramPipeline,
    DistogramToMaskPipeline,
    DistogramToCoords,
)

# from bioimage_embed.models import VQ_VAE, BioimageEmbed, VAE
# from bioimage_embed.lightning import AutoEncoderUnsupervised
# from bioimage_embed.models.legacy import vae

interp_size = 128 * 4
window_size = 64
latent_dim = 64
channels = 1
input_dim = (channels, window_size, window_size)

# Assuming create_model is a function that creates models
from bioimage_embed import create_model

# Model names for parametrization
model_names = ["resnet18_vae", "resnet18_vqvae"]

@pytest.fixture(scope="session")
def model(request):
    return create_model(request.param, input_dim, latent_dim)

@pytest.fixture(scope="session", params=model_names)
def model_fixture(request):
    return model(request)

@pytest.fixture(scope="session", params=[(32, (64, 64)), (64, (128, 128))])
def circle_contour(request):
    radius, image_size = request.param
    image = np.zeros(image_size, dtype=np.uint8)
    center = (image_size[0] // 2, image_size[1] // 2)
    y, x = np.ogrid[:image_size[0], :image_size[1]]
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    image[np.abs(distance - radius) < 1] = 255  # Set contour to white
    return image, distance

@pytest.fixture
def prepared_circle_contour(circle_contour):
    test_img, test_dist = circle_contour
    test_img_torch = torch.tensor(test_img).unsqueeze(0).unsqueeze(0)
    test_dist_torch = torch.tensor(test_dist).unsqueeze(0).unsqueeze(0)
    return test_img_torch, test_dist_torch

# @pytest.mark.parametrize(indirect=True)
# class TestMask:
#     def test_pipeline_forward(self, prepared_circle_contour):
#         test_img_torch, _ = prepared_circle_contour
#         dist = MaskToDistogramPipeline(window_size)(test_img_torch)
#         plt.imshow(dist.squeeze())
#         plt.savefig("tests/test_pipeline_forward.png")
#         plt.close()
#         mask = DistogramToMaskPipeline(window_size)(dist)
#         plt.imshow(mask.squeeze())
#         plt.savefig("tests/test_dist_to_mask.png")
#         plt.close()

#     def test_mask_to_dist(self, prepared_circle_contour ):
#         test_img_torch, test_dist_torch = prepared_circle_contour
#         dist = MaskToDistogramPipeline(window_size)(test_img_torch)
#         assert torch.allclose(dist, test_dist_torch), "Distogram does not match expected output"

#     def test_dist_to_coord(self, prepared_circle_contour,):
#         test_img_torch, _ = prepared_circle_contour
#         coords = DistogramToCoords(window_size)(test_img_torch)
#         plt.scatter(coords[-1][0][:, 0], coords[0][0][:, 1])
#         plt.savefig("tests/test_dist_to_coord.png")
#         plt.close()

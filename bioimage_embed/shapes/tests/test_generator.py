#  %%

import pytest
import torch
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
import bioimage_embed.models
import bioimage_embed
import matplotlib.pyplot as plt

from bioimage_embed.shapes.transforms import (
    DistogramToCoords,
    CropCentroidPipeline,
)
from bioimage_embed.shapes.transforms import (
    MaskToDistogramPipeline,
    AsymmetricDistogramToMaskPipeline,
)


@pytest.fixture(scope="session")
def gray2rgb():
    return T.Lambda(lambda x: x.repeat(3, 1, 1))


@pytest.fixture(scope="session")
def window_size():
    return 128 - 32


@pytest.fixture(scope="session")
def interp_size(window_size):
    return window_size * 4


@pytest.fixture(scope="session")
def input_dim(window_size):
    return (3, window_size, window_size)


@pytest.fixture(scope="session")
def latent_dim():
    return 16


@pytest.fixture(scope="session")
def model(input_dim, latent_dim):
    return bioimage_embed.models.create_model(
        "resnet18_vqvae", input_dim, latent_dim, pretrained=False, progress=False
    )


@pytest.fixture(scope="session")
def voc_dataset(window_size):
    voc_transform = T.Compose([T.ToTensor()])
    dataset = VOCSegmentation(
        root="/tmp",
        year="2012",
        image_set="train",
        download=True,
        transform=voc_transform,
        target_transform=voc_transform,
    )
    return dataset


@pytest.fixture(scope="session")
def binary_mask_tensor(voc_dataset):
    _, mask = voc_dataset[0]
    return mask[0]


@pytest.fixture(scope="session")
def binary_mask(binary_mask_tensor):
    return T.ToPILImage()(binary_mask_tensor)


@pytest.fixture(scope="session")
def transformer_crop(window_size):
    return CropCentroidPipeline(window_size)


@pytest.fixture(scope="session")
def transformer_dist(window_size, interp_size):
    return MaskToDistogramPipeline(window_size, interp_size)


@pytest.fixture(scope="session")
def transformer_coords(window_size):
    return DistogramToCoords(window_size)


@pytest.fixture(scope="session")
def transform(transformer_crop, transformer_dist, binary_mask):
    gray2rgb = T.Lambda(lambda x: x.repeat(3, 1, 1))
    transformer = T.Compose(
        [
            T.Grayscale(1),
            transformer_crop,
            transformer_dist,
            # transformer_coords,
            T.ToTensor(),
            gray2rgb,
        ]
    )
    return transformer


@pytest.fixture(scope="session")
def distance_matrix(transform, binary_mask):
    return transform(binary_mask)

@pytest.fixture(scope="session")
def distance_matrix_tensor(distance_matrix):
    return distance_matrix.unsqueeze(0).float()


def test_decoder(
    model,
    distance_matrix_tensor,
    window_size,
):
    # Test the decoder part of the model
    output_e = model.encoder(distance_matrix_tensor)
    output_d = model.decoder(output_e.embedding)
    y_prime = output_d.reconstruction
    assert y_prime.shape == (
        1,
        3,
        window_size,
        window_size,
    ), "Decoder output shape mismatch"


def test_generate(model, binary_mask_tensor, distance_matrix_tensor, window_size):
    # Test the generate functionality
    z = model.encoder(distance_matrix_tensor)
    z_random = torch.normal(2*(torch.ones_like(z.embedding,)))
    generated_image_dist = model.decoder(z_random)
    np_dist = generated_image_dist.reconstruction.detach().numpy()
    mask = AsymmetricDistogramToMaskPipeline(window_size)(np_dist)

    plt.imshow(binary_mask_tensor, cmap="gray")
    plt.savefig("tests/test_generate_img_crop.png")
    plt.close()
    
    plt.imshow(mask.transpose([0,3,2,1])[0].astype(float))
    plt.savefig("tests/test_generate_mask.png")
    plt.close()
    assert mask is not None, "Generated mask is None"


def test_encoder(model, distance_matrix_tensor):
    # Test the encoder part of the model
    z = model.encoder(distance_matrix_tensor)
    assert z is not None, "Encoder output is None"

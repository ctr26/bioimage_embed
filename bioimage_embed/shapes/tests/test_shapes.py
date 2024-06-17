import pytest
import torch
import matplotlib.pyplot as plt


# Note - you must have torchvision installed for this example

# from bioimage_embed.datasets import BroadDataset, DatasetGlob
from bioimage_embed.shapes.transforms import (
    MaskToDistogramPipeline,
    DistogramToMaskPipeline,
)
from bioimage_embed.shapes.transforms import (
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

models = []
import numpy as np


def create_circle_contour(radius, image_size):
    image = np.zeros(image_size, dtype=np.uint8)
    center = (image_size[0] // 2, image_size[1] // 2)

    y, x = np.ogrid[: image_size[0], : image_size[1]]
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    image[np.abs(distance - radius) < 1] = 255  # Set contour to white

    return image, distance


(test_imgs,) = [create_circle_contour(32, (64, 64))]


@pytest.mark.parametrize("model", models)
# @pytest.mark.parametrize("test_img", test_imgs)
class TestMask:
    test_img, test_dist = create_circle_contour(32, (64, 64))

    test_img_torch = torch.tensor(test_img).unsqueeze(0).unsqueeze(0)
    test_dist_torch = torch.tensor(test_dist).unsqueeze(0).unsqueeze(0)

    def test_pipeline_forward(self, model):
        # dist = MaskToDistogramPipeline(window_size)(train_dataset_raw[0])
        # plt.imshow(dist)
        # plt.savefig("tests/test_mask_to_dist.png")
        # plt.close()
        # plt.close()
        dist = MaskToDistogramPipeline(window_size)(self.test_img_torch)
        plt.imshow(dist.squeeze())
        plt.savefig("tests/test_pipeline_forward.png")
        plt.close()
        mask = DistogramToMaskPipeline(window_size)(dist)
        plt.imshow(mask.squeeze())
        plt.savefig("tests/test_dist_to_mask.png")
        plt.close()

    def test_mask_to_dist(self, model):
        dist = MaskToDistogramPipeline(window_size)(self.test_img_torch)
        assert dist == self.test_dist_torch
        pass

    def test_dist_to_coord(self, model, test_img):
        # dist = transformer_dist(train_dataset[-1][0])
        # TODO Faulty?
        # test_img = get_test_image(dataset)
        coords = DistogramToCoords(window_size)(test_img)
        plt.scatter(coords[-1][0][:, 0], coords[0][0][:, 1])
        plt.savefig("tests/test_dist_to_coord.png")
        plt.close()


# @pytest.mark.parametrize("model", models)
# class TestModels:


#     def test_dist_to_coord(self, model, test_img):
#         # test_img = get_test_image(dataset)
#         # dist = transformer_dist(train_dataset[0][0])
#         coords = DistogramToCoords(window_size)(test_img)
#         plt.scatter(coords[0][:, 0], coords[0][:, 1])
#         plt.savefig("tests/test_dist_to_coord.png")
#         plt.close()

#     def test_models(self, model, test_img):
#         # vae = AutoEncoder(1, 1)
#         # vae = VQ_VAE(channels=1)

#         # test_img = get_test_image(dataset)
#         # loss, x_recon, perplexity = model(img)
#         result = model(test_img)
#         z, log_var = model.encode(test_img)
#         y_prime = model.decode(z)
#         # print(f"img_dims:{img.shape} y:_dims:{x_recon.shape}")
#         print(f"img_dims:{test_img.shape}, z:_dims:{z.shape}")

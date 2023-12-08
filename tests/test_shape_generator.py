# import os
# import torch
# import matplotlib.pyplot as plt
# import bioimage_embed

# from bioimage_embed.shapes.transforms import (
#     CropCentroidPipeline,
#     MaskToDistogramPipeline,
#     AsymmetricDistogramToMaskPipeline,
#     DistogramToCoords,
# )

# # Constants
# interp_size = 128 * 4
# window_size = 128 - 32

# # Initialize transformers
# transformer_crop = CropCentroidPipeline(window_size)
# transformer_dist = MaskToDistogramPipeline(window_size, interp_size)
# transformer_coords = DistogramToCoords(window_size)

# # Set up dataset (assuming the dataset setup is handled elsewhere)
# train_dataset = None
# train_dataset_crop = None

# # Model setup
# input_dim = (3, window_size, window_size)
# latent_dim = 16
# model = bioimage_embed.models.create_model(
#     "resnet18_vqvae_legacy", input_dim, latent_dim, pretrained=False, progress=False
# )

# def test_img_test(test_img):
#     plt.imshow(test_img[0][0])
#     plt.close()

# def test_forward_test(test_img):
#     results = model.forward(test_img)
#     # Add assertions or checks on results if needed

# def test_encoder_decoder(test_img):
#     z = model.encoder(test_img)
#     y_prime = model.decoder(z).detach().numpy()
#     plt.imshow(y_prime[0][0])
#     plt.close()

# def test_generate(test_img):
#     z = model.encoder(test_img)
#     z_random = torch.normal(torch.zeros_like(z), torch.ones_like(z))
#     z_random = z + torch.normal(torch.zeros_like(z), torch.ones_like(z)) / 20
#     generated_image_dist = model.decoder(z_random).detach().numpy()

#     mask = AsymmetricDistogramToMaskPipeline(window_size)(generated_image_dist)

#     plt.imshow(img_crop[0][0])
#     plt.savefig("tests/test_generate_img_crop.png")
#     plt.close()
#     plt.imshow(mask[0][0])
#     plt.savefig("tests/test_generate_mask.png")
#     plt.close()

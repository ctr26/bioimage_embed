#  %%
import matplotlib.pyplot as plt
import numpy as np
import pythae
import torch
import umap
import umap.plot

#  %%

# Note - you must have torchvision installed for this example
from torchvision import transforms
from tqdm import tqdm

from bioimage_embed.datasets import DatasetGlob
from bioimage_embed.lightning import AutoEncoderUnsupervised
from bioimage_embed.models.legacy import VQ_VAE, BioimageEmbed

latent_dim = 64
window_size = 64 * 2

batch_size = 32
num_training_updates = 15000

num_hiddens = 64
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3
dataset = "ivy_gap"
data_dir = "data"
channels = 3

input_dim = (channels, window_size, window_size)
model_name = "VQ_VAE"
train_dataset_glob = f"{data_dir}/{dataset}/random/*png"
model_dir = f"models/{dataset}_{model_name}"
ckpt_file = "models/ivy_gap_BioimageEmbed/last.ckpt"

model_config_vqvae = pythae.models.VQVAEConfig(
    input_dim=input_dim, latent_dim=latent_dim, num_embeddings=num_embeddings
)
model = BioimageEmbed("VQ_VAE", model_config=model_config_vqvae, channels=channels)

args = SimpleNamespace(**params, **optimizer_params, **lr_scheduler_params)
lit_model = AutoEncoderUnsupervised(model, args)
model = AutoEncoderUnsupervised(model).load_from_checkpoint(ckpt_file, model=model)
train_dataset = DatasetGlob(train_dataset_glob)

train_dataset = DatasetGlob(train_dataset_glob, transform=transforms.ToTensor())

# %%
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(4.0, 4.0))
grid = ImageGrid(
    fig,
    111,  # similar to subplot(111)
    nrows_ncols=(4, 4),  # creates 2x2 grid of axes
    axes_pad=0.1,  # pad between axes in inch.
)

for i, ax in enumerate(grid):
    ax.imshow(train_dataset[i][0])
plt.show()
plt.close()
# %%

image_index = 5
test_img_in = train_dataset[image_index][:, 0:window_size, 0:window_size].unsqueeze(0)


# I have upstreamed this function in the vqvae class but use this for now
def vqvae_to_latent(model: VQ_VAE, img: torch.Tensor) -> torch.Tensor:
    vq = model.get_model().model._vq_vae
    embedding_torch = vq._embedding
    embedding_in = model.get_model().model.encoder_z(img)
    embedding_out = vq(embedding_in)
    latent = embedding_torch(embedding_out[-1].argmax(axis=1))

    return latent


tensor = vqvae_to_latent(model, test_img_in)
axd = plt.figure(constrained_layout=True, figsize=(12, 8)).subplot_mosaic(
    """
    Aa
    Bb
    Cc
    """,
)

axd["A"].imshow(test_img_in[0][0])
axd["A"].set_title("test_img_in")

plt.show()
plt.close()
# %%
z_list = []
z_dict = {}
for i, data in enumerate(tqdm(train_dataset)):
    if data is not None:
        z = vqvae_to_latent(model, data.unsqueeze(0))
        z_list.append(z)
        z_dict[i] = z
    if len(z_list) >= 100:
        break
latent = torch.stack(z_list).detach().numpy()
#  %%

latent_umap = latent.reshape(latent.shape[0], -1)
unfit_umap = umap.UMAP(n_neighbors=3, min_dist=0.1, metric="cosine", random_state=42)
unfit_umap = umap.UMAP(random_state=42)

fit_umap = unfit_umap.fit(latent_umap)
proj = fit_umap.transform(latent_umap)

umap_z = fit_umap.transform(z.detach().numpy().reshape((1, latent_umap[0].shape[0])))

umap.plot.points(fit_umap)
plt.show()
plt.close()
plt.scatter(proj[:, 0], proj[:, 1])
plt.savefig("latent_space.pdf")
plt.show()
plt.close()
# %%
# Sorted images

fig = plt.figure(figsize=(4.0, 4.0))
grid = ImageGrid(
    fig,
    111,  # similar to subplot(111)
    nrows_ncols=(10, 10),  # creates 2x2 grid of axes
    axes_pad=0.1,  # pad between axes in inch.
)

indices = np.argsort(proj[:, 0])

for i, ax in enumerate(grid):
    ax.imshow(train_dataset[indices[i]][0])
plt.show()
plt.close()

from .lightning import MaskEmbed, MaskEmbedLatentAugment
import torch
import torch.nn.functional as F
from .transforms import DistogramToMaskPipeline

def mask_from_latent(self, z, window_size):
    # This should be class-method based
    # I.e. self.decoder(z)
    dist = self.decoder(z).detach().numpy()
    mask = DistogramToMaskPipeline(window_size)(dist)
    return mask


def loss_function(self, *args, recons, input, distance_matrix_loss=True, **kwargs):
    # decode_z, input, mu, log_var = kwargs
    # # Check to see if distance matrix creates a shape without intersecting edges
    # x_diff = torch.diff(recons,1,-1)-torch.diff(recons,2,-1)
    # y_diff = torch.diff(recons,1,-2)

    # Need to invent metric for ensuring that the final shape is a simple polygon

    diag_loss = F.mse_loss(
        torch.diagonal(recons), torch.zeros_like(torch.diagonal(recons))
    )
    symmetry_loss = F.mse_loss(recons, recons.transpose(3, 2))
    vae_loss = self.model.loss_function(*args, recons=recons, input=input, **kwargs)
    if distance_matrix_loss:
        vae_loss["loss"] = (
            8 / 10 * vae_loss["loss"] + 1 / 10 * diag_loss + 1 / 10 * symmetry_loss
        )

    return vae_loss

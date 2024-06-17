# import torch

# import torch.nn.functional as F

# Note - you must have torchvision installed for this example
# from torch.utils.data import DataLoader

# from bioimage_embed.transforms import DistogramToMaskPipeline


# from .vae_bio import Mask_VAE, Image_VAE

# from .bolts import ResNet18VAEEncoder, ResNet18VAEDecoder

import pythae
from .pythae import legacy
from . import bolts
from functools import partial


class ModelFactory:
    def __init__(
        self, input_dim, latent_dim, pretrained=False, progress=True, **kwargs
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.pretrained = pretrained
        self.progress = progress
        self.kwargs = kwargs

    def create_model(
        self, model_config_class, model_class, encoder_class, decoder_class
    ):
        model_config = model_config_class(
            input_dim=tuple(self.input_dim),
            latent_dim=self.latent_dim,
        )
        encoder = encoder_class(model_config)
        decoder = decoder_class(model_config)
        # TODO Fix this
        return model_class(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )

    def dummy_model(self):
        return self.create_model(
            pythae.models.VAEConfig,
            pythae.models.VAE,
            lambda x: None,
            lambda x: None,
        )

    def resnet_vae_bolt(
        self,
        enc_type,
        enc_out_dim,
        first_conv=False,
        maxpool1=False,
        kl_coeff=1.0,
    ):
        return self.create_model(
            pythae.models.VAEConfig,
            partial(
                bolts.vae.VAEPythaeWrapper,
                input_height=self.input_dim[1],
                enc_type=enc_type,
                enc_out_dim=enc_out_dim,
                first_conv=first_conv,
                maxpool1=maxpool1,
                kl_coeff=kl_coeff,
            ),
            encoder_class=lambda x: None,
            decoder_class=lambda x: None,
        )

    # bolts.vae.VAEPythaeWrapper(
    #         input_height=self.input_dim[1],
    #         enc_type=enc_type,
    #         enc_out_dim=enc_out_dim,
    #         first_conv=first_conv,
    #         maxpool1=maxpool1,
    #         kl_coeff=kl_coeff,
    #         latent_dim=self.latent_dim,
    #     )

    def resnet18_vae_bolt(self, **kwargs):
        return self.resnet_vae_bolt(enc_type="resnet18", enc_out_dim=512, **kwargs)

    def resnet50_vae_bolt(self, **kwargs):
        return self.resnet_vae_bolt(enc_type="resnet50", enc_out_dim=2048, **kwargs)

    def resnet18_vae(self):
        return self.create_model(
            partial(
                pythae.models.VAEConfig,
                use_default_encoder=False,
                use_default_decoder=False,
                **self.kwargs,
            ),
            pythae.models.VAE,
            bolts.ResNet18VAEEncoder,
            bolts.ResNet18VAEDecoder,
        )

    def resnet50_vae(self):
        return self.create_model(
            partial(
                pythae.models.VAEConfig,
                use_default_encoder=False,
                use_default_decoder=False,
                **self.kwargs,
            ),
            pythae.models.VAE,
            bolts.ResNet50VAEEncoder,
            bolts.ResNet50VAEDecoder,
        )

    def resnet18_vqvae(self):
        return self.create_model(
            partial(
                pythae.models.VQVAEConfig,
                use_default_encoder=False,
                use_default_decoder=False,
                **self.kwargs,
            ),
            pythae.models.VQVAE,
            bolts.ResNet18VQVAEEncoder,
            bolts.ResNet18VQVAEDecoder,
        )

    def resnet50_vqvae(self):
        return self.create_model(
            partial(
                pythae.models.VQVAEConfig,
                use_default_encoder=False,
                use_default_decoder=False,
                **self.kwargs,
            ),
            pythae.models.VQVAE,
            bolts.ResNet50VQVAEEncoder,
            bolts.ResNet50VQVAEDecoder,
        )

    def resnet_vae_legacy(self, depth):
        return self.create_model(
            pythae.models.VAEConfig,
            partial(legacy.VAE, num_residual_hiddens=depth),
            encoder_class=lambda x: None,
            decoder_class=lambda x: None,
        )

    def resnet18_vae_legacy(self):
        return self.resnet_vae_legacy(18)

    def resnet50_vae_legacy(self):
        return self.resnet_vae_legacy(50)

    def resnet_vqvae_legacy(self, depth):
        return self.create_model(
            pythae.models.VQVAEConfig,
            # partial(legacy.vq_vae.VQVAE,**self.kwargs,num_hidden_residuals=depth),
            partial(legacy.vq_vae.VQVAE, depth=depth),
            encoder_class=lambda x: None,
            decoder_class=lambda x: None,
        )

    def resnet18_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(18)

    def resnet50_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(50)

    def resnet101_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(101)

    def resnet110_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(150)

    def resnet152_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(152)

    def __call__(self, model):
        return getattr(self, model)()

    #    return getattr(self
    #         (
    #             self.input_dim, self.latent_dim, self.pretrained, self.progress),
    #         ),
    #         model,
    #     )


MODELS = [
    "resnet18_vae",
    "resnet50_vae",
    "resnet18_vae_bolt",
    "resnet50_vae_bolt",
    "resnet18_vqvae",
    "resnet50_vqvae",
    "resnet18_vqvae_legacy",
    "resnet50_vqvae_legacy",
    "resnet101_vqvae_legacy",
    "resnet110_vqvae_legacy",
    "resnet152_vqvae_legacy",
    "resnet18_vae_legacy",
    "resnet50_vae_legacy",
    "dummy_model",
]

from typing import Tuple


def create_model(
    model: str,
    input_dim: Tuple[int, int, int],
    latent_dim: int,
    pretrained=False,
    progress=True,
    **kwargs,
):
    factory = ModelFactory(input_dim, latent_dim, pretrained, progress, **kwargs)
    return getattr(factory, model)()

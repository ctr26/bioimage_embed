from .._mae.models_mae import MAEMaskedAutoencoderViT

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder

# class Encoder(BaseEncoder,MAEMaskedAutoencoderViT):


class Encoder(BaseEncoder, MAEMaskedAutoencoderViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super(BaseEncoder, self).__init__(*args, **kwargs)

    def forward(self, x, mask_ratio=0.75):
        x, mask, ids_restore = self.forward_encoder(x)
        return ModelOutput(embedding=x, mask=mask, ids_restore=ids_restore)


class Decoder(BaseDecoder, MAEMaskedAutoencoderViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super(BaseDecoder, self).__init__(*args, **kwargs)

    def forward(self, x, mask_ratio=0.75):
        x = self.forward_decoder(x=x.embedding, ids_restore=x.ids_restore)
        return ModelOutput(reconstruction=x)


# # set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

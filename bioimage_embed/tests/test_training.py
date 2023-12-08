# import pytest
# import torch
# from torch.utils.data import DataLoader

# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# import pythae

# from torch.utils.data import DataLoader

# from bioimage_embed.utils import collate_none

# from bioimage_embed.models import MODELS

# from bioimage_embed.lightning import LitAutoEncoderTorch
# from bioimage_embed.models.legacy import vae
# import  numpy as np
# from bioimage_embed.models import create_model, MODELS


# @pytest.fixture(params=[(3, 64, 64), (1, 64, 64)])
# def input_dim(request):
#     return request.param

# @pytest.fixture(params=np.power(2, np.arange(2, 5)))
# def latent_dim(request):
#     return request.param

# @pytest.fixture(params=[1, 2, 16s])
# def batch_size(request):
#     return request.param

# @pytest.fixture()
# def dataset(batch_size,input_dim):
#     return torch.rand(batch_size,*input_dim)

# @pytest.fixture()
# def model(model_name,input_dim,latent_dim):
#     return create_model(model_name, input_dim=input_dim, latent_dim=latent_dim)

# class TestModels:
#     def test_training(self,model, input_dim, latent_dim,batch_size):
#         # model = create_model(model, input_dim=input_dim, latent_dim=latent_dim)
#         lit_model = LitAutoEncoderTorch(model)
#         dataloader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=1,
#             pin_memory=False,
#             collate_fn=collate_none,
#         )
#         trainer = pl.Trainer(
#             max_steps=1,
#             max_epochs=1,
#         )  # .from_argparse_args(args)
#         trainer.test(lit_model, dataloader)
#         # trainer.fit(lit_model, dataloader)

#     def test_latent_space(self,model,input_dim,latent_dim,batch_size):
#         dataset = torch.rand(batch_size,*input_dim)
#         model = create_model(model, input_dim=input_dim, latent_dim=latent_dim)
#         z = model({"data":dataset})["z"]
#         assert tuple(z.shape) == (batch_size,latent_dim)
from bioimage_embed.models import create_model
import pytest
import torch

import pytorch_lightning as pl

from bioimage_embed.models import create_model, MODELS

from bioimage_embed.models import MODELS
from bioimage_embed.lightning import DataModule
from bioimage_embed.lightning.torch import _3c_model_classes


@pytest.fixture(params=_3c_model_classes)
def model_class(request):
    return request.param

@pytest.fixture(params=MODELS)
def model_name(request):
    return request.param


@pytest.fixture()
def image_dim():
    return (256, 256)


@pytest.fixture()
def channel_dim():
    return 3


@pytest.fixture()
def latent_dim():
    return 16


@pytest.fixture(params=[1, 2, 16])
def batch_size(request):
    return request.param


@pytest.fixture()
def pretrained():
    return True


@pytest.fixture()
def progress():
    return True


@pytest.fixture
def model(model_name, image_dim, channel_dim, latent_dim, pretrained, progress):
    input_dim = (channel_dim, *image_dim)
    return create_model(
        model_name,
        input_dim,
        latent_dim,
        pretrained,
        progress,
    )


@pytest.fixture()
def input_dim(image_dim, channel_dim):
    return (channel_dim, *image_dim)


@pytest.fixture()
def data(input_dim):
    return torch.rand(1, *input_dim)



@pytest.fixture()
def lit_model(model, model_class):
    return model_class(model)
    return LitAutoEncoderTorch(model)

def test_export_onxx(data, lit_model):
    return lit_model.to_onnx("model.onnx", data)

@pytest.fixture()
def model_torchscript(lit_model):
    return lit_model.to_torchscript()


@pytest.mark.skip(reason="Upstream bug with pythae")
def test_export_jit(data, model_torchscript):
    return model_torchscript.save("model.pt")


@pytest.mark.skip(reason="Upstream bug with pythae")
def test_jit_save(model_torchscript):
    return torch.jit.save(model_torchscript, "model.pt", method="script")


@pytest.fixture()
def data(input_dim):
    return torch.rand(1, *input_dim)


@pytest.fixture()
def dataset(data):
    return data


@pytest.fixture()
def dataloader(dataset, batch_size):
    return DataModule(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )


@pytest.fixture()
def trainer():
    return pl.Trainer(
        max_steps=1,
        max_epochs=1,
    )


@pytest.mark.skip(reason="Expensive to run")
def test_trainer_fit(trainer, lit_model, dataloader):
    trainer.fit(lit_model, dataloader)


def test_dataset_trainer(trainer, lit_model, dataset):
    trainer.test(lit_model, dataset.unsqueeze(0))


def test_dataloader_trainer(trainer, lit_model, dataloader):
    trainer.test(lit_model, dataloader)

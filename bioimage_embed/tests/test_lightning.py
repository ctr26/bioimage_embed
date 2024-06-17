import pytest
import torch
import pytorch_lightning as pl
from bioimage_embed.models import MODELS
from bioimage_embed.lightning import (
    DataModule,
    AutoEncoderSupervised,
    AutoEncoderUnsupervised,
)
from bioimage_embed.lightning.torch import _3c_model_classes
from bioimage_embed.models import create_model
from torch.utils.data import TensorDataset


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
    return torch.rand(*input_dim)


@pytest.fixture()
def dataset(data):
    return data.unsqueeze(0)


@pytest.fixture()
def labelled_dataset(data):
    return TensorDataset(data, torch.zeros(1))


@pytest.fixture()
def unlabelled_dataset(data):
    return data


@pytest.fixture()
def supervised_lit_model(model):
    return AutoEncoderSupervised(model)


@pytest.fixture()
def unsupervised_lit_model(model):
    return AutoEncoderUnsupervised(model)


def test_export_onxx_unsupervised(data, unsupervised_lit_model):
    return unsupervised_lit_model.to_onnx("model.onnx", data)


@pytest.fixture()
def labelled_data(data):
    return data, torch.tensor([0])


@pytest.mark.skip(reason="Dictionaries not allowed")
def test_export_onxx_supervised(data, supervised_lit_model):
    return supervised_lit_model.to_onnx("model.onnx", data)


@pytest.fixture()
def data(input_dim):
    return torch.rand(1, *input_dim)


@pytest.fixture()
def labelled_dataloader(labelled_dataset, batch_size):
    return DataModule(
        labelled_dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=1,
        pin_memory=False,
    )


@pytest.fixture()
def unlabelled_dataloader(unlabelled_dataset, batch_size):
    return DataModule(
        unlabelled_dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=1,
        pin_memory=False,
    )


@pytest.fixture()
def dataloader(dataset, batch_size):
    return DataModule(
        dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=1,
        pin_memory=False,
    )

@pytest.fixture()
def trainer():
    return pl.Trainer(
        max_steps=1,
        max_epochs=1,
    )


def test_trainer_test_supervised(trainer, supervised_lit_model, labelled_dataloader):
    return trainer.test(supervised_lit_model, labelled_dataloader)


def test_trainer_test_unsupervised(
    trainer, unsupervised_lit_model, unlabelled_dataloader
):
    return trainer.test(unsupervised_lit_model, unlabelled_dataloader)


@pytest.mark.skip(reason="Expensive")
def test_trainer_fit_supervised(trainer, supervised_lit_model, labelled_dataloader):
    return trainer.fit(supervised_lit_model, labelled_dataloader)


@pytest.mark.skip(reason="Expensive")
def test_trainer_fit_unsupervised(
    trainer, unsupervised_lit_model, unlabelled_dataloader
):
    return trainer.fit(unsupervised_lit_model, unlabelled_dataloader)


def test_dataset_trainer(trainer, supervised_lit_model, labelled_dataset):
    return trainer.test(supervised_lit_model, labelled_dataset)


def test_dataset_trainer(trainer, unsupervised_lit_model, unlabelled_dataset):
    return trainer.test(unsupervised_lit_model, unlabelled_dataset.unsqueeze(0))


@pytest.mark.skip(reason="Dictionaries not allowed")
def test_export_onnx_supervised(data, supervised_lit_model):
    return supervised_lit_model.to_onnx("model.onnx", data)


def test_export_onnx_unsupervised(data, unsupervised_lit_model):
    return unsupervised_lit_model.to_onnx("model.onnx", data)


@pytest.mark.skip(reason="Upstream bug with pythae")
def test_export_jit(data, model_torchscript):
    return model_torchscript.save("model.pt")


@pytest.mark.skip(reason="Upstream bug with pythae")
def test_jit_save(model_torchscript):
    return torch.jit.save(model_torchscript, "model.pt", method="script")

import os
import pytest
from hydra import initialize, compose
from .. import cli
from pathlib import Path
import os
import pytest
from ..cli import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def config_dir():
    return "test_conf"


@pytest.fixture
def config_file():
    return "config.yaml"


@pytest.fixture
def config_path(config_dir, config_file):
    return Path(config_dir).joinpath(config_file)


@pytest.fixture
def config_directory_setup(config_dir, config_file, config_path):
    if config_path.is_file():
        config_path.unlink()

    config_path.parent.mkdir(parents=True, exist_ok=True)

    yield config_dir, config_file, config_path

    if config_path.is_file():
        config_path.unlink()
    if config_dir.is_dir():
        config_dir.rmdir()


def test_write_default_config_file(
    config_path, config_dir, config_file, config_directory_setup
):
    # config_path, config_file = config_directory_setup
    cli.write_default_config_file(config_path)
    assert config_path.is_file(), "Default config file was not created"


from .. import config


@pytest.fixture
def cfg():
    mock_dataset = config.ImageFolderDataset(
        _target_="bioimage_embed.datasets.FakeImageFolder",
    )
    cfg = cli.get_default_config()
    cfg.dataset = mock_dataset
    return cfg


def test_get_default_config(cfg):
    assert cfg is not None, "Default config should not be None"
    # Further assertions can be added to check specific config properties


# def test_main_with_default_config(
#     cfg, config_path, config_dir, config_file, config_directory_setup
# ):
#     test_get_default_config

#     # cli.main(config_dir=config_dir, config_file=config_file, job_name="test_app")


@pytest.mark.skip("Computationally heavy")
def test_hydra():
    #  bie_train model.model="resnet50_vqvae" dataset._target_="bioimage_embed.datasets.FakeImageFolder"
    cfg = config.Config()
    cfg.dataset._target_ = "bioimage_embed.datasets.FakeImageFolder"
    cfg.model.model = "resnet18_vae"
    cfg.recipe.max_epochs = 1

    cli.train(cfg)


#     result = runner.invoke(app, ["main", "+dataset.root=data", "--config_dir", "tests/sample_conf", "--config_file", "sample_config.yaml"])

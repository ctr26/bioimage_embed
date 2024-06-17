import pytest
from hydra import initialize, compose
from .. import cli
from pathlib import Path
from typer.testing import CliRunner
from .. import config

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


@pytest.fixture
def cfg():
    mock_dataset = config.ImageFolderDataset(
        _target_="bioimage_embed.datasets.FakeImageFolder",
    )
    cfg = cli.get_default_config()
    cfg.recipe.data = mock_dataset
    return cfg


def test_get_default_config(cfg):
    assert cfg is not None, "Default config should not be None"
    # Further assertions can be added to check specific config properties


def test_main_with_default_config(
    cfg, config_path, config_dir, config_file, config_directory_setup
):
    test_get_default_config

    # cli.main(config_dir=config_dir, config_file=config_file, job_name="test_app")


# @pytest.mark.skip("Computationally heavy")
def test_hydra():
    #  bie_train model.model="resnet50_vqvae" dataset._target_="bioimage_embed.datasets.FakeImageFolder"
    input_dim = [3, 224, 224]
    cfg = config.Config()
    cfg.dataloader.dataset._target_ = "bioimage_embed.datasets.FakeImageFolder"
    cfg.dataloader.dataset.image_size = input_dim
    cfg.recipe.model = "resnet18_vae"
    cfg.recipe.max_epochs = 1


# def test_cli():
#     # This test checks if the CLI correctly handles the dataset target input
#     result = runner.invoke(app, ["bie_train", "--dataset-target", "bioimage_embed.datasets.FakeImageFolder"])
#     assert result.exit_code == 0
#     assert "Dataset target set to: bioimage_embed.datasets.FakeImageFolder" in result.stdout

#     result = runner.invoke(app, ["main", "+dataset.root=data", "--config_dir", "tests/sample_conf", "--config_file", "sample_config.yaml"])
# def test_init_hydra_with_default_values():
#     config = init_hydra()
#     assert config is not None, "Config should not be None"

# def test_init_hydra_with_custom_values():
#     config_dir = "custom_conf"
#     config_file = "custom_config.yaml"
#     job_name = "custom_job"
#     config = init_hydra(config_dir, config_file, job_name)
#     assert config is not None, "Config should not be None"
#     assert config["config_dir"] == config_dir, "Config directory should match"
#     assert config["config_file"] == config_file, "Config file should match"
#     assert config["job_name"] == job_name, "Job name should match"


def test_init_hydra_with_invalid_config_dir():
    with pytest.raises(Exception):
        cli.init_hydra(config_dir="invalid_dir")


def test_init_hydra_with_invalid_config_file():
    with pytest.raises(Exception):
        cli.init_hydra(config_file="invalid_config.yaml")


@pytest.fixture
def hydra_cfg():
    with initialize(config_path="."):
        # cfg = compose(config_name="config", overrides=[
        #     'dataloader.dataset._target_=bioimage_embed.datasets.FakeImageFolder'
        # ])
        cfg = compose(config_name="config")
        cfg.dataloader.dataset._target_ = "bioimage_embed.datasets.FakeImageFolder"
        return cfg


@pytest.mark.skip("Computationally heavy")
def test_train(hydra_cfg):
    cli.train(hydra_cfg)


def test_check(hydra_cfg):
    cli.check(hydra_cfg)

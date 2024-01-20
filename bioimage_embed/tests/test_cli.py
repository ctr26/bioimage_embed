import os
import pytest
from .. import hydra_cli
from hydra import initialize, compose
from pathlib import Path
import os
import pytest


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
    hydra_cli.write_default_config_file(config_path)
    assert config_path.is_file(), "Default config file was not created"


def test_get_default_config():
    cfg = hydra_cli.get_default_config()
    assert cfg is not None, "Default config should not be None"
    # Further assertions can be added to check specific config properties


def test_main_with_default_config(
    config_path, config_dir, config_file, config_directory_setup
):
    hydra_cli.write_default_config_file(config_path)
    assert config_path.is_file()
    # assert os.path.isfile(config_path), "Default config file was not created"

    # Now, test if main function works correctly with this default configuration
    hydra_cli.main(config_path=config_dir, job_name="test_app")
    # Add your logic here to validate the proper initialization of the configuration
    # This might involve checking if certain expected values are set in the configuration

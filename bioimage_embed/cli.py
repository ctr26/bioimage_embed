# TODO: CLI autocomplete is currently quite slow
from bioimage_embed import BioImageEmbed, Config

from omegaconf import OmegaConf
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

import hydra

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def write_default_config_file(config_path):
    cfg = get_default_config()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as file:
        file.write(OmegaConf.to_yaml(cfg))


# TODO make this work with typer (hard)
# @typer.command()
# @hydra.main(config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     print(cfg)


def init_hydra(config_dir="conf", config_file="config.yaml", job_name="bie"):
    hydra.initialize(
        version_base=None,
        config_path=config_dir,
        job_name=job_name,
    )
    return hydra.compose(config_name=config_file)


def get_default_config(config_name="config"):
    with initialize(config_path=None, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


# TODO smarter way to handle this
@hydra.main(config_path=".", config_name="config", version_base="1.1.0")
def infer():
    pass


@hydra.main(config_path=".", config_name="config", version_base="1.1.0")
def train(cfg: Config):
    bie = BioImageEmbed(cfg)
    bie.train()
    pass


@hydra.main(config_path=".", config_name="config", version_base="1.1.0")
def check(cfg: Config):
    bie = BioImageEmbed(cfg)
    bie.check()


@hydra.main(config_path=".", config_name="config", version_base="1.1.0")
def finetune(cfg: Config):
    pass
    bie = BioImageEmbed(cfg)
    bie.finetune()


# app.command()(train)
# app.command()(infer)

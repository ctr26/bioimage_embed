
from pl_bolts.utils.stability import UnderReviewWarning
import warnings
warnings.simplefilter(action="ignore", category=UnderReviewWarning)

from typer import Typer
from bioimage_embed.bioimage_embed import BioImageEmbed

from omegaconf import OmegaConf
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from .config import Config
import hydra

app = Typer()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def write_default_config_file(config_path):
    cfg = get_default_config()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as file:
        file.write(OmegaConf.to_yaml(cfg))


# TODO make this work with typer (hard)
# @hydra.main(config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     print(cfg)


@hydra.main(config_path=".", config_name="config")
def train(cfg: Config):
# def train(cfg):
    bie = BioImageEmbed(cfg)
    bie.train()
    pass


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


if __name__ == "__main__":
    train()


def infer():
    pass


# app.command()(train)
# app.command()(infer)

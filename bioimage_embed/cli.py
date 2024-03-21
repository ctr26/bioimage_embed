from typer import Typer
from bioimage_embed.bioimage_embed import BioImageEmbed
from dataclasses import dataclass
import torch

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


# @hydra.main(config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     print(cfg)


@hydra.main(config_path=".", config_name="config")
def train(cfg: Config):
    # print(OmegaConf.to_yaml(cfg))
    bie = BioImageEmbed(cfg)
    bie.train()


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


def filter_dataset(dataset: torch.Tensor):
    valid_indices = []
    # Iterate through the dataset and apply the transform to each image
    for idx in range(len(dataset)):
        try:
            image, label = dataset[idx]
            # If the transform works without errors, add the index to the list of valid indices
            valid_indices.append(idx)
        except Exception as e:
            # A better way to do with would be with batch collation
            print(f"Error occurred for image {idx}: {e}")
        return torch.utils.data.Subset(dataset, valid_indices)


if __name__ == "__main__":
    train()

# def infer():
#     main(job_name="test_app")


# def main():
#     app()

# if __name__ == "__main__":
#     main()


# app.command()(train)
# app.command()(infer)

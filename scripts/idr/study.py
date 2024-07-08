import bioimage_embed
import bioimage_embed.config as config
# from ray.tune.integration.pytorch_lightning import (
#     TuneReportCallback,
#     TuneReportCheckpointCallback,
    
# )
from ray import tune
import numpy as np
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from hydra.utils import instantiate
import ray
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from pytorch_lightning import loggers as pl_loggers

if __name__ == "__main__":

    ray.init()
    input_dim = [3, 224, 224]
    # trainer = instantiate(cfg.trainer)
    params_space = {
            "model": tune.choice(
                [
                    "resnet50_vqvae",
                    "resnet110_vqvae_legacy",
                    "resnet152_vqvae_legacy",
                ]
            ),
            # "data": "data",
            "opt": tune.choice(["adamw","LAMB"]),
            "max_epochs": 1000,
            "max_steps": -1,
            "weight_decay": tune.uniform(0.0001, 0.01),
            "momentum": tune.uniform(0.8, 0.99),
            # "sched": "cosine",
            "epochs": 1000,
            "lr": tune.loguniform(1e-6, 1e-2),
            "batch_size": tune.choice([2 **x for x in range(4,12)])
            # tune.qlograndint(4, 4096,q=1,base=2),
            # "min_lr": 1e-6,
            # "t_initial": 10,
            # "t_mul": 2,
            # "decay_rate": 0.1,
            # "warmup_lr": 1e-6,
            # "warmup_lr_init": 1e-6,
            # "warmup_epochs": 5,
            # "cycle_limit": None,
            # "t_in_epochs": False,
            # "noisy": False,
            # "noise_std": 0.1,
            # "noise_pct": 0.67,
            # "cooldown_epochs": 5,
            # "warmup_t": 0,
            # "seed": 42
    }
        
    # root = "/nfs/ftp/public/databases/IDR/idr0093-mueller-perturbation"

    # mock_dataset = config.ImageFolderDataset(
    #     image_size=input_dim,
    #     root="/nfs/",
    # )

    mock_dataset = config.ImageFolderDataset(
        _target_="bioimage_embed.datasets.FakeImageFolder",
        image_size=input_dim,
        num_classes=1,
    )

    dataloader = config.DataLoader(dataset=mock_dataset)
    # breakpoint()
    model = config.Model(input_dim=input_dim)
    

    lit_model = config.LightningModel(
        _target_="bioimage_embed.lightning.torch.AutoEncoderSupervisedNChannels",
        model=model,
    )

    trainer = config.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        # callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
    )
    
    def task():
        cfg = config.Config(dataloader=dataloader, model=model,trainer=trainer)
        bie = bioimage_embed.BioImageEmbed(cfg)
        # bie.icfg.trainer = prepare_trainer(bie.icfg.trainer)
        bie.check()
        return True

    assert task()
    task = ray.remote(task)
    gen = task.remote()

    def train(params):
        
        cfg = config.Config(dataloader=dataloader,
                            model=model,
                            trainer=trainer,
                            recipe=config.Recipe(**params))
        
        
        bie = bioimage_embed.BioImageEmbed(cfg)
        wandb = pl_loggers.WandbLogger(project="bioimage-embed", name="shapes")
        # bie.icfg.trainer = prepare_trainer(bie.icfg.trainer)
        wandb.watch(bie.icfg.lit_model, log="all")
        bie.train()
        wandb.finish()
        return bie


    analysis = tune.run(
        tune.with_parameters(train),
        # resources_per_trial={"cpu": 32, "gpu": 1},
        config=params_space,
        # metric="loss",
        # mode="min",
        num_samples=1,
        scheduler=tune.schedulers.ASHAScheduler(
            metric="val/loss",
            mode="min",
            max_t=10,
            grace_period=1,
            reduction_factor=2,
        ),
    )
    # results = tuner.fit()
    print("Best hyperparameters found were: ", analysis.best_config)


    # bie.export("model")

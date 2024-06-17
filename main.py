# %%
from hydra.utils import instantiate
import torchvision
import albumentations as A
import hydra
from omegaconf import DictConfig, OmegaConf

from bioimage_embed.models.bolts import ResNet18VAEEncoder, ResNet18VAEDecoder
from pythae.models import VAE, VAEConfig
import torch
from bioimage_embed.datasets import DatasetGlob

import pytorch_lightning as pl

# @hydra.main(config_path="conf", config_name="config", version_base="1.2")
# def main(cfg: DictConfig) -> None:
#     # OmegaConf.resolve(cfg)
#     # cli = LightningCLI(Bio_VAE)
#     seed_everything(cfg.seed)
#     # args = cfg.timm
#     # transform = A.from_dict(OmegaConf.to_container(cfg.albumentations, resolve=True))
#     transform = A.Compose(
#         [
#             A.RandomCrop(width=256, height=256, p=1.0),
#             A.Resize(width=224, height=224, p=1.0),
#             A.ToFloat(max_value=1, p=1.0),
#             ToTensorV2(),
#         ]
#     )
#     # dataset = instantiate(cfg.dataset)
#     # pythae = instantiate(cfg.pythae)

#     model_config = VAEConfig(
#         latent_dim=64,
#         input_dim=(3, 64, 64),
#     )

#     encoder = ResNet18VAEEncoder(model_config)
#     decoder = ResNet18VAEDecoder(model_config)

#     pythae = VAE(
#         model_config=model_config,
#         encoder=encoder,
#         decoder=decoder,
#     )
#     # model = instantiate(cfg.model)
#     # dataloader = instantiate(cfg.dataloader, transform=transform)
#     # dataloader.setup()
#     data = DatasetGlob(
#             "/home/ctr26/gdrive/+projects/idr_autoencode_torch/data/ivy_gap/random/*png",
#             transform=transform,
#         )
#     dataloader = torch.utils.data.DataLoader(data, batch_size=4, num_workers=4)
#     trainer = pl.Trainer(
#         gpus=1, max_epochs=100, precision=16
#         )
#     lightning = LitAutoEncoderTorch(pythae,args=None)
#     # test_data = dataloader.get_dataset()[0].unsqueeze(dim=0)
#     # test = pythae({"data": test_data[:, :, :64, :64]})
#     # model = instantiate(cfg.model)
#     # pythae = instantiate(cfg.pythae)
#     # data = DatasetGlob(
#     #     "/home/ctr26/gdrive/+projects/idr_autoencode_torch/data/ivy_gap/random/*png",
#     #     transform=transform,
#     #     )

#     # dataloader = torch.utils.data.DataLoader(data, batch_size=4, num_workers=4)
#     # lightning = instantiate(cfg.lightning, model=pythae)
#     # logger = instantiate(cfg.logger)
#     # checkpoint_callback = instantiate(cfg.checkpoints)
#     # trainer = instantiate(cfg.trainer)

#     # trainer = pl.Trainer(gpus=1, max_epochs=100, precision=16)
#     try:
#         trainer.fit(
#             lightning,
#             datamodule=dataloader,
#             ckpt_path=f"{cfg.checkpoints.dirpath}/last.ckpt",
#         )
#     except:
#         trainer.fit(lightning, datamodule=dataloader)


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    model_config = VAEConfig(
        latent_dim=int(224),
        input_dim=(3, int(224), int(224)),
    )

    # transform = A.Compose(
    #     [
    #         # A.RandomCrop(width=256, height=256, p=1.0),
    #         A.Resize(width=int(224), height=int(224), p=1.0),
    #         A.ToFloat(max_value=1, p=1.0),
    #         ToTensorV2(),
    #     ]
    # )
    transform = A.from_dict(OmegaConf.to_container(cfg.albumentations, resolve=True))
    data = DatasetGlob(
        "data/ivy_gap/random/*png",
        transform=transform,
    )

    # train_loader = torch.utils.data.DataLoader(data, batch_size=4, num_workers=4)
    encoder = ResNet18VAEEncoder(model_config)
    decoder = ResNet18VAEDecoder(model_config)

    model = VAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder,
    )

    class VAEModel(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model = self.model.to(self.device)

        def forward(self, x):
            return self.model({"data": x})

        def training_step(self, batch, batch_idx):
            self.model.train()
            x = {"data": batch}
            model_output = self.model(x, epoch=batch_idx)
            self.loss = model_output.loss
            self.logger.experiment.add_scalar("Loss/train", self.loss, batch_idx)

            self.logger.experiment.add_image(
                "input", torchvision.utils.make_grid(batch), batch_idx
            )
            self.logger.experiment.add_image(
                "output",
                torchvision.utils.make_grid(model_output["recon_x"]),
                batch_idx,
            )
            self.log("train_loss", self.loss)
            return self.loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    # vae_model = VAEModel(model)
    # model = instantiate(cfg.pythae)
    lightning_model = instantiate(cfg.lightning, model=model)
    # train_loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=4)
    train_loader = instantiate(cfg.dataloader, transform=transform)
    trainer = instantiate(cfg.trainer)

    # trainer = pl.Trainer(
    #     devices="auto",
    #     accelerator="gpu",
    #     max_epochs=100,
    #     precision=16,
    #     logger=TensorBoardLogger(save_dir="logs/"),
    #     accumulate_grad_batches=1,
    # )
    trainer.fit(
        lightning_model,
        train_loader,
    )


if __name__ == "__main__":
    main()

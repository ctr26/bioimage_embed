import pyro
import pytorch_lightning as pl

# Note - you must have torchvision installed for this example
import torch
import torchvision


class LitAutoEncoderPyro(pl.LightningModule):
    def __init__(self, model, batch_size=1, learning_rate=1e-3):
        super().__init__()
        # self.autoencoder = AutoEncoder(batch_size, 1)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.autoencoder = model
        # self.vae = VAE()
        # self.vae_flag = vae_flag
        self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

    def forward(self, x):
        return self.autoencoder.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def torch_training_step(self, train_batch, batch_idx):
        inputs = train_batch
        output = self.forward(inputs)
        loss = self.loss_fn(output, inputs)
        self.log("train_loss", loss)
        # tensorboard = self.logger.experiment
        self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)
        # torchvision.utils.make_grid(output)
        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(inputs), batch_idx
        )
        self.logger.experiment.add_image(
            "output",
            torchvision.utils.make_grid(torch.sigmoid(output)),
            batch_idx,
        )

    def pyro_training_step(self, train_batch, batch_idx):
        inputs = train_batch
        output = self.vae.reconstruct_img(inputs)
        loss = self.loss_fn(self.vae.model, self.vae.guide, inputs)
        self.log("train_loss", loss)
        self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)
        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(inputs), batch_idx
        )
        self.logger.experiment.add_image(
            "output",
            torchvision.utils.make_grid(torch.sigmoid(output)),
            batch_idx,
        )
        return loss

    def training_step(self, train_batch, batch_idx):
        return self.torch_training_step(train_batch, batch_idx)

    def training_step(self, train_batch, batch_idx):
        return self.pyro_training_step(train_batch, batch_idx)

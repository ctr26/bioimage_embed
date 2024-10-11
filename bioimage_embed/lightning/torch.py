import torchvision
import torch
import pytorch_lightning as pl
from timm import optim, scheduler
from types import SimpleNamespace
import argparse
from transformers.utils import ModelOutput
import torch.nn.functional as F
from monai import losses

"""
x_recon -> output of the model
z -> latent space
data -> input to the model
target -> target for supervised learning
recon_loss -> reconstruction loss
loss -> total loss
variational_loss -> loss - recon_loss
"""


class AutoEncoder(pl.LightningModule):
    args = argparse.Namespace(
        opt="adamw",
        weight_decay=0.001,
        momentum=0.9,
        sched="cosine",
        epochs=50,
        lr=1e-4,
        min_lr=1e-6,
        t_initial=10,
        t_mul=2,
        lr_min=None,
        decay_rate=0.1,
        warmup_lr=1e-6,
        warmup_lr_init=1e-6,
        warmup_epochs=5,
        cycle_limit=None,
        t_in_epochs=False,
        noisy=False,
        noise_std=0.1,
        noise_pct=0.67,
        noise_seed=None,
        cooldown_epochs=5,
        warmup_t=0,
    )

    def __init__(self, model, args=SimpleNamespace()):
        super().__init__()
        self.model = model
        self.model = self.model.to(self.device)
        # Flatten hparams
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        if args:
            self.args = SimpleNamespace(**{**vars(args), **vars(self.args)})
        self.save_hyperparameters(vars(self.args))
        # keep a handle on metrics logged by the model
        self.metrics = {}

        # TODO update all models to use this for export to onxx
        # self.example_input_array = torch.randn(1, *self.model.input_dim)
        # self.model.train()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass of the model
        Pythae models take in ModelOutput objects, and return ModelOutput objects so that we can pass in and return multiple tensors
        """
        return self.model(ModelOutput(data=x.float()))

    def predict_step(
        self, batch: tuple, batch_idx: int, dataloader_idx=0
    ) -> ModelOutput:
        return self.batch_to_tensor(batch)

    def batch_to_tensor(self, batch) -> ModelOutput:
        """
        This takes in a batch and returns a ModelOutput object.
        Lightning batches are x,y pairs of tensors, but we only need the x tensor for the model.
        x is fed into the self.forward method
        """
        x, y = self.batch_to_xy(batch)
        model_output = self.forward(x)
        model_output.data = x
        model_output.target = y
        return model_output

    def embedding(self, model_output: ModelOutput) -> torch.Tensor:
        return model_output.z.view(model_output.z.shape[0], -1)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        self.model.train()
        model_output = self.eval_step(batch, batch_idx)
        metrics = (
            {
                "loss/train": model_output.loss,
                "mse/train": F.mse_loss(model_output.recon_x, model_output.data),
                "recon_loss/train": model_output.recon_loss,
                "variational_loss/train": model_output.loss - model_output.recon_loss,
            },
        )
        self.log_dict(
            metrics,
            # on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.metrics = {**self.metrics, **metrics}

        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            self.log_tensorboard(model_output, model_output.data)
        return model_output.loss

    def validation_step(self, batch, batch_idx):
        model_output = self.eval_step(batch, batch_idx)
        self.log_dict(
            {
                "loss/val": model_output.loss,
                "mse/val": F.mse_loss(model_output.recon_x, model_output.data),
                "recon_loss/val": model_output.recon_loss,
                "variational_loss/val": model_output.loss - model_output.recon_loss,
            }
        )
        return model_output.loss

    def test_step(self, batch, batch_idx):
        # x, y = batch
        model_output = self.eval_step(batch, batch_idx)
        self.log_dict(
            {
                "loss/test": model_output.loss,
                "mse/test": F.mse_loss(model_output.recon_x, model_output.data),
                "recon_loss/test": model_output.recon_loss,
                "variational_loss/test": model_output.loss - model_output.recon_loss,
            }
        )
        return model_output.loss

    def batch_to_xy(self, batch):
        """
        Fangless function to be overloaded later
        """
        x, y = batch
        return x, y

    def eval_step(self, batch, batch_idx):
        """
        This function should be overloaded in the child class to implement the evaluation logic.
        """
        return self.predict_step(batch, batch_idx)

    # def lr_scheduler_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    #     # Implement your own logic for updating the lr scheduler
    #     # This method will be called at each training step
    #     # Update the lr scheduler based on the provided arguments
    #     # You can access the lr scheduler using `self.lr_schedulers()`

    #     # Example:
    #     for lr_scheduler in self.lr_schedulers():
    #         lr_scheduler.step()

    def timm_optimizers(self, model):
        optimizer = optim.create_optimizer(self.args, model.parameters())
        lr_scheduler = scheduler.create_scheduler(self.args, optimizer)[0]
        return optimizer, lr_scheduler

    def timm_to_lightning(self, optimizer, lr_scheduler):
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # or 'epoch' for step vs epoch training, respectively
            },
        }

    def configure_optimizers(self):
        # optimizer = optim.create_optimizer(self.args, self.model.parameters())
        # lr_scheduler = scheduler.create_scheduler(self.args, optimizer)[0]
        optimizer, lr_scheduler = self.timm_optimizers(self.model)
        return self.timm_to_lightning(optimizer, lr_scheduler)

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch, metric=metric)

    def log_wandb(self):
        pass

    def log_tensorboard(self, model_output, x):
        # Optionally you can add more logging, for example, visualizations:
        self.logger.experiment.add_image(
            "test_input",
            torchvision.utils.make_grid(model_output.data),
            self.global_step,
        )
        self.logger.experiment.add_image(
            "test_output",
            torchvision.utils.make_grid(model_output.recon_x),
            self.global_step,
        )


class AE(AutoEncoder):
    pass


class AutoEncoderUnsupervised(AutoEncoder):
    pass


class AEUnsupervised(AutoEncoder):
    pass


"""
This function generates positive pairs of feature vectors (`input_pairs` and `target_pairs`)
based on the class labels provided.

For each unique class in the labels:
- It selects all samples from the same class and creates pairs of feature vectors.
- Only pairs within the same class are generated, no cross-class pairs.
- If there is only one sample in a class, no pairs are created for that class.

The resulting `input_pairs` and `target_pairs`:
- `input_pairs`: Feature vectors of the first sample in each pair.
- `target_pairs`: Feature vectors of the second sample in each pair.

### Example 1: Two Classes
Suppose `X` (features) and `y` (labels) are:
X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
y = [[0], [0], [1], [1]]

For class 0:
- Input pair: [1.0, 2.0, 3.0]
- Target pair: [4.0, 5.0, 6.0]

For class 1:
- Input pair: [7.0, 8.0, 9.0]
- Target pair: [10.0, 11.0, 12.0]

Final pairs:
input_pairs = [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]
target_pairs = [[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]

### Example 2: Multiple Classes
Suppose `X` and `y` have three classes:
X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0],
     [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
y = [[0], [0], [1], [1], [2], [2]]

For class 0:
- Input pair: [1.0, 2.0, 3.0]
- Target pair: [4.0, 5.0, 6.0]

For class 1:
- Input pair: [7.0, 8.0, 9.0]
- Target pair: [10.0, 11.0, 12.0]

For class 2:
- Input pair: [13.0, 14.0, 15.0]
- Target pair: [16.0, 17.0, 18.0]

Final pairs:
input_pairs = [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0], [13.0, 14.0, 15.0]]
target_pairs = [[4.0, 5.0, 6.0], [10.0, 11.0, 12.0], [16.0, 17.0, 18.0]]

This is used in contrastive learning settings like SimCLR, where pairs from the same class
are treated as positive examples to learn class-consistent embeddings.
"""


def create_label_based_pairs(
    features: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create positive pairs based on labels for contrastive learning (SimCLR/MoCo).

    Args:
    features: Tensor of shape (b, latent_dim)
    labels: Tensor of shape (b, 1)

    Returns:
    tuple of two tensors, each of shape (n, latent_dim), where n is the number of pairs
    """
    labels = labels.squeeze()  # Convert (b, 1) to (b,)
    unique_labels = torch.unique(labels)

    # If there's only one unique label or no samples, return empty tensors
    if len(unique_labels) == 1 or features.size(0) == 1:
        return torch.empty(0, features.size(1)), torch.empty(0, features.size(1))

    positive_pairs = []

    for label in unique_labels:
        mask = labels == label
        class_samples = features[mask]
        if class_samples.size(0) > 1:  # Need at least 2 samples to form pairs
            # Generate all possible pairs of samples within this class
            num_samples = class_samples.size(0)
            pairs = torch.combinations(torch.arange(num_samples), r=2)
            positive_pairs.append(
                (class_samples[pairs[:, 0]], class_samples[pairs[:, 1]])
            )

    # If no valid pairs were found, return empty tensors
    if not positive_pairs:
        return torch.empty(0, features.size(1)), torch.empty(0, features.size(1))

    # Concatenate all positive pairs across classes
    input_pairs = torch.cat([pair[0] for pair in positive_pairs])
    target_pairs = torch.cat([pair[1] for pair in positive_pairs])

    return input_pairs, target_pairs


def compute_contrastive_loss(
    X: torch.Tensor, y: torch.Tensor, criterion=losses.ContrastiveLoss()
):
    """
    Wrapper function that computes contrastive loss using the MONAI ContrastiveLoss function.

    Args:
    - X (torch.Tensor): The feature tensor of shape (batch_size, latent_dim).
    - y (torch.Tensor): The label tensor of shape (batch_size, 1).
    - contrastive_criterion: The criterion to compute contrastive loss. If None, defaults to monai.losses.ContrastiveLoss.

    Returns:
    - loss (torch.Tensor): The computed contrastive loss.
    """

    # Create positive pairs from X and y
    input_pairs, target_pairs = create_label_based_pairs(X, y)

    # If no pairs are created, return zero loss
    if input_pairs.numel() == 0 or target_pairs.numel() == 0:
        return torch.tensor(0.0, device=X.device)

    # Compute the contrastive loss
    contrastive_loss = criterion(input_pairs, target_pairs)

    return contrastive_loss


class AutoEncoderSupervised(AutoEncoder):
    criterion = losses.ContrastiveLoss()

    def eval_step(self, batch, batch_idx):
        # x, y = batch
        # TODO check this
        # Scale is used as the rest of the loss functions are sums rather than means, which may mean we need to scale up the contrastive loss
        model_output = self.predict_step(batch, batch_idx)
        scale = torch.prod(torch.tensor(model_output.z.shape[1:]))
        if model_output.target.unique().size(0) == 1:
            return model_output
        contrastive_loss = compute_contrastive_loss(
            # Belt and braces on this view
            model_output.z.view(-1, self.model.latent_dim),
            model_output.target,
            criterion=self.criterion,
        )
        model_output.contrastive_loss = scale * contrastive_loss
        model_output.loss += model_output.contrastive_loss
        return model_output


class AESupervised(AutoEncoderSupervised):
    pass


class NDAutoEncoder(AESupervised):
    def batch_to_xy(self, batch):
        x, y = super().batch_to_xy(batch)

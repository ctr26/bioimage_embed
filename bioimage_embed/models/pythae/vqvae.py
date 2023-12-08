# Note - you must have torchvision installed for this example
import torch
from torch import nn
import torch
import torch.nn.functional as F

from ..nets.resnet import ResnetDecoder, ResnetEncoder

# https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fknqLRCvdJ4I


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class VQ_VAE(nn.Module):
    def __init__(
        self,
        num_hiddens=32,
        num_residual_hiddens=64,
        num_residual_layers=2,
        embedding_dim=32,
        num_embeddings=32,
        commitment_cost=0.25,
        decay=0.99,
        channels=1,
    ):
        super(VQ_VAE, self).__init__()

        self._encoder = ResnetEncoder(
            num_hiddens, num_residual_layers, num_residual_hiddens, in_channels=channels
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        self._decoder = ResnetDecoder(
            embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            out_channels=channels,
        )
    def forward(self, x):
        z = self.encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)["reconstruction"]
        return loss, x_recon, perplexity

    def recon(self, x):
        loss, x_recon, perplexity = self.forward(x)
        return x_recon

    def encoder_z(self, x):
        # z = self._encoder(x)
        # z = self._pre_vq_conv(z)
        # z = self._encoder(x)["embedding"]
        # z = self._pre_vq_conv(z)
        return self._encoder(x)["embedding"]

    def encoder_zq(self, x):
        z = self.encoder_z(x)
        loss, z_q, perplexity, _ = self._vq_vae(z)
        return z_q

    def decoder_z(self, z):
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return x_recon

    def decoder_zq(self, quantized):
        return self._decoder(quantized)

    def encoder(self, x):
        return self.encoder_z(x)
        # return self.encoder_zq(x)

    def decoder(self, z):
        return self.decoder_z(z)
        # return self.decoder_zq(z)

    # def forward(self, x):
    #     vq_output_eval = self._pre_vq_conv(self._encoder(x))
    #     _, quantize, _, _ = self._vq_vae(vq_output_eval)
    #     reconstructions = self._decoder(quantize)

    def model(self, x):
        loss, quantized, perplexity, _ = self.forward(x)
        x_recon = self.decoder(quantized)
        return x_recon

    def get_embedding(self):
        return self._vq_vae._embedding.weight.data.cpu()

    def encode(self, x):
        return [self.encoder(x), None]

    def decode(self, x):
        return self.decoder(x)

    def output_from_results(self, loss, x_recon, perplexity):
        return x_recon

    def loss_function(self, *args, recons, input, **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        vq_loss, recons, perplexity = args

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "VQ_Loss": vq_loss}

        # vq_loss, output, perplexity = self.forward(inputs)
        # output = x_recon
        # loss = self.loss_fn(output, inputs)

        # vq_loss, data_recon, perplexity = model(inputs)
        # recon_error = F.mse_loss(output, inputs)
        # recon_error = self.loss_fn(output, inputs)

    def vqvae_to_latent(self, img: torch.Tensor) -> torch.Tensor:

        vq = self._vq_vae
        embedding_torch = vq._embedding
        embedding_in = self.encoder_z(img)
        embedding_out = self._vq_vae(embedding_in)
        latent = embedding_torch(embedding_out[-1].argmax(axis=1))

        return latent

class VAE(nn.Module):
    def __init__(self, num_hiddens=32, num_residual_hiddens=64, num_residual_layers=2, embedding_dim=32, channels=1):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            ResnetEncoder(num_hiddens, num_residual_layers, num_residual_hiddens, in_channels=channels),
            nn.Flatten(),
            nn.Linear(num_hiddens * 8 * 8, embedding_dim * 2),  # Assuming input size is 64x64
        )
        self.decoder = ResnetDecoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=channels)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.split(h, h.size(1) // 2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z.view(z.size(0), z.size(1), 1, 1))
        return x_recon, mu, log_var

    def loss_function(self, recons, input, mu, log_var):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recons_loss + kld_loss
        return {"loss": loss, "Reconstruction_Loss":recons_loss, "KLD":kld_loss}

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from typing import Tuple

from .abstract import AbstractModel


class VAEReconstruction(AbstractModel):
    """Reconstruction class for Variational Autoencoder."""

    name = "Variational Autoencoder"

    def __init__(self, *args, **kwargs):
        """Initialize attributes."""
        super().__init__(*args, **kwargs)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.latent_dim = kwargs.get("latent_dim")
        self.hidden_dim = kwargs.get("hidden_dim")
        self.kl_weight = kwargs.get("kl_weight")
        self.epochs = kwargs.get("epochs")
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")

        if not self.latent_dim:
            raise ValueError("Missing 'latent_dim' key-argument.")
        if not self.hidden_dim:
            raise ValueError("Missing 'hidden_dim' key-argument.")
        if not self.kl_weight:
            raise ValueError("Missing 'kl_weight' key-argument.")
        if not self.epochs:
            raise ValueError("Missing 'epochs' key-argument.")
        if not self.batch_size:
            raise ValueError("Missing 'batch_size' key-argument.")
        if not self.lr:
            raise ValueError("Missing 'lr' key-argument.")

        self._model = VariationalAutoencoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            kl_weight=self.kl_weight,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device
        )

    @property
    def hyperparameters(self):
        """Get hyperparameters."""
        return {
            "re_quantile": self.re_quantile,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "kl_weight": self.kl_weight,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr
        }

    def _calculate_reconstruction_errors(
        self, data: pd.DataFrame
    ) -> np.ndarray:
        return self._model.calculate_reconstruction_errors(data)


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder class."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        kl_weight: float = 1.0,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu")
    ):
        """Initialize attributes."""
        super().__init__()
        self.encoder_fc1 = nn.Linear(768, hidden_dim)
        self.encoder_fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, 768)

        self.kl_weight = kl_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.to(self.device)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the input into latent distribution parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 768).

        Returns
        -------
        mu : torch.Tensor
            Mean vector of latent distribution.
        logvar : torch.Tensor
            Log-variance vector of latent distribution.
        """
        h = F.relu(self.encoder_fc1(x))
        mu = self.encoder_fc2_mu(h)
        logvar = self.encoder_fc2_logvar(h)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Apply the reparameterization trick.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of latent distribution.
        logvar : torch.Tensor
            Log-variance of latent distribution.

        Returns
        -------
        z : torch.Tensor
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Parameters
        ----------
        z : torch.Tensor
            Latent variable tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor.
        """
        h = F.relu(self.decoder_fc1(z))
        return self.decoder_fc2(h)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform forward pass through VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        reconstructed : torch.Tensor
            Output reconstruction.
        mu : torch.Tensor
            Latent mean.
        logvar : torch.Tensor
            Latent log-variance.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(
        self,
        reconstructed: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Compute total VAE loss.

        Parameters
        ----------
        reconstructed : torch.Tensor
            Output reconstruction.
        x : torch.Tensor
            Original input.
        mu : torch.Tensor
            Latent mean.
        logvar : torch.Tensor
            Latent log-variance.

        Returns
        -------
        torch.Tensor
            Combined reconstruction + KL loss.
        """
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + self.kl_weight * kl_divergence

    def fit(self, data: pd.DataFrame):
        """Train the VAE on provided data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data of shape (n_samples, 768).
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        X_tensor = torch.tensor(
            data.values, dtype=torch.float32
        ).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for _ in range(self.epochs):
            for batch in loader:
                x_batch = batch[0]
                optimizer.zero_grad()
                reconstructed, mu, logvar = self(x_batch)
                loss = self.loss_function(reconstructed, x_batch, mu, logvar)
                loss.backward()
                optimizer.step()

    def calculate_reconstruction_errors(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """Calculate reconstruction errors for given data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data of shape (n_samples, 768).

        Returns
        -------
        np.ndarray
            Reconstruction errors per sample.
        """
        self.eval()
        X_tensor = torch.tensor(
            data.values, dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            reconstructed, _, _ = self(X_tensor)
            errors = torch.norm(X_tensor - reconstructed, dim=1).cpu().numpy()
        return errors

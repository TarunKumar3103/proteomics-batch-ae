"""Adversarial guided autoencoder for batch correction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class SmallClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 32):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ProteomicsBatchAE(nn.Module):
    """Small guided autoencoder with explicit batch and biology latent slices.

    The model is generic: it never knows anything about the data source. It only
    receives X, missingness mask M, and one-hot covariates prepared by the runner.
    """

    def __init__(
        self,
        n_features: int,
        n_covariates: int,
        n_batches: int,
        n_biology_classes: int,
        latent_dim: int = 32,
        batch_dim: int = 8,
        biology_dim: int = 6,
        hidden_classifier_dim: int = 32,
        variational: bool = False,
    ):
        super().__init__()
        if batch_dim + biology_dim > latent_dim:
            raise ValueError("batch_dim + biology_dim must be <= latent_dim")

        self.n_features = n_features
        self.n_covariates = n_covariates
        self.n_batches = n_batches
        self.n_biology_classes = n_biology_classes
        self.latent_dim = latent_dim
        self.batch_dim = batch_dim
        self.biology_dim = biology_dim
        self.variational = variational

        self.batch_slice = slice(0, batch_dim)
        self.biology_slice = slice(batch_dim, batch_dim + biology_dim)
        self.other_slice = slice(batch_dim + biology_dim, latent_dim)

        encoder_in = n_features * 2
        self.encoder_mu = nn.Linear(encoder_in, latent_dim)
        self.encoder_logvar = nn.Linear(encoder_in, latent_dim) if variational else None

        decoder_in = latent_dim + n_covariates
        self.decoder_x = nn.Linear(decoder_in, n_features)
        self.decoder_m = nn.Linear(decoder_in, n_features)

        self.batch_classifier = SmallClassifier(batch_dim, n_batches, hidden_classifier_dim)
        self.biology_classifier = SmallClassifier(biology_dim, n_biology_classes, hidden_classifier_dim)

        nonbatch_dim = latent_dim - batch_dim
        self.adv_batch_classifier = SmallClassifier(nonbatch_dim, n_batches, hidden_classifier_dim)

    def encode(self, X, M):
        inp = torch.cat([X, M], dim=-1)
        mu = self.encoder_mu(inp)
        if self.variational:
            logvar = torch.clamp(self.encoder_logvar(inp), min=-4.0, max=4.0)
            return mu, logvar
        return mu, None

    def reparameterize(self, mu, logvar):
        if self.variational and self.training and logvar is not None:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z, covariates):
        inp = torch.cat([z, covariates], dim=-1)
        X_hat = self.decoder_x(inp)
        M_hat = torch.sigmoid(self.decoder_m(inp))
        return X_hat, M_hat

    def forward(self, X, M, covariates, grl_lambda: float = 1.0):
        mu, logvar = self.encode(X, M)
        z = self.reparameterize(mu, logvar)

        z_batch = z[:, self.batch_slice]
        z_bio = z[:, self.biology_slice]
        z_other = z[:, self.other_slice]
        z_nonbatch = torch.cat([z_bio, z_other], dim=-1)

        X_hat, M_hat = self.decode(z, covariates)
        return {
            "X_hat": X_hat,
            "M_hat": M_hat,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "batch_logits": self.batch_classifier(z_batch),
            "biology_logits": self.biology_classifier(z_bio),
            "adv_batch_logits": self.adv_batch_classifier(grad_reverse(z_nonbatch, grl_lambda)),
        }

    @torch.no_grad()
    def correct(self, X, M, covariates, batch_cov_dim: int):
        self.eval()
        mu, _ = self.encode(X, M)
        z_corr = mu.clone()
        z_corr[:, self.batch_slice] = 0.0

        cov_corr = covariates.clone()
        cov_corr[:, :batch_cov_dim] = 0.0

        X_corr, _ = self.decode(z_corr, cov_corr)
        return X_corr, mu

    def init_weights_svd(self, X_centered: np.ndarray) -> None:
        """Initialize encoder/decoder with PCA-like directions for stability."""
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        k = min(self.latent_dim, Vt.shape[0])

        with torch.no_grad():
            self.encoder_mu.weight.mul_(0.01)
            self.encoder_mu.bias.zero_()
            self.encoder_mu.weight[:k, : self.n_features] = torch.tensor(Vt[:k], dtype=torch.float32)

            self.decoder_x.weight.mul_(0.01)
            self.decoder_x.bias.zero_()
            self.decoder_x.weight[:, :k] = torch.tensor(Vt[:k].T, dtype=torch.float32)

        explained = float((S[:k] ** 2).sum() / max((S ** 2).sum(), 1e-12))
        print(f"SVD init: top {k} components, explained variance: {explained:.1%}")

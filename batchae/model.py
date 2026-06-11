"""Adversarial proteomics CVAE model."""

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


def grad_reverse(x, lambd: float = 1.0):
    return GradReverse.apply(x, lambd)


class BatchClassifier(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, z_sub):
        return self.net(z_sub)


class ProteomicsCVAE(nn.Module):
    """Guided autoencoder with explicit batch/cell-line latent slices.

    The corrected output is produced by zeroing the batch latent slice and batch
    covariates before decoding.
    """

    def __init__(
        self,
        n_proteins: int,
        n_covariates: int,
        latent_dim: int = 24,
        batch_dim: int = 8,
        cellline_dim: int = 4,
        n_batches: int = 6,
        n_cell_lines: int = 3,
        variational: bool = False,
        hidden_dim: int = 32,
    ):
        super().__init__()

        if batch_dim + cellline_dim >= latent_dim:
            raise ValueError("latent_dim must exceed batch_dim + cellline_dim")

        self.n_proteins = n_proteins
        self.n_covariates = n_covariates
        self.latent_dim = latent_dim
        self.batch_dim = batch_dim
        self.cellline_dim = cellline_dim
        self.variational = variational
        self.n_batches = n_batches
        self.n_cell_lines = n_cell_lines

        self.batch_slice = slice(0, batch_dim)
        self.cellline_slice = slice(batch_dim, batch_dim + cellline_dim)
        self.other_slice = slice(batch_dim + cellline_dim, latent_dim)

        encoder_in = n_proteins * 2
        self.encoder = nn.Linear(encoder_in, latent_dim)
        if variational:
            self.encoder_logvar = nn.Linear(encoder_in, latent_dim)

        decoder_in = latent_dim + n_covariates
        self.decoder_intensity = nn.Linear(decoder_in, n_proteins)
        self.decoder_missingness = nn.Linear(decoder_in, n_proteins)

        self.classifier_batch = BatchClassifier(batch_dim, n_batches, hidden_dim=hidden_dim)
        self.classifier_cellline = BatchClassifier(cellline_dim, n_cell_lines, hidden_dim=hidden_dim)

        nonbatch_dim = latent_dim - batch_dim
        self.adversarial_batch_classifier = BatchClassifier(nonbatch_dim, n_batches, hidden_dim=hidden_dim)

    def encode(self, X, M):
        inp = torch.cat([X, M], dim=-1)
        mu = self.encoder(inp)
        if self.variational:
            log_var = torch.clamp(self.encoder_logvar(inp), min=-4.0, max=4.0)
            return mu, log_var
        return mu, None

    def reparameterize(self, mu, log_var):
        if self.variational and self.training and log_var is not None:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z, covariates):
        inp = torch.cat([z, covariates], dim=-1)
        X_hat = self.decoder_intensity(inp)
        M_hat = torch.sigmoid(self.decoder_missingness(inp))
        return X_hat, M_hat

    def forward(self, X, M, covariates, grl_lambda: float = 1.0):
        mu, log_var = self.encode(X, M)
        z = self.reparameterize(mu, log_var)

        z_batch = z[:, self.batch_slice]
        z_cellline = z[:, self.cellline_slice]
        z_other = z[:, self.other_slice]
        z_nonbatch = torch.cat([z_cellline, z_other], dim=-1)

        batch_logits = self.classifier_batch(z_batch)
        cellline_logits = self.classifier_cellline(z_cellline)
        adv_batch_logits = self.adversarial_batch_classifier(grad_reverse(z_nonbatch, lambd=grl_lambda))

        X_hat, M_hat = self.decode(z, covariates)

        return {
            "X_hat": X_hat,
            "M_hat": M_hat,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "batch_logits": batch_logits,
            "cellline_logits": cellline_logits,
            "adv_batch_logits": adv_batch_logits,
        }

    def correct(self, X, M, covariates, batch_cov_dim: int):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(X, M)
            z_corrected = mu.clone()
            z_corrected[:, self.batch_slice] = 0.0

            cov_corrected = covariates.clone()
            cov_corrected[:, :batch_cov_dim] = 0.0

            X_corrected, _ = self.decode(z_corrected, cov_corrected)
        return X_corrected, mu

    def init_weights_svd(self, X_centered: np.ndarray) -> None:
        """Initialize encoder/decoder with leading right singular vectors."""
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        k = min(self.latent_dim, Vt.shape[0])
        with torch.no_grad():
            self.encoder.weight[:, :] *= 0.01
            self.encoder.weight[:k, : self.n_proteins] = torch.tensor(Vt[:k], dtype=torch.float32)
            self.decoder_intensity.weight[:, :] *= 0.01
            self.decoder_intensity.weight[:, :k] = torch.tensor(Vt[:k].T, dtype=torch.float32)
        explained = (S[:k] ** 2).sum() / max((S**2).sum(), 1e-12)
        print(f"SVD init: top {k} components, explained variance: {explained:.1%}")

"""Training and correction utilities for the adversarial autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import ProteomicsCVAE


@dataclass(frozen=True)
class AEHyperParams:
    lambda_: float
    eta_b: float
    eta_c: float
    eta_adv: float
    gamma: float
    batch_dim: int
    cellline_dim: int
    latent_dim: int

    def asdict(self) -> dict[str, Any]:
        return {
            "lambda_": self.lambda_,
            "eta_b": self.eta_b,
            "eta_c": self.eta_c,
            "eta_adv": self.eta_adv,
            "gamma": self.gamma,
            "batch_dim": self.batch_dim,
            "cellline_dim": self.cellline_dim,
            "latent_dim": self.latent_dim,
        }


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(X: np.ndarray, M: np.ndarray, meta: pd.DataFrame, covariate_cols=("batch", "cell_line")) -> TensorDataset:
    X_t = torch.tensor(X, dtype=torch.float32)
    M_t = torch.tensor(M, dtype=torch.float32)

    cov_parts = []
    for col in covariate_cols:
        vals = meta[col].values.astype(int)
        n_classes = int(vals.max()) + 1
        cov_parts.append(np.eye(n_classes, dtype=np.float32)[vals])

    cov_t = torch.tensor(np.concatenate(cov_parts, axis=1), dtype=torch.float32)
    batch_t = torch.tensor(meta["batch"].values, dtype=torch.long)
    cell_t = torch.tensor(meta["cell_line"].values, dtype=torch.long)
    return TensorDataset(X_t, M_t, cov_t, batch_t, cell_t)


def masked_mse(X_hat, X, M):
    observed = M.bool()
    if observed.sum() == 0:
        return torch.tensor(0.0, device=X.device)
    return F.mse_loss(X_hat[observed], X[observed])


def missingness_bce(M_hat, M):
    return F.binary_cross_entropy(M_hat, M)


def classifier_ce(logits, labels):
    return F.cross_entropy(logits, labels)


def independence_penalty(z):
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / max(z.shape[0] - 1, 1)
    eye = torch.eye(cov.shape[0], device=z.device)
    return torch.norm(cov - eye, p="fro")


def compute_loss(outputs, X, M, batch_labels, cellline_labels, hparams: dict[str, float]):
    loss_mse = masked_mse(outputs["X_hat"], X, M)
    loss_bce = missingness_bce(outputs["M_hat"], M)
    loss_batch = classifier_ce(outputs["batch_logits"], batch_labels)
    loss_cl = classifier_ce(outputs["cellline_logits"], cellline_labels)
    loss_adv = classifier_ce(outputs["adv_batch_logits"], batch_labels)
    loss_indep = independence_penalty(outputs["z"])

    total = (
        loss_mse
        + hparams["lambda_"] * loss_bce
        + hparams["eta_b"] * loss_batch
        + hparams["eta_c"] * loss_cl
        + hparams["eta_adv"] * loss_adv
        + hparams["gamma"] * loss_indep
    )

    return total, {
        "total": float(total.item()),
        "mse": float(loss_mse.item()),
        "bce": float(loss_bce.item()),
        "h_batch": float(loss_batch.item()),
        "h_cl": float(loss_cl.item()),
        "adv": float(loss_adv.item()),
        "indep": float(loss_indep.item()),
    }


def make_model(
    X: np.ndarray,
    meta: pd.DataFrame,
    hparams: AEHyperParams,
    device: str = "cpu",
    hidden_dim: int = 32,
) -> ProteomicsCVAE:
    n_b = meta["batch"].nunique()
    n_cl = meta["cell_line"].nunique()
    model = ProteomicsCVAE(
        n_proteins=X.shape[1],
        n_covariates=n_b + n_cl,
        latent_dim=hparams.latent_dim,
        batch_dim=hparams.batch_dim,
        cellline_dim=hparams.cellline_dim,
        n_batches=n_b,
        n_cell_lines=n_cl,
        variational=False,
        hidden_dim=hidden_dim,
    ).to(device)
    model.init_weights_svd(X - X.mean(axis=0, keepdims=True))
    return model


def train_model(
    model: ProteomicsCVAE,
    dataset: TensorDataset,
    hparams: dict[str, float],
    n_epochs: int = 400,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: str = "cpu",
    dataloader_workers: int = 0,
    verbose: bool = True,
    log_every: int = 50,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=max(0, int(dataloader_workers)),
        pin_memory=(device.startswith("cuda") and dataloader_workers > 0),
    )

    history = {k: [] for k in ["total", "mse", "bce", "h_batch", "h_cl", "adv", "indep"]}

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = {k: 0.0 for k in history}
        n_seen = 0
        grl_lambda = min(1.0, epoch / max(1, n_epochs * 0.4))

        for X_b, M_b, cov_b, batch_b, cell_b in loader:
            X_b = X_b.to(device)
            M_b = M_b.to(device)
            cov_b = cov_b.to(device)
            batch_b = batch_b.to(device)
            cell_b = cell_b.to(device)

            optimizer.zero_grad()
            outputs = model(X_b, M_b, cov_b, grl_lambda=grl_lambda)
            loss, components = compute_loss(outputs, X_b, M_b, batch_b, cell_b, hparams)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k, v in components.items():
                epoch_losses[k] += v
            n_seen += 1

        for k in history:
            history[k].append(epoch_losses[k] / max(n_seen, 1))

        if verbose and epoch % log_every == 0:
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"total={history['total'][-1]:8.3f} | "
                f"mse={history['mse'][-1]:.3f} | "
                f"bce={history['bce'][-1]:.3f} | "
                f"h_batch={history['h_batch'][-1]:.3f} | "
                f"h_cl={history['h_cl'][-1]:.3f} | "
                f"adv={history['adv'][-1]:.3f} | "
                f"indep={history['indep'][-1]:.3f}"
            )

    return history


def correct_model(model: ProteomicsCVAE, X: np.ndarray, M: np.ndarray, meta: pd.DataFrame, device: str = "cpu"):
    n_b = meta["batch"].nunique()
    n_cl = meta["cell_line"].nunique()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    M_t = torch.tensor(M, dtype=torch.float32).to(device)
    cov_batch = np.eye(n_b, dtype=np.float32)[meta["batch"].values.astype(int)]
    cov_cl = np.eye(n_cl, dtype=np.float32)[meta["cell_line"].values.astype(int)]
    cov = torch.tensor(np.concatenate([cov_batch, cov_cl], axis=1), dtype=torch.float32).to(device)

    X_corrected_t, z_t = model.correct(X_t, M_t, cov, batch_cov_dim=n_b)
    return X_corrected_t.cpu().numpy(), z_t.cpu().numpy()


def suggest_hparams(trial) -> AEHyperParams:
    eta_b = trial.suggest_float("eta_b", 0.5, 25.0, log=True)
    eta_c = trial.suggest_float("eta_c", 0.5, 25.0, log=True)
    eta_adv = trial.suggest_float("eta_adv", 0.05, 25.0, log=True)
    gamma = trial.suggest_float("gamma", 0.005, 1.0, log=True)
    lam = trial.suggest_float("lambda_", 0.05, 1.0)
    batch_dim = trial.suggest_int("batch_dim", 4, 14)
    cellline_dim = trial.suggest_int("cellline_dim", 2, 8)
    latent_dim = max(28, batch_dim + cellline_dim + 10)
    return AEHyperParams(
        lambda_=lam,
        eta_b=eta_b,
        eta_c=eta_c,
        eta_adv=eta_adv,
        gamma=gamma,
        batch_dim=batch_dim,
        cellline_dim=cellline_dim,
        latent_dim=latent_dim,
    )

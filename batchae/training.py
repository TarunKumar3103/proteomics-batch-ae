"""Training utilities for the adversarial batch-correction autoencoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class EncodedDataset:
    dataset: TensorDataset
    batch_classes: list[str]
    biology_classes: list[str]
    covariate_dim: int
    batch_cov_dim: int


def encode_labels(values) -> tuple[np.ndarray, list[str]]:
    classes = sorted([str(v) for v in set(values)])
    mapping = {c: i for i, c in enumerate(classes)}
    return np.array([mapping[str(v)] for v in values], dtype=np.int64), classes


def onehot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes, dtype=np.float32)[labels]


def build_torch_dataset(X: np.ndarray, M: np.ndarray, meta, batch_col: str, biology_col: str) -> EncodedDataset:
    batch_labels, batch_classes = encode_labels(meta[batch_col].values)

    if biology_col in meta.columns and meta[biology_col].nunique() >= 2:
        bio_labels, bio_classes = encode_labels(meta[biology_col].values)
    else:
        bio_labels = np.zeros(len(meta), dtype=np.int64)
        bio_classes = ["unknown"]

    batch_oh = onehot(batch_labels, len(batch_classes))
    bio_oh = onehot(bio_labels, len(bio_classes))
    cov = np.concatenate([batch_oh, bio_oh], axis=1).astype(np.float32)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(M, dtype=torch.float32),
        torch.tensor(cov, dtype=torch.float32),
        torch.tensor(batch_labels, dtype=torch.long),
        torch.tensor(bio_labels, dtype=torch.long),
    )
    return EncodedDataset(
        dataset=dataset,
        batch_classes=batch_classes,
        biology_classes=bio_classes,
        covariate_dim=cov.shape[1],
        batch_cov_dim=len(batch_classes),
    )


def masked_mse(X_hat, X, M):
    observed = M.bool()
    if observed.sum() == 0:
        return torch.tensor(0.0, device=X.device)
    return F.mse_loss(X_hat[observed], X[observed])


def missingness_bce(M_hat, M):
    return F.binary_cross_entropy(M_hat, M)


def independence_penalty(z):
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / max(z.shape[0] - 1, 1)
    eye = torch.eye(cov.shape[0], device=z.device)
    return torch.norm(cov - eye, p="fro")


def compute_loss(outputs, X, M, batch_labels, bio_labels, hparams):
    loss_mse = masked_mse(outputs["X_hat"], X, M)
    loss_bce = missingness_bce(outputs["M_hat"], M)
    loss_batch = F.cross_entropy(outputs["batch_logits"], batch_labels)
    loss_bio = F.cross_entropy(outputs["biology_logits"], bio_labels)
    loss_adv = F.cross_entropy(outputs["adv_batch_logits"], batch_labels)
    loss_indep = independence_penalty(outputs["z"])

    total = (
        loss_mse
        + hparams.get("lambda_missing", 0.1) * loss_bce
        + hparams.get("eta_batch", 1.0) * loss_batch
        + hparams.get("eta_biology", 1.0) * loss_bio
        + hparams.get("eta_adv", 1.0) * loss_adv
        + hparams.get("gamma_indep", 0.01) * loss_indep
    )
    return total, {
        "total": float(total.detach().cpu()),
        "mse": float(loss_mse.detach().cpu()),
        "bce": float(loss_bce.detach().cpu()),
        "batch_ce": float(loss_batch.detach().cpu()),
        "biology_ce": float(loss_bio.detach().cpu()),
        "adv_ce": float(loss_adv.detach().cpu()),
        "indep": float(loss_indep.detach().cpu()),
    }


def train_model(
    model,
    encoded: EncodedDataset,
    hparams: dict,
    n_epochs: int = 300,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: str = "cpu",
    dataloader_workers: int = 0,
    verbose: bool = True,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        encoded.dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=int(dataloader_workers),
    )

    history = {k: [] for k in ["total", "mse", "bce", "batch_ce", "biology_ce", "adv_ce", "indep"]}

    for epoch in range(1, n_epochs + 1):
        model.train()
        sums = {k: 0.0 for k in history}
        n_batches = 0
        grl_lambda = min(1.0, epoch / max(1, int(n_epochs * 0.4)))

        for X_b, M_b, cov_b, batch_b, bio_b in loader:
            X_b = X_b.to(device)
            M_b = M_b.to(device)
            cov_b = cov_b.to(device)
            batch_b = batch_b.to(device)
            bio_b = bio_b.to(device)

            optimizer.zero_grad()
            outputs = model(X_b, M_b, cov_b, grl_lambda=grl_lambda)
            loss, components = compute_loss(outputs, X_b, M_b, batch_b, bio_b, hparams)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k, v in components.items():
                sums[k] += v
            n_batches += 1

        for k in history:
            history[k].append(sums[k] / max(n_batches, 1))

        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == n_epochs):
            print(
                f"Epoch {epoch:4d}/{n_epochs} | total={history['total'][-1]:.4f} "
                f"mse={history['mse'][-1]:.4f} batch_ce={history['batch_ce'][-1]:.4f} "
                f"bio_ce={history['biology_ce'][-1]:.4f} adv={history['adv_ce'][-1]:.4f}"
            )

    return history


@torch.no_grad()
def correct_with_model(model, X: np.ndarray, M: np.ndarray, encoded: EncodedDataset, device: str = "cpu"):
    model = model.to(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    M_t = torch.tensor(M, dtype=torch.float32, device=device)
    cov_t = encoded.dataset.tensors[2].to(device)
    X_corr, z = model.correct(X_t, M_t, cov_t, batch_cov_dim=encoded.batch_cov_dim)
    return X_corr.cpu().numpy(), z.cpu().numpy()

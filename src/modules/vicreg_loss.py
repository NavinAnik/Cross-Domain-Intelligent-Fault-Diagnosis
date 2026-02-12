"""
VICReg (Variance-Invariance-Covariance Regularization) loss.

No negative pairs, no momentum encoder. Three terms:
- Invariance: MSE between embeddings of two views (pull same-sample views together).
- Variance: Hinge loss so each dimension has std >= gamma (prevents collapse).
- Covariance: Off-diagonal of covariance matrix → 0 (decorrelate dimensions).

Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning", 2022.
"""

import torch
import torch.nn.functional as F


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambda_inv: float = 25.0,
    mu_var: float = 25.0,
    nu_cov: float = 1.0,
    gamma_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    VICReg loss for two embedding views z1, z2 (e.g. from two augmentations).

    Args:
        z1: Embeddings from view 1, shape (N, D).
        z2: Embeddings from view 2, shape (N, D).
        lambda_inv: Weight for invariance term (MSE between z1 and z2).
        mu_var: Weight for variance term (hinge on std).
        nu_cov: Weight for covariance term (off-diagonal to zero).
        gamma_std: Minimum std for variance hinge (default 1.0).
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss tensor.
    """
    batch_size, dim = z1.shape
    device = z1.device

    # --- Invariance: MSE between two views ---
    loss_inv = F.mse_loss(z1, z2)

    # --- Variance: std per dimension should be >= gamma ---
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    loss_var = torch.mean(F.relu(gamma_std - std_z1)) + torch.mean(F.relu(gamma_std - std_z2))

    # --- Covariance: off-diagonal of (Z^T Z / (n-1)) should be 0 ---
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
    cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)
    off_diag_mask = ~torch.eye(dim, dtype=torch.bool, device=device)
    loss_cov = (cov_z1[off_diag_mask].pow(2).sum() + cov_z2[off_diag_mask].pow(2).sum()) / dim

    loss = lambda_inv * loss_inv + mu_var * loss_var + nu_cov * loss_cov
    return loss

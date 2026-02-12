"""
BYOL (Bootstrap Your Own Latent) loss.

No negative pairs. Uses a predictor on the online branch and a momentum target.
Loss = 2 - 2 * cosine_similarity(prediction, target), i.e. MSE between
normalized prediction and normalized target (symmetric optional).

Caller is responsible for:
- Online branch: encoder -> projector -> predictor -> prediction
- Target branch: momentum_encoder -> momentum_projector -> target (no predictor)
- EMA update of momentum encoder/projector.

This module only implements the loss given (prediction, target).
"""

import torch
import torch.nn.functional as F


def byol_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    BYOL regression loss: encourages prediction to match target in direction.

    Uses negative cosine similarity: 2 - 2 * cos_sim(pred, target).
    pred and target are normalized internally.

    Args:
        prediction: Output of online predictor, shape (N, D). Not normalized.
        target: Output of target (momentum) projection, shape (N, D). Not normalized.
        reduction: "mean" or "sum".

    Returns:
        Scalar loss tensor.
    """
    pred_norm = F.normalize(prediction, dim=1)
    target_norm = F.normalize(target.detach(), dim=1)  # stop gradient on target
    cos_sim = (pred_norm * target_norm).sum(dim=1)
    loss = 2.0 - 2.0 * cos_sim
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


def byol_loss_symmetric(
    pred_1: torch.Tensor,
    target_2: torch.Tensor,
    pred_2: torch.Tensor,
    target_1: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Symmetric BYOL loss: L(pred1->t2) + L(pred2->t1).
    Use when you have two views and both online and target process both.
    """
    l1 = byol_loss(pred_1, target_2, reduction="none")
    l2 = byol_loss(pred_2, target_1, reduction="none")
    loss = (l1 + l2) / 2
    if reduction == "mean":
        return loss.mean()
    return loss.sum()

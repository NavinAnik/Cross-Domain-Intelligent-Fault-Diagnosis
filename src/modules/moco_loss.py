"""
MoCo v3-style contrastive loss with a queue of negative keys.

Uses momentum encoder (EMA) for keys and a FIFO queue to simulate a larger
batch size. InfoNCE loss: query (online) vs key (momentum) positive, queue as negatives.

Caller is responsible for:
- Computing query = online_encoder(x), key = momentum_encoder(x) (no grad on key).
- Updating momentum encoder: theta_k = m * theta_k + (1 - m) * theta_q.
- Enqueueing new keys and dequeueing old ones (see MoCoQueue below).
"""

import torch
import torch.nn.functional as F


def moco_info_nce_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    queue: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    InfoNCE loss for MoCo: positive (query, key), negatives = queue.

    query and key are L2-normalized. Logits = (query @ key.T) / temp for positives
    and (query @ queue.T) / temp for negatives; labels are positive indices.

    Args:
        query: Online embeddings (N, D), will be normalized.
        key: Momentum embeddings (N, D), will be normalized (no grad).
        queue: Negative keys (K, D), will be normalized.
        temperature: Softmax temperature (default 0.2 for MoCo).

    Returns:
        Scalar loss tensor.
    """
    query = F.normalize(query, dim=1)
    key = F.normalize(key.detach(), dim=1)
    if queue.numel() > 0:
        queue = F.normalize(queue, dim=1)

    batch_size = query.size(0)
    device = query.device

    # Positive logits: (N,) per-sample (query_i, key_i)
    l_pos = (query * key).sum(dim=1, keepdim=True)  # (N, 1)

    if queue.size(0) > 0:
        # Negative logits: (N, K)
        l_neg = query @ queue.T
        # Logits: [l_pos, l_neg] -> (N, 1 + K)
        logits = torch.cat([l_pos, l_neg], dim=1)
    else:
        logits = l_pos

    logits = logits / temperature
    # Labels: positive is index 0 for each row
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)


class MoCoQueue:
    """
    FIFO queue of key embeddings for MoCo. Enqueue new keys and dequeue old ones.
    """

    def __init__(self, dim: int, K: int, device: torch.device):
        """
        Args:
            dim: Embedding dimension.
            K: Queue size (number of keys).
            device: Device for the queue tensor.
        """
        self.K = K
        self.dim = dim
        self.device = device
        self.queue = torch.zeros(K, dim, device=device)

    @torch.no_grad()
    def enqueue_dequeue(self, keys: torch.Tensor) -> None:
        """
        Enqueue keys and dequeue oldest. keys shape (N, D).
        """
        keys = F.normalize(keys, dim=1)
        n = keys.size(0)
        if n >= self.K:
            self.queue = keys[-self.K :].clone()
            return
        self.queue = torch.cat([self.queue[n:], keys], dim=0)

    def get_queue(self) -> torch.Tensor:
        """Return current queue (K, D)."""
        return self.queue

    def __len__(self) -> int:
        return self.K

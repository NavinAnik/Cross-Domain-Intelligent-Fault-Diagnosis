import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2*B, D]
    z = F.normalize(z, dim=1)

    similarity_matrix = torch.matmul(z, z.T)  # [2*B, 2*B]
    sim = similarity_matrix / temperature

    # Mask to remove similarity with itself
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, -float('inf'))

    # Labels: for each i in [0..B), the positive sample is i + B, else i - B
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)  # [0..B) -> [B..2B)

    return F.cross_entropy(sim, labels)

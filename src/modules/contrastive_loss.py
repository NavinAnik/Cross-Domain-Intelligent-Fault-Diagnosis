import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2N x D

    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # 2N x 2N

    # Mask to remove similarity of sample with itself
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(DEVICE)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    positives = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
    ).to(DEVICE)

    logits = sim_matrix / temperature
    labels = positives

    loss = F.cross_entropy(logits, labels)
    return loss


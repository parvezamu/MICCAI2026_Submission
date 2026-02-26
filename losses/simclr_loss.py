"""
losses/simclr_loss.py

NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR

Author: Parvez
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i, z_j):
        N = z_i.size(0)
        device = z_i.device

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # compute similarity in fp32 for AMP safety
        sim = torch.mm(z.float(), z.float().t()) / self.temperature
        sim = sim.to(z.dtype)

        # fp16-safe masking
        sim.fill_diagonal_(-1e4)

        labels = torch.arange(N, device=device)
        labels = torch.cat([labels + N, labels], dim=0)  # (2N,)

        return self.criterion(sim, labels)



class NTXentLoss2(nn.Module):
    """
    Correct NT-Xent / InfoNCE loss for SimCLR (single GPU).

    z_i, z_j: (N, D)
    Uses 2N samples, positives are (i, i+N) and (i+N, i)
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i, z_j):
        N = z_i.size(0)
        device = z_i.device

        # Normalize so dot-product == cosine similarity
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # (2N, 2N) similarity
        sim = torch.mm(z, z.t()) / self.temperature

        # mask self-similarity
        sim.fill_diagonal_(-1e9)

        # labels: for i in [0..N-1], positive is i+N; for i in [N..2N-1], positive is i-N
        labels = torch.arange(N, device=device)
        labels = torch.cat([labels + N, labels], dim=0)  # (2N,)

        loss = self.criterion(sim, labels)
        return loss




class NTXentLoss1(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for SimCLR
    
    This loss encourages similar representations for different augmentations
    of the same image (positive pairs) while pushing apart representations
    from different images (negative pairs).
    """
    
    def __init__(self, temperature=0.5):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss
        
        Args:
            z_i: Projections from view 1, shape (N, projection_dim)
            z_j: Projections from view 2, shape (N, projection_dim)
        
        Returns:
            loss: Scalar loss value
        """
        batch_size = z_i.shape[0]
        
        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate all projections: [z_i, z_j]
        representations = torch.cat([z_i, z_j], dim=0)  # (2N, projection_dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(
            representations, representations.T
        )  # (2N, 2N)
        
        # Create mask to identify positive pairs
        # For each sample i, its positive pair is at position i + N (or i - N)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        
        # Remove diagonal (self-similarity)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # Positive pairs are at position N-1 (after removing diagonal)
        positives = torch.cat([
            torch.diag(torch.matmul(z_i, z_j.T)),  # i to i+N
            torch.diag(torch.matmul(z_j, z_i.T))   # i+N to i
        ], dim=0).unsqueeze(1)  # (2N, 1)
        
        # Concatenate positives and negatives
        logits = torch.cat([positives, similarity_matrix], dim=1)  # (2N, 2N)
        
        # Scale by temperature
        logits = logits / self.temperature
        
        # Labels: positive pair is at index 0
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class InfoNCELoss(nn.Module):
    """
    Alternative implementation of NT-Xent loss
    More memory efficient for large batches
    """
    
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: (N, D) tensor
            z_j: (N, D) tensor
        """
        batch_size = z_i.shape[0]
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Gather all representations if using multiple GPUs
        # For single GPU, this is just concatenation
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t().contiguous()) / self.temperature  # (2N, 2N)
        
        # Create labels
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        sim = sim.masked_fill(mask, -9e15)
        
        # Compute loss
        loss = self.criterion(sim, labels) / (2 * batch_size)
        
        return loss


if __name__ == '__main__':
    # Test loss functions
    batch_size = 4
    projection_dim = 128
    
    z_i = torch.randn(batch_size, projection_dim)
    z_j = torch.randn(batch_size, projection_dim)
    
    # Test NT-Xent loss
    loss_fn = NTXentLoss(temperature=0.5)
    loss = loss_fn(z_i, z_j)
    print(f"NT-Xent Loss: {loss.item():.4f}")
    
    # Test InfoNCE loss
    loss_fn2 = InfoNCELoss(temperature=0.5)
    loss2 = loss_fn2(z_i, z_j)
    print(f"InfoNCE Loss: {loss2.item():.4f}")

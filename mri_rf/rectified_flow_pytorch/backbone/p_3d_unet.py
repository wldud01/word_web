import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# 1) Depth pooling head (그대로 사용)
# ============================================================
class VoxelEmbed3D(nn.Module):
    """
    alpha: (B, C, D, H, W) -> pooled: (B, C, H, W)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv3d(channels, 1, kernel_size=1, bias=False)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        alpha: (B, C, D, H, W)
        """
        logits = self.score(alpha)                 # (B, 1, D, H, W)
        weights = torch.softmax(logits, dim=2)     # (B, 1, D, H, W)
        pooled = (alpha * weights).sum(dim=2)      # (B, C, H, W)
        return pooled
    

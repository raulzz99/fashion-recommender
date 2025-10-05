from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    """
    The core triplet network that takes anchor, positive, and negative images
    and outputs their L2-normalized embeddings.

    This ensures the training objective matches the evaluation metric.
    """
    def __init__(self, feature_extractor):
        super(TripletNet, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, anchor, positive=None, negative=None):
        if positive is None or negative is None:
            z = self.feature_extractor(anchor)
            return F.normalize(z, p=2, dim=1)
        x = torch.cat([anchor, positive, negative], dim=0)   # (3B, C, H, W)
        z = self.feature_extractor(x)                        # (3B, D)
        a, p, n = torch.chunk(z, 3, dim=0)
        a = F.normalize(a, p=2, dim=1)
        p = F.normalize(p, p=2, dim=1)
        n = F.normalize(n, p=2, dim=1)
        return a, p, n
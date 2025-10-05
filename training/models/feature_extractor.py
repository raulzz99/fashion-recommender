# training/models/feature_extractor.py
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models

# --- weights helpers to be robust across torchvision versions ---
try:
    ResNet18_Weights = models.ResNet18_Weights
    ResNet50_Weights = models.ResNet50_Weights
    ViT_B_16_Weights = models.ViT_B_16_Weights
except Exception:
    ResNet18_Weights = ResNet50_Weights = ViT_B_16_Weights = None

class ViTTokenExtractor(nn.Module):
    """
    Torchvision ViT that returns CLS or mean-pooled patch tokens.
    Mirrors ViT forward up to encoder + final layer norm (as in your notebook logic).
    NOTE: Keeps your exact behavior (no explicit vit.ln call here).
    """
    def __init__(self, pooling='mean_pool', pretrained=True):
        super().__init__()
        assert pooling in ('cls', 'mean_pool')
        self.pooling = pooling

        vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        in_dim = vit.heads.head.in_features    # 768 for ViT-B/16
        vit.heads = nn.Identity()              # remove classifier

        self.model = vit
        self.proj = nn.Identity()              # keep as Identity unless we want a smaller D
        self.output_size = in_dim

        self.model = vit
        self.proj = nn.Identity()              # keep Identity unless you later want a smaller D
        self.output_size = in_dim

    def forward(self, x):
        B = x.shape[0]
        # Patch embedding
        x = self.model._process_input(x)                   # (B, N, D)
        # Add CLS token
        cls_tok = self.model.class_token.expand(B, -1, -1) # (B, 1, D)
        x = torch.cat((cls_tok, x), dim=1)                 # (B, N+1, D)
        # Encoder
        x = self.model.encoder(x)                          # (B, N+1, D)

        if self.pooling == 'cls':
            h = x[:, 0]                                    # (B, D)
        else:
            h = x[:, 1:].mean(dim=1)                       # (B, D)

        return self.proj(h)                                # (B, D)

class FeatureExtractor(nn.Module):
    """
    Outputs (B, D) with D > 1 for resnet18/resnet50/vit_b_16.
    For ViT, uses token extractor with 'cls' or 'mean_pool'.
    """
    def __init__(self, model_name='resnet18', pooling='mean_pool', pretrained=True):
        super().__init__()
        self.model_name = model_name

        if model_name == 'resnet18':
            m = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            in_dim = m.fc.in_features            # 512
            m.fc = nn.Identity()                 # -> (B, 512)
            self.model = m
            self.output_size = in_dim

        elif model_name == 'resnet50':
            m = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            in_dim = m.fc.in_features            # 2048
            m.fc = nn.Identity()                 # -> (B, 2048)
            self.model = m
            self.output_size = in_dim

        elif model_name == 'vit_b_16':
            self.model = ViTTokenExtractor(pooling=pooling, pretrained=pretrained)
            self.output_size = self.model.output_size

        else:
            raise ValueError(f"Model '{model_name}' not supported.")

    def forward(self, x):
        return self.model(x)   # (B, D)

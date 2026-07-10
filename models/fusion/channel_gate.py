import torch
import torch.nn as nn

class ChannelGate(nn.Module):
    """
    Computes a channel-wise gating vector for a single modality.
    Uses Global Average Pooling followed by a lightweight MLP to estimate 
    channel importance (reliability) between 0 and 1.
    """
    def __init__(self, channels: int = 320, reduction_ratio: int = 4):
        super().__init__()
        hidden = channels // reduction_ratio
        
        # 3D AdaptiveAvgPool to reduce spatial dimensions to 1x1x1
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize gates near 0.5 (neutral importance) by initializing the final linear layer
        # with small weights
        nn.init.kaiming_normal_(self.mlp[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.mlp[2].weight, std=0.01)

    def forward(self, x):
        b, c, _, _, _ = x.shape
        # GAP: [B, C, D, H, W] -> [B, C, 1, 1, 1] -> [B, C]
        y = self.gap(x).view(b, c)
        
        # MLP: [B, C] -> [B, C]
        gate = self.mlp(y)
        
        # Expand gate to match spatial dimensions: [B, C, 1, 1, 1]
        gate = gate.view(b, c, 1, 1, 1)
        
        # Scale input features
        return x * gate

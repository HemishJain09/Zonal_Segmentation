import torch
import torch.nn as nn

class AdaptiveReconstructionAdapter(nn.Module):
    """
    Module 2: Lightweight residual adapter applied to the nnU-Net bottleneck.
    Follows ponytail design: minimal code, stdlib components (nn.Conv3d).
    
    Architecture:
      Input (320) -> 1x1x1 Conv (80) -> ReLU -> 3x3x3 Conv (80) -> ReLU -> 1x1x1 Conv (320) -> Residual Add
    """
    def __init__(self, in_channels: int = 320, reduction_ratio: int = 4):
        super().__init__()
        hidden_channels = in_channels // reduction_ratio
        
        self.adapter = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1, bias=False)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming init for first two layers, Zero init for the last layer.
        This ensures the adapter acts as an identity mapping at the start of training,
        preserving the baseline's performance completely.
        """
        nn.init.kaiming_normal_(self.adapter[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.adapter[2].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.adapter[4].weight)

    def forward(self, x):
        return x + self.adapter(x)


class AdapterFactory(nn.ModuleDict):
    """
    Creates a dictionary of adapters for the modalities.
    Usage:
        adapters = AdapterFactory(["t2", "adc", "hbv"])
        adapted_t2 = adapters["t2"](shared_bottleneck)
    """
    def __init__(self, modalities: list, in_channels: int = 320, reduction_ratio: int = 4):
        adapters = {
            mod: AdaptiveReconstructionAdapter(in_channels, reduction_ratio)
            for mod in modalities
        }
        super().__init__(adapters)

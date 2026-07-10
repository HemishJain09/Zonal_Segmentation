import torch
import torch.nn as nn
from models.fusion.channel_gate import ChannelGate

class ChannelGatedFusion(nn.Module):
    """
    Module 5: Channel-Gated Feature Fusion.
    Fuses the adapted latents from T2, ADC, and HBV into a single bottleneck 
    representation compatible with the pretrained nnU-Net decoder.
    """
    def __init__(self, modalities: list, channels: int = 320, reduction_ratio: int = 4):
        super().__init__()
        
        # Create an independent channel gate for each modality
        self.gates = nn.ModuleDict({
            mod: ChannelGate(channels, reduction_ratio)
            for mod in modalities
        })
        
        # 1x1x1 Conv to reduce the concatenated 960 channels back to 320
        in_channels = channels * len(modalities)
        self.projection = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier init for the projection layer
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, adapted_latents: dict):
        """
        Args:
            adapted_latents: Dict mapping modality name to its adapted latent tensor
                             (e.g., {"t2": z_t2, "adc": z_adc, "hbv": z_hbv})
        Returns:
            ZFused: A single fused latent tensor [B, 320, D, H, W]
        """
        # Apply gating to each modality
        gated_features = []
        # Ensure we always concatenate in the exact same order
        for mod in sorted(self.gates.keys()):
            gate_mod = self.gates[mod]
            latent = adapted_latents[mod]
            gated = gate_mod(latent)
            gated_features.append(gated)
            
        # Concatenate along the channel dimension
        # 3 * [B, 320, D, H, W] -> [B, 960, D, H, W]
        concat = torch.cat(gated_features, dim=1)
        
        # Project back to decoder dimension
        # [B, 960, D, H, W] -> [B, 320, D, H, W]
        fused = self.projection(concat)
        
        return fused

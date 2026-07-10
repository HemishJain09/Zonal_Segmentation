import torch
import torch.nn as nn

class SelfReconstructionHead(nn.Module):
    """
    Module 3: Self-Reconstruction Branch.
    Learns to reconstruct the adapted latent representation Z' to ensure 
    intra-modality consistency during Test-Time Adaptation.
    
    Architecture:
      Input (320) -> 3x3x3 Conv (320) -> ReLU -> 3x3x3 Conv (320) -> Reconstructed Z''
    """
    def __init__(self, channels: int = 320):
        super().__init__()
        self.reconstructor = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier init as specified in the document
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.reconstructor(x)


class SelfReconstructionFactory(nn.ModuleDict):
    """
    Creates a dictionary of self-reconstruction heads for the modalities.
    """
    def __init__(self, modalities: list, channels: int = 320):
        heads = {
            mod: SelfReconstructionHead(channels)
            for mod in modalities
        }
        super().__init__(heads)

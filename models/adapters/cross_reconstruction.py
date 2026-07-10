import torch
import torch.nn as nn

class CrossReconstructionHead(nn.Module):
    """
    Module 4: Cross-Reconstruction Branch.
    Learns to predict the latent representation of one modality given the other two.
    
    Architecture:
      Input (320*2=640) -> 1x1x1 Conv (320) -> ReLU -> 3x3x3 Conv (320) -> ReLU -> 3x3x3 Conv (320) -> Predicted Z
    """
    def __init__(self, in_channels: int = 640, out_channels: int = 320):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1, x2):
        # Concatenate the two input modalities along the channel dimension
        x = torch.cat([x1, x2], dim=1)
        return self.predictor(x)


class CrossReconstructionFactory(nn.ModuleDict):
    """
    Creates a dictionary of cross-reconstruction heads for the 3 prediction tasks.
    Tasks:
      - predict_hbv: from t2 and adc
      - predict_adc: from t2 and hbv
      - predict_t2: from adc and hbv
    """
    def __init__(self, channels: int = 320):
        heads = {
            "predict_hbv": CrossReconstructionHead(in_channels=channels*2, out_channels=channels),
            "predict_adc": CrossReconstructionHead(in_channels=channels*2, out_channels=channels),
            "predict_t2": CrossReconstructionHead(in_channels=channels*2, out_channels=channels)
        }
        super().__init__(heads)

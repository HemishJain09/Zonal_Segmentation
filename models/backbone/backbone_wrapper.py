import torch
import torch.nn as nn
from models.backbone.nnunet_loader import load_nnunet_model

class BaselineWrapper(nn.Module):
    """
    Wraps the loaded PlainConvUNet to expose explicit encode() and decode() methods.
    This allows us to insert the adaptation framework between the encoder and decoder.
    """
    def __init__(self, plans_path: str, checkpoint_path: str, device="cpu"):
        super().__init__()
        self.model = load_nnunet_model(plans_path, checkpoint_path, device)
        
    def encode(self, x):
        """
        Runs the encoder and returns (bottleneck, skips).
        """
        skips = []
        current = x
        for stage in self.model.encoder.stages:
            current = stage(current)
            skips.append(current)
            
        bottleneck = current
        return bottleneck, skips

    def decode(self, bottleneck, skips):
        """
        Runs the decoder given a modified bottleneck and the original skip connections.
        """
        current = bottleneck
        for i, (transpconv, stage) in enumerate(zip(self.model.decoder.transpconvs, self.model.decoder.stages)):
            current = transpconv(current)
            skip_idx = len(skips) - 2 - i
            current = torch.cat([current, skips[skip_idx]], dim=1)
            current = stage(current)
            
        return self.model.decoder.seg_layers[-1](current)

    def forward(self, x):
        """
        Standard forward pass (identical to baseline).
        """
        bottleneck, skips = self.encode(x)
        return self.decode(bottleneck, skips)

    def freeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        for param in self.model.decoder.parameters():
            param.requires_grad = False
            
    def freeze_all(self):
        self.freeze_encoder()
        self.freeze_decoder()

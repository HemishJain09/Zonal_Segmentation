import torch
import torch.nn as nn

class MockBaselineWrapper(nn.Module):
    """
    Mocks the BaselineWrapper to allow local testing without the heavy nnU-Net
    dependencies and checkpoints.
    It returns tensors of the exact expected shape.
    """
    def __init__(self, plans_path: str = "", checkpoint_path: str = "", device="cpu"):
        super().__init__()
        # We add some dummy parameters so the optimizer doesn't complain if it looks at it
        self.encoder = nn.Sequential(nn.Conv3d(3, 320, 3, padding=1))
        self.decoder = nn.Sequential(nn.Conv3d(320, 3, 3, padding=1))
        
    def encode(self, x):
        # x is [B, 3, 16, 320, 320]
        b = x.shape[0]
        # Return a dummy bottleneck of shape [B, 320, 4, 5, 5]
        bottleneck = torch.randn(b, 320, 4, 5, 5, device=x.device, requires_grad=True)
        # Dummy skips (just random tensors to pass through)
        skips = [torch.randn(1) for _ in range(6)]
        return bottleneck, skips

    def decode(self, bottleneck, skips):
        b = bottleneck.shape[0]
        # Use the bottleneck so gradients flow back through it!
        # Just do a simple projection to the expected output shape
        x = bottleneck.view(b, -1).sum(dim=1).view(b, 1, 1, 1, 1)
        # Create a dummy segmentation of shape [B, 3, 16, 320, 320]
        dummy = torch.randn(b, 3, 16, 320, 320, device=bottleneck.device)
        seg = x * dummy # Now seg depends on bottleneck
        return seg

    def forward(self, x):
        bottleneck, skips = self.encode(x)
        return self.decode(bottleneck, skips)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
            
    def freeze_all(self):
        self.freeze_encoder()
        self.freeze_decoder()

import torch
import torch.nn as nn
from losses.segmentation import DC_and_CE_loss
from losses.self_loss import compute_self_reconstruction_loss
from losses.cross_loss import compute_cross_reconstruction_loss

class bpMRILoss(nn.Module):
    """
    Computes the total loss for source training:
    L_total = L_seg + lambda_self * L_self + lambda_cross * L_cross
    """
    def __init__(self, lambda_self: float = 0.1, lambda_cross: float = 0.1):
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_cross = lambda_cross
        
        # Standard nnU-Net segmentation loss
        self.seg_loss = DC_and_CE_loss(
            soft_dice_kwargs={'batch_dice': True, 'smooth': 1e-5, 'do_bg': False},
            ce_kwargs={}
        )

    def forward(self, network_outputs: dict, targets: torch.Tensor):
        """
        Args:
            network_outputs: Output dictionary from network.forward_train()
            targets: Ground truth segmentation mask [B, 1, D, H, W]
        """
        # 1. Segmentation Loss
        l_seg = self.seg_loss(network_outputs["segmentation"], targets)
        
        # 2. Self-Reconstruction Loss
        l_self = compute_self_reconstruction_loss(
            predictions=network_outputs["self_outputs"],
            targets=network_outputs["adapted_latents"]
        )
        
        # 3. Cross-Reconstruction Loss
        l_cross = compute_cross_reconstruction_loss(
            predictions=network_outputs["cross_outputs"],
            targets=network_outputs["adapted_latents"]
        )
        
        # Total Loss
        l_total = l_seg + (self.lambda_self * l_self) + (self.lambda_cross * l_cross)
        
        return {
            "total": l_total,
            "seg": l_seg,
            "self": l_self,
            "cross": l_cross
        }

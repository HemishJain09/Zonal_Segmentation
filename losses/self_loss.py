import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_self_reconstruction_loss(predictions: dict, targets: dict) -> torch.Tensor:
    """
    Computes the self-reconstruction loss (L1) across all modalities.
    
    Args:
        predictions: Dict mapping modality name to reconstructed latent (e.g. {"t2": z_t2_recon})
        targets: Dict mapping modality name to original adapted latent (e.g. {"t2": z_t2_adapted})
        
    Returns:
        Scalar tensor containing the averaged L1 loss.
    """
    losses = []
    for mod in predictions.keys():
        loss = F.l1_loss(predictions[mod], targets[mod])
        losses.append(loss)
        
    return torch.stack(losses).mean()

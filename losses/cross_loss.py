import torch
import torch.nn.functional as F

def compute_cross_reconstruction_loss(predictions: dict, targets: dict) -> torch.Tensor:
    """
    Computes the cross-reconstruction loss (L1) across the 3 prediction tasks.
    
    Args:
        predictions: Dict mapping task to predicted latent 
                     (e.g. {"predict_hbv": pred_hbv, "predict_adc": pred_adc, "predict_t2": pred_t2})
        targets: Dict mapping target modality to original adapted latent 
                 (e.g. {"hbv": z_hbv, "adc": z_adc, "t2": z_t2})
        
    Returns:
        Scalar tensor containing the averaged L1 loss.
    """
    losses = [
        F.l1_loss(predictions["predict_hbv"], targets["hbv"]),
        F.l1_loss(predictions["predict_adc"], targets["adc"]),
        F.l1_loss(predictions["predict_t2"], targets["t2"])
    ]
    
    return torch.stack(losses).mean()

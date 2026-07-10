import torch
from torch.optim import AdamW

from models.model_factory import create_adaptation_network
from losses.self_loss import compute_self_reconstruction_loss
from losses.cross_loss import compute_cross_reconstruction_loss

class ContinualTTASession:
    """
    Module 8: Continual Test-Time Adaptation Session.
    Adapts the network continually for a stream of patients from a new hospital (e.g. PCNN).
    Only updates the adapters using self-supervised reconstruction losses.
    Labels are NOT used.
    """
    def __init__(self, plans_path: str, checkpoint_path: str, source_adapters_path: str, device="cuda"):
        self.device = device
        self.network = create_adaptation_network(plans_path, checkpoint_path, device)
        
        # Load the source-trained adaptation modules
        self._load_source_adapters(source_adapters_path)
        
        # Freeze everything EXCEPT the adapters
        self.network.freeze_for_tta()
        
        # Optimizer only contains adapter parameters
        adapter_params = [p for p in self.network.adapters.parameters() if p.requires_grad]
        self.optimizer = AdamW(adapter_params, lr=1e-5) # Smaller learning rate for TTA
        
        self.lambda_self = 0.1
        self.lambda_cross = 0.1
        
        self.patient_logs = []

    def _load_source_adapters(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.network.adapters.load_state_dict(state['adapters'])
        self.network.self_heads.load_state_dict(state['self_heads'])
        self.network.cross_heads.load_state_dict(state['cross_heads'])
        self.network.fusion.load_state_dict(state['fusion'])
        print(f"Loaded source adaptation modules from {path}")

    def adapt_and_predict(self, patient_id: str, image_tensor: torch.Tensor, k_iterations: int = 10):
        """
        Adapts the network to a single patient using K iterations, then returns the 
        final segmentation prediction.
        Adapter weights persist to the next patient in the session.
        """
        self.network.train() # Adapters in train mode
        image_tensor = image_tensor.to(self.device)
        
        # 1. Adaptation Loop (Self-Supervised)
        for k in range(k_iterations):
            self.optimizer.zero_grad()
            
            # Forward TTA (only computes recon branches, not segmentation)
            outputs = self.network.forward_tta(image_tensor)
            
            # Compute self-supervised losses
            l_self = compute_self_reconstruction_loss(
                predictions=outputs["self_outputs"],
                targets=outputs["adapted_latents"]
            )
            
            l_cross = compute_cross_reconstruction_loss(
                predictions=outputs["cross_outputs"],
                targets=outputs["adapted_latents"]
            )
            
            l_tta = (self.lambda_self * l_self) + (self.lambda_cross * l_cross)
            
            # Update ONLY the adapters
            l_tta.backward()
            self.optimizer.step()
            
        # 2. Final Inference (predict segmentation)
        self.network.eval()
        with torch.no_grad():
            segmentation = self.network.forward_inference(image_tensor)
            
        # Log adaptation for this patient
        self.patient_logs.append({
            "patient_id": patient_id,
            "final_l_self": l_self.item(),
            "final_l_cross": l_cross.item(),
            "final_l_tta": l_tta.item()
        })
            
        return segmentation

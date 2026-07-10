import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.model_factory import create_adaptation_network
from losses.total_loss import bpMRILoss

class SourceTrainer:
    """
    Module 7: Source Training Pipeline.
    Trains the adaptation modules (Adapters, Recon Heads, Fusion) using labeled source data
    while keeping the nnU-Net Encoder and Decoder completely frozen.
    """
    def __init__(self, plans_path: str, checkpoint_path: str, device="cuda"):
        self.device = device
        
        # 1. Network setup
        self.network = create_adaptation_network(plans_path, checkpoint_path, device)
        self.network.freeze_for_source_training()
        
        # 2. Loss setup
        self.criterion = bpMRILoss(lambda_self=0.1, lambda_cross=0.1).to(device)
        
        # 3. Optimizer setup - ONLY pass parameters that require gradients
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
        
        # 4. State
        self.current_epoch = 0

    def train_epoch(self, dataloader, num_epochs=1):
        """
        Executes one epoch of source training.
        """
        self.network.train()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        epoch_losses = {"total": 0, "seg": 0, "self": 0, "cross": 0}
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            # nnU-Net dataloaders return dictionaries
            data = batch['data'].to(self.device)
            target = batch['target'][0].to(self.device) # Target [0] is the full resolution mask
            
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.network.forward_train(data)
            
            # Loss
            loss_dict = self.criterion(outputs, target)
            
            # Backward
            loss_dict["total"].backward()
            self.optimizer.step()
            
            # Logging
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k].item()
                
            pbar.set_postfix({
                'Total': f"{loss_dict['total'].item():.4f}",
                'Seg': f"{loss_dict['seg'].item():.4f}",
                'Recon': f"{(loss_dict['self'].item() + loss_dict['cross'].item()):.4f}"
            })
            
        self.scheduler.step()
        self.current_epoch += 1
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= len(dataloader)
            
        return epoch_losses

    def save_checkpoint(self, path: str):
        """
        Saves ONLY the adaptation modules. We don't save the encoder/decoder since
        they are frozen and we already have the baseline checkpoint.
        """
        state = {
            'epoch': self.current_epoch,
            'adapters': self.network.adapters.state_dict(),
            'self_heads': self.network.self_heads.state_dict(),
            'cross_heads': self.network.cross_heads.state_dict(),
            'fusion': self.network.fusion.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)
        print(f"Saved adaptation modules to {path}")

    def load_checkpoint(self, path: str):
        """Loads previously trained adaptation modules."""
        state = torch.load(path, map_location=self.device)
        self.network.adapters.load_state_dict(state['adapters'])
        self.network.self_heads.load_state_dict(state['self_heads'])
        self.network.cross_heads.load_state_dict(state['cross_heads'])
        self.network.fusion.load_state_dict(state['fusion'])
        
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        if 'epoch' in state:
            self.current_epoch = state['epoch']
            
        print(f"Loaded adaptation modules from {path}")

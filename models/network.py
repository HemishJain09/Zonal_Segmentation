import torch
import torch.nn as nn
from models.backbone.backbone_wrapper import BaselineWrapper
from models.adapters.residual_adapter import AdapterFactory
from models.adapters.self_reconstruction import SelfReconstructionFactory
from models.adapters.cross_reconstruction import CrossReconstructionFactory
from models.fusion.gated_fusion import ChannelGatedFusion

class bpMRITTA(nn.Module):
    """
    Module 6: Complete Network Assembly.
    Assembles the frozen baseline nnU-Net, Adapters, Reconstruction Branches, and Fusion
    into a single cohesive network with 3 explicit forward execution modes.
    """
    def __init__(self, plans_path: str, checkpoint_path: str, device="cpu"):
        super().__init__()
        
        # Modalities in exact order: T2, ADC, HBV (matching nnU-Net input channels 0, 1, 2)
        self.modalities = ["t2", "adc", "hbv"]
        
        # 1. Baseline Backbone
        self.backbone = BaselineWrapper(plans_path, checkpoint_path, device)
        
        # 2. Adaptive Reconstruction Adapters (ARA)
        self.adapters = AdapterFactory(self.modalities, in_channels=320, reduction_ratio=4)
        
        # 3. Self-Reconstruction Branches
        self.self_heads = SelfReconstructionFactory(self.modalities, channels=320)
        
        # 4. Cross-Reconstruction Branches
        self.cross_heads = CrossReconstructionFactory(channels=320)
        
        # 5. Channel-Gated Fusion
        self.fusion = ChannelGatedFusion(self.modalities, channels=320, reduction_ratio=4)

    def _get_channel_latents(self, shared_latent):
        """
        Since our encoder processes all modalities jointly and outputs a single shared
        bottleneck, we pass the same shared latent to all 3 modality-specific adapters.
        Each adapter learns to extract/correct its specific modality's features.
        """
        return {
            "t2": shared_latent,
            "adc": shared_latent,
            "hbv": shared_latent
        }

    def forward_train(self, x):
        """
        Execution Mode 1: Source Training Mode.
        Computes all branches so that all losses (Seg, Self, Cross) can be calculated.
        """
        # 1. Shared Encoder
        shared_bottleneck, skips = self.backbone.encode(x)
        
        # Map shared bottleneck to inputs for each adapter
        inputs = self._get_channel_latents(shared_bottleneck)
        
        # 2. Adapters
        adapted = {mod: self.adapters[mod](inputs[mod]) for mod in self.modalities}
        
        # 3. Self-Reconstruction
        self_recon = {mod: self.self_heads[mod](adapted[mod]) for mod in self.modalities}
        
        # 4. Cross-Reconstruction
        cross_recon = {
            "predict_hbv": self.cross_heads["predict_hbv"](adapted["t2"], adapted["adc"]),
            "predict_adc": self.cross_heads["predict_adc"](adapted["t2"], adapted["hbv"]),
            "predict_t2": self.cross_heads["predict_t2"](adapted["adc"], adapted["hbv"])
        }
        
        # 5. Fusion
        fused_latent = self.fusion(adapted)
        
        # 6. Shared Decoder
        segmentation = self.backbone.decode(fused_latent, skips)
        
        return {
            "segmentation": segmentation,
            "adapted_latents": adapted,
            "self_outputs": self_recon,
            "cross_outputs": cross_recon,
            "fused_latent": fused_latent
        }

    def forward_tta(self, x):
        """
        Execution Mode 2: Test-Time Adaptation Mode.
        Computes reconstruction branches for optimization. Does not compute segmentation
        to save memory, as it's not used for TTA loss.
        """
        # 1. Shared Encoder
        shared_bottleneck, _ = self.backbone.encode(x)
        inputs = self._get_channel_latents(shared_bottleneck)
        
        # 2. Adapters (these are the ONLY things being updated)
        adapted = {mod: self.adapters[mod](inputs[mod]) for mod in self.modalities}
        
        # 3. Self-Reconstruction (frozen)
        self_recon = {mod: self.self_heads[mod](adapted[mod]) for mod in self.modalities}
        
        # 4. Cross-Reconstruction (frozen)
        cross_recon = {
            "predict_hbv": self.cross_heads["predict_hbv"](adapted["t2"], adapted["adc"]),
            "predict_adc": self.cross_heads["predict_adc"](adapted["t2"], adapted["hbv"]),
            "predict_t2": self.cross_heads["predict_t2"](adapted["adc"], adapted["hbv"])
        }
        
        return {
            "adapted_latents": adapted,
            "self_outputs": self_recon,
            "cross_outputs": cross_recon
        }

    def forward_inference(self, x):
        """
        Execution Mode 3: Inference Mode.
        Fastest path. Skips all reconstruction branches.
        """
        # 1. Shared Encoder
        shared_bottleneck, skips = self.backbone.encode(x)
        inputs = self._get_channel_latents(shared_bottleneck)
        
        # 2. Adapters
        adapted = {mod: self.adapters[mod](inputs[mod]) for mod in self.modalities}
        
        # 3. Fusion
        fused_latent = self.fusion(adapted)
        
        # 4. Shared Decoder
        return self.backbone.decode(fused_latent, skips)

    def freeze_for_source_training(self):
        """Freezes backbone. Leaves adapters, recon heads, and fusion trainable."""
        self.backbone.freeze_all()
        # Ensure new modules require gradients
        for p in self.adapters.parameters(): p.requires_grad = True
        for p in self.self_heads.parameters(): p.requires_grad = True
        for p in self.cross_heads.parameters(): p.requires_grad = True
        for p in self.fusion.parameters(): p.requires_grad = True

    def freeze_for_tta(self):
        """Freezes everything EXCEPT adapters."""
        self.backbone.freeze_all()
        for p in self.self_heads.parameters(): p.requires_grad = False
        for p in self.cross_heads.parameters(): p.requires_grad = False
        for p in self.fusion.parameters(): p.requires_grad = False
        
        # ONLY adapters learn during TTA
        for p in self.adapters.parameters(): p.requires_grad = True

    def freeze_for_inference(self):
        """Freezes entire network."""
        for p in self.parameters():
            p.requires_grad = False

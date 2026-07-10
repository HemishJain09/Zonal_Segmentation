import unittest
import torch

from models.adapters.residual_adapter import AdaptiveReconstructionAdapter, AdapterFactory
from models.adapters.self_reconstruction import SelfReconstructionHead, SelfReconstructionFactory
from models.adapters.cross_reconstruction import CrossReconstructionHead, CrossReconstructionFactory
from models.fusion.channel_gate import ChannelGate
from models.fusion.gated_fusion import ChannelGatedFusion

class TestModules(unittest.TestCase):
    def setUp(self):
        self.b, self.c, self.d, self.h, self.w = 2, 320, 4, 5, 5
        self.dummy_input = torch.randn(self.b, self.c, self.d, self.h, self.w)
        self.modalities = ["t2", "adc", "hbv"]

    def test_adapter_shape_and_identity(self):
        # Module 2
        adapter = AdaptiveReconstructionAdapter(in_channels=self.c, reduction_ratio=4)
        out = adapter(self.dummy_input)
        
        # Shape should be preserved
        self.assertEqual(out.shape, self.dummy_input.shape)
        
        # Identity initialization check: The residual path should be exactly 0 at initialization
        # So output should perfectly match input
        diff = torch.abs(out - self.dummy_input).max().item()
        self.assertLess(diff, 1e-6, "Adapter should act as identity at initialization")

    def test_self_reconstruction_shape(self):
        # Module 3
        head = SelfReconstructionHead(channels=self.c)
        out = head(self.dummy_input)
        
        # Shape should be preserved
        self.assertEqual(out.shape, self.dummy_input.shape)

    def test_cross_reconstruction_shape(self):
        # Module 4
        head = CrossReconstructionHead(in_channels=self.c * 2, out_channels=self.c)
        dummy_input2 = torch.randn(self.b, self.c, self.d, self.h, self.w)
        out = head(self.dummy_input, dummy_input2)
        
        # Shape should be predicted latent of same dimensions
        self.assertEqual(out.shape, self.dummy_input.shape)

    def test_channel_gate_shape(self):
        # Module 5 - Gate
        gate = ChannelGate(channels=self.c, reduction_ratio=4)
        out = gate(self.dummy_input)
        
        # Gating should preserve shape (it's element-wise multiplication across channels)
        self.assertEqual(out.shape, self.dummy_input.shape)

    def test_fusion_shape(self):
        # Module 5 - Fusion
        fusion = ChannelGatedFusion(self.modalities, channels=self.c, reduction_ratio=4)
        
        adapted_latents = {
            "t2": torch.randn(self.b, self.c, self.d, self.h, self.w),
            "adc": torch.randn(self.b, self.c, self.d, self.h, self.w),
            "hbv": torch.randn(self.b, self.c, self.d, self.h, self.w)
        }
        
        out = fusion(adapted_latents)
        
        # The fused output must have exactly the original bottleneck shape for decoder compatibility
        self.assertEqual(out.shape, self.dummy_input.shape)

if __name__ == '__main__':
    unittest.main()

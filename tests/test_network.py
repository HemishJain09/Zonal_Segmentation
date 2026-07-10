import unittest
import torch
import unittest.mock as mock

from models.network import bpMRITTA
from tests.mock_backbone import MockBaselineWrapper

class TestNetworkAssembly(unittest.TestCase):
    def setUp(self):
        # We need to mock the baseline wrapper inside bpMRITTA 
        # so we don't try to load actual nnUNet checkpoints
        with mock.patch('models.network.BaselineWrapper', MockBaselineWrapper):
            self.network = bpMRITTA(plans_path="", checkpoint_path="")
            
        self.b, self.c, self.d, self.h, self.w = 2, 3, 16, 320, 320
        self.dummy_input = torch.randn(self.b, self.c, self.d, self.h, self.w)
        
    def test_forward_train(self):
        out = self.network.forward_train(self.dummy_input)
        
        # Check that it returns all expected dictionary keys
        expected_keys = ["segmentation", "adapted_latents", "self_outputs", "cross_outputs", "fused_latent"]
        for k in expected_keys:
            self.assertIn(k, out)
            
        # Check shapes
        self.assertEqual(out["segmentation"].shape, (self.b, 3, 16, 320, 320))
        self.assertEqual(out["fused_latent"].shape, (self.b, 320, 4, 5, 5))
        
        for mod in ["t2", "adc", "hbv"]:
            self.assertEqual(out["adapted_latents"][mod].shape, (self.b, 320, 4, 5, 5))
            self.assertEqual(out["self_outputs"][mod].shape, (self.b, 320, 4, 5, 5))
            
        for task in ["predict_hbv", "predict_adc", "predict_t2"]:
            self.assertEqual(out["cross_outputs"][task].shape, (self.b, 320, 4, 5, 5))

    def test_forward_tta(self):
        out = self.network.forward_tta(self.dummy_input)
        
        # Should NOT compute segmentation or fused_latent
        self.assertNotIn("segmentation", out)
        self.assertNotIn("fused_latent", out)
        
        # Should have recon outputs for adaptation loss
        self.assertIn("adapted_latents", out)
        self.assertIn("self_outputs", out)
        self.assertIn("cross_outputs", out)

    def test_forward_inference(self):
        out = self.network.forward_inference(self.dummy_input)
        
        # Inference mode returns only the segmentation tensor
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (self.b, 3, 16, 320, 320))

    def test_freeze_for_source_training(self):
        self.network.freeze_for_source_training()
        
        # Backbone should be frozen
        for p in self.network.backbone.parameters():
            self.assertFalse(p.requires_grad)
            
        # Adapters, recon, and fusion should be trainable
        self.assertTrue(any(p.requires_grad for p in self.network.adapters.parameters()))
        self.assertTrue(any(p.requires_grad for p in self.network.self_heads.parameters()))
        self.assertTrue(any(p.requires_grad for p in self.network.cross_heads.parameters()))
        self.assertTrue(any(p.requires_grad for p in self.network.fusion.parameters()))

    def test_freeze_for_tta(self):
        self.network.freeze_for_tta()
        
        # Backbone, recon, fusion should be frozen
        for p in self.network.backbone.parameters():
            self.assertFalse(p.requires_grad)
        for p in self.network.self_heads.parameters():
            self.assertFalse(p.requires_grad)
        for p in self.network.cross_heads.parameters():
            self.assertFalse(p.requires_grad)
        for p in self.network.fusion.parameters():
            self.assertFalse(p.requires_grad)
            
        # ONLY adapters should be trainable
        self.assertTrue(any(p.requires_grad for p in self.network.adapters.parameters()))

if __name__ == '__main__':
    unittest.main()

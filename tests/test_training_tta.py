import unittest
import torch
import unittest.mock as mock
from torch.optim import AdamW

from models.network import bpMRITTA
from tests.mock_backbone import MockBaselineWrapper

class TestTrainingAndTTA(unittest.TestCase):
    def setUp(self):
        with mock.patch('models.network.BaselineWrapper', MockBaselineWrapper):
            self.network = bpMRITTA(plans_path="", checkpoint_path="")
            
        self.b, self.c, self.d, self.h, self.w = 2, 3, 16, 320, 320
        self.dummy_input = torch.randn(self.b, self.c, self.d, self.h, self.w)
        # Dummy target (one-hot or class indices, let's use class indices 0,1,2 for CE)
        self.dummy_target = torch.randint(0, 3, (self.b, 1, 16, 320, 320)).float()

    def test_source_training_gradient_flow(self):
        self.network.freeze_for_source_training()
        
        # Ensure gradients are zeroed
        self.network.zero_grad()
        
        # Forward
        outputs = self.network.forward_train(self.dummy_input)
        
        # We need a dummy loss since DC_and_CE_loss might fail without a proper softmax output
        # Let's just use MSE against dummy target to test gradient flow
        loss = outputs["segmentation"].sum() + sum([v.sum() for v in outputs["self_outputs"].values()])
        loss.backward()
        
        # Verify gradients reached expected modules
        self.assertTrue(any(p.grad is not None and p.grad.sum() != 0 for p in self.network.adapters.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.sum() != 0 for p in self.network.self_heads.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.sum() != 0 for p in self.network.fusion.parameters()))
        
        # Verify gradients did NOT reach backbone
        for p in self.network.backbone.parameters():
            self.assertTrue(p.grad is None or p.grad.sum() == 0)

    def test_tta_gradient_flow(self):
        self.network.freeze_for_tta()
        self.network.zero_grad()
        
        outputs = self.network.forward_tta(self.dummy_input)
        
        # Only self and cross outputs are used for TTA loss
        loss = sum([v.sum() for v in outputs["self_outputs"].values()]) + \
               sum([v.sum() for v in outputs["cross_outputs"].values()])
               
        loss.backward()
        
        # Verify gradients reached ONLY adapters
        self.assertTrue(any(p.grad is not None and p.grad.sum() != 0 for p in self.network.adapters.parameters()))
        
        # Verify gradients did NOT reach anything else
        for p in self.network.self_heads.parameters():
            self.assertTrue(p.grad is None or p.grad.sum() == 0)
        for p in self.network.cross_heads.parameters():
            self.assertTrue(p.grad is None or p.grad.sum() == 0)
        for p in self.network.fusion.parameters():
            self.assertTrue(p.grad is None or p.grad.sum() == 0)
        for p in self.network.backbone.parameters():
            self.assertTrue(p.grad is None or p.grad.sum() == 0)

if __name__ == '__main__':
    unittest.main()

from models.network import bpMRITTA

def create_adaptation_network(plans_path: str, checkpoint_path: str, device="cpu"):
    """
    Factory function to instantiate the full bpMRI-TTA adaptation framework.
    """
    return bpMRITTA(plans_path, checkpoint_path, device)

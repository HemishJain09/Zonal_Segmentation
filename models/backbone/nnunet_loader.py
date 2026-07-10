import json
import torch
from pathlib import Path


def load_nnunet_model(plans_path: str, checkpoint_path: str, device="cpu"):
    """
    Loads the pretrained nnU-Net baseline from disk.
    Follows the minimal-dependency "ponytail" philosophy: directly uses 
    dynamic_network_architectures rather than building custom classes.
    """
    with open(plans_path) as f:
        plans = json.load(f)

    config = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]

    # Import the exact class nnU-Net used during training
    import importlib
    def resolve(class_string):
        module_path, class_name = class_string.rsplit(".", 1)
        return getattr(importlib.import_module(module_path), class_name)

    arch = plans["configurations"]["3d_fullres"]["architecture"]
    kw = dict(config)
    for key in arch.get("_kw_requires_import", []):
        if kw.get(key):
            kw[key] = resolve(kw[key])

    # Hardcode input/output channels since we know we are doing Zonal Segmentation (3 MRI, 3 classes)
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    model = PlainConvUNet(
        input_channels=3,
        num_classes=3,
        **kw
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('network_weights', ckpt.get('state_dict', ckpt))
    
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

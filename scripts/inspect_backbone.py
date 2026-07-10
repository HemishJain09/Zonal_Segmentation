"""
Module 0 — Backbone Inspector
Traces a forward pass through the trained nnU-Net, prints every tensor shape,
and verifies encode/decode separability.

Run on Colab after loading the trained model:
  python /content/ZonalSeg/scripts/inspect_backbone.py \
      --plans /content/nnUNet_preprocessed/Dataset501_ZonalSeg/nnUNetPlans.json \
      --checkpoint /content/drive/MyDrive/ZonalSeg_Results/nnUNet_results/Dataset501_ZonalSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth \
      --dataset_json /content/nnUNet_preprocessed/Dataset501_ZonalSeg/dataset.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


def load_plans(plans_path: str) -> dict:
    with open(plans_path) as f:
        return json.load(f)


def build_network(plans: dict, num_input_channels: int, num_classes: int):
    """Build the PlainConvUNet from nnU-Net plans — reuses the installed package."""
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    config = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]

    # Import the actual classes referenced in plans
    import importlib
    def resolve(class_string):
        module_path, class_name = class_string.rsplit(".", 1)
        return getattr(importlib.import_module(module_path), class_name)

    arch = plans["configurations"]["3d_fullres"]["architecture"]
    kw = dict(config)
    for key in arch.get("_kw_requires_import", []):
        if kw.get(key):
            kw[key] = resolve(kw[key])

    model = PlainConvUNet(
        input_channels=num_input_channels,
        num_classes=num_classes,
        **kw
    )
    return model


def trace_encoder_decoder(model, patch_size, num_input_channels, device="cpu"):
    """Run a dummy forward pass, intercepting every stage."""
    model.eval()
    model.to(device)

    x = torch.randn(1, num_input_channels, *patch_size, device=device)
    print(f"\n{'='*70}")
    print(f"INPUT: {list(x.shape)}")
    print(f"{'='*70}")

    # === ENCODER ===
    print(f"\n--- ENCODER ---")
    skips = []
    current = x
    for i, stage in enumerate(model.encoder.stages):
        current = stage(current)
        skips.append(current)
        print(f"  Stage {i}: {list(current.shape)}  (features={current.shape[1]})")

    bottleneck = current
    print(f"\n  ★ BOTTLENECK: {list(bottleneck.shape)}")
    print(f"    Channels: {bottleneck.shape[1]}")
    print(f"    Spatial:  {list(bottleneck.shape[2:])}")
    print(f"    Total elements per sample: {bottleneck[0].numel()}")

    # === SKIP CONNECTIONS ===
    print(f"\n--- SKIP CONNECTIONS (encoder → decoder) ---")
    # In PlainConvUNet, decoder uses skips from encoder stages [0..N-2]
    # (the last encoder stage IS the bottleneck, not a skip)
    for i, s in enumerate(skips[:-1]):
        print(f"  Skip {i}: {list(s.shape)}")

    # === DECODER ===
    print(f"\n--- DECODER ---")
    # PlainConvUNet decoder: transpconv → concat skip → conv stages
    seg_layers = []
    current = bottleneck
    for i, (transpconv, stage) in enumerate(zip(model.decoder.transpconvs, model.decoder.stages)):
        current = transpconv(current)
        skip_idx = len(skips) - 2 - i  # reverse order
        current = torch.cat([current, skips[skip_idx]], dim=1)
        current = stage(current)
        print(f"  Decoder stage {i}: {list(current.shape)}  (after concat+conv)")

        # Check if there's a segmentation layer at this stage
        if hasattr(model.decoder, 'seg_layers') and i < len(model.decoder.seg_layers):
            seg_out = model.decoder.seg_layers[i](current)
            seg_layers.append(seg_out)

    print(f"\n--- SEGMENTATION OUTPUT ---")
    with torch.no_grad():
        full_output = model(x)
    if isinstance(full_output, (list, tuple)):
        for j, o in enumerate(full_output):
            print(f"  Output[{j}]: {list(o.shape)}")
        final = full_output[0]
    else:
        print(f"  Output: {list(full_output.shape)}")
        final = full_output
    print(f"  Final segmentation: {list(final.shape)} (classes={final.shape[1]})")

    return bottleneck


def verify_checkpoint(model, checkpoint_path, device="cpu"):
    """Load checkpoint and verify all keys match."""
    print(f"\n{'='*70}")
    print(f"CHECKPOINT VERIFICATION")
    print(f"{'='*70}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # nnU-Net v2 stores weights under 'network_weights'
    if 'network_weights' in ckpt:
        state_dict = ckpt['network_weights']
        print(f"  Checkpoint keys: network_weights found")
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"  Checkpoint keys: state_dict found")
    else:
        state_dict = ckpt
        print(f"  Checkpoint keys: raw state dict")

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if not missing and not unexpected:
        print(f"  ✅ Perfect match: {len(model_keys)} parameters")
        model.load_state_dict(state_dict)
        print(f"  ✅ Weights loaded successfully")
    else:
        if missing:
            print(f"  ⚠️ Missing from checkpoint ({len(missing)}):")
            for k in sorted(missing)[:10]:
                print(f"    - {k}")
        if unexpected:
            print(f"  ⚠️ Unexpected in checkpoint ({len(unexpected)}):")
            for k in sorted(unexpected)[:10]:
                print(f"    - {k}")

    # Print first/last 5 keys for structure understanding
    print(f"\n  State dict key structure (first 10):")
    for k in sorted(state_dict.keys())[:10]:
        print(f"    {k}: {list(state_dict[k].shape)}")

    return state_dict


def count_parameters(model):
    """Count trainable vs total parameters."""
    print(f"\n{'='*70}")
    print(f"PARAMETER COUNT")
    print(f"{'='*70}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")

    # Per-component breakdown
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"  Encoder parameters:   {enc_params:>12,}")
    print(f"  Decoder parameters:   {dec_params:>12,}")


def verify_separability(model, patch_size, num_input_channels, device="cpu"):
    """Verify that encode() and decode() can run independently."""
    print(f"\n{'='*70}")
    print(f"ENCODE/DECODE SEPARABILITY TEST")
    print(f"{'='*70}")

    model.eval()
    model.to(device)
    x = torch.randn(1, num_input_channels, *patch_size, device=device)

    with torch.no_grad():
        # Full forward
        full_out = model(x)
        if isinstance(full_out, (list, tuple)):
            full_out = full_out[0]

        # Separate encode
        skips = []
        current = x
        for stage in model.encoder.stages:
            current = stage(current)
            skips.append(current)
        bottleneck = current

        # Separate decode (feed bottleneck + skips)
        current = bottleneck
        for i, (transpconv, stage) in enumerate(zip(model.decoder.transpconvs, model.decoder.stages)):
            current = transpconv(current)
            skip_idx = len(skips) - 2 - i
            current = torch.cat([current, skips[skip_idx]], dim=1)
            current = stage(current)

        # Final segmentation layer
        seg_out = model.decoder.seg_layers[-1](current)

    diff = torch.abs(full_out - seg_out).max().item()
    if diff < 1e-5:
        print(f"  ✅ PASS — encode/decode separation verified (max diff: {diff:.2e})")
    else:
        print(f"  ⚠️ MISMATCH — max diff: {diff:.2e}")
        print(f"     Full output shape: {list(full_out.shape)}")
        print(f"     Separated output:  {list(seg_out.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Module 0: Inspect nnU-Net backbone")
    parser.add_argument("--plans", type=str, required=True, help="Path to nnUNetPlans.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional)")
    parser.add_argument("--dataset_json", type=str, default=None, help="Path to dataset.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    plans = load_plans(args.plans)

    # Determine channels/classes from plans or dataset.json
    num_input_channels = 3  # T2W, ADC, HBV
    num_classes = 3  # background, PZ, TZ
    if args.dataset_json and Path(args.dataset_json).exists():
        with open(args.dataset_json) as f:
            ds = json.load(f)
        num_input_channels = len(ds.get("channel_names", {0: "T2", 1: "ADC", 2: "HBV"}))
        num_classes = len(ds.get("labels", {0: "bg", 1: "PZ", 2: "TZ"}))
        print(f"Dataset: {num_input_channels} input channels, {num_classes} classes")

    # Architecture config
    config = plans["configurations"]["3d_fullres"]
    patch_size = config["patch_size"]
    print(f"Patch size: {patch_size}")

    # Build
    model = build_network(plans, num_input_channels, num_classes)

    # Trace
    bottleneck = trace_encoder_decoder(model, patch_size, num_input_channels, args.device)

    # Parameters
    count_parameters(model)

    # Separability
    verify_separability(model, patch_size, num_input_channels, args.device)

    # Checkpoint
    if args.checkpoint and Path(args.checkpoint).exists():
        verify_checkpoint(model, args.checkpoint, args.device)

    # Summary for manifest
    print(f"\n{'='*70}")
    print(f"BASELINE MANIFEST VALUES (copy to baseline_manifest.yaml)")
    print(f"{'='*70}")
    arch = config["architecture"]["arch_kwargs"]
    print(f"  n_stages: {arch['n_stages']}")
    print(f"  features_per_stage: {arch['features_per_stage']}")
    print(f"  strides: {arch['strides']}")
    print(f"  bottleneck_channels: {arch['features_per_stage'][-1]}")
    print(f"  bottleneck_spatial: {list(bottleneck.shape[2:])}")
    print(f"  bottleneck_shape: [{bottleneck.shape[1]}, {', '.join(str(s) for s in bottleneck.shape[2:])}]")
    print(f"  patch_size: {patch_size}")
    print(f"  spacing: {config['spacing']}")
    print(f"  num_input_channels: {num_input_channels}")
    print(f"  num_classes: {num_classes}")


if __name__ == "__main__":
    main()

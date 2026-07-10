# Module 0 — Baseline Analysis

## 1. Architecture Summary

Our trained model is a **7-stage PlainConvUNet** from nnU-Net v2.

```
Input [B, 3, 16, 320, 320]   ← 3-channel bpMRI (T2W + ADC + HBV)
  ↓
Encoder Stage 0: [B,  32, 16, 320, 320]   stride [1,1,1]
Encoder Stage 1: [B,  64, 16, 160, 160]   stride [1,2,2]
Encoder Stage 2: [B, 128, 16,  80,  80]   stride [1,2,2]
Encoder Stage 3: [B, 256,  8,  40,  40]   stride [2,2,2]
Encoder Stage 4: [B, 320,  4,  20,  20]   stride [2,2,2]
Encoder Stage 5: [B, 320,  4,  10,  10]   stride [1,2,2]
Encoder Stage 6: [B, 320,  4,   5,   5]   stride [1,2,2]  ★ BOTTLENECK
  ↓
Decoder Stage 0: [B, 320,  4,  10,  10]   ← bottleneck + skip 5
Decoder Stage 1: [B, 320,  4,  20,  20]   ← + skip 4
Decoder Stage 2: [B, 256,  8,  40,  40]   ← + skip 3
Decoder Stage 3: [B, 128, 16,  80,  80]   ← + skip 2
Decoder Stage 4: [B,  64, 16, 160, 160]   ← + skip 1
Decoder Stage 5: [B,  32, 16, 320, 320]   ← + skip 0
  ↓
Segmentation:  [B, 3, 16, 320, 320]   ← 3 classes (bg, PZ, TZ)
```

## 2. Bottleneck Specification

| Property | Value |
|----------|-------|
| **Shape** | `[B, 320, 4, 5, 5]` |
| **Channels** | 320 |
| **Spatial** | 4 × 5 × 5 = 100 voxels |
| **Elements per sample** | 320 × 100 = 32,000 |

> **Key finding**: Our bottleneck shape `[320, 4, 5, 5]` matches the bpMRI-TTA document's assumed shape exactly. This means all adapter architectures from the spec (reduction ratio, conv sizes) can be used without modification.

## 3. Skip Connections

The decoder uses 6 skip connections from encoder stages 0–5:

| Skip | Shape | Channels |
|------|-------|----------|
| 0 | `[B, 32, 16, 320, 320]` | 32 |
| 1 | `[B, 64, 16, 160, 160]` | 64 |
| 2 | `[B, 128, 16, 80, 80]` | 128 |
| 3 | `[B, 256, 8, 40, 40]` | 256 |
| 4 | `[B, 320, 4, 20, 20]` | 320 |
| 5 | `[B, 320, 4, 10, 10]` | 320 |

Skip connections are **not modified** by our adaptation framework. They pass through unchanged.

## 4. Encode/Decode Separability

The PlainConvUNet cleanly separates into:
- **`model.encoder.stages`** — list of 7 sequential conv blocks
- **`model.decoder.transpconvs`** — transposed convolutions for upsampling
- **`model.decoder.stages`** — conv blocks after skip concatenation
- **`model.decoder.seg_layers`** — 1×1×1 conv producing class logits

This means we can:
1. Run encoder stages 0-6 → get bottleneck + skips ✅
2. Insert adapters at the bottleneck ✅
3. Feed modified bottleneck + original skips into decoder ✅
4. Get segmentation output ✅

## 5. Checkpoint Structure

nnU-Net v2 checkpoints store weights under the `network_weights` key:
```python
ckpt = torch.load("checkpoint_final.pth")
state_dict = ckpt['network_weights']
# Keys look like: encoder.stages.0.0.conv.weight, decoder.transpconvs.0.weight, etc.
```

## 6. Adaptation Point

Our adapters will be inserted **between encoder stage 6 and decoder stage 0**:

```
Encoder Stage 6 output: [B, 320, 4, 5, 5]
        ↓
  ┌─────┼─────┐
  ↓     ↓     ↓
ARA_T2 ARA_ADC ARA_HBV    ← 3 Adaptive Reconstruction Adapters
  ↓     ↓     ↓
  └─────┼─────┘
        ↓
Channel-Gated Fusion: [B, 320, 4, 5, 5]
        ↓
Decoder Stage 0 (transpconv + skip 5 + conv)
```

## 7. Modality Handling Decision

Our encoder processes all 3 MRI modalities as a **single 3-channel input** `[B, 3, D, H, W]`. There is no per-modality encoder path.

**Decision**: The 3 adapters operate on the **same shared bottleneck**. Each adapter learns to apply a different residual correction to the shared latent representation. This is justified because:
- The bottleneck already encodes joint multi-modal features
- Separate adapters allow different modality-specific drift corrections
- At initialization (zero-init residual), all 3 adapters produce identity, so the model behaves identically to baseline

## 8. Verification Script

Run on Colab:
```bash
python /content/ZonalSeg/scripts/inspect_backbone.py \
    --plans /content/nnUNet_preprocessed/Dataset501_ZonalSeg/nnUNetPlans.json \
    --checkpoint /content/drive/MyDrive/ZonalSeg_Results/nnUNet_results/Dataset501_ZonalSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth \
    --dataset_json /content/nnUNet_preprocessed/Dataset501_ZonalSeg/dataset.json \
    --device cuda:0
```

This will verify all shapes match what's documented here.

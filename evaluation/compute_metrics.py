"""
===============================================================================
Evaluation Metrics for Prostate Zonal Segmentation
===============================================================================
Computes per-zone metrics: Dice, HD95, ASD, Precision, Recall
Supports both internal validation (5-fold CV) and external validation (PCNN).

Usage:
  # Internal validation (fold-by-fold):
  python evaluation/compute_metrics.py \
    --predictions /path/to/nnUNet_results/fold_0/validation_raw \
    --ground_truth /path/to/nnUNet_raw/Dataset501_ZonalSeg/labelsTr \
    --output_dir /path/to/results \
    --mode internal

  # External validation (PCNN):
  python evaluation/compute_metrics.py \
    --predictions /path/to/predictions \
    --ground_truth /path/to/pcnn_labels \
    --output_dir /path/to/results \
    --mode external
===============================================================================
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed. Run: pip install nibabel")
    sys.exit(1)

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    print("scipy not installed. Run: pip install scipy")
    sys.exit(1)


def compute_dice(pred, gt, label):
    """Compute Dice coefficient for a specific label."""
    pred_mask = (pred == label).astype(np.float32)
    gt_mask = (gt == label).astype(np.float32)
    
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    if union == 0:
        return 1.0 if np.sum(gt_mask) == 0 else 0.0
    
    return (2.0 * intersection) / union


def compute_surface_distances(pred, gt, spacing):
    """Compute surface distances between prediction and ground truth."""
    pred_border = pred ^ np.pad(pred, 1, mode='edge')[:-2, :-2, :-2] | \
                  pred ^ np.pad(pred, 1, mode='edge')[2:, 2:, 2:]
    gt_border = gt ^ np.pad(gt, 1, mode='edge')[:-2, :-2, :-2] | \
                gt ^ np.pad(gt, 1, mode='edge')[2:, 2:, 2:]
    
    # Use distance transform for efficiency
    if not np.any(pred):
        return np.array([]), np.array([])
    if not np.any(gt):
        return np.array([]), np.array([])
    
    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)
    
    pred_surface = pred_border & pred
    gt_surface = gt_border & gt
    
    if not np.any(pred_surface) or not np.any(gt_surface):
        return np.array([]), np.array([])
    
    dist_pred_to_gt = dt_gt[pred_surface]
    dist_gt_to_pred = dt_pred[gt_surface]
    
    return dist_pred_to_gt, dist_gt_to_pred


def compute_hd95(pred, gt, label, spacing=(1, 1, 1)):
    """Compute 95th percentile Hausdorff Distance for a specific label."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    if not np.any(pred_mask) and not np.any(gt_mask):
        return 0.0
    if not np.any(pred_mask) or not np.any(gt_mask):
        return np.inf
    
    d1, d2 = compute_surface_distances(pred_mask, gt_mask, spacing)
    
    if len(d1) == 0 or len(d2) == 0:
        return np.inf
    
    all_distances = np.concatenate([d1, d2])
    return np.percentile(all_distances, 95)


def compute_asd(pred, gt, label, spacing=(1, 1, 1)):
    """Compute Average Surface Distance for a specific label."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    if not np.any(pred_mask) and not np.any(gt_mask):
        return 0.0
    if not np.any(pred_mask) or not np.any(gt_mask):
        return np.inf
    
    d1, d2 = compute_surface_distances(pred_mask, gt_mask, spacing)
    
    if len(d1) == 0 or len(d2) == 0:
        return np.inf
    
    return (np.mean(d1) + np.mean(d2)) / 2.0


def compute_precision_recall(pred, gt, label):
    """Compute Precision and Recall for a specific label."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall


def evaluate_case(pred_path, gt_path, labels):
    """Evaluate a single case across all labels."""
    pred_img = nib.load(str(pred_path))
    gt_img = nib.load(str(gt_path))
    
    pred = pred_img.get_fdata().astype(np.uint8)
    gt = gt_img.get_fdata().astype(np.uint8)
    
    spacing = gt_img.header.get_zooms()[:3]
    
    results = {}
    for label_name, label_val in labels.items():
        if label_name == "background":
            continue  # Skip background metrics
        
        dice = compute_dice(pred, gt, label_val)
        hd95 = compute_hd95(pred, gt, label_val, spacing)
        asd = compute_asd(pred, gt, label_val, spacing)
        precision, recall = compute_precision_recall(pred, gt, label_val)
        
        results[label_name] = {
            "Dice": dice,
            "HD95": hd95,
            "ASD": asd,
            "Precision": precision,
            "Recall": recall
        }
    
    return results


def compute_metrics(predictions_dir: str, ground_truth_dir: str, output_dir: str,
                    mode: str = "internal"):
    """Compute metrics across all cases."""
    predictions_dir = Path(predictions_dir)
    ground_truth_dir = Path(ground_truth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default label mapping (will be adjusted based on dataset)
    labels = {"background": 0, "PZ": 1, "TZ": 2}
    
    # Find prediction files
    pred_files = sorted(predictions_dir.glob("*.nii.gz"))
    
    if not pred_files:
        print(f"No prediction files found in {predictions_dir}")
        return
    
    print(f"Evaluating {len(pred_files)} cases ({mode} validation)")
    
    all_results = []
    
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        case_id = pred_file.stem.replace(".nii", "")
        gt_file = ground_truth_dir / f"{case_id}.nii.gz"
        
        if not gt_file.exists():
            print(f"  ⚠️ Ground truth not found for {case_id}, skipping")
            continue
        
        try:
            case_results = evaluate_case(pred_file, gt_file, labels)
            case_results["case_id"] = case_id
            all_results.append(case_results)
        except Exception as e:
            print(f"  ⚠️ Error evaluating {case_id}: {e}")
    
    # Build results DataFrame
    rows = []
    for result in all_results:
        case_id = result["case_id"]
        for zone_name in ["PZ", "TZ"]:
            if zone_name in result:
                row = {"case_id": case_id, "zone": zone_name}
                row.update(result[zone_name])
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Results Summary ({mode} validation)")
    print(f"{'='*60}")
    
    for zone in ["PZ", "TZ"]:
        zone_df = df[df["zone"] == zone]
        print(f"\n{zone}:")
        for metric in ["Dice", "HD95", "ASD", "Precision", "Recall"]:
            values = zone_df[metric].replace([np.inf], np.nan).dropna()
            mean = values.mean()
            std = values.std()
            print(f"  {metric:10s}: {mean:.4f} ± {std:.4f}")
    
    # Overall (average across zones)
    print(f"\nOverall (mean across PZ + TZ):")
    for metric in ["Dice", "HD95", "ASD", "Precision", "Recall"]:
        values = df[metric].replace([np.inf], np.nan).dropna()
        mean = values.mean()
        std = values.std()
        print(f"  {metric:10s}: {mean:.4f} ± {std:.4f}")
    
    # Save detailed results
    csv_path = output_dir / f"metrics_{mode}.csv"
    df.to_csv(csv_path, index=False)
    
    # Save summary
    summary = {}
    for zone in ["PZ", "TZ", "Overall"]:
        if zone == "Overall":
            zone_df = df
        else:
            zone_df = df[df["zone"] == zone]
        
        summary[zone] = {}
        for metric in ["Dice", "HD95", "ASD", "Precision", "Recall"]:
            values = zone_df[metric].replace([np.inf], np.nan).dropna()
            summary[zone][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max())
            }
    
    summary_path = output_dir / f"summary_{mode}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Detailed results: {csv_path}")
    print(f"📋 Summary: {summary_path}")
    
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute zonal segmentation metrics")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="internal",
                        choices=["internal", "external"])
    
    args = parser.parse_args()
    compute_metrics(args.predictions, args.ground_truth, args.output_dir, args.mode)

"""
===============================================================================
Evaluation Metrics for Prostate Zonal Segmentation
===============================================================================
Computes per-zone metrics: Dice, HD95, ASD, Precision, Recall
Uses Multi-threading to process the distance transforms extremely quickly.

Usage:
  python evaluation/compute_metrics.py \
    --predictions /path/to/predictions \
    --ground_truth /path/to/labels \
    --output_dir /path/to/results \
    --mode external
===============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
import concurrent.futures

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed. Run: pip install nibabel")
    sys.exit(1)

try:
    from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
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
    """Compute exact surface distances using morphological erosion."""
    # Use 3D 6-connectivity for robust border extraction
    struct = generate_binary_structure(3, 1)
    pred_border = pred ^ binary_erosion(pred, structure=struct)
    gt_border = gt ^ binary_erosion(gt, structure=struct)
    
    if not np.any(pred):
        return np.array([]), np.array([])
    if not np.any(gt):
        return np.array([]), np.array([])
    
    # Distance transforms automatically release the GIL for multi-threading!
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
    """Compute true mathematical 95th percentile Hausdorff Distance."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    if not np.any(pred_mask) and not np.any(gt_mask):
        return 0.0
    if not np.any(pred_mask) or not np.any(gt_mask):
        return np.inf
    
    d1, d2 = compute_surface_distances(pred_mask, gt_mask, spacing)
    
    if len(d1) == 0 or len(d2) == 0:
        return np.inf
    
    # HD95 is defined as the max of the two directed 95th percentiles
    return max(np.percentile(d1, 95), np.percentile(d2, 95))


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


def evaluate_single_case_task(pred_file: Path, ground_truth_dir: Path, labels: dict):
    """Thread-worker function to evaluate a single NIfTI file."""
    case_id = pred_file.stem.replace(".nii", "")
    gt_file = ground_truth_dir / f"{case_id}.nii.gz"
    
    if not gt_file.exists():
        return {"case_id": case_id, "error": "Ground truth not found"}
    
    try:
        pred_img = nib.load(str(pred_file))
        gt_img = nib.load(str(gt_file))
        
        pred = pred_img.get_fdata().astype(np.uint8)
        gt = gt_img.get_fdata().astype(np.uint8)
        
        spacing = gt_img.header.get_zooms()[:3]
        
        results = {"case_id": case_id}
        for label_name, label_val in labels.items():
            if label_name == "background":
                continue
            
            results[label_name] = {
                "Dice": compute_dice(pred, gt, label_val),
                "HD95": compute_hd95(pred, gt, label_val, spacing),
                "ASD": compute_asd(pred, gt, label_val, spacing),
                "Precision": compute_precision_recall(pred, gt, label_val)[0],
                "Recall": compute_precision_recall(pred, gt, label_val)[1]
            }
        return results
    except Exception as e:
        return {"case_id": case_id, "error": str(e)}


def compute_metrics(predictions_dir: str, ground_truth_dir: str, output_dir: str, mode: str = "internal"):
    """Compute metrics across all cases in parallel."""
    predictions_dir = Path(predictions_dir)
    ground_truth_dir = Path(ground_truth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels = {"background": 0, "PZ": 1, "TZ": 2}
    pred_files = sorted(predictions_dir.glob("*.nii.gz"))
    
    if not pred_files:
        print(f"No prediction files found in {predictions_dir}")
        return
    
    print(f"Evaluating {len(pred_files)} cases with 16-Core PARALLEL PROCESSING ({mode} validation)...")
    
    all_results = []
    
    # Run heavy distance transforms in parallel across 16 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(evaluate_single_case_task, pf, ground_truth_dir, labels): pf 
            for pf in pred_files
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating"):
            res = future.result()
            if "error" in res:
                print(f"  ⚠️ Error evaluating {res['case_id']}: {res['error']}")
            else:
                all_results.append(res)
    
    if not all_results:
        print("No cases successfully evaluated.")
        return
        
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
        zone_df = df if zone == "Overall" else df[df["zone"] == zone]
        summary[zone] = {}
        for metric in ["Dice", "HD95", "ASD", "Precision", "Recall"]:
            values = zone_df[metric].replace([np.inf], np.nan).dropna()
            if not values.empty:
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
    parser = argparse.ArgumentParser(description="Compute zonal segmentation metrics (Multi-Threaded)")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="internal", choices=["internal", "external"])
    
    args = parser.parse_args()
    compute_metrics(args.predictions, args.ground_truth, args.output_dir, args.mode)

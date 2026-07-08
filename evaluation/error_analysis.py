"""
===============================================================================
Error Analysis for Prostate Zonal Segmentation
===============================================================================
Analyzes worst and best cases, generates overlay visualizations.

Usage:
  python evaluation/error_analysis.py \
    --metrics_csv /path/to/metrics_internal.csv \
    --predictions /path/to/predictions \
    --ground_truth /path/to/ground_truth \
    --images /path/to/images \
    --output_dir /path/to/error_analysis
===============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed. Run: pip install nibabel")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


# Custom colormap for zonal segmentation
ZONE_COLORS = ListedColormap([
    [0, 0, 0, 0],        # Background (transparent)
    [0.2, 0.6, 1.0, 0.5],  # PZ (blue, semi-transparent)
    [1.0, 0.3, 0.3, 0.5],  # TZ (red, semi-transparent)
])


def create_overlay_figure(t2_slice, gt_slice, pred_slice, case_id, slice_idx,
                          dice_pz, dice_tz):
    """Create a 3-panel overlay figure: T2W | Ground Truth | Prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # T2W image
    axes[0].imshow(t2_slice.T, cmap='gray', origin='lower')
    axes[0].set_title('T2W Image', fontsize=12)
    axes[0].axis('off')
    
    # Ground Truth overlay
    axes[1].imshow(t2_slice.T, cmap='gray', origin='lower')
    axes[1].imshow(gt_slice.T, cmap=ZONE_COLORS, vmin=0, vmax=2, 
                   origin='lower', alpha=0.5)
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].axis('off')
    
    # Prediction overlay
    axes[2].imshow(t2_slice.T, cmap='gray', origin='lower')
    axes[2].imshow(pred_slice.T, cmap=ZONE_COLORS, vmin=0, vmax=2,
                   origin='lower', alpha=0.5)
    axes[2].set_title('Prediction', fontsize=12)
    axes[2].axis('off')
    
    fig.suptitle(f'{case_id} (Slice {slice_idx}) | PZ Dice: {dice_pz:.3f} | TZ Dice: {dice_tz:.3f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def analyze_errors(metrics_csv: str, predictions_dir: str, ground_truth_dir: str,
                   images_dir: str, output_dir: str, n_cases: int = 10):
    """Perform error analysis on worst and best cases."""
    metrics_csv = Path(metrics_csv)
    predictions_dir = Path(predictions_dir)
    ground_truth_dir = Path(ground_truth_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    df = pd.read_csv(metrics_csv)
    
    # Compute per-case average Dice across zones
    case_dice = df.groupby("case_id")["Dice"].mean().reset_index()
    case_dice.columns = ["case_id", "mean_dice"]
    case_dice = case_dice.sort_values("mean_dice")
    
    # Worst N cases
    worst_cases = case_dice.head(n_cases)
    best_cases = case_dice.tail(n_cases)
    
    print(f"{'='*60}")
    print(f"Error Analysis")
    print(f"{'='*60}")
    
    print(f"\n🔴 Worst {n_cases} Cases (lowest mean Dice):")
    for _, row in worst_cases.iterrows():
        print(f"  {row.case_id}: Dice = {row.mean_dice:.4f}")
    
    print(f"\n🟢 Best {n_cases} Cases (highest mean Dice):")
    for _, row in best_cases.iterrows():
        print(f"  {row.case_id}: Dice = {row.mean_dice:.4f}")
    
    # Generate overlay visualizations for worst cases
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    for category, cases_df in [("worst", worst_cases), ("best", best_cases)]:
        cat_dir = vis_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        for _, row in cases_df.iterrows():
            case_id = row.case_id
            
            # Load images
            t2_path = images_dir / f"{case_id}_0000.nii.gz"
            pred_path = predictions_dir / f"{case_id}.nii.gz"
            gt_path = ground_truth_dir / f"{case_id}.nii.gz"
            
            if not all(p.exists() for p in [t2_path, pred_path, gt_path]):
                print(f"  ⚠️ Missing files for {case_id}, skipping visualization")
                continue
            
            try:
                t2_data = nib.load(str(t2_path)).get_fdata()
                pred_data = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
                gt_data = nib.load(str(gt_path)).get_fdata().astype(np.uint8)
                
                # Find the slice with the most zonal anatomy
                gt_counts = np.sum(gt_data > 0, axis=(0, 1))
                best_slice = np.argmax(gt_counts)
                
                # Get per-zone Dice
                case_metrics = df[df["case_id"] == case_id]
                dice_pz = case_metrics[case_metrics["zone"] == "PZ"]["Dice"].values
                dice_tz = case_metrics[case_metrics["zone"] == "TZ"]["Dice"].values
                dice_pz = dice_pz[0] if len(dice_pz) > 0 else 0
                dice_tz = dice_tz[0] if len(dice_tz) > 0 else 0
                
                fig = create_overlay_figure(
                    t2_data[:, :, best_slice],
                    gt_data[:, :, best_slice],
                    pred_data[:, :, best_slice],
                    case_id, best_slice, dice_pz, dice_tz
                )
                
                fig.savefig(cat_dir / f"{case_id}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"  ⚠️ Error visualizing {case_id}: {e}")
    
    # Save summary report
    report = {
        "worst_cases": worst_cases.to_dict(orient="records"),
        "best_cases": best_cases.to_dict(orient="records"),
        "overall_stats": {
            "mean_dice": float(case_dice["mean_dice"].mean()),
            "std_dice": float(case_dice["mean_dice"].std()),
            "median_dice": float(case_dice["mean_dice"].median()),
        }
    }
    
    import json
    with open(output_dir / "error_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Visualizations saved to: {vis_dir}")
    print(f"📋 Report saved to: {output_dir / 'error_analysis.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error analysis for zonal segmentation")
    parser.add_argument("--metrics_csv", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_cases", type=int, default=10,
                        help="Number of worst/best cases to analyze")
    
    args = parser.parse_args()
    analyze_errors(args.metrics_csv, args.predictions, args.ground_truth,
                   args.images, args.output_dir, args.n_cases)

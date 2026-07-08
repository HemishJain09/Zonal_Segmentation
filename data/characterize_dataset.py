"""
===============================================================================
PI-CAI Dataset Characterization Script — Phase 1
===============================================================================
Produces a comprehensive dataset report verifying all data before training.

Usage (in Colab):
  python data/characterize_dataset.py \
    --picai_dir /content/drive/MyDrive/PI-CAI_pre-processed \
    --marksheet /content/drive/MyDrive/marksheet.csv \
    --output_dir /content/drive/MyDrive/ZonalSeg_Results

Checks:
  1. Patient/scan/follow-up counts per centre
  2. Modality completeness (T2, ADC, HBV, zonal mask)
  3. Label encoding (np.unique on zonal masks)
  4. Voxel spacing distributions
  5. Image dimensions
  6. Class imbalance (BG vs PZ vs TZ voxel ratios)
  7. Orientation consistency
===============================================================================
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed. Run: pip install nibabel")
    sys.exit(1)


def find_modality_files(picai_dir: Path):
    """
    Discover the folder structure of the PI-CAI pre-processed dataset.
    Expected structure:
      picai_dir/
        t2w/ or t2/          → T2-weighted images
        adc/                  → ADC maps
        hbv/                  → High b-value images
        zonal_masks/          → Zonal segmentation masks
        whole_gland_masks/    → Whole gland masks (optional)
    """
    folder_map = {}
    
    # Try common folder names
    t2_candidates = ["t2w", "t2", "T2W", "T2"]
    adc_candidates = ["adc", "ADC"]
    hbv_candidates = ["hbv", "HBV", "highbvalue", "high_b_value"]
    zonal_candidates = ["zonal_masks", "zonal", "zonal_pz_tz"]
    
    for name in t2_candidates:
        p = picai_dir / name
        if p.exists():
            folder_map["t2w"] = p
            break
    
    for name in adc_candidates:
        p = picai_dir / name
        if p.exists():
            folder_map["adc"] = p
            break
    
    for name in hbv_candidates:
        p = picai_dir / name
        if p.exists():
            folder_map["hbv"] = p
            break
    
    for name in zonal_candidates:
        p = picai_dir / name
        if p.exists():
            folder_map["zonal"] = p
            break
    
    return folder_map


def characterize_dataset(picai_dir: str, marksheet_path: str, output_dir: str, 
                         max_cases: int = None):
    """Main characterization function."""
    picai_dir = Path(picai_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    
    def log(msg):
        print(msg)
        report_lines.append(msg)
    
    log("=" * 70)
    log("PI-CAI Dataset Characterization Report")
    log("=" * 70)
    log("")
    
    # =========================================================================
    # 1. Discover folder structure
    # =========================================================================
    log("## 1. Folder Structure Discovery")
    log("")
    
    folder_map = find_modality_files(picai_dir)
    
    if not folder_map:
        # Try listing all subdirectories
        log(f"Could not auto-detect folders in: {picai_dir}")
        log("Contents:")
        for item in sorted(picai_dir.iterdir()):
            log(f"  {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
        log("")
        log("Please verify the folder structure and adjust folder names.")
        # Save partial report
        report_path = output_dir / "dataset_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        return
    
    for mod, path in folder_map.items():
        n_files = len(list(path.glob("*.nii.gz")))
        log(f"  {mod:15s} → {path.name}/ ({n_files} files)")
    log("")
    
    # =========================================================================
    # 2. Load marksheet for centre information
    # =========================================================================
    log("## 2. Centre Distribution")
    log("")
    
    marksheet_path = Path(marksheet_path)
    if not marksheet_path.exists():
        log(f"❌ ERROR: Marksheet not found at {marksheet_path}")
        log("Please update the MARKSHEET path in the Colab notebook.")
        
        report_path = output_dir / "dataset_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        sys.exit(1)
        
    marksheet = pd.read_csv(marksheet_path)    log(f"Total rows in marksheet: {len(marksheet)}")
    log(f"Columns: {list(marksheet.columns)}")
    log("")
    
    # Centre distribution
    centre_counts = marksheet["center"].value_counts()
    for centre, count in centre_counts.items():
        n_patients = marksheet[marksheet["center"] == centre]["patient_id"].nunique()
        log(f"  {centre}: {count} scans, {n_patients} unique patients")
    
    # Follow-up detection
    patient_scan_counts = marksheet.groupby("patient_id")["study_id"].count()
    follow_up_patients = patient_scan_counts[patient_scan_counts > 1]
    log(f"\nPatients with follow-up scans: {len(follow_up_patients)}")
    log("")
    
    # =========================================================================
    # 3. Verify modality completeness
    # =========================================================================
    log("## 3. Modality Completeness Check")
    log("")
    
    # Build case IDs from marksheet
    case_ids = [f"{row.patient_id}_{row.study_id}" for _, row in marksheet.iterrows()]
    
    if max_cases:
        case_ids = case_ids[:max_cases]
        log(f"(Limited to {max_cases} cases for quick check)")
    
    missing_modalities = defaultdict(list)
    complete_cases = []
    
    for case_id in tqdm(case_ids, desc="Checking completeness"):
        has_all = True
        for mod, folder in folder_map.items():
            fpath = folder / f"{case_id}.nii.gz"
            if not fpath.exists():
                missing_modalities[mod].append(case_id)
                has_all = False
        if has_all:
            complete_cases.append(case_id)
    
    log(f"Complete cases (all modalities): {len(complete_cases)} / {len(case_ids)}")
    for mod, missing in missing_modalities.items():
        log(f"  Missing {mod}: {len(missing)} cases")
        if len(missing) <= 5:
            for m in missing:
                log(f"    - {m}")
    log("")
    
    # =========================================================================
    # 4. Label Encoding Analysis (CRITICAL)
    # =========================================================================
    log("## 4. Zonal Mask Label Encoding (CRITICAL)")
    log("")
    
    if "zonal" not in folder_map:
        log("⚠️  No zonal mask folder found! Cannot proceed with label analysis.")
        log("    Check if zonal masks exist in your dataset.")
        log("")
    else:
        zonal_dir = folder_map["zonal"]
        all_unique_values = set()
        label_counts = Counter()
        sample_count = 0
        
        zonal_files = sorted(zonal_dir.glob("*.nii.gz"))
        if max_cases:
            zonal_files = zonal_files[:max_cases]
        
        for zf in tqdm(zonal_files, desc="Analyzing zonal labels"):
            try:
                mask = nib.load(str(zf)).get_fdata()
                unique_vals = np.unique(mask)
                all_unique_values.update(unique_vals.tolist())
                
                # Count voxels per label
                for val in unique_vals:
                    label_counts[int(val)] += np.sum(mask == val)
                
                sample_count += 1
            except Exception as e:
                log(f"  ⚠️ Error loading {zf.name}: {e}")
        
        log(f"Analyzed {sample_count} zonal masks")
        log(f"Unique label values found: {sorted(all_unique_values)}")
        log("")
        log("Label → Voxel Count:")
        total_voxels = sum(label_counts.values())
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = (count / total_voxels) * 100
            log(f"  Label {label}: {count:>15,} voxels ({pct:.2f}%)")
        log("")
        
        # Guess label mapping
        sorted_labels = sorted(all_unique_values)
        if sorted_labels == [0, 1, 2]:
            log("✅ Standard 3-class encoding detected: {0: BG, 1: PZ(?), 2: TZ(?)}")
            log("   (Verify by visual inspection which label is PZ vs TZ)")
        elif sorted_labels == [0, 1, 2, 3]:
            log("⚠️  4-class encoding detected — may include AFMS or urethra")
        else:
            log(f"⚠️  Non-standard encoding: {sorted_labels}")
            log("   Manual verification required!")
        log("")
    
    # =========================================================================
    # 5. Voxel Spacing & Dimensions
    # =========================================================================
    log("## 5. Voxel Spacing & Dimensions")
    log("")
    
    spacings = []
    shapes = []
    orientations = []
    
    # Sample from T2W (reference modality)
    if "t2w" in folder_map:
        t2_dir = folder_map["t2w"]
        t2_files = sorted(t2_dir.glob("*.nii.gz"))
        if max_cases:
            t2_files = t2_files[:max_cases]
        
        for tf in tqdm(t2_files, desc="Analyzing geometry"):
            try:
                img = nib.load(str(tf))
                spacings.append(img.header.get_zooms()[:3])
                shapes.append(img.shape[:3])
                orientations.append(nib.aff2axcodes(img.affine))
            except Exception as e:
                log(f"  ⚠️ Error loading {tf.name}: {e}")
        
        if spacings:
            spacings_arr = np.array(spacings)
            log(f"Spacing (mm) — min:  [{spacings_arr[:, 0].min():.3f}, {spacings_arr[:, 1].min():.3f}, {spacings_arr[:, 2].min():.3f}]")
            log(f"Spacing (mm) — max:  [{spacings_arr[:, 0].max():.3f}, {spacings_arr[:, 1].max():.3f}, {spacings_arr[:, 2].max():.3f}]")
            log(f"Spacing (mm) — mean: [{spacings_arr[:, 0].mean():.3f}, {spacings_arr[:, 1].mean():.3f}, {spacings_arr[:, 2].mean():.3f}]")
            log("")
            
            # Shape analysis
            shapes_arr = np.array(shapes)
            log(f"Shape — min:  {shapes_arr.min(axis=0)}")
            log(f"Shape — max:  {shapes_arr.max(axis=0)}")
            log("")
            
            # Orientation consistency
            orient_counter = Counter(orientations)
            log("Orientations:")
            for orient, count in orient_counter.most_common():
                log(f"  {orient}: {count} cases")
            log("")
    
    # =========================================================================
    # 6. Save Report
    # =========================================================================
    log("=" * 70)
    log("End of Report")
    log("=" * 70)
    
    report_path = output_dir / "dataset_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PI-CAI Dataset Characterization")
    parser.add_argument("--picai_dir", type=str, required=True,
                        help="Path to PI-CAI pre-processed dataset root")
    parser.add_argument("--marksheet", type=str, required=True,
                        help="Path to marksheet.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the report")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="Limit analysis to N cases (for quick check)")
    
    args = parser.parse_args()
    characterize_dataset(args.picai_dir, args.marksheet, args.output_dir, args.max_cases)

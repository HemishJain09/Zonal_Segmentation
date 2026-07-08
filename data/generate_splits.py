"""
===============================================================================
Patient-Level Splits Generator for nnU-Net v2 — Zonal Segmentation
===============================================================================
Generates a custom splits_final.json for nnU-Net v2 that:
  1. Reads marksheet.csv to identify patient centres
  2. Excludes the external test centre (PCNN)
  3. Performs patient-level GroupKFold to prevent follow-up leakage
  4. Saves splits_final.json to the nnU-Net preprocessed directory

Usage:
  python data/generate_splits.py \
    --nnunet_raw /path/to/nnUNet_raw \
    --nnunet_preprocessed /path/to/nnUNet_preprocessed \
    --marksheet /path/to/marksheet.csv \
    --train_centers RUMC ZGT \
    --n_folds 5
===============================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


DATASET_NAME = "Dataset501_ZonalSeg"


def generate_splits(nnunet_raw: str, nnunet_preprocessed: str, marksheet_path: str,
                    train_centers: list, n_folds: int = 5):
    """Generate patient-level 5-fold CV splits excluding external test centre."""
    
    nnunet_raw = Path(nnunet_raw)
    nnunet_preprocessed = Path(nnunet_preprocessed)
    
    # Load marksheet
    marksheet = pd.read_csv(marksheet_path)
    
    # Build patient-to-centre mapping
    patient_centre = {}
    for _, row in marksheet.iterrows():
        patient_centre[str(row.patient_id)] = row.center
    
    # Get all case IDs from the converted dataset
    labels_dir = nnunet_raw / DATASET_NAME / "labelsTr"
    all_cases = sorted([f.name.replace(".nii.gz", "") for f in labels_dir.glob("*.nii.gz")])
    
    print(f"Total cases in dataset: {len(all_cases)}")
    
    # Filter to training centres only
    train_cases = []
    excluded_cases = []
    
    for case_id in all_cases:
        patient_id = case_id.split("_")[0]
        centre = patient_centre.get(patient_id, "UNKNOWN")
        
        if centre in train_centers:
            train_cases.append(case_id)
        else:
            excluded_cases.append((case_id, centre))
    
    print(f"Training cases ({', '.join(train_centers)}): {len(train_cases)}")
    print(f"Excluded cases: {len(excluded_cases)}")
    
    if excluded_cases:
        excluded_centres = defaultdict(int)
        for _, centre in excluded_cases:
            excluded_centres[centre] += 1
        for centre, count in excluded_centres.items():
            print(f"  Excluded {centre}: {count} cases")
    
    # Extract patient IDs as groups (for GroupKFold)
    patient_ids = [case_id.split("_")[0] for case_id in train_cases]
    unique_patients = sorted(set(patient_ids))
    
    # Map patient IDs to integer groups
    patient_to_group = {pid: idx for idx, pid in enumerate(unique_patients)}
    groups = np.array([patient_to_group[pid] for pid in patient_ids])
    
    print(f"\nUnique patients: {len(unique_patients)}")
    
    # Detect longitudinal patients (multiple scans per patient)
    patient_scan_counts = defaultdict(list)
    for case_id in train_cases:
        pid = case_id.split("_")[0]
        patient_scan_counts[pid].append(case_id)
    
    longitudinal = {pid: cases for pid, cases in patient_scan_counts.items() if len(cases) > 1}
    if longitudinal:
        print(f"Longitudinal patients (multiple scans): {len(longitudinal)}")
        for pid, cases in list(longitudinal.items())[:5]:
            print(f"  Patient {pid}: {cases}")
    
    # Perform GroupKFold
    gkf = GroupKFold(n_splits=n_folds)
    X_dummy = np.zeros(len(train_cases))
    y_dummy = np.zeros(len(train_cases))
    
    splits = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, y_dummy, groups)):
        train_keys = [train_cases[i] for i in train_idx]
        val_keys = [train_cases[i] for i in val_idx]
        
        # Verify no patient overlap
        train_patients = set(k.split("_")[0] for k in train_keys)
        val_patients = set(k.split("_")[0] for k in val_keys)
        overlap = train_patients & val_patients
        
        assert len(overlap) == 0, f"Data leakage in fold {fold_idx}! Patients in both: {overlap}"
        
        splits.append({
            "train": sorted(train_keys),
            "val": sorted(val_keys)
        })
        
        print(f"Fold {fold_idx}: train={len(train_keys)} cases ({len(train_patients)} patients), "
              f"val={len(val_keys)} cases ({len(val_patients)} patients)")
    
    # Save splits
    output_dir = nnunet_preprocessed / DATASET_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_path = output_dir / "splits_final.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✅ Splits saved to: {splits_path}")
    print(f"   {n_folds} folds, patient-level split, zero data leakage verified")
    
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate patient-level CV splits for zonal segmentation"
    )
    parser.add_argument("--nnunet_raw", type=str, required=True)
    parser.add_argument("--nnunet_preprocessed", type=str, required=True)
    parser.add_argument("--marksheet", type=str, required=True)
    parser.add_argument("--train_centers", nargs="+", required=True,
                        help="Centers to include (e.g., RUMC ZGT)")
    parser.add_argument("--n_folds", type=int, default=5)
    
    args = parser.parse_args()
    generate_splits(
        args.nnunet_raw, args.nnunet_preprocessed, args.marksheet,
        args.train_centers, args.n_folds
    )

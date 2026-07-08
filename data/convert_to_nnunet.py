"""
===============================================================================
PI-CAI → nnU-Net v2 Data Conversion for Zonal Segmentation
===============================================================================
Converts the PI-CAI pre-processed dataset into nnU-Net v2 format for
multi-class prostate zonal segmentation.

Input:  3 channels (T2W=_0000, ADC=_0001, HBV=_0002)
Target: Zonal mask (multi-class: 0=BG, 1=PZ, 2=TZ — verified by Phase 1)

Usage:
  # Full dataset:
  python data/convert_to_nnunet.py \
    --picai_dir /path/to/PI-CAI_pre-processed \
    --nnunet_raw /path/to/nnUNet_raw \
    --marksheet /path/to/marksheet.csv

  # Sanity check (10 patients only):
  python data/convert_to_nnunet.py \
    --picai_dir /path/to/PI-CAI_pre-processed \
    --nnunet_raw /path/to/nnUNet_raw \
    --marksheet /path/to/marksheet.csv \
    --max_cases 10

  # Filter by centres:
  python data/convert_to_nnunet.py \
    --picai_dir /path/to/PI-CAI_pre-processed \
    --nnunet_raw /path/to/nnUNet_raw \
    --marksheet /path/to/marksheet.csv \
    --train_centers RUMC ZGT
===============================================================================
"""

import argparse
import json
import gc
import shutil
import sys
from pathlib import Path
from collections import Counter
import concurrent.futures

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed. Run: pip install nibabel")
    sys.exit(1)


DATASET_NAME = "Dataset501_ZonalSeg"


def verify_and_copy_mask(zonal_nifti_path: Path, output_path: Path):
    """
    Load a zonal segmentation mask, verify its contents, and save.
    
    Expected values: {0: Background, 1: PZ, 2: TZ} (or similar multi-class).
    Logs warnings for unexpected label values.
    """
    img = nib.load(str(zonal_nifti_path))
    data = img.get_fdata().astype(np.uint8)
    
    unique_vals = np.unique(data)
    
    # Save as uint8 NIfTI
    out_img = nib.Nifti1Image(data, img.affine, img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(output_path))
    
    return unique_vals


def convert_picai_to_nnunet(picai_dir: str, nnunet_raw: str, marksheet_path: str,
                             max_cases: int = None, train_centers: list = None):
    """Main conversion function."""
    picai_dir = Path(picai_dir)
    nnunet_raw = Path(nnunet_raw)
    
    # Discover folder structure
    t2_dir = None
    adc_dir = None
    hbv_dir = None
    zonal_dir = None
    
    for name in ["t2w", "t2", "T2W", "T2"]:
        if (picai_dir / name).exists():
            t2_dir = picai_dir / name
            break
    
    for name in ["adc", "ADC", "adc_reg"]:
        if (picai_dir / name).exists():
            adc_dir = picai_dir / name
            break
    
    for name in ["hbv", "HBV", "highbvalue", "hbv_reg"]:
        if (picai_dir / name).exists():
            hbv_dir = picai_dir / name
            break
    
    for name in ["zonal_masks", "zonal", "zonal_pz_tz"]:
        if (picai_dir / name).exists():
            zonal_dir = picai_dir / name
            break
    
    # Verify all folders found
    assert t2_dir is not None, f"T2W folder not found in {picai_dir}"
    assert adc_dir is not None, f"ADC folder not found in {picai_dir}"
    assert hbv_dir is not None, f"HBV folder not found in {picai_dir}"
    assert zonal_dir is not None, f"Zonal mask folder not found in {picai_dir}"
    
    print(f"T2W:   {t2_dir}")
    print(f"ADC:   {adc_dir}")
    print(f"HBV:   {hbv_dir}")
    print(f"Zonal: {zonal_dir}")
    
    # Load marksheet for centre filtering
    marksheet = pd.read_csv(marksheet_path)
    
    # Build case list
    case_ids = []
    for _, row in marksheet.iterrows():
        case_id = f"{row.patient_id}_{row.study_id}"
        
        # Filter by centre if specified
        if train_centers and row.center not in train_centers:
            continue
        
        # Verify all files exist
        t2_file = t2_dir / f"{case_id}.nii.gz"
        adc_file = adc_dir / f"{case_id}.nii.gz"
        hbv_file = hbv_dir / f"{case_id}.nii.gz"
        zonal_file = zonal_dir / f"{case_id}.nii.gz"
        
        if t2_file.exists() and adc_file.exists() and hbv_file.exists() and zonal_file.exists():
            case_ids.append(case_id)
    
    if max_cases:
        case_ids = case_ids[:max_cases]
    
    print(f"\nTotal cases to convert: {len(case_ids)}")
    if train_centers:
        print(f"Filtered to centres: {train_centers}")
    
    # Define output directories
    dataset_dir = nnunet_raw / DATASET_NAME
    
    # Clean output directory if it exists (prevent leftover files)
    if dataset_dir.exists():
        print(f"Cleaning existing {dataset_dir}...")
        shutil.rmtree(dataset_dir)
    
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Track label statistics
    all_label_values = Counter()
    n_converted = 0
    
    # Helper function for parallel processing
    def process_case(case_id):
        t2_file = t2_dir / f"{case_id}.nii.gz"
        adc_file = adc_dir / f"{case_id}.nii.gz"
        hbv_file = hbv_dir / f"{case_id}.nii.gz"
        zonal_file = zonal_dir / f"{case_id}.nii.gz"
        
        # Copy 3 image channels
        shutil.copy2(str(t2_file), str(images_dir / f"{case_id}_0000.nii.gz"))
        shutil.copy2(str(adc_file), str(images_dir / f"{case_id}_0001.nii.gz"))
        shutil.copy2(str(hbv_file), str(images_dir / f"{case_id}_0002.nii.gz"))
        
        # Copy and verify zonal mask
        label_vals = verify_and_copy_mask(
            zonal_file,
            labels_dir / f"{case_id}.nii.gz"
        )
        return label_vals
        
    print(f"Starting parallel conversion using up to 16 threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_case, cid): cid for cid in case_ids}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(case_ids), desc="Converting"):
            label_vals = future.result()
            for v in label_vals:
                all_label_values[int(v)] += 1
            n_converted += 1
    
    # Determine labels from observed data
    observed_labels = sorted(all_label_values.keys())
    print(f"\nObserved label values across all masks: {observed_labels}")
    print(f"Label frequency (number of masks containing each value):")
    for val in observed_labels:
        print(f"  Label {val}: appears in {all_label_values[val]} masks")
    
    # Build label mapping (will be verified by characterize_dataset.py)
    label_mapping = {"background": 0}
    if 1 in observed_labels:
        label_mapping["PZ"] = 1
    if 2 in observed_labels:
        label_mapping["TZ"] = 2
    # Handle any additional labels
    for val in observed_labels:
        if val not in [0, 1, 2]:
            label_mapping[f"label_{val}"] = val
    
    # Generate dataset.json
    dataset_json = {
        "channel_names": {
            "0": "T2W",
            "1": "ADC",
            "2": "HBV"
        },
        "labels": label_mapping,
        "numTraining": n_converted,
        "file_ending": ".nii.gz"
    }
    
    json_path = dataset_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Cases: {n_converted}")
    print(f"   Images: {images_dir} ({n_converted * 3} files)")
    print(f"   Labels: {labels_dir} ({n_converted} files)")
    print(f"   Config: {json_path}")
    print(f"\nDataset JSON:")
    print(json.dumps(dataset_json, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PI-CAI to nnU-Net v2 format for zonal segmentation"
    )
    parser.add_argument("--picai_dir", type=str, required=True,
                        help="Path to PI-CAI pre-processed dataset root")
    parser.add_argument("--nnunet_raw", type=str, required=True,
                        help="Path to nnU-Net raw data directory")
    parser.add_argument("--marksheet", type=str, required=True,
                        help="Path to marksheet.csv")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="Limit to N cases (for sanity check)")
    parser.add_argument("--train_centers", nargs="+", default=None,
                        help="Only include cases from these centres (e.g., RUMC ZGT)")
    
    args = parser.parse_args()
    convert_picai_to_nnunet(
        args.picai_dir, args.nnunet_raw, args.marksheet,
        args.max_cases, args.train_centers
    )

import json
import numpy as np
from pathlib import Path
import argparse
import sys

def run_preflight_check(nnunet_preprocessed: str, dataset_name: str = "Dataset501_ZonalSeg"):
    print("======================================================================")
    print("🚀 nnU-Net v2 Pre-flight Check")
    print("======================================================================\n")
    
    preprocessed_dir = Path(nnunet_preprocessed) / dataset_name
    
    if not preprocessed_dir.exists():
        print(f"❌ Error: Preprocessed directory not found at {preprocessed_dir}")
        sys.exit(1)
        
    # 1. Check Plans (Hyperparameters)
    plans_file = preprocessed_dir / "nnUNetPlans.json"
    if not plans_file.exists():
        print(f"❌ Error: {plans_file.name} missing.")
        sys.exit(1)
        
    with open(plans_file, "r") as f:
        plans = json.load(f)
        
    print("✅ 1. Hyperparameters & Plans")
    print(f"  - Target Spacing (Z, X, Y): {plans.get('original_median_spacing_after_transp')}")
    
    config_3d = plans.get("configurations", {}).get("3d_fullres", {})
    print(f"  - 3D Patch Size: {config_3d.get('patch_size')}")
    print(f"  - 3D Batch Size: {config_3d.get('batch_size')}")
    print(f"  - Normalization Schemes: {config_3d.get('normalization_schemes')}")
    
    arch = config_3d.get("architecture", {}).get("arch_kwargs", {})
    print(f"  - U-Net Stages: {arch.get('n_stages')}")
    print(f"  - Base Features: {arch.get('features_per_stage', [])[0] if arch.get('features_per_stage') else 'Unknown'}")
    print("")

    # 2. Check Preprocessed Data Tensors
    data_dir = preprocessed_dir / "nnUNetPlans_3d_fullres"
    npy_files = list(data_dir.glob("*.npy"))
    
    print("✅ 2. Preprocessed Data Integrity")
    print(f"  - Preprocessed cases found: {len(npy_files)}")
    
    if not npy_files:
        print(f"  ❌ No .npy files found in {data_dir.name}!")
        print(f"     Contents of {preprocessed_dir.name}:")
        for f in preprocessed_dir.iterdir():
            print(f"       - {f.name} (IsDir: {f.is_dir()})")
            
        if data_dir.exists():
            print(f"     Contents of {data_dir.name}:")
            for f in list(data_dir.glob("*"))[:10]:
                print(f"       - {f.name}")
        else:
            print(f"     Folder {data_dir.name} DOES NOT EXIST!")
    else:
        sample_file = npy_files[0]
        data = np.load(sample_file)
        # nnU-Net preprocessed data shape: (C, Z, X, Y) where C = 3 modalities + 1 segmentation
        print(f"  - Sample Tensor: {sample_file.name}")
        print(f"  - Tensor Shape: {data.shape} (Channels, Z, X, Y)")
        print(f"  - Tensor Dtype: {data.dtype}")
        
        if data.shape[0] != 4:
            print(f"  ❌ Expected 4 channels (T2, ADC, HBV, Seg), got {data.shape[0]}")
        else:
            print(f"  - Channel mapping valid (3 modalities + 1 target mask)")
            
    print("")
            
    # 3. Check Splits
    splits_file = preprocessed_dir / "splits_final.json"
    if not splits_file.exists():
        print(f"❌ Error: {splits_file.name} missing. Please run Phase 3 Step 3 (Generate Splits).")
    else:
        with open(splits_file, "r") as f:
            splits = json.load(f)
            
        print("✅ 3. Cross-Validation Splits")
        print(f"  - Number of Folds: {len(splits)}")
        
        # Verify Fold 0 Leakage
        fold_0 = splits[0]
        train_patients = set([k.split('_')[0] for k in fold_0['train']])
        val_patients = set([k.split('_')[0] for k in fold_0['val']])
        overlap = train_patients & val_patients
        
        print(f"  - Fold 0 Train Cases: {len(fold_0['train'])} ({len(train_patients)} unique patients)")
        print(f"  - Fold 0 Val Cases:   {len(fold_0['val'])} ({len(val_patients)} unique patients)")
        
        if len(overlap) == 0:
            print("  - Longitudinal Leakage Check: PASS (0 overlapping patients)")
        else:
            print(f"  - Longitudinal Leakage Check: FAIL! Overlapping patients: {overlap}")
            
    print("\n======================================================================")
    print("If all checks display ✅ and PASS, you are cleared for Phase 3 Training!")
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-flight check for nnU-Net preprocessed data.")
    parser.add_argument("--nnunet_preprocessed", type=str, required=True)
    args = parser.parse_args()
    
    run_preflight_check(args.nnunet_preprocessed)

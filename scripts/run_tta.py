import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from tta.session import ContinualTTASession

def main():
    parser = argparse.ArgumentParser(description="Run Continual TTA on PCNN Dataset")
    parser.add_argument("--plans", type=str, required=True, help="Path to nnUNetPlans.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained baseline checkpoint")
    parser.add_argument("--adapters", type=str, required=True, help="Path to trained source adapters")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing PCNN target cases")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save TTA segmentations")
    parser.add_argument("--iterations", type=int, default=10, help="Number of TTA iterations per case")
    args = parser.parse_args()

    print("=== bpMRI-TTA: Continual Test-Time Adaptation ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialize TTA Session
    session = ContinualTTASession(args.plans, args.checkpoint, args.adapters, device=device)
    
    # 2. Get target cases (we expect T2, ADC, HBV files for each case)
    in_path = Path(args.input_folder)
    out_path = Path(args.output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find unique case IDs (assuming standard nnU-Net format: caseID_0000.nii.gz)
    files = list(in_path.glob("*_0000.nii.gz"))
    case_ids = [f.name.replace("_0000.nii.gz", "") for f in files]
    
    # Sort them to simulate a sequential stream of patients arriving at the clinic
    case_ids.sort()
    
    print(f"Found {len(case_ids)} cases in target stream. Starting adaptation...")
    
    for case_id in tqdm(case_ids):
        # In a real implementation, you'd use nnU-Net's preprocessing pipeline here
        # to ensure the image is resampled to the patch size [16, 320, 320].
        # Since nnUNetv2_predict handles this internally, integrating TTA cleanly 
        # requires hooking into nnUNet's predict_from_files function.
        # For the sake of this standalone script, we simulate it.
        
        print(f"\nProcessing {case_id}...")
        
        # Load modalities
        t2 = nib.load(in_path / f"{case_id}_0000.nii.gz").get_fdata()
        adc = nib.load(in_path / f"{case_id}_0001.nii.gz").get_fdata()
        hbv = nib.load(in_path / f"{case_id}_0002.nii.gz").get_fdata()
        
        # Stack into [1, 3, D, H, W] tensor (Simulating preprocessed input)
        # Note: Actual pipeline requires running nnU-Net preprocessing!
        # This script assumes the input is ALREADY PREPROCESSED to match network spacing/size.
        stacked = np.stack([t2, adc, hbv], axis=0)
        tensor = torch.from_numpy(stacked).float().unsqueeze(0)
        
        # Run TTA for this patient!
        # The adapter weights update on this patient and carry over to the next
        segmentation = session.adapt_and_predict(case_id, tensor, k_iterations=args.iterations)
        
        # Convert prediction to classes (argmax)
        pred_classes = torch.argmax(segmentation, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        # Save output (using T2 header as reference)
        ref_img = nib.load(in_path / f"{case_id}_0000.nii.gz")
        out_img = nib.Nifti1Image(pred_classes, ref_img.affine, ref_img.header)
        nib.save(out_img, out_path / f"{case_id}.nii.gz")

    print("\nContinual Adaptation Complete!")
    print(f"Processed {len(session.patient_logs)} patients sequentially.")

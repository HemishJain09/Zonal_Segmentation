import argparse
import torch
import sys
from pathlib import Path

# Add project root to python path so we can import local modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.trainer import SourceTrainer

# We need a dummy dataloader that yields batches from the nnU-Net preprocessed data
# In a real nnU-Net pipeline, we would use their nnUNetDataLoader3D, but we can build
# a simple wrapper that loads the .npy files if needed. For now, since the user has
# access to the nnU-Net trainer, we can load it.

def main():
    parser = argparse.ArgumentParser(description="Run Source Training for bpMRI-TTA")
    parser.add_argument("--plans", type=str, required=True, help="Path to nnUNetPlans.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained baseline checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to nnUNet_preprocessed dataset folder")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train adaptation modules")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to save the trained adapters")
    args = parser.parse_args()

    print("=== bpMRI-TTA: Source Training ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialize Trainer
    trainer = SourceTrainer(args.plans, args.checkpoint, device=device)
    
    # 2. Setup Dataloader
    # Note: To fully integrate with nnU-Net's data augmentation, we instantiate
    # the nnU-Net trainer just to hijack its dataloader.
    import os
    os.environ['nnUNet_preprocessed'] = str(Path(args.dataset).parent)
    
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    
    import json
    with open(args.plans, 'r') as f:
        plans_dict = json.load(f)
        
    dataset_json_path = Path(args.dataset) / "dataset.json"
    with open(dataset_json_path, 'r') as f:
        dataset_json = json.load(f)
        
    # We just need the dataloader from it
    nnunet_trainer = nnUNetTrainer(plans=plans_dict, configuration="3d_fullres", fold=0, dataset_json=dataset_json)
    nnunet_trainer.dataset_tr = nnunet_trainer.get_tr_and_val_datasets()[0] # get training dataset
    dl_tr, _ = nnunet_trainer.get_dataloaders()
    
    print(f"Starting training for {args.epochs} epochs...")
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        losses = trainer.train_epoch(dl_tr, num_epochs=args.epochs)
        
        # Save best model
        if losses["total"] < best_loss:
            best_loss = losses["total"]
            trainer.save_checkpoint(f"{args.out_dir}/adapters_best.pth")
            
        # Save latest model periodically
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"{args.out_dir}/adapters_ep{epoch+1}.pth")

    trainer.save_checkpoint(f"{args.out_dir}/adapters_final.pth")
    print("Training Complete!")

if __name__ == "__main__":
    main()

# Robust Cross-Centre Prostate Zonal Segmentation

**nnU-Net v2 Baseline for Prostate Zonal Segmentation on Biparametric MRI**

## Objective

Develop a publication-grade nnU-Net v2 baseline for prostate zonal segmentation (Peripheral Zone / Transition Zone) using biparametric MRI (T2W, ADC, HBV) across three clinical centres (RUMC, ZGT, PCNN), and quantify cross-centre domain shift before introducing domain adaptation techniques.

## Dataset

PI-CAI Public Training and Development Dataset (1500 cases, 3 centres):
- **Training**: RUMC + ZGT (5-fold cross-validation)
- **External Test**: PCNN (never used for training or model selection)

## Architecture

- **Model**: nnU-Net v2 (vanilla defaults)
- **Input**: 3-channel biparametric MRI (T2W, ADC, HBV)
- **Output**: Multi-class zonal segmentation (Background, PZ, TZ)

## Project Structure

```
├── configs/                    # Experiment configurations
│   └── experiment_config.yaml
├── data/                       # Data processing scripts
│   ├── characterize_dataset.py # Phase 1: Dataset analysis
│   ├── convert_to_nnunet.py    # Phase 3: nnU-Net format conversion
│   └── generate_splits.py     # Phase 2: Patient-level CV splits
├── evaluation/                 # Evaluation scripts
│   ├── compute_metrics.py     # Dice, HD95, ASD per zone
│   └── error_analysis.py     # Failure case analysis
├── notebooks/                  # Colab training notebooks
│   └── zonal_seg_colab.ipynb
├── docs/                       # Setup guides
│   └── colab_setup_guide.md
├── problem.md                  # Research protocol
└── requirements.txt
```

## Quick Start

1. Clone this repo
2. Open `notebooks/zonal_seg_colab.ipynb` in Colab (or via VS Code Colab extension)
3. Follow the 2-phase workflow (sanity check → full training)

## Success Criteria

| Metric | Internal (RUMC+ZGT) | External (PCNN) | Domain Gap |
|--------|---------------------|------------------|------------|
| Dice   | > 0.91              | > 0.86           | < 0.05     |

## Citation

If using the PI-CAI dataset, please cite:
> Saha A, Bosma JS, Twilt JJ, et al. Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI). Lancet Oncol 2024; 25: 879–887.

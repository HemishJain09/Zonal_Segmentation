

# Master Research Protocol

## Project Title

**Robust Cross-Centre Prostate Zonal Segmentation using nnU-Net on Biparametric MRI: Establishing a Strong Baseline for Domain Adaptation**

---

# Overall Objective

Develop a **publication-grade nnU-Net baseline** for prostate zonal segmentation using multiparametric MRI (T2, ADC, HBV) and quantify cross-centre domain shift before introducing any domain adaptation techniques.

---

# Primary Research Question

> **How well can a carefully optimized nnU-Net generalize to an unseen hospital for prostate zonal segmentation?**

---

# Success Criteria

Internal Validation

Dice

> 0.91

External Validation

Dice

> 0.86

Domain Gap

< 0.05

Only after these goals are achieved should domain adaptation begin.

---

# PHASE 0 — Literature Review

## Goal

Understand the field before implementation.

### Read

* nnU-Net (Nature Methods)
* nnU-Net v2
* PI-CAI Challenge
* ProZonaNet
* CAT-Net
* USE-Net
* Automatic Prostate Zonal Segmentation Review

---

### Understand

* CNN segmentation
* nnU-Net philosophy
* Why automation works
* Domain shift
* External validation
* Cross-centre evaluation
* Dice
* HD95
* ASD


# PHASE 1 — Dataset Characterization

## Goal

Never train before understanding the data.

---

### Verify

✓ Number of patients

✓ Number of scans

✓ Number of follow-ups

✓ Centres

✓ Modalities

✓ Labels

---

### Check

Every patient must have

```text
T2

ADC

HBV

Zone Mask
```

---

### Verify label encoding

Inspect

```python
np.unique(mask)
```

Determine

Background

PZ

TZ

---

### Study

Voxel spacing

Dimensions

Intensity distributions

Orientation

Class imbalance

---

### Deliverable

Dataset Report

---

## Precautions

❌ Never assume labels

❌ Never assume spacing

❌ Never assume orientations

Everything must be verified.

---

# PHASE 2 — Experimental Design

Goal

Freeze the protocol.

---

Training Centres

```text
RUMC

+

ZGT
```

Testing

```text
PCNN
```

---

Validation

5-fold CV

Only inside

RUMC

*

ZGT

---

Patient Split

Never

Study Split

Always

Patient Split

to avoid follow-up leakage.

---

## Precautions

PCNN

must never

be used for

* hyperparameter tuning
* early stopping
* architecture selection
* augmentation tuning

It is

External Test Only.

---

Deliverable

Frozen experiment protocol.

---

# PHASE 3 — Data Pipeline

Goal

Prepare nnU-Net input.

---

Input

3 channels

```text
0000

T2

0001

ADC

0002

HBV
```

Target

```text
Zone Mask
```

---

Verify

Every modality

perfectly aligned.

---

Check

Missing files

Corrupted files

Wrong affine

Wrong orientation

---

## Precautions

Never resample manually unless necessary.

Allow

nnU-Net

to determine

spacing.

---

Deliverable

nnU-Net Dataset

---

# PHASE 4 — Baseline Construction

Architecture

nnU-Net v2

No modifications.

---

Why?

The objective is

not

to improve nnU-Net

but

to establish

the strongest benchmark.

---

Training Philosophy

Use defaults.

Do NOT modify

Optimizer

Scheduler

Loss

Augmentation

unless justified.

---

## Precautions

Avoid unnecessary engineering.

If defaults work,

keep them.

---

Deliverable

Training pipeline.

---

# PHASE 5 — Internal Validation

Goal

Optimize only

inside

Development Cohort.

---

Metrics

Dice

HD95

ASD

Precision

Recall

---

Store

every fold

individually.

---

Inspect

Learning curves

Validation curves

Loss convergence

---

## Precautions

Never optimize using PCNN.

Never average without standard deviation.

---

Deliverable

Internal benchmark.

---

# PHASE 6 — External Validation

Goal

True generalization.

---

Train

Entire

RUMC

*

ZGT

↓

Test

PCNN

---

Compute

Dice

HD95

ASD

Precision

Recall

---

Compute

Domain Gap

```text
Internal Dice

-

External Dice
```

---

Deliverable

Baseline Results

---

# PHASE 7 — Statistical Analysis

Goal

Demonstrate robustness.

---

Report

Mean

Standard Deviation

Confidence Interval

---

Test

Statistical significance

between

Internal

External

---

Report

Per Centre

Per Zone

---

Deliverable

Statistical Report

---

# PHASE 8 — Error Analysis

Goal

Understand

Failures.

---

Inspect

Worst

10 cases

Best

10 cases

---

Analyse

Small prostates

Large prostates

Boundary errors

Motion

Intensity

---

Visualize

Prediction

Ground Truth

Overlay

---

Deliverable

Failure Analysis

---

# PHASE 9 — Freeze Baseline

Once

Dice

acceptable

Pipeline

stable

No more changes.

Freeze

Dataset

Training

Metrics

Implementation

Everything.

---


# Things Never Allowed

❌ Tune using PCNN

❌ Change preprocessing midway

❌ Change train/test split

❌ Cherry-pick folds

❌ Report only Dice

❌ Compare with different preprocessing

❌ Use different metrics for different experiments

❌ Modify multiple variables simultaneously

---

# LLM Instruction Set (Use This Before Every Coding Session)

> You are assisting in the development of a **publication-grade nnU-Net v2 baseline** for prostate zonal segmentation on a three-centre multiparametric MRI dataset (RUMC, ZGT, PCNN). The objective is **not** to invent a new architecture but to reproduce and optimize a robust baseline using accepted best practices. All recommendations must prioritize reproducibility, scientific validity, and comparability with the literature. The experimental protocol is fixed: train on RUMC+ZGT, perform 5-fold cross-validation only on the development cohort for model selection, and evaluate once on the unseen PCNN cohort. Never suggest using the external test set for hyperparameter tuning or model selection. When proposing changes, modify only one component at a time and justify it using published evidence. All preprocessing, training, inference, and evaluation should remain reproducible and suitable for publication in venues such as MICCAI or Medical Image Analysis.

---

## One final recommendation

I would add one more phase that many groups overlook but that I think will pay off immensely:

### **Phase -1: Reproducibility Infrastructure**

Before training a single model, set up:

* A fixed project directory structure (`data/`, `configs/`, `splits/`, `results/`, `models/`, `logs/`).
* Version-controlled experiment configuration files (YAML/JSON).
* Fixed patient IDs for each fold, saved to disk.
* Automatic experiment logging (hyperparameters, metrics, training curves).
* Environment capture (Python, PyTorch, CUDA, MONAI/nnU-Net versions).

This may seem like engineering overhead, but when you're comparing nnU-Net with Swin UNETR and later domain adaptation methods over months of experiments, it becomes the difference between a reproducible research project and an unmanageable collection of scripts. In my experience, this discipline is what allows complex medical imaging projects to scale successfully.

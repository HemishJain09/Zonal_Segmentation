Project Overview

# **bpMRI-TTA**

## **Continual Test-Time Self-Supervised Domain Adaptation for Clinically Significant Prostate Cancer Detection using Multi-Modal MRI**

# **1\. Project Overview**

This project implements a **continual test-time self-supervised domain adaptation framework** for the detection of clinically significant prostate cancer (csPCa) from biparametric MRI (bpMRI).

Rather than proposing a new segmentation network, this work extends the official **PI-CAI nnU-Net baseline** with lightweight adaptation modules that enable the model to continually adapt to previously unseen hospitals and MRI scanners during deployment.

The framework is designed around one key observation:

Domain shift in prostate MRI is **modality-specific**, while prostate cancer is defined through the **joint relationship between multiple MRI modalities**.

Therefore, instead of adapting the entire segmentation network, we introduce a lightweight latent-space adaptation framework capable of learning scanner-specific characteristics while preserving the anatomical and multimodal knowledge already learned by the baseline segmentation model.

# **2\. Research Motivation**

Deep neural networks trained for prostate cancer detection achieve excellent performance on the scanners and hospitals used during training.

However, performance drops significantly when deployed to new hospitals because MRI appearance varies across institutions.

Unlike natural images, domain shift in prostate MRI is not a single global style change.

Each imaging modality changes differently.

### **T2-weighted MRI**

Affected by

- scanner vendor
- receive coil
- reconstruction algorithm
- acquisition protocol

### **ADC**

Affected by

- gradient calibration
- selected b-values
- post-processing pipeline

### **High-b DWI**

Affected by

- acquired vs computed high-b images
- maximum b-value
- reconstruction pipeline
- noise characteristics

Although these modalities shift independently, clinicians diagnose prostate cancer using the combined information from all three modalities.

Consequently,

adapting each modality independently is insufficient,

and adapting them jointly without preserving modality-specific information is equally problematic.

This motivates our proposed framework.

# **3\. Project Objective**

The objective of this project is **not** to improve the segmentation capability of nnU-Net.

Instead,

the objective is to develop a lightweight adaptation framework that enables an already-trained segmentation model to continuously adapt to unseen target domains without requiring manual annotations.

The proposed framework must satisfy four goals.

- Preserve the pretrained segmentation knowledge.
- Adapt to unseen scanners.
- Preserve multimodal relationships.
- Operate without segmentation labels during deployment.

# **4\. Core Hypothesis**

We hypothesize that domain adaptation can be achieved by operating entirely in the bottleneck latent space of a pretrained segmentation network.

Instead of modifying the encoder or decoder, we learn modality-specific latent corrections that align target-domain representations with the source-domain latent manifold.

During deployment, these corrections are refined through self-supervised reconstruction objectives without requiring segmentation labels.

# **5\. Overall Framework**

The complete framework consists of two stages.

## **Stage 1 - Baseline Training**

The official PI-CAI nnU-Net is trained exactly as provided by the authors.

No architectural modifications are introduced.

Training Dataset

│

▼

Official PI-CAI nnU-Net

│

▼

Pretrained model_best.pth

This stage produces the segmentation backbone used throughout the remainder of the project.

## **Stage 2 - Proposed Framework**

The pretrained backbone is imported into our repository through the Backbone Integration Layer.

The encoder and decoder are frozen permanently.

New adaptation modules are inserted only at the bottleneck.

Pretrained Encoder

│

▼

Adaptive Reconstruction Adapters

│

├────────► Self Reconstruction

│

├────────► Cross Reconstruction

│

▼

Channel-Gated Fusion

│

▼

Pretrained Decoder

Only these newly introduced modules are optimized.

# **6\. Training Strategy**

Source-domain training has one purpose:

Learn adaptation priors.

The encoder and decoder remain frozen.

The adapters, reconstruction heads, and fusion module are optimized using

- Dice Loss
- Adaptive Focal Loss
- Self Reconstruction Loss
- Cross Reconstruction Loss

After convergence,

the reconstruction heads become fixed latent teachers.

# **7\. Test-Time Adaptation Strategy**

During deployment,

no segmentation labels are available.

Only

- T2
- ADC
- High-b DWI

are provided.

The frozen reconstruction heads generate self-supervised reconstruction objectives.

Only the modality-specific adapters remain trainable.

These adapters continually evolve as additional patients from the same hospital are processed.

The encoder,

decoder,

fusion,

and reconstruction heads

remain unchanged.

# **8\. Continual Hospital-Level Adaptation**

Unlike conventional patient-wise TTA,

our framework performs continual adaptation across an entire hospital deployment session.

Hospital A

Patient 1

↓

Patient 2

↓

Patient 3

↓

...

↓

Patient N

Adapter parameters persist across patients.

When deployment moves to a different hospital,

the source-trained adapter checkpoint is reloaded,

starting a new adaptation session.

# **9\. Architectural Components**

The framework is composed of eight implementation modules.

## **Module 1 - Baseline Integration Layer**

Responsible for integrating the official PI-CAI nnU-Net into our framework.

Responsibilities

- load pretrained checkpoint
- expose encoder
- expose decoder
- expose bottleneck features
- freeze backbone
- hide PI-CAI implementation details

Produces

Stable backbone interface.

## **Module 2 - Adaptive Reconstruction Adapter**

Introduces modality-specific residual adapters operating only on bottleneck latent features.

Responsibilities

- modality-specific latent adaptation
- residual feature refinement
- bottleneck correction

Produces

Adapted latent representations.

## **Module 3 - Self-Reconstruction Branch**

Learns to reconstruct each modality from its own adapted latent representation.

Responsibilities

- learn intra-modality latent manifold
- provide self-supervised latent prior

Produces

Self reconstruction supervision.

## **Module 4 - Cross-Reconstruction Branch**

Learns to reconstruct one modality from the remaining two.

Responsibilities

- learn cross-modal relationships
- preserve multimodal consistency

Produces

Cross-modal supervision.

## **Module 5 - Channel-Gated Fusion**

Combines the three adapted latent representations into a single bottleneck representation.

Responsibilities

- modality weighting
- adaptive feature fusion

Produces

Unified latent representation for segmentation.

## **Module 6 - Complete Network Assembly**

Combines all architectural components into a single executable network.

Responsibilities

- forward pass orchestration
- tensor routing
- interface between modules

Produces

Complete segmentation network.

## **Module 7 - Source Training Pipeline**

Optimizes the newly introduced modules using labelled source-domain data.

Responsibilities

- training loop
- optimizer
- checkpointing
- validation
- supervised and self-supervised losses

Produces

Source-trained adaptation checkpoint.

## **Module 8 - Continual Test-Time Adaptation**

Performs continual adaptation during deployment without segmentation labels.

Responsibilities

- adapter optimization
- hospital session management
- continual self-supervised learning

Produces

Adapted latent representations for each deployment site.

# **10\. Software Architecture**

The repository follows a strict modular architecture.

Each folder owns exactly one responsibility.

| **Folder**  | **Responsibility**                                  |
| ----------- | --------------------------------------------------- |
| baseline/   | Official PI-CAI repository (read-only)              |
| ---         | ---                                                 |
| models/     | Neural network architecture introduced by this work |
| ---         | ---                                                 |
| losses/     | Supervised and self-supervised objectives           |
| ---         | ---                                                 |
| training/   | Source-domain optimization                          |
| ---         | ---                                                 |
| tta/        | Continual test-time adaptation                      |
| ---         | ---                                                 |
| datasets/   | Data loading and preprocessing                      |
| ---         | ---                                                 |
| evaluation/ | Metrics and benchmarking                            |
| ---         | ---                                                 |
| utils/      | Shared utilities                                    |
| ---         | ---                                                 |
| scripts/    | Entry points                                        |
| ---         | ---                                                 |

# **11\. Development Philosophy**

Every module is developed using the same lifecycle:

- Define the research problem.
- Review relevant literature.
- Brainstorm design alternatives.
- Select the final design.
- Specify mathematical formulation.
- Produce an implementation specification.
- Implement the module.
- Unit test the module.
- Integrate with the framework.
- Freeze the module before proceeding.

This ensures every component is scientifically justified, independently testable, and easy to maintain.

# **12\. Expected Contributions**

The proposed framework contributes:

- A lightweight latent-space adaptation framework built on top of an existing medical segmentation model.
- Modality-specific residual adapters for MRI domain adaptation.
- Self- and cross-reconstruction objectives operating entirely in latent space.
- A continual hospital-level test-time adaptation strategy requiring no segmentation labels.
- A modular software architecture that cleanly separates baseline functionality from novel research contributions.

# **13\. End-to-End Pipeline**

STAGE 1

Official PI-CAI Baseline Training

────────────────────────────────────────────────────

Training Dataset

│

▼

Official nnU-Net

│

▼

Pretrained model_best.pth

STAGE 2

Source Training of Adaptation Framework

────────────────────────────────────────────────────

Load Pretrained Backbone

│

▼

Freeze Encoder & Decoder

│

▼

Insert Adapters

│

▼

Self Reconstruction

│

▼

Cross Reconstruction

│

▼

Channel-Gated Fusion

│

▼

Frozen Decoder

│

▼

Dice + Adaptive Focal +

Self + Cross Losses

│

▼

Train Only Adaptation Modules

│

▼

Source Adaptation Checkpoint

STAGE 3

Continual Test-Time Adaptation

────────────────────────────────────────────────────

New Hospital

│

▼

Load Source Adaptation Checkpoint

│

▼

Patient Stream

│

▼

Forward Pass

│

▼

Self + Cross Reconstruction Loss

│

▼

Update Adapters Only

│

▼

Repeat Across Hospital Session

│

▼

Final Adapted Segmentation

Code Repo Design

# **Final Repository (Version 2.0)**

bpMRI-TTA/

│

├── baseline/ # Official PI-CAI (Read Only)

│

├── configs/

│

├── datasets/

│

├── models/

│

├── losses/

│

├── training/

│

├── tta/

│

├── evaluation/

│

├── utils/

│

├── scripts/

│

├── experiments/

│

├── tests/

│

└── README.md

Already much cleaner.

# **MODELS**

Instead of dozens of tiny files:

models/

├── backbone/

├── adaptation/

├── network.py

├── factory.py

## **backbone/**

backbone/

nnunet_loader.py

backbone_wrapper.py

That's enough.

No need for

encoder_wrapper.py

decoder_wrapper.py

backbone_factory.py

backbone_utils.py

because they are only used by the wrapper.

### **Responsibilities**

- load pretrained checkpoint
- expose encoder
- expose decoder
- freeze backbone

Nothing else.

## **adaptation/**

This becomes the heart of the project.

adaptation/

adapter.py

reconstruction.py

fusion.py

initialization.py

adaptation_factory.py

Notice

Self Reconstruction

Cross Reconstruction

are now

inside

reconstruction.py

because they always evolve together.

I would NOT separate them.

# **Why?**

Because

every change to

Self

will affect

Cross.

Every experiment

will modify

both.

Keeping them together

reduces duplicated code.

# **network.py**

Responsible only for

assembling

Encoder

↓

Adapter

↓

Reconstruction

↓

Fusion

↓

Decoder

Nothing else.

# **LOSSES**

Instead of

segmentation/

reconstruction/

I would simplify.

losses/

segmentation.py

reconstruction.py

total_training.py

total_tta.py

Each file computes one logical loss group.

# **TRAINING**

Instead of

trainer

manager

callbacks

optimizer

scheduler

...

I'd keep

training/

trainer.py

engine.py

optimizer.py

checkpoint.py

validator.py

### **Responsibilities**

trainer.py

Epoch loop

engine.py

One training iteration

forward

loss

backward

optimizer.py

parameter groups

checkpoint.py

saving

validator.py

validation

This is enough.

# **TTA**

This is where I think the biggest redesign should happen.

Instead of

patient_adapter.py

adaptation_manager.py

inference.py

those names no longer reflect what we decided.

We decided

Hospital

not

Patient.

So

tta/

session.py

adaptation_loop.py

optimizer.py

freeze.py

ttn.py

Notice

No

patient_adapter.py

because

adapter

already exists

inside

models.

# **session.py**

Responsible for

Hospital lifecycle

Hospital Start

↓

Load Checkpoint

↓

Patients

↓

Hospital End

# **adaptation_loop.py**

Implements

Algorithm 2.

Nothing else.

# **optimizer.py**

Registers

only

Adapter.

# **freeze.py**

Freezes

everything except

Adapter.

# **EVALUATION**

Simplify.

evaluation/

evaluator.py

metrics.py

visualization.py

Enough.

# **DATASETS**

Keep exactly as

dataset.py

loader.py

transforms.py

No need

dataset_factory.py

dataset_utils.py

unless

multiple datasets

appear later.

# **UTILS**

Keep

config.py

checkpoint.py

logger.py

seed.py

Everything else

can be added later.

# **TESTS**

This folder is currently missing.

Very important.

tests/

test_backbone.py

test_adapter.py

test_reconstruction.py

test_fusion.py

test_training.py

test_tta.py

# **Scripts**

Very clean.

scripts/

train.py

adapt.py

evaluate.py

predict.py

Enough.

# **How Modules Map to Files**

Now the important part.

## **Module 1**

Backbone Wrapper

Implements

models/backbone/

nnunet_loader.py

backbone_wrapper.py

## **Module 2**

Adaptive Reconstruction Adapter

Implements

models/adaptation/

adapter.py

initialization.py

## **Module 3**

Self Reconstruction

Implements

models/adaptation/

reconstruction.py

## **Module 4**

Cross Reconstruction

Same file

reconstruction.py

Different class.

## **Module 5**

Feature Fusion

models/adaptation/

fusion.py

## **Module 6**

Network Assembly

models/

network.py

## **Module 7**

Source Training

Implements

training/

trainer.py

engine.py

optimizer.py

checkpoint.py

validator.py

losses/

segmentation.py

reconstruction.py

total_training.py

## **Module 8**

Continual TTA

Implements

tta/

session.py

adaptation_loop.py

optimizer.py

freeze.py

ttn.py

losses/

total_tta.py

# **Files Already Covered by Modules**

After Modules 1-8, these are effectively designed:

models/

backbone/

adaptation/

network.py

losses/

training/

tta/

That is approximately **80-85% of the repository's core logic**.

# **Files Still Remaining**

These are infrastructure rather than research modules:

datasets/

Dataset interface.

configs/

YAML configurations.

evaluation/

Metrics and PI-CAI evaluation.

utils/

Generic utilities.

scripts/

Entry points.

tests/

Unit and integration tests.

# **Final Architecture Ownership**

This is the key principle I would ask your agent to follow:

| **Folder**         | **Responsibility**                     | **Never Contains**         |
| ------------------ | -------------------------------------- | -------------------------- |
| baseline/          | Official PI-CAI code                   | Your research logic        |
| ---                | ---                                    | ---                        |
| models/backbone/   | Load and wrap nnU-Net                  | Training or losses         |
| ---                | ---                                    | ---                        |
| models/adaptation/ | Adapters, reconstruction, fusion       | Optimizers or loops        |
| ---                | ---                                    | ---                        |
| models/network.py  | Assemble the complete forward graph    | Loss computation           |
| ---                | ---                                    | ---                        |
| losses/            | Compute objectives only                | Parameter updates          |
| ---                | ---                                    | ---                        |
| training/          | Source-domain optimization             | TTA logic                  |
| ---                | ---                                    | ---                        |
| tta/               | Continual adaptation during deployment | Source training            |
| ---                | ---                                    | ---                        |
| evaluation/        | Metrics and visualization              | Model updates              |
| ---                | ---                                    | ---                        |
| datasets/          | Data loading and transforms            | Neural network logic       |
| ---                | ---                                    | ---                        |
| utils/             | Shared helpers                         | Domain-specific algorithms |
| ---                | ---                                    | ---                        |
| scripts/           | Thin entry points                      | Business logic             |
| ---                | ---                                    | ---                        |

Module0-Compatibilty

# **Module 0 - Baseline Analysis & Integration Planning**

## **Implementation Specification Document**

# **Module Name**

**Baseline Analysis & Integration Planning**

# **Module Purpose**

Module 0 is responsible for analyzing the existing PI-CAI nnU-Net codebase and defining a stable integration strategy for our research framework.

Unlike the remaining modules, Module 0 introduces **no new neural network components** and performs **no model training**.

Instead, it serves as a one-time engineering phase that answers a single question:

**How do we safely reuse the pretrained PI-CAI backbone while keeping our research framework completely independent from the baseline implementation?**

The output of this module is **not a trained model**.

The output is a verified software architecture, dependency analysis, migration strategy, and integration contract that all remaining modules depend upon.

# **Why This Module Exists**

The official PI-CAI repository is a complete standalone segmentation project.

Our repository is a continual test-time adaptation framework.

Although both use the same pretrained backbone, they solve fundamentally different problems.

The PI-CAI repository was designed to perform

MRI

│

▼

nnU-Net

│

▼

Segmentation

Our framework requires

MRI

│

▼

Encoder

│

▼

Latent Features

│

▼

Adapters

│

▼

Self Reconstruction

│

▼

Cross Reconstruction

│

▼

Fusion

│

▼

Decoder

│

▼

Segmentation

The official implementation was never designed to expose intermediate latent representations or support modular insertion of adaptation components.

Therefore, before implementing any research module, we must first understand exactly how the baseline is implemented and define a clean integration strategy.

# **Scope**

Module 0 is responsible for

- Understanding the PI-CAI repository
- Identifying required dependencies
- Identifying unnecessary dependencies
- Verifying pretrained checkpoint compatibility
- Identifying encoder, bottleneck and decoder
- Understanding checkpoint loading
- Understanding preprocessing assumptions
- Designing repository migration strategy
- Defining integration contract
- Producing implementation documentation

Module 0 must **never**

- implement adapters
- implement reconstruction
- implement fusion
- implement losses
- implement optimizers
- implement training
- modify the baseline architecture

# **Inputs**

Module 0 assumes the following resources already exist.

## **1\. Official PI-CAI Repository**

This includes

- customized nnU-Net implementation
- PI-CAI specific modifications
- preprocessing pipeline
- inference pipeline
- evaluation pipeline

## **2\. Pretrained Checkpoints**

Example

model_best.pth

These checkpoints already contain

- trained encoder
- trained decoder
- segmentation knowledge

## **3\. nnU-Net Planning Files**

Example

plans.pkl

plans.json

These define

- architecture configuration
- number of stages
- feature channels
- patch size
- pooling hierarchy

## **4\. Dataset Metadata**

Example

dataset.json

Defines

- modality order
- labels
- spacing
- normalization information

## **5\. Preprocessed Dataset**

The project assumes that the dataset has already been preprocessed using the official nnU-Net preprocessing pipeline.

No preprocessing implementation is performed inside our repository.

# **Objectives**

Module 0 has eight objectives.

## **Objective 1 - Repository Analysis**

Completely understand the existing PI-CAI repository.

The goal is to identify

- project structure
- execution flow
- customized nnU-Net components
- PI-CAI specific modifications

Deliverable

Complete repository analysis.

## **Objective 2 - Execution Flow Analysis**

Trace one complete forward execution.

Dataset

↓

Preprocessing

↓

Network

↓

Encoder

↓

Bottleneck

↓

Decoder

↓

Segmentation

Every important function call should be identified.

Deliverable

Forward execution graph.

## **Objective 3 - Backbone Analysis**

Identify

- encoder implementation
- decoder implementation
- bottleneck location
- skip connections
- latent tensor dimensions
- feature channel dimensions

Deliverable

Backbone architecture specification.

## **Objective 4 - Dependency Analysis**

Determine exactly which baseline files are required by our framework.

Deliverable

Dependency graph.

## **Objective 5 - Artifact Analysis**

Identify every artifact required after baseline training.

Expected artifacts

model_best.pth

plans.pkl

dataset.json

network definition

Everything else should be classified as

Required

Optional

Unused

## **Objective 6 - Integration Feasibility**

Verify that

Encoder

↓

Latent

↓

Decoder

can be separated safely.

Questions to answer

Can encoder execute independently?

Can decoder execute independently?

Can bottleneck tensors be intercepted?

Can decoder consume externally modified latent tensors?

Deliverable

Integration feasibility report.

## **Objective 7 - Repository Migration Strategy**

Design how the baseline will interact with our repository.

Questions

Should baseline remain external?

Should files be copied?

Should wrappers be created?

How will checkpoints be loaded?

Deliverable

Migration strategy.

## **Objective 8 - Integration Contract**

Define the software interface between the baseline and our framework.

Deliverable

Stable integration specification used by Module 1.

# **Objective 9 - Baseline Asset Manifest**

## **Purpose**

The final responsibility of Module 0 is to produce a **Baseline Asset Manifest**, which serves as the single source of truth describing every external asset required by our research framework.

Instead of hardcoding file paths, architecture assumptions, or checkpoint locations throughout the codebase, all baseline-related information is centralized into one manifest.

This allows the framework to remain independent of the PI-CAI repository structure and makes it straightforward to switch between different checkpoints, folds, or even future backbone architectures.

The manifest is consumed by the **Baseline Integration Layer (Module 1)** and should never be modified by downstream modules.

# **Why This Manifest Exists**

Without a centralized manifest:

- different modules may hardcode checkpoint locations,
- tensor dimensions may be duplicated across files,
- changing folds requires modifying multiple scripts,
- replacing the backbone becomes error-prone.

Instead, every module queries the Baseline Integration Layer, which in turn reads the manifest.

The result is:

Baseline Assets

│

▼

Baseline Manifest

│

▼

Baseline Integration Layer

│

▼

Research Framework

Only Module 1 reads the manifest directly.

All remaining modules communicate only with Module 1.

# **Manifest Location**

configs/

baseline_manifest.yaml

This file becomes part of the repository configuration.

# **Manifest Responsibilities**

The manifest should describe:

### **1\. Backbone Information**

- nnU-Net version
- PI-CAI implementation version
- model architecture
- trainer name
- fold number

### **2\. Checkpoint Information**

Required checkpoint

Example

model_best.pth

Optional checkpoint

model_final.pth

Checkpoint directory

### **3\. Architecture Information**

Required metadata

- bottleneck tensor shape
- bottleneck channels
- encoder stages
- decoder stages
- feature map sizes
- skip connection levels

These values are verified during Module 0 and **must never be hardcoded elsewhere**.

### **4\. Dataset Information**

The manifest should define

- dataset identifier
- modality order
- label definitions
- preprocessing version

This ensures that the framework always interprets the pretrained backbone correctly.

### **5\. Runtime Configuration**

Store runtime assumptions such as

- input patch size
- inference patch size
- normalization strategy
- spacing assumptions

### **6\. Repository Mapping**

The manifest should explicitly state where baseline components originate.

For example

Architecture

↓

PI-CAI Repository

Checkpoint

↓

checkpoints/

Metadata

↓

plans.pkl

# **Example Manifest**

baseline:

name: picai_nnunet

version: v1

framework: nnUNet

trainer: nnUNetTrainerV2_Loss_FL_and_CE

fold: 0

\# ----------------------------------------

checkpoint:

model: checkpoints/model_best.pth

plans: checkpoints/plans.pkl

dataset_json: checkpoints/dataset.json

\# ----------------------------------------

architecture:

encoder_stages: 6

decoder_stages: 5

bottleneck_channels: 320

bottleneck_shape: \[4, 5, 5\]

latent_tensor_shape: \[320, 4, 5, 5\]

\# ----------------------------------------

modalities:

\- T2W

\- ADC

\- HBV

\# ----------------------------------------

labels:

background: 0

csPCa: 1

\# ----------------------------------------

runtime:

patch_size: \[16, 320, 320\]

spacing: inherited_from_plans

preprocessing: nnUNet

\# ----------------------------------------

repository:

backbone: baseline/

weights: checkpoints/

metadata: configs/

# **How Module 1 Uses the Manifest**

Module 1 should never assume anything about the backbone.

Instead, it performs

manifest = load_manifest()

model = build_backbone(manifest)

model.load_checkpoint(manifest.checkpoint)

freeze_backbone(model)

This completely removes hardcoded assumptions from the implementation.

# **How Other Modules Use the Manifest**

Importantly, Modules 2-8 **never read the manifest directly**.

Instead, they request information through the Baseline Integration Layer.

For example

channels = backbone.get_feature_channels()

shape = backbone.get_bottleneck_shape()

modalities = backbone.get_modalities()

This ensures that only one component in the entire project understands the baseline configuration.

# **Validation Requirements**

Before Module 1 begins, the manifest must be automatically validated.

The validation should verify:

- All required files exist.
- Checkpoint loads successfully.
- Architecture matches the checkpoint.
- Bottleneck dimensions match the actual model.
- Modality ordering is correct.
- Label definitions are consistent.
- Runtime configuration is complete.

If any validation fails, Module 1 must not start.

# **Repository Impact**

Module 0 should now additionally create:

configs/

└── baseline_manifest.yaml

and

scripts/

├── generate_manifest.py

├── validate_manifest.py

└── inspect_backbone.py

The generate_manifest.py script extracts verified information from the PI-CAI baseline after analysis, while validate_manifest.py ensures the manifest remains consistent with the available checkpoint and architecture.

# **Baseline Dependency Classification**

Module 0 must classify every baseline component into one of three categories.

## **Category A - Required**

These components are mandatory.

| **Component**        | **Reason**                 |
| -------------------- | -------------------------- |
| Network Architecture | Construct pretrained model |
| ---                  | ---                        |
| model_best.pth       | Load pretrained weights    |
| ---                  | ---                        |
| plans.pkl            | Build correct architecture |
| ---                  | ---                        |
| dataset.json         | Preserve modality ordering |
| ---                  | ---                        |
| Metadata             | Architecture configuration |
| ---                  | ---                        |

## **Category B - Optional**

Useful but not mandatory.

Examples

- preprocessing utilities
- inference utilities

## **Category C - Not Required**

These remain inside the baseline repository.

- trainer classes
- optimizer
- scheduler
- loss functions
- evaluation scripts
- Docker
- submission code
- cross-validation scripts

# **Repository Migration Plan**

After Module 0, the relationship between the two repositories becomes

Official PI-CAI Repository

│

├── Model Definition

├── Pretrained Weights

├── Metadata

│

▼

Baseline Integration Layer

│

▼

Research Framework

The research repository never imports training logic from PI-CAI.

Only the pretrained backbone is reused.

# **Required Deliverables**

Module 0 should produce the following documentation.

docs/

baseline_analysis.md

dependency_graph.md

integration_contract.md

migration_plan.md

backbone_specification.md

These documents become permanent project references.

# **Repository Impact**

The following utilities may be implemented.

scripts/

inspect_backbone.py

verify_checkpoint.py

trace_forward.py

Unit tests

tests/

test_checkpoint_loading.py

test_forward_equivalence.py

test_backbone_structure.py

These utilities exist only to validate the baseline before Module 1 begins.

# **Prerequisites for Module 1**

Module 1 must not begin until all of the following are verified.

✓ Official PI-CAI repository understood.

✓ nnU-Net architecture identified.

✓ Encoder identified.

✓ Decoder identified.

✓ Bottleneck identified.

✓ Bottleneck tensor dimensions verified.

✓ Skip connections understood.

✓ Checkpoint loading verified.

✓ Forward inference reproduced.

✓ Required baseline files documented.

✓ Repository migration strategy finalized.

✓ Integration contract approved.

# **Success Criteria**

Module 0 is successful when every engineering assumption about the baseline has been replaced by verified knowledge.

After completion,

the development team should know exactly

- how the backbone is constructed,
- how checkpoints are loaded,
- where latent representations are extracted,
- how the decoder is invoked,
- which files belong to the baseline,
- which files belong to the research framework,
- and how both repositories interact.

No downstream module should need to inspect the original PI-CAI repository again.

# **Definition of Done (DoD)**

Module 0 is complete when:

- The complete PI-CAI repository has been analyzed.
- The execution flow from input MRI to segmentation output is fully documented.
- The encoder, bottleneck, decoder, and skip connections have been identified.
- The pretrained checkpoint can be instantiated independently.
- The minimal set of required baseline artifacts has been defined.
- The dependency graph has been documented.
- The repository migration strategy has been finalized.
- The Baseline Integration Layer contract has been specified.
- Module 1 can be implemented without any further inspection of the PI-CAI repository.
- A complete baseline_manifest.yaml has been generated.
- The manifest accurately describes the pretrained backbone, checkpoint, architecture, dataset metadata, and runtime assumptions.
- The manifest passes automated validation.
- Module 1 can instantiate the pretrained backbone using only the manifest and the referenced baseline assets.
- No downstream module requires hardcoded backbone paths, tensor dimensions, or architectural constants.

Module1-BackboneWrapper

# **Module 1 - Baseline Integration Layer**

## **Implementation Specification Document**

# **Module Name**

**Baseline Integration Layer**

# **Module Purpose**

The Baseline Integration Layer is responsible for integrating the pretrained PI-CAI nnU-Net model into our proposed Test-Time Self-Supervised Adaptation Framework.

Unlike the official PI-CAI repository, which was designed as a standalone segmentation pipeline, our framework requires direct access to the encoder, bottleneck latent representations, and decoder separately.

Therefore, this module transforms the pretrained PI-CAI model into a reusable backbone interface without modifying its architecture or learned segmentation capability.

This module establishes the software contract between the baseline segmentation model and every research component implemented in this project.

It is the only module allowed to interact directly with the PI-CAI implementation.

# **Research Philosophy**

This module is **not** part of our scientific contribution.

It introduces:

- no new neural network layers,
- no trainable parameters,
- no additional losses,
- no architectural modifications.

Its sole purpose is to expose the pretrained backbone through a stable and reusable API.

All research contributions begin **after** this module.

# **Backbone Lifecycle**

The backbone follows two completely independent stages.

## **Stage 1 - Baseline Training (External)**

The official PI-CAI repository is trained independently using its original training pipeline.

PI-CAI Dataset

│

▼

Official PI-CAI nnU-Net

│

▼

Train Exactly As Authors Intended

│

▼

model_best.pth

At this stage,

our framework **does not exist**.

There are

- no adapters,
- no reconstruction heads,
- no fusion,
- no test-time adaptation.

The only objective is to obtain a high-quality segmentation backbone.

This stage is completely external to our repository.

## **Stage 2 - Research Framework**

Once the pretrained checkpoint is available,

our framework begins.

model_best.pth

│

▼

Baseline Integration Layer

│

▼

Frozen Encoder

│

▼

Research Modules

│

▼

Frozen Decoder

The pretrained backbone now becomes the foundation upon which all adaptation modules are built.

# **Why Do We Need a Baseline Integration Layer?**

The official PI-CAI implementation was designed for a single inference pipeline.

Input MRI

│

▼

nnU-Net

│

▼

Segmentation

However, our framework requires direct manipulation of bottleneck features.

The required execution graph becomes

Input MRI

│

▼

Encoder

│

▼

Latent Bottleneck

│

▼

Adaptive Reconstruction Adapters

│

▼

Self Reconstruction

│

▼

Cross Reconstruction

│

▼

Channel-Gated Fusion

│

▼

Decoder

│

▼

Segmentation

The official repository does not expose these intermediate stages.

Therefore, this module decomposes the original forward pass into reusable interfaces while preserving identical segmentation behaviour.

# **What Are We Actually Wrapping?**

We are **not** wrapping only the nnU-Net architecture.

We are wrapping the complete pretrained segmentation model.

The Baseline Integration Layer requires:

Official PI-CAI Network Architecture

-

Pretrained model_best.pth

-

nnU-Net Plans

-

Dataset Metadata

Together these components define the complete segmentation backbone.

# **Baseline Dependencies**

The following artifacts are required from the official PI-CAI repository.

| **Required Artifact**             | **Purpose**                                      |
| --------------------------------- | ------------------------------------------------ |
| Official nnU-Net Model Definition | Instantiate the network architecture             |
| ---                               | ---                                              |
| model_best.pth                    | Load pretrained encoder and decoder weights      |
| ---                               | ---                                              |
| plans.pkl / architecture plans    | Reconstruct the exact network configuration      |
| ---                               | ---                                              |
| dataset.json                      | Preserve modality ordering and label definitions |
| ---                               | ---                                              |
| nnU-Net preprocessing assumptions | Ensure compatibility with pretrained weights     |
| ---                               | ---                                              |

# **What We Do NOT Use**

The following components remain inside the official PI-CAI repository and are **not** imported into our framework.

- Training scripts
- Trainer classes
- Optimizers
- Learning-rate schedulers
- Loss functions
- Cross-validation code
- Evaluation scripts
- Docker configurations
- Submission utilities

These components were only required to produce the pretrained checkpoint.

They play no role in our research framework.

# **Responsibilities**

The Baseline Integration Layer has six responsibilities.

### **1\. Construct the Backbone**

Instantiate the official nnU-Net architecture using the provided planning files.

### **2\. Load Pretrained Weights**

Restore the complete segmentation model from model_best.pth.

### **3\. Expose Stable Interfaces**

Provide reusable access to

- encoder,
- decoder,
- forward inference,
- metadata.

### **4\. Freeze Backbone Parameters**

Provide utilities to freeze

- encoder,
- decoder,
- complete backbone.

This module **does not decide when to freeze**; it only exposes the functionality. Module 7 invokes these APIs before source training.

### **5\. Provide Backbone Metadata**

Expose

- bottleneck tensor shape,
- feature channels,
- architecture information,

to downstream modules.

This prevents hardcoded architectural assumptions.

### **6\. Hide PI-CAI Internals**

No other module in the repository may directly import or depend on the PI-CAI implementation.

All interaction must occur through this layer.

# **Repository Structure**

models/

└── backbone/

nnunet_loader.py

backbone_wrapper.py

metadata.py

freeze.py

# **File Responsibilities**

## **nnunet_loader.py**

Responsible for

- locating architecture files,
- constructing the official nnU-Net,
- loading pretrained checkpoints,
- validating successful initialization.

Public API

load_model()

load_checkpoint()

## **backbone_wrapper.py**

Provides the public backbone interface.

Public API

encode()

decode()

forward()

Internally,

it manages the interaction between the encoder and decoder without exposing PI-CAI implementation details.

## **metadata.py**

Provides architectural information.

Public API

get_bottleneck_shape()

get_feature_channels()

get_num_encoder_stages()

get_num_decoder_stages()

get_metadata()

No downstream module should hardcode architectural constants such as the bottleneck channel dimension.

## **freeze.py**

Provides freezing utilities.

Public API

freeze_encoder()

freeze_decoder()

freeze_backbone()

unfreeze()

No optimization logic belongs here.

# **Public Interface**

The Baseline Integration Layer exposes only the following APIs.

backbone.load_checkpoint()

backbone.encode()

backbone.decode()

backbone.forward()

backbone.freeze_encoder()

backbone.freeze_decoder()

backbone.freeze_backbone()

backbone.get_bottleneck_shape()

backbone.get_feature_channels()

backbone.get_metadata()

These APIs form the stable contract between the pretrained backbone and the research framework.

# **Source Training Integration**

Module 7 interacts with this module as follows.

Load Pretrained Backbone

│

▼

Freeze Encoder

│

▼

Freeze Decoder

│

▼

Insert Adapters

│

▼

Insert Reconstruction Heads

│

▼

Insert Fusion

│

▼

Train Only Research Modules

The backbone weights remain unchanged throughout source training.

# **Test-Time Adaptation Integration**

Module 8 interacts with this module as follows.

Load Adaptation Checkpoint

│

▼

Load Frozen Backbone

│

▼

Hospital Session

│

▼

Adapter Updates Only

Neither the encoder nor decoder are ever updated during deployment.

# **Mathematical Representation**

Let

- xxx denote the multi-modal MRI input,
- E(⋅)E(\\cdot)E(⋅) denote the pretrained encoder,
- D(⋅)D(\\cdot)D(⋅) denote the pretrained decoder.

The backbone computes

z = E(x)

ŷ = D(z)

This module introduces **no additional mathematical operations**.

It simply exposes E(⋅)E(\\cdot)E(⋅) and D(⋅)D(\\cdot)D(⋅) as reusable components for the downstream adaptation framework.

# **Module Ownership**

This module owns

- backbone construction,
- checkpoint loading,
- encoder interface,
- decoder interface,
- forward inference,
- freezing utilities,
- architecture metadata.

This module does **not** own

- adapters,
- reconstruction heads,
- fusion,
- losses,
- optimizers,
- source training,
- test-time adaptation,
- evaluation.

# **Unit Testing Requirements**

The following tests must pass before the module is considered complete.

| **Test**                   | **Expected Outcome**                               |
| -------------------------- | -------------------------------------------------- |
| Load pretrained checkpoint | Successful initialization                          |
| ---                        | ---                                                |
| Construct architecture     | Matches PI-CAI implementation                      |
| ---                        | ---                                                |
| Forward inference          | Identical prediction to baseline                   |
| ---                        | ---                                                |
| Encoder API                | Returns bottleneck latent representation           |
| ---                        | ---                                                |
| Decoder API                | Produces valid segmentation from bottleneck        |
| ---                        | ---                                                |
| Metadata API               | Returns correct bottleneck shape and channel count |
| ---                        | ---                                                |
| Freeze encoder             | Encoder parameters become non-trainable            |
| ---                        | ---                                                |
| Freeze decoder             | Decoder parameters become non-trainable            |
| ---                        | ---                                                |
| Freeze backbone            | Entire backbone becomes non-trainable              |
| ---                        | ---                                                |
| Prediction equivalence     | Wrapper output equals original PI-CAI output       |
| ---                        | ---                                                |

# **Integration Requirements**

Every downstream module must communicate exclusively through the Baseline Integration Layer.

The following modules may depend on this interface:

- Module 2 - Adaptive Reconstruction Adapter
- Module 5 - Channel-Gated Fusion
- Module 6 - Network Assembly
- Module 7 - Source Training
- Module 8 - Continual Test-Time Adaptation

No downstream module may import the PI-CAI repository directly.

# **Success Criteria**

Module 1 is considered complete when:

- The pretrained PI-CAI backbone can be instantiated from the official architecture and checkpoint.
- The wrapper reproduces identical segmentation outputs to the original implementation.
- Encoder and decoder can be accessed independently.
- Backbone metadata is available through stable APIs.
- Backbone freezing utilities are implemented.
- No downstream module requires direct interaction with PI-CAI internals.

# **Definition of Done (DoD)**

Module 1 is complete when the Baseline Integration Layer provides a stable, implementation-independent interface between the pretrained PI-CAI segmentation backbone and the remainder of the research framework.

After completion:

- the official PI-CAI repository becomes a **read-only dependency**,
- every research component depends only on the Baseline Integration Layer,
- the pretrained encoder and decoder can be reused without modification,
- and the remaining modules (2-8) can be developed without any knowledge of the internal structure of the PI-CAI codebase.

## **One final architectural recommendation**

There is one small refinement I would make based on your current assets.

Since you already possess:

- the customized PI-CAI nnU-Net implementation,
- the pretrained model_best.pth weights,
- and the preprocessed dataset,

I would make the **Baseline Integration Layer completely agnostic to where those assets come from**. It should accept configurable paths (through your configuration files) to the model definition, checkpoint, and metadata rather than assuming a fixed directory layout. This keeps the integration layer reusable across different checkpoints, folds, or future backbone variants while preserving the same public API for the rest of your framework.

Module2-ResidualAdapters

# **Updated Module 2 - Adaptive Reconstruction Adapter (ARA)**

## **Module Name**

**Adaptive Reconstruction Adapter (ARA)**

# **Module Category**

**Core Research Module**

This is the primary novel architectural component of the proposed Test-Time Adaptation framework.

Unlike a conventional residual adapter, the ARA integrates

- feature adaptation,
- self-supervision,
- reconstruction-guided learning

into a single trainable unit.

# **Research Motivation**

Deep models trained on source-domain bpMRI produce strong latent representations.

However, when deployed to unseen scanners, the latent feature distribution shifts because of modality-specific acquisition differences.

Instead of adapting the entire segmentation network, we adapt only the latent representation using lightweight modality-specific adapters.

The adapter is not merely a feature transformation block.

It also contains its own self-supervision branches that provide the learning signal during both source training and test-time adaptation.

# **Research Hypothesis**

Domain shift is sequence-specific.

Therefore,

each modality should have its own trainable adaptation unit.

Furthermore,

since test-time adaptation has no segmentation labels,

the adaptation unit must generate its own supervision through reconstruction objectives learned during source training.

# **Research Objective**

Develop a lightweight modality-specific adaptation unit that

- adapts latent features,
- preserves modality-specific information,
- preserves cross-modal relationships,
- provides self-supervision during source training,
- remains the only trainable component during test-time adaptation.

# **Position in Network**

T2

\\

ADC --------> Shared nnU-Net Encoder

/

DWI

│

▼

Bottleneck Feature

│

┌────────┼─────────┐

▼ ▼ ▼

Adaptive Adaptive Adaptive

Adapter Adapter Adapter

T2 ADC DWI

│ │ │

└────────┼─────────┘

▼

Feature Fusion

▼

Shared nnU-Net Decoder

# **Internal Architecture**

Each adapter is now a complete adaptation unit.

Bottleneck Feature

\[320 × 4 × 5 × 5\]

│

▼

Residual Transformation

1×1×1 → ReLU → 3×3×3 → ReLU → 1×1×1

│

Adapted Feature

┌──────────┴──────────┐

▼ ▼

Self Reconstruction Cross Reconstruction

Branch Branch

│ │

└──────────┬──────────┘

▼

Self-Supervised Losses

Notice that the reconstruction branches belong to the adapter itself.

They are **not independent modules**.

# **Responsibilities**

The Adaptive Reconstruction Adapter has three responsibilities.

## **1\. Feature Adaptation**

Transform the encoder bottleneck into a domain-adapted latent representation.

## **2\. Self Reconstruction**

Ensure the adapted latent representation preserves information specific to its own MRI sequence.

## **3\. Cross Reconstruction**

Ensure the adapted latent representation preserves relationships across MRI sequences.

# **Adapter Architecture**

Residual Transformation

Input

↓

1×1×1 Conv

↓

ReLU

↓

3×3×3 Conv

↓

ReLU

↓

1×1×1 Conv

↓

Residual Add

↓

Adapted Feature

# **Reconstruction Branches**

Each adapter contains

Residual Block

├── Self Reconstruction Branch

└── Cross Reconstruction Branch

These branches are auxiliary during training and adaptation.

They are **not used for segmentation inference**.

# **Tensor Dimensions**

Input

\[B,320,4,5,5\]

Output

\[B,320,4,5,5\]

The residual path preserves tensor dimensions exactly.

# **Initialization**

Initialization remains part of Module 2.

The residual branch is initialized as an identity mapping.

- First convolution → Kaiming.
- Middle convolution → Kaiming.
- Final projection → Zero initialization.

This ensures

y=xy=xy=x

at initialization.

# **Training Behaviour**

## **Source Training**

Trainable

- Encoder
- Decoder
- Adaptive Reconstruction Adapters

The adapter learns

- segmentation,
- self reconstruction,
- cross reconstruction

simultaneously.

## **Test-Time Adaptation**

Frozen

- Encoder
- Decoder
- Fusion
- Segmentation Head

Trainable

- Adaptive Reconstruction Adapter only

The reconstruction losses become the optimization signal for adapting the adapter parameters.

# **Public Interface**

The adapter no longer returns a single tensor.

Conceptually it returns

{

"adapted_feature": ...,

"self_reconstruction": ...,

"cross_reconstruction": ...

}

This gives downstream modules access to both the adapted latent feature and the auxiliary outputs required for computing self-supervised losses.

# **Updated Codebase Impact**

The previous repository layout should be updated to reflect that reconstruction is part of the adapter.

## **Folder Structure**

models/

adapters/

├── residual_adapter.py

├── self_reconstruction.py

├── cross_reconstruction.py

├── adapter_block.py

├── adapter_factory.py

├── adapter_utils.py

└── adapter_config.py

The separate models/reconstruction/ directory is removed.

# **Primary Files**

### **residual_adapter.py**

Responsibilities

- Residual bottleneck transformation
- Identity initialization
- Adapter forward pass
- Calls reconstruction branches

### **self_reconstruction.py**

Responsibilities

- Self reconstruction branch
- Produces self-reconstruction output
- Used only for self-supervised loss

### **cross_reconstruction.py**

Responsibilities

- Cross reconstruction branch
- Produces cross-reconstruction output
- Used only for self-supervised loss

### **adapter_block.py**

Defines the internal bottleneck residual block.

### **adapter_factory.py**

Creates

- T2 adapter
- ADC adapter
- DWI adapter

### **adapter_utils.py**

Contains

- initialization helpers
- shape validation
- parameter counting
- debugging utilities

### **adapter_config.py**

Stores

- reduction ratio
- activation
- initialization strategy
- branch configuration

# **Updated Files**

### **models/network.py**

Now receives

adapter_outputs = {

"adapted_feature": ...,

"self_reconstruction": ...,

"cross_reconstruction": ...

}

Only adapted_feature continues to the fusion and decoder.

The reconstruction outputs are routed to the loss computation during training and test-time adaptation.

### **forward_manager.py**

Updated flow

Encoder

↓

Adaptive Reconstruction Adapter

├── Adapted Feature

├── Self Reconstruction

└── Cross Reconstruction

↓

Feature Fusion

↓

Decoder

### **training/trainer.py**

Computes

Segmentation Loss

-

Self Reconstruction Loss

-

Cross Reconstruction Loss

### **adaptation/adaptation_loop.py**

Computes

Self Reconstruction Loss

-

Cross Reconstruction Loss

Only the adapter receives gradients.

# **Unit Tests**

The updated adapter should satisfy:

- Correct input/output tensor shape.
- Identity behaviour immediately after initialization.
- Residual addition correctness.
- Self reconstruction branch produces valid outputs.
- Cross reconstruction branch produces valid outputs.
- Decoder accepts adapted features without modification.
- Gradients propagate only through adapter parameters during TTA.

# **Updated Definition of Done**

Module 2 is complete when:

- Three modality-specific Adaptive Reconstruction Adapters are implemented.
- Each adapter combines residual feature adaptation with self- and cross-reconstruction branches.
- The adapter preserves bottleneck dimensions (\[B,320,4,5,5\]).
- Identity initialization is correctly applied.
- Only the adapted latent feature proceeds to fusion and segmentation.
- Reconstruction outputs are exposed for self-supervised optimization.
- During source training, the adapter is optimized jointly with the segmentation network.
- During test-time adaptation, the adapter becomes the **only trainable component**, using reconstruction losses as its supervision signal.
- The adapter integrates with the frozen backbone without modifying the official PI-CAI implementation.

Module3-SelfReconstruction

# **Module 3 - Self-Reconstruction Branch**

## **Status**

**Core Self-Supervision Module**

This module provides the **modality-specific self-supervised learning signal** that enables the Adaptive Reconstruction Adapter (ARA) to adapt without segmentation labels.

Unlike traditional autoencoders, this module **does not reconstruct MRI images**. Instead, it learns the structure of **well-adapted latent representations** produced by the adapter during source training.

# **Research Motivation**

During deployment we only receive

Patient MRI

There is no

- segmentation mask
- lesion annotation
- supervision

Therefore the adapter has no direct optimization signal.

The Self-Reconstruction Branch provides an auxiliary objective that allows the adapter to continue learning without labels.

# **Research Hypothesis**

Well-adapted latent representations produced from the **source domain** occupy a stable latent manifold.

If a latent representation from a new scanner deviates from this manifold,

the reconstruction error increases.

Minimizing this reconstruction error encourages the adapter to map unseen-domain features back toward the latent distribution learned during source training.

# **Core Idea**

During source training

Encoder

↓

Latent Z

↓

Adaptive Reconstruction Adapter

↓

Adapted Latent Z'

↓

Self Reconstruction Branch

↓

Reconstructed Latent Ẑ'

↓

L1(Z', Ẑ')

Notice carefully:

The reconstruction target is

Adapted Latent Z'

NOT

Original Encoder Latent Z

This is a fundamental design decision.

# **Why Not Reconstruct the Original Latent?**

Suppose we reconstruct

Z

Then the optimization becomes

Adapter

↓

Identity

because the easiest solution is simply

Z' = Z

That directly opposes adaptation.

Our goal is not to preserve the original latent.

Our goal is to preserve the **adapted latent manifold** learned during source training.

# **What is the Reconstruction Branch Learning?**

It is **not** learning segmentation.

It is **not** learning image synthesis.

It is learning

"What does a valid adapted latent representation look like?"

The branch acts as a learned latent prior.

# **Architecture**

Input

Adapted Latent

\[B,320,4,5,5\]

Output

Reconstructed Latent

\[B,320,4,5,5\]

Recommended architecture

Adapted Latent

↓

3×3×3 Conv

↓

ReLU

↓

3×3×3 Conv

↓

Output

A small CNN is sufficient because

- latent resolution is already tiny,
- spatial information is preserved,
- computational cost remains negligible.

# **Loss Function**

Prediction

Z^′\\hat Z'Z^′

Target

Z′Z'Z′

Loss

Lself=∣∣Z′−Z^′∣∣1L\_{self} = ||Z'-\\hat Z'||\_1Lself​=∣∣Z′−Z^′∣∣1​

L1 loss is preferred because

- stable,
- robust,
- commonly used for feature reconstruction.

# **Source Training Behaviour**

During source training

Trainable

- Encoder
- Decoder
- Adaptive Reconstruction Adapter
- Self-Reconstruction Branch

Optimization

Segmentation Loss

-

Self Reconstruction Loss

-

Cross Reconstruction Loss

The reconstruction branch learns the latent manifold jointly with the adapter.

# **Test-Time Behaviour**

This is the most important part.

After source training

freeze

Self Reconstruction Branch

Only

Adaptive Reconstruction Adapter

remains trainable.

# **Test-Time Pipeline**

New Patient

↓

Encoder (Frozen)

↓

Latent Znew

↓

Adapter (Trainable)

↓

Adapted Latent Z'new

↓

Frozen Self Reconstruction Branch

↓

Reconstructed Latent Ẑ'new

↓

Lself

↓

Update Adapter Only

The reconstruction branch never changes.

It acts as a fixed evaluator.

# **Why Freeze the Reconstruction Branch?**

If both

Adapter

-

Reconstruction Branch

continue learning,

then both can adapt together.

The reconstruction loss decreases,

but not because the adapter has improved.

Instead,

the evaluator itself changes.

That weakens the self-supervision signal.

By freezing the reconstruction branch,

we obtain a stable latent-space prior learned from the source domain.

The adapter alone must change to satisfy this prior.

# **Responsibilities**

The Self-Reconstruction Branch is responsible for

- learning the source-domain latent manifold,
- reconstructing adapted latent representations,
- producing the self-supervised loss,
- remaining frozen during test-time adaptation.

It is **not** responsible for

- feature adaptation,
- segmentation,
- modality fusion,
- optimization.

# **Public Interface**

Conceptually

reconstructed_latent = self_reconstruction(adapted_latent)

Input

Adapted Latent

Output

Reconstructed Latent

# **Codebase Impact**

Since we decided that the reconstruction branches belong **inside the ARA**, this module affects only the adapter package.

### **New File**

models/

└── adapters/

└── self_reconstruction.py

Responsibilities

- Define the lightweight CNN reconstruction head.
- Implement the forward pass.
- Expose reconstructed latent output.

### **Updated File**

models/

└── adapters/

└── residual_adapter.py

Responsibilities

- Instantiate the Self-Reconstruction Branch.
- Call it during the forward pass.
- Return both the adapted latent and the reconstruction output.

### **Updated Factory**

models/

└── adapters/

└── adapter_factory.py

Responsibilities

- Create adapters with integrated self-reconstruction branches.

### **Updated Network**

models/

└── network.py

The adapter now returns

{

"adapted_feature": ...,

"self_reconstruction": ...

}

The adapted feature continues to fusion and segmentation, while the reconstruction output is consumed by the loss computation.

### **Updated Losses**

losses/

├── self_reconstruction_loss.py

└── total_training_loss.py

The total source-training loss becomes

Ltotal=Lseg+λselfLself+λcrossLcrossL*{total} = L*{seg} + \\lambda*{self}L*{self} + \\lambda*{cross}L*{cross}Ltotal​=Lseg​+λself​Lself​+λcross​Lcross​

where the cross term will be added in Module 4.

# **Unit Tests**

The module is complete when the following are verified:

- Input and output shapes match (\[B,320,4,5,5\]).
- The branch reconstructs adapted latent features.
- L1 reconstruction loss computes correctly.
- During source training, gradients flow through both the adapter and the reconstruction branch.
- During test-time adaptation, gradients flow **only** through the adapter, while the reconstruction branch remains frozen.
- The branch integrates seamlessly inside the Adaptive Reconstruction Adapter.

# **Definition of Done (DoD)**

Module 3 is considered **complete** when all the following criteria are satisfied.

## **Architecture**

- A lightweight **Self-Reconstruction Branch** is implemented inside every Adaptive Reconstruction Adapter (ARA).
- The branch operates entirely in **latent space**.
- The branch reconstructs the **adapted latent representation (Z')**, not the input MRI and not the original encoder latent (Z).
- Input and output tensor dimensions are identical (\[B, 320, 4, 5, 5\]).
- The branch uses the finalized lightweight 3D CNN architecture.

## **Forward Pass**

Given

Adapted Latent Z'

the branch produces

Reconstructed Latent Ẑ'

The forward interface is finalized as

adapted_feature,

self_reconstruction

or

{

"adapted_feature": ...,

"self_reconstruction": ...

}

depending on the final adapter interface.

## **Loss**

- Self-Reconstruction Loss is implemented.
- Target is the adapted latent (Z').
- Prediction is the reconstructed latent (Ẑ').
- Default loss is L1 Loss.
- Loss integrates into the total training objective.

## **Source Training Behaviour**

During source training

- Adapter parameters are trainable.
- Self-Reconstruction Branch parameters are trainable.
- Both are optimized jointly.
- Reconstruction branch learns the source-domain latent manifold.

## **Test-Time Behaviour**

During test-time adaptation

- Encoder is frozen.
- Decoder is frozen.
- Fusion module is frozen.
- Self-Reconstruction Branch is frozen.
- Only the Adaptive Reconstruction Adapter receives gradients.
- Reconstruction loss updates only the adapter parameters.

This is the most important requirement of the module.

## **Repository Integration**

Module 3 is completely integrated with

models/

└── adapters/

├── residual_adapter.py

├── self_reconstruction.py

└── adapter_factory.py

The following files are updated

models/network.py

training/trainer.py

losses/self_reconstruction_loss.py

losses/total_training_loss.py

adaptation/adaptation_loop.py

## **Unit Tests**

The following tests must pass.

### **Shape Test**

Input

\[B,320,4,5,5\]

↓

Output

\[B,320,4,5,5\]

### **Reconstruction Test**

Verify

Self Reconstruction

produces

valid latent reconstruction

### **Gradient Test (Training)**

Verify gradients flow through

- Adapter
- Self-Reconstruction Branch

during source training.

### **Gradient Test (TTA)**

Verify gradients flow through

Adapter Only

The Self-Reconstruction Branch must remain frozen.

### **Loss Test**

Verify

L1(Z', Ẑ')

computes correctly.

### **Integration Test**

Verify the complete forward pass

Encoder

↓

Adapter

↓

Self Reconstruction

↓

Fusion

↓

Decoder

runs without tensor mismatch.

## **Scientific Validation**

The module should satisfy the following research objective.

The Self-Reconstruction Branch learns the latent manifold of well-adapted source-domain representations during source training and serves as a **fixed latent-space prior** during test-time adaptation, providing a self-supervised optimization signal that updates only the modality-specific adapters.

# **🎯 Success Criteria**

By the completion of Module 3, we should be able to say:

"Given a pretrained model and an unseen patient, the framework can compute a meaningful self-supervised reconstruction loss entirely in latent space without requiring segmentation labels. This loss is sufficient to optimize only the modality-specific adapter while leaving the remainder of the segmentation network frozen."

If that statement is true, **Module 3 is complete**.

Module4-CrossConstruction

# **Module 4 - Cross-Reconstruction Branch**

## **Status**

**Core Self-Supervision Module**

This module provides the **cross-modal self-supervised learning signal** that preserves the relationships among T2W, ADC, and High-b DWI during both source training and test-time adaptation.

Unlike conventional cross-reconstruction methods that translate one modality into another, this module operates entirely in **latent space**, predicting the latent representation of one modality from the remaining two.

# **Research Motivation**

Prostate cancer is **not diagnosed from a single MRI sequence**.

Instead, radiologists jointly interpret

- T2W
- ADC
- High-b DWI

The suspiciousness of a lesion emerges from the **combined appearance** across all sequences.

Therefore,

during adaptation,

it is insufficient to preserve each modality independently.

The framework must also preserve the **cross-modal relationships** that define clinically significant prostate cancer.

# **Research Hypothesis**

If two adapted modality-specific latent representations are known,

they should contain sufficient complementary information to infer the adapted latent representation of the third modality.

Therefore,

learning to predict one modality from the remaining two encourages the latent representations to preserve clinically meaningful cross-modal relationships.

# **Research Objective**

Develop a lightweight cross-reconstruction branch that

- learns the relationships among MRI modalities,
- operates completely in latent space,
- provides an auxiliary self-supervised learning objective,
- remains frozen during test-time adaptation,
- supervises only the modality-specific adapters.

# **Core Idea**

Unlike image reconstruction,

the branch predicts

**adapted latent representations**.

For example

ZT2'

-

ZADC'

↓

Cross Reconstruction Branch

↓

Predicted ZDWI'

↓

L1 Loss

↓

Target ZDWI'

This process is rotated across all three modalities.

# **Cross-Reconstruction Tasks**

Three prediction tasks are learned.

## **Task 1**

Input

ZT2'

-

ZADC'

↓

Predict

ZDWI'

## **Task 2**

Input

ZT2'

-

ZDWI'

↓

Predict

ZADC'

## **Task 3**

Input

ZADC'

-

ZDWI'

↓

Predict

ZT2'

Each modality becomes the prediction target exactly once.

# **Why This Design?**

This design directly models the complementary nature of bpMRI.

Unlike predicting one modality from another,

using two modalities provides richer contextual information and better reflects how radiologists interpret prostate cancer.

It also aligns directly with our hypothesis:

Preserve the joint cross-sequence relationships that define the disease.

# **Input**

Each prediction branch receives

Two Adapted Latent Features

\[B,320,4,5,5\]

-

\[B,320,4,5,5\]

# **Feature Combination**

The two latent tensors are concatenated

Concatenate

↓

\[B,640,4,5,5\]

# **Architecture**

Recommended architecture

Concatenated Latent

↓

1×1×1 Conv

640 → 320

↓

ReLU

↓

3×3×3 Conv

320 → 320

↓

ReLU

↓

3×3×3 Conv

320 → 320

↓

Predicted Target Latent

This keeps the branch lightweight while preserving spatial information.

# **Output**

Example

Predicted ZDWI'

\[B,320,4,5,5\]

Target

Actual ZDWI'

\[B,320,4,5,5\]

# **Loss Function**

Prediction

Z^m′\\hat Z'\_mZ^m′​

Target

Zm′Z'\_mZm′​

Loss

Lcross=∣∣Zm′−Z^m′∣∣1L\_{cross} = ||Z'\_m-\\hat Z'\_m||\_1Lcross​=∣∣Zm′​−Z^m′​∣∣1​

The three prediction losses are averaged

Lcross=LT2+LADC+LDWI3L*{cross} = \\frac{ L*{T2} + L*{ADC} + L*{DWI} }{3}Lcross​=3LT2​+LADC​+LDWI​​

# **Source Training Behaviour**

During source training

Trainable

- Encoder
- Decoder
- Adaptive Reconstruction Adapter
- Self-Reconstruction Branch
- Cross-Reconstruction Branch

Optimization

Ltotal=Lseg+λselfLself+λcrossLcrossL*{total} = L*{seg} + \\lambda*{self}L*{self} + \\lambda*{cross}L*{cross}Ltotal​=Lseg​+λself​Lself​+λcross​Lcross​

The Cross-Reconstruction Branch learns the cross-modal latent manifold jointly with the adapters.

# **Test-Time Behaviour**

During test-time adaptation

Frozen

- Encoder
- Decoder
- Fusion
- Self-Reconstruction Branch
- Cross-Reconstruction Branch

Trainable

- Adaptive Reconstruction Adapter only

# **Test-Time Pipeline**

New Patient

↓

Encoder

↓

ZT2'

ZADC'

ZDWI'

↓

Frozen Cross Reconstruction Branch

↓

Predict Missing Latent

↓

Cross Reconstruction Loss

↓

Update Adapter Only

The Cross-Reconstruction Branch acts as a fixed evaluator of cross-modal consistency.

# **Why Freeze the Cross-Reconstruction Branch?**

The Cross-Reconstruction Branch learns the source-domain relationships among the modalities.

If it also adapts during test time,

the latent relationship itself would drift,

weakening the supervision signal.

Instead,

the branch is frozen,

forcing the adapters to modify the latent representations until they satisfy the learned source-domain relationships.

# **Responsibilities**

The Cross-Reconstruction Branch is responsible for

- learning cross-modal latent relationships,
- predicting one latent modality from the remaining two,
- producing the cross-reconstruction loss,
- acting as a fixed latent prior during test-time adaptation.

It is **not** responsible for

- segmentation,
- modality adaptation,
- feature fusion,
- optimization.

# **Public Interface**

Conceptually

predicted_target = cross_reconstruction(

latent_modality_a,

latent_modality_b

)

Input

Two Adapted Latent Features

Output

Predicted Target Latent

# **Codebase Impact**

This module belongs inside the adapter package.

## **New File**

models/

└── adapters/

└── cross_reconstruction.py

Responsibilities

- Implement the cross-reconstruction network.
- Predict the missing latent representation.
- Return reconstructed latent.

## **Updated Files**

### **residual_adapter.py**

Responsibilities

- Instantiate the Cross-Reconstruction Branch.
- Route adapted latent features to the branch.
- Return cross-reconstruction outputs.

### **adapter_factory.py**

Responsibilities

Instantiate

- T2 Adapter
- ADC Adapter
- DWI Adapter

each with an integrated Cross-Reconstruction Branch.

### **network.py**

Updated output

{

"adapted_feature": ...,

"self_reconstruction": ...,

"cross_reconstruction": ...

}

### **forward_manager.py**

Updated pipeline

Encoder

↓

Adaptive Reconstruction Adapter

├── Adapted Feature

├── Self Reconstruction

└── Cross Reconstruction

↓

Feature Fusion

↓

Decoder

### **losses/**

New file

cross_reconstruction_loss.py

Updated

total_training_loss.py

to include

LcrossL\_{cross}Lcross​

### **training/trainer.py**

Updated to compute

Segmentation Loss

-

Self Reconstruction Loss

-

Cross Reconstruction Loss

### **adaptation/adaptation_loop.py**

Updated to compute

Self Reconstruction Loss

-

Cross Reconstruction Loss

while updating

only

the adapters.

# **Unit Tests**

The following tests must pass.

## **Shape Test**

Input

\[B,320,4,5,5\]

-

\[B,320,4,5,5\]

Output

\[B,320,4,5,5\]

## **Prediction Test**

Verify each branch predicts

- T2
- ADC
- DWI

correctly.

## **Loss Test**

Verify

L1

computes correctly.

## **Gradient Test (Training)**

Verify gradients flow through

- Adapter
- Cross-Reconstruction Branch

during source training.

## **Gradient Test (TTA)**

Verify gradients flow only through

Adapter

The Cross-Reconstruction Branch remains frozen.

## **Integration Test**

Verify

Encoder

↓

Adapter

↓

Cross Reconstruction

↓

Fusion

↓

Decoder

executes without tensor mismatch.

# **Definition of Done (DoD)**

Module 4 is complete when

- Three cross-reconstruction prediction tasks are implemented.
- The branch predicts the adapted latent of one modality using the remaining two adapted latent representations.
- The branch operates entirely in latent space.
- Input/output tensor dimensions remain \[B,320,4,5,5\].
- The lightweight 3D CNN predictor is implemented.
- L1 cross-reconstruction loss is implemented.
- Cross-Reconstruction Branch is trainable during source training.
- Cross-Reconstruction Branch is frozen during test-time adaptation.
- Only the modality-specific adapters receive gradients during TTA.
- Module integrates seamlessly with the Adaptive Reconstruction Adapter.
- Unit tests pass.

# **Scientific Validation**

By the completion of Module 4, the framework should satisfy the following research statement:

**The Cross-Reconstruction Branch learns the inter-modality latent manifold from the source domain by predicting the adapted latent representation of one MRI sequence from the remaining two. During test-time adaptation, this branch is frozen and acts as a fixed cross-modal prior, while only the modality-specific adapters are optimized to restore latent relationships disrupted by unseen scanner domains.**

Module5-FeatureFusion

# **Module 5 - Channel-Gated Feature Fusion**

## **Status**

**Supporting Architectural Module**

This module fuses the three modality-specific adapted latent representations into a single latent representation that can be consumed by the shared nnU-Net decoder.

Unlike attention-based fusion, this module uses lightweight **channel-wise gating**, allowing the network to learn the relative importance of each modality while remaining computationally efficient.

# **Research Motivation**

After Modules 2-4, the framework has three adapted latent representations

ZT2'

ZADC'

ZDWI'

each containing modality-specific and cross-modal information.

However, the pretrained nnU-Net decoder expects a **single bottleneck tensor**.

Therefore, a fusion mechanism is required.

The fusion module should

- preserve complementary information,
- suppress unreliable latent channels,
- remain lightweight,
- avoid introducing a competing research contribution.

# **Design Philosophy**

The purpose of this module is **not to learn new representations**.

Instead, it should answer one simple question:

**How much should each modality contribute to the final segmentation?**

The adapters have already corrected the domain shift.

The reconstruction branches have already ensured

- intra-modality consistency
- inter-modality consistency.

Now the fusion module simply decides how much to trust each corrected modality.

# **Research Objective**

Develop a lightweight feature fusion mechanism that

- estimates channel-wise importance for each adapted modality,
- suppresses unreliable latent channels,
- combines all modalities into a single bottleneck representation,
- produces a latent tensor fully compatible with the frozen nnU-Net decoder.

# **Why Channel-Gated Fusion?**

Simple weighted averaging assumes that every modality contributes equally.

This assumption rarely holds in clinical MRI because

- ADC may contain motion artefacts,
- High-b DWI may exhibit scanner-specific noise,
- T2 contrast may vary across vendors.

Instead, each modality should contribute **according to its estimated reliability**.

# **Core Idea**

For every adapted latent representation

ZT'

↓

Estimate Channel Importance

↓

Channel Gates

↓

Multiply

↓

Gated Feature

The gated features are then fused into a single bottleneck representation.

# **Complete Pipeline**

ZT2'

│

▼

Channel Gate

│

▼

Gated ZT2'

ZADC'

│

▼

Channel Gate

│

▼

Gated ZADC'

ZDWI'

│

▼

Channel Gate

│

▼

Gated ZDWI'

│

▼

Concatenate

↓

1×1×1 Convolution

↓

Fused Latent

↓

Shared nnU-Net Decoder

# **Channel Gate Architecture**

Each modality has an independent gate.

Input

\[B,320,4,5,5\]

## **Step 1 - Global Average Pooling**

320×4×5×5

↓

320

Each channel is summarized into a single descriptor.

## **Step 2 - Gate Network**

320

↓

Linear

↓

80

↓

ReLU

↓

Linear

↓

320

↓

Sigmoid

Output

320 channel weights

between

0 and 1

## **Step 3 - Feature Scaling**

Each channel is multiplied by its corresponding gate value.

Example

Channel

↓

Feature = 12

↓

Gate = 0.20

↓

Output = 2.4

High-confidence channels remain unchanged.

Low-confidence channels are suppressed.

# **Feature Fusion**

After gating

Gated T2

-

Gated ADC

-

Gated DWI

are concatenated

\[B,960,4,5,5\]

A lightweight projection layer compresses

960

↓

320

using a

1×1×1 Convolution

The final output

\[B,320,4,5,5\]

is passed directly to the shared nnU-Net decoder.

# **Why Concatenation Instead of Summation?**

Concatenation preserves modality-specific information.

Summation mixes features irreversibly.

The subsequent 1×1×1 convolution learns how to combine complementary channels while maintaining decoder compatibility.

# **Source Training Behaviour**

During source training

Trainable

- Encoder
- Decoder
- Adaptive Reconstruction Adapters
- Self-Reconstruction Branch
- Cross-Reconstruction Branch
- Channel-Gated Fusion Module

Optimization

Ltotal=Lseg+λselfLself+λcrossLcrossL*{total} = L*{seg} + \\lambda*{self}L*{self} + \\lambda*{cross}L*{cross}Ltotal​=Lseg​+λself​Lself​+λcross​Lcross​

No additional fusion loss is required.

The gating module is optimized implicitly through the segmentation objective.

# **Test-Time Behaviour**

During test-time adaptation

Frozen

- Encoder
- Decoder
- Self-Reconstruction Branch
- Cross-Reconstruction Branch
- Channel-Gated Fusion Module

Trainable

- Adaptive Reconstruction Adapters only

The fusion module remains fixed, ensuring that only the latent representations adapt while the learned modality integration strategy remains stable.

# **Responsibilities**

The Channel-Gated Fusion Module is responsible for

- estimating channel-wise modality importance,
- suppressing unreliable latent channels,
- combining adapted latent representations,
- generating a decoder-compatible fused bottleneck.

It is **not** responsible for

- domain adaptation,
- reconstruction,
- segmentation,
- optimization,
- self-supervision.

# **Public Interface**

Conceptually

fused_latent = fusion(

z_t2,

z_adc,

z_dwi

)

Input

ZT2'

ZADC'

ZDWI'

Output

ZFused

# **Codebase Impact**

## **New Folder**

models/

└── fusion/

## **New Files**

models/

└── fusion/

├── gated_fusion.py

├── channel_gate.py

├── fusion_block.py

├── fusion_utils.py

└── fusion_config.py

## **Responsibilities**

### **channel_gate.py**

Implements

- Global Average Pooling
- Gate MLP
- Sigmoid activation
- Channel-wise scaling

### **gated_fusion.py**

Implements

- three modality gates,
- feature concatenation,
- projection layer,
- fused latent generation.

### **fusion_block.py**

Contains reusable fusion operations

- concatenation
- projection
- tensor validation.

### **fusion_utils.py**

Contains

- debugging utilities,
- parameter counting,
- shape validation,
- helper functions.

### **fusion_config.py**

Stores

- reduction ratio,
- hidden dimension,
- activation,
- projection settings.

# **Updated Files**

### **models/network.py**

Updated forward pipeline

Encoder

↓

Adaptive Reconstruction Adapters

↓

Channel-Gated Fusion

↓

Shared Decoder

### **forward_manager.py**

Updated outputs

{

"adapted_latents": ...,

"gated_latents": ...,

"fused_latent": ...

}

### **model_factory.py**

Creates

- Fusion Module
- connects adapters to decoder.

### **training/trainer.py**

No new losses.

Fusion parameters receive gradients only through segmentation loss.

### **adaptation/adaptation_loop.py**

Fusion module remains frozen.

Only adapters receive gradients.

# **Unit Tests**

The following tests must pass.

## **Shape Test**

Input

3 × \[B,320,4,5,5\]

Output

\[B,320,4,5,5\]

## **Gate Test**

Verify

Sigmoid

↓

All gate values

∈ \[0,1\]

## **Scaling Test**

Verify channel-wise multiplication produces correctly scaled features.

## **Fusion Test**

Verify concatenation

320

-

320

-

320

↓

960

followed by

1×1×1 Conv

↓

320

## **Decoder Compatibility**

Verify the fused latent is accepted by the frozen nnU-Net decoder without modification.

## **Gradient Test (Training)**

Verify gradients flow through

- Channel Gates
- Projection Layer

during source training.

## **Gradient Test (TTA)**

Verify

Fusion Module

Frozen

Only adapters receive gradients.

# **Definition of Done (DoD)**

Module 5 is complete when

- Independent channel-wise gates are implemented for T2W, ADC, and High-b DWI.
- Each gate produces 320 channel importance weights using Global Average Pooling followed by a lightweight MLP.
- Channel-wise feature scaling is correctly applied to each adapted latent representation.
- The gated latent features are concatenated into a 960-channel tensor.
- A 1×1×1 projection layer reduces the concatenated representation back to 320 channels.
- The fused latent tensor has shape \[B,320,4,5,5\].
- The fused latent is fully compatible with the pretrained nnU-Net decoder.
- The fusion module is trainable during source training.
- The fusion module is frozen during test-time adaptation.
- No additional fusion-specific loss is introduced.
- All unit tests pass.

# **Scientific Validation**

By completion of Module 5, the framework should satisfy the following architectural statement:

**The Channel-Gated Fusion Module learns a source-domain modality integration strategy by estimating channel-wise reliability for each adapted latent representation. During test-time adaptation, this strategy remains fixed while only the modality-specific adapters evolve, ensuring that domain adaptation occurs solely in the latent representations rather than in the fusion mechanism.**

Module6-NetworkAssembly

# **Module 6 - Complete Network Assembly & Forward Orchestration**

## **Status**

**Core System Integration Module**

This module assembles all previously designed components into a single end-to-end architecture and orchestrates the complete forward pipeline for

- source training,
- test-time adaptation,
- inference.

It is responsible for execution flow, **not learning**.

# **Research Motivation**

After implementing

- Shared nnU-Net Backbone
- Adaptive Reconstruction Adapters
- Self-Reconstruction Branches
- Cross-Reconstruction Branches
- Channel-Gated Fusion

all components exist independently.

However, they must execute in the correct order while maintaining

- modularity,
- reproducibility,
- clean software design,
- compatibility with different execution modes.

Module 6 defines this execution pipeline.

# **Design Philosophy**

Module 6 follows a single architectural principle.

**Modules never orchestrate each other.**

Instead,

the network itself controls

- execution order,
- data flow,
- freezing strategy,
- execution mode.

Each module performs exactly one responsibility.

# **Research Objective**

Develop a clean network orchestration framework that

- connects every architectural component,
- exposes dedicated forward modes,
- centralizes execution logic,
- isolates research modules,
- minimizes coupling between components.

# **System Overview**

Input Patient

│

▼

┌──────────────────────────┐

│ Shared nnU-Net Encoder │

└──────────────────────────┘

│

┌──────────────┼──────────────┐

▼ ▼ ▼

ZT2 ZADC ZDWI

│ │ │

▼ ▼ ▼

Adaptive Reconstruction Adapter (ARA)

│ │ │

ZT2' ZADC' ZDWI'

│ │ │

│ │ │

├──────────────┼──────────────┐

│ │ │

▼ ▼ ▼

Self Reconstruction Branches

│

Cross Reconstruction Branches

│

▼

Channel-Gated Feature Fusion

│

▼

Shared nnU-Net Decoder

│

▼

Segmentation Prediction

Notice that

- reconstruction branches execute in parallel,
- fusion receives only adapted latent representations,
- decoder receives only the fused latent.

# **Module Responsibilities**

Module 6 is responsible for

- network assembly,
- execution order,
- execution modes,
- parameter freezing,
- output packaging.

It is **not** responsible for

- adaptation,
- reconstruction,
- segmentation,
- optimization,
- loss computation.

# **Execution Pipeline**

The complete forward pipeline is fixed as

Input MRI

↓

Shared Encoder

↓

Three Bottleneck Latent Features

↓

Three Adaptive Reconstruction Adapters

↓

Three Adapted Latent Features

↓

Self-Reconstruction Branches (Parallel)

↓

Cross-Reconstruction Branches (Parallel)

↓

Channel-Gated Feature Fusion

↓

Shared Decoder

↓

Segmentation Output

# **Parallel Execution Principle**

Self-Reconstruction and Cross-Reconstruction do **not** execute sequentially.

Instead,

they independently consume the adapted latent representations.

ZT2'

├────────► Self Reconstruction

├────────► Cross Reconstruction

└────────► Fusion

The adapted latent remains the shared feature representation throughout the pipeline.

# **Forward Modes**

The assembled network exposes three execution modes.

## **1\. Source Training Mode**

Purpose

Supervised source-domain optimization.

Pipeline

Encoder

↓

Adapters

↓

Self Reconstruction

↓

Cross Reconstruction

↓

Fusion

↓

Decoder

Returns

{

"segmentation": ...,

"adapted_latents": ...,

"self_outputs": ...,

"cross_outputs": ...,

"fused_latent": ...

}

This provides all information required for computing

- Dice Loss
- Adaptive Focal Loss
- Self-Reconstruction Loss
- Cross-Reconstruction Loss

## **2\. Test-Time Adaptation Mode**

Purpose

Patient-specific adaptation without labels.

Pipeline

Exactly identical.

Difference

Only adapters remain trainable.

Returns

{

"segmentation": ...,

"self_outputs": ...,

"cross_outputs": ...,

"adapted_latents": ...

}

The segmentation prediction is still generated but is **not** used for optimization.

## **3\. Inference Mode**

Purpose

Clinical deployment after adaptation.

Pipeline

Encoder

↓

Adapters

↓

Fusion

↓

Decoder

Reconstruction branches are skipped.

Returns

segmentation_prediction

This minimizes inference time.

# **Parameter Freezing Strategy**

Module 6 centralizes freezing behaviour.

## **Source Training**

Trainable

- Encoder
- Decoder
- Adapters
- Self-Reconstruction Branches
- Cross-Reconstruction Branches
- Fusion Module

## **Test-Time Adaptation**

Frozen

- Encoder
- Decoder
- Self-Reconstruction Branches
- Cross-Reconstruction Branches
- Fusion Module

Trainable

- Adaptive Reconstruction Adapters

## **Inference**

Entire network frozen.

# **Public API**

The assembled network exposes

network.forward_train(...)

network.forward_tta(...)

network.forward_inference(...)

Additionally,

network.freeze_for_tta()

automatically freezes every module except the adapters.

This avoids scattering

requires_grad = False

throughout the training code.

# **Output Contract**

Every forward method returns structured outputs.

Example

{

"segmentation": segmentation,

"adapted_latents": {

"t2": ...,

"adc": ...,

"dwi": ...

},

"self_outputs": {

...

},

"cross_outputs": {

...

},

"fused_latent": ...

}

Loss computation remains outside Module 6.

# **Codebase Impact**

## **Primary Files**

models/

network.py

forward_manager.py

model_factory.py

### **network.py**

Responsibilities

- Assemble complete architecture
- Instantiate every module
- Expose public forward API

### **forward_manager.py**

Responsibilities

- Define execution order
- Execute all forward modes
- Package outputs
- Coordinate module interaction

### **model_factory.py**

Responsibilities

Construct

- Backbone
- Adapters
- Reconstruction Branches
- Fusion Module

Return complete network.

# **Updated Files**

### **adaptation/freezing.py**

Provides

freeze_for_tta()

freeze_for_inference()

unfreeze_source_training()

### **training/trainer.py**

Uses

network.forward_train()

### **adaptation/adaptation_loop.py**

Uses

network.forward_tta()

### **inference/predictor.py**

Uses

network.forward_inference()

# **Unit Tests**

## **Assembly Test**

Verify every module is instantiated correctly.

## **Forward Test**

Verify complete forward pass executes successfully.

## **Shape Test**

Verify

Encoder

↓

Adapters

↓

Fusion

↓

Decoder

preserves tensor compatibility.

## **Output Contract Test**

Verify

every forward mode returns the expected dictionary.

## **Freezing Test**

Verify

Source Training

↓

Everything Trainable

Test-Time Adaptation

↓

Only Adapters Trainable

Inference

↓

Everything Frozen

## **Parallel Branch Test**

Verify

Self-Reconstruction

Cross-Reconstruction

Fusion

receive identical adapted latent representations.

## **Decoder Compatibility**

Verify

Fusion Output

↓

Shared Decoder

executes successfully.

# **Definition of Done (DoD)**

Module 6 is complete when

- All architectural modules are assembled into a single network.
- The execution order follows the finalized pipeline.
- Three dedicated forward modes (forward_train, forward_tta, forward_inference) are implemented.
- Reconstruction branches execute in parallel from the adapted latent representations.
- The Channel-Gated Fusion Module receives only adapted latent features.
- The shared decoder receives a single fused latent representation.
- Parameter freezing is centralized through helper functions.
- Loss computation is completely separated from the forward pass.
- The output contract is consistent across execution modes.
- All unit tests pass.

# **Scientific Validation**

By completion of Module 6, the framework should satisfy the following architectural statement:

**The Complete Network Assembly module provides a unified orchestration layer that integrates modality-specific adaptation, latent-space self-supervision, cross-modal reasoning, and channel-gated fusion into a single execution pipeline while maintaining strict separation of responsibilities between learning components and system control.**

Module7-SourceTraining

# **MODULE 7 - SOURCE TRAINING IMPLEMENTATION SPECIFICATION**

# **Module Objective**

Implement the complete source-domain training pipeline that learns the newly introduced adaptation modules while preserving the segmentation capability of the pretrained PI-CAI nnU-Net.

The source training stage has one objective:

Learn latent adaptation priors that will later supervise Test-Time Adaptation.

The nnU-Net segmentation backbone is **not retrained**.

# **Final Architecture State**

At the start of source training the network consists of

Pretrained nnUNet Encoder

↓

Adaptive Reconstruction Adapters

↓

Self Reconstruction Heads

↓

Cross Reconstruction Heads

↓

Channel Gated Fusion

↓

Pretrained nnUNet Decoder

Every module already exists.

This module implements only the optimization pipeline.

# **Parameter Initialization**

| **Component**              | **Initialization**               |
| -------------------------- | -------------------------------- |
| Encoder                    | Load pretrained baseline weights |
| ---                        | ---                              |
| Decoder                    | Load pretrained baseline weights |
| ---                        | ---                              |
| Adapters                   | Random (Kaiming Initialization)  |
| ---                        | ---                              |
| Self Reconstruction Heads  | Xavier Initialization            |
| ---                        | ---                              |
| Cross Reconstruction Heads | Xavier Initialization            |
| ---                        | ---                              |
| Fusion Module              | Xavier Initialization            |
| ---                        | ---                              |

No other initialization is required.

# **Frozen Parameters**

Freeze permanently

Encoder

Decoder

No gradients.

No optimizer.

No updates.

# **Trainable Parameters**

Adaptive Reconstruction Adapters

Self Reconstruction Heads

Cross Reconstruction Heads

Fusion Module

Only these parameters are registered inside AdamW.

# **Input**

Every iteration receives

T2

ADC

High-b DWI

Ground Truth Mask

Batch comes directly from Dataset Loader.

No preprocessing inside trainer.

# **Forward Pipeline**

Exactly

MRI

↓

Encoder

↓

ZT2

ZADC

ZDWI

↓

Adapters

↓

ZT2'

ZADC'

ZDWI'

↓

Self Reconstruction

↓

ZT2''

ZADC''

ZDWI''

↓

Cross Reconstruction

↓

Predict T2

Predict ADC

Predict DWI

↓

Fusion

↓

ZFused

↓

Decoder

↓

Segmentation

Execution order must never change.

# **Self Reconstruction**

Input

ZT2'

Output

ZT2''

Target

ZT2'

Same

ADC

Same

DWI

Loss

L1

No Decoder involved.

# **Cross Reconstruction**

Input

ZT2'

-

ZADC'

Output

Predict DWI'

Target

ZDWI'

Repeat

ZT2'

-

ZDWI'

↓

Predict ADC'

Repeat

ZADC'

-

ZDWI'

↓

Predict T2'

Loss

L1

# **Fusion**

Receives

ZT2'

ZADC'

ZDWI'

Returns

ZFused

Decoder receives only

ZFused

# **Segmentation**

Decoder predicts

Mask

Target

Ground Truth

# **Losses**

Segmentation

Dice

-

Adaptive Focal

Self

Average

L1

Cross

Average

L1

Total

Ltotal

\=

Dice

-

Adaptive Focal

-

λself Lself

-

λcross Lcross

Initial

λself

\=

0.1

λcross

\=

0.1

Configurable.

# **Gradient Flow**

## **Dice**

Updates

Fusion

↓

Adapters

Stops.

Never updates

Encoder

Decoder

## **Adaptive Focal**

Exactly identical.

## **Self Loss**

Updates

Self Heads

↓

Adapters

Nothing else.

## **Cross Loss**

Updates

Cross Heads

↓

Adapters

Nothing else.

# **Final Parameter Updates**

After summing gradients

Update

Adapters

Self Heads

Cross Heads

Fusion

Encoder

Decoder

must remain bit-identical.

# **Optimizer**

AdamW

Parameter Groups

Group 1

Adapters

Group 2

Self Heads

Group 3

Cross Heads

Group 4

Fusion

No encoder.

No decoder.

# **One Training Iteration**

Exactly

Load Batch

↓

Forward

↓

Compute Dice

↓

Compute Adaptive Focal

↓

Compute Self Loss

↓

Compute Cross Loss

↓

Sum Loss

↓

Backward

↓

Optimizer Step

↓

Next Batch

Nothing else.

# **Checkpoint**

Save

Adapters

Self Heads

Cross Heads

Fusion

Optimizer

Scheduler

Epoch

Do NOT save encoder/decoder again.

Baseline checkpoint already exists.

# **Validation**

Validation

does NOT update parameters.

Compute

Segmentation

Dice

PI-CAI Metrics

Self Loss

Cross Loss

# **Repository Files**

training/

trainer.py

Implements

training loop.

training/

optimizer.py

Creates

AdamW.

Registers trainable modules.

training/

checkpoint.py

Save

resume

checkpoint.

training/

validator.py

Validation.

losses/

dice.py

Dice.

losses/

adaptive_focal.py

Adaptive focal.

losses/

self_loss.py

Self.

losses/

cross_loss.py

Cross.

losses/

total_loss.py

Aggregates

all losses.

models/

network.py

Returns

{

segmentation,

adapted_latents,

self_predictions,

cross_predictions,

fused_latent

}

# **Unit Tests**

Must verify

✓ Encoder frozen

✓ Decoder frozen

✓ Optimizer contains only trainable modules

✓ Dice updates Fusion

✓ Dice updates Adapters

✓ Self updates Self Heads

✓ Self updates Adapters

✓ Cross updates Cross Heads

✓ Cross updates Adapters

✓ Encoder receives zero gradients

✓ Decoder receives zero gradients

✓ Forward shapes correct

✓ Checkpoint restores correctly

# **Definition of Done**

Module 7 is complete when

- Pretrained nnU-Net loads successfully.
- Encoder and decoder remain frozen for the entire training process.
- Adapters, self heads, cross heads, and fusion are initialized correctly.
- A complete forward pass executes without shape mismatches.
- Dice + Adaptive Focal + Self + Cross losses are computed in every iteration.
- Gradients reach only the intended trainable modules.
- The optimizer updates only the trainable modules.
- Checkpoint save/resume works.
- Validation runs independently of training.
- The trained checkpoint contains all adaptation modules required for deployment in Module 8.

Module8-TestAdaptaion

# **MODULE 8 - CONTINUAL TEST-TIME ADAPTATION IMPLEMENTATION SPECIFICATION**

# **Module Objective**

Implement the continual Test-Time Adaptation (TTA) algorithm that updates only the modality-specific adapters using self-supervised latent reconstruction objectives.

The objective of this module is **not** to improve segmentation using labels.

Instead,

the objective is

Adapt the latent representations to the new scanner/site while preserving the segmentation knowledge learned during source training.

No segmentation labels are available.

# **Inputs**

Incoming deployment stream

Hospital B

↓

Patient 1

↓

Patient 2

↓

Patient 3

↓

...

↓

Patient N

Each patient contains

T2

ADC

High-b DWI

Ground Truth

NOT AVAILABLE

# **Initial Network State**

Load

Pretrained nnUNet Encoder

↓

Source Trained Adapters

↓

Frozen Self Heads

↓

Frozen Cross Heads

↓

Frozen Fusion

↓

Pretrained Decoder

Exactly the checkpoint generated by Module 7.

# **Parameter State Before Adaptation**

| **Module**                      | **State** |
| ------------------------------- | --------- |
| Encoder                         | Frozen    |
| ---                             | ---       |
| Decoder                         | Frozen    |
| ---                             | ---       |
| Adaptive Reconstruction Adapter | Trainable |
| ---                             | ---       |
| Self Reconstruction Heads       | Frozen    |
| ---                             | ---       |
| Cross Reconstruction Heads      | Frozen    |
| ---                             | ---       |
| Fusion Module                   | Frozen    |
| ---                             | ---       |

# **Why Only Adapters Are Trainable**

Adapters are the only modules responsible for learning domain-specific latent corrections.

Everything else already represents the source-domain knowledge.

If reconstruction heads were updated,

their supervision would continuously change,

making reconstruction loss meaningless.

Therefore

they remain fixed

and become

**latent teachers.**

# **Deployment Session**

One deployment session corresponds to

one hospital.

Example

Hospital B

↓

Patient 1

↓

Update Adapter

↓

Patient 2

↓

Update Adapter

↓

Patient 3

↓

Update Adapter

↓

...

↓

Hospital Ends

Adapter weights persist throughout the session.

# **Hospital Transition**

When deployment moves

Hospital B

↓

Hospital C

Do

Reload Source Adapter Checkpoint

Do NOT continue adapting

Hospital B adapters.

# **Input Pipeline**

Each patient

MRI

↓

TTN

↓

Encoder

↓

Latent

No segmentation label

ever enters

the pipeline.

# **Forward Pipeline**

Exactly

Incoming MRI

↓

Frozen Encoder

↓

ZT2

ZADC

ZDWI

↓

Adapters

↓

ZT2'

ZADC'

ZDWI'

↓

Self Reconstruction

↓

ZT2''

ZADC''

ZDWI''

↓

Cross Reconstruction

↓

Predict T2'

Predict ADC'

Predict DWI'

↓

Compute Self Loss

↓

Compute Cross Loss

↓

Backward

↓

Update Adapter

↓

Repeat K iterations

↓

Final Forward Pass

↓

Fusion

↓

Frozen Decoder

↓

Final Segmentation

Execution order

must never change.

# **Adaptation Objective**

Segmentation Loss

NOT COMPUTED

Adaptive Focal

NOT COMPUTED

Only

Lself

-

Lcross

Total

LTTA

\=

λself

Lself

-

λcross

Lcross

Recommended

λself

\=

0.1

λcross

\=

0.1

Configurable.

# **Self Reconstruction**

Input

ZT2'

Prediction

ZT2''

Target

ZT2'

Loss

L1

Repeat

ADC

Repeat

DWI

Average

# **Cross Reconstruction**

Input

ZT2'

-

ZADC'

Prediction

Predict DWI'

Target

ZDWI'

Repeat

ZT2'

-

ZDWI'

↓

Predict ADC'

Repeat

ZADC'

-

ZDWI'

↓

Predict T2'

Average

# **Gradient Flow**

This section is the most important.

## **Self Reconstruction**

Lself

↓

Frozen Self Head

↓

Adapter

Updates

Adapter

Only.

Stops.

## **Cross Reconstruction**

Lcross

↓

Frozen Cross Head

↓

Adapter

Updates

Adapter

Only.

Stops.

## **Encoder**

Never updated.

Gradient

blocked.

## **Decoder**

Never updated.

Gradient

blocked.

## **Fusion**

Never updated.

Gradient

blocked.

## **Self Heads**

Never updated.

They act as

fixed latent priors.

## **Cross Heads**

Never updated.

They act as

fixed cross-modal priors.

# **Adaptation Iterations**

Each patient

runs

for

k

\=

1

...

K

Recommended

K

\=

10

Configurable.

Every iteration

Forward

↓

Compute

Lself

↓

Compute

Lcross

↓

LTTA

↓

Backward

↓

Update Adapter

# **Final Segmentation**

After

last

adapter update

perform

one clean

forward pass.

Updated Adapter

↓

Fusion

↓

Decoder

↓

Segmentation

Return

only

this segmentation.

# **Adapter Persistence**

Do NOT reset

after

Patient 1.

Instead

Patient 1

↓

Adapter Updated

↓

Patient 2

↓

Continue

↓

Patient 3

↓

Continue

↓

...

↓

Hospital Ends

# **Session Manager**

Responsible for

Hospital Start

↓

Load Source Adapter

↓

Adapt Patients

↓

Hospital End

↓

Save Optional Logs

↓

Discard Session

# **Optimizer**

Only

Adapter

registered.

AdamW.

No parameter groups

for

anything else.

# **One Adaptation Iteration**

Exactly

Load Patient

↓

Forward

↓

Compute Self Loss

↓

Compute Cross Loss

↓

Compute LTTA

↓

Backward

↓

Optimizer Step

↓

Repeat

# **End of Patient**

Run

one additional

forward

without

gradient

Return segmentation.

# **Repository Files**

adaptation/

adaptation_loop.py

Main adaptation loop.

adaptation/

session_manager.py

Handles

hospital lifecycle.

adaptation/

optimizer.py

Creates

Adapter optimizer.

adaptation/

freeze.py

Freezes

everything

except

Adapter.

adaptation/

checkpoint.py

Loads

source-trained

adapter checkpoint.

models/

network.py

Exposes

forward_tta()

# **Unit Tests**

Must verify

✓ Encoder frozen

✓ Decoder frozen

✓ Fusion frozen

✓ Self Heads frozen

✓ Cross Heads frozen

✓ Adapter trainable

✓ Optimizer contains only Adapter

✓ Self Loss updates Adapter

✓ Cross Loss updates Adapter

✓ Forward pass correct

✓ Adaptation iterations execute correctly

✓ Final segmentation uses updated Adapter

✓ Session manager preserves adapter state

✓ New hospital reloads source checkpoint

# **Failure Cases**

## **Wrong Freezing**

Encoder receives gradients.

Immediate failure.

## **Reconstruction Collapse**

Loss becomes

NaN.

Stop adaptation.

## **Adapter Drift**

Loss increases

continuously.

Trigger warning.

## **Wrong Session**

Hospital changes

but

checkpoint

not reloaded.

Invalid experiment.

## **Shape Mismatch**

Latent tensor

must always remain

320 × 4 × 5 × 5

No modification

allowed.

# **Logging**

For every patient

store

Patient ID

Self Loss

Cross Loss

Total Loss

Adaptation Steps

Inference Time

No segmentation metrics

during deployment.

# **Outputs**

For every patient

return

{

"segmentation": segmentation,

"adapter_loss": LTTA,

"self_loss": Lself,

"cross_loss": Lcross,

"adaptation_steps": K

}

# **Definition of Done**

Module 8 is complete when

- Module 7 checkpoint loads successfully.
- Encoder remains frozen throughout deployment.
- Decoder remains frozen throughout deployment.
- Fusion module remains frozen.
- Self Reconstruction Heads remain frozen.
- Cross Reconstruction Heads remain frozen.
- Only adapters are registered in the optimizer.
- Self Reconstruction Loss updates only adapter parameters.
- Cross Reconstruction Loss updates only adapter parameters.
- The total adaptation objective is computed as:  
   <br/>LTTA = λself × Lself + λcross × Lcross

- K adaptation iterations execute successfully for every patient.
- A final inference pass is performed after the last adapter update.
- The segmentation returned corresponds to the final adapted adapter state.
- Adapter parameters persist across patients within the same hospital session.
- The source-trained adapter checkpoint is reloaded only when a new hospital session begins.
- Adaptation logs are saved for every patient.

Tab 11
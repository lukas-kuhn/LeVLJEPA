# LeVLJEPA

Official implementation of **LeVLJEPA** and **LeVLJEPA+**, non-contrastive methods for vision-language pretraining based on joint-embedding prediction and SIGReg regularization.

LeVLJEPA replaces contrastive image-text alignment with cross-modal prediction using stop-gradient targets and per-modality isotropic Gaussian regularization. LeVLJEPA+ extends this framework with visual multi-view prediction, adding DINO-style global/local crops to improve frozen vision representations.

## Overview

Vision-language pretraining is typically dominated by contrastive objectives such as CLIP and SigLIP, which rely on negative pairs and benefit strongly from large batch sizes. LeVLJEPA explores whether non-contrastive joint-embedding prediction can provide a viable alternative for paired image-caption data.

The core idea is to align image and text embeddings through predictive losses rather than contrastive classification. To avoid collapse, each modality is regularized independently with SIGReg, which encourages the marginal embedding distribution to match an isotropic Gaussian. LeVLJEPA+ further adds visual multi-view prediction, encouraging multiple augmented views of the same image to agree in representation space.

## Method

### LeVLJEPA (`train_levljepa.py`)

The single-view variant trains an image encoder and a text encoder with two components:

1. **Cross-modal prediction:** image embeddings predict stop-gradient text embeddings, and text embeddings predict stop-gradient image embeddings through modality-specific MLP predictors.
2. **SIGReg regularization:** vision and text embeddings are independently regularized toward isotropic Gaussian marginals to prevent collapse.

The objective has no negatives, no temperature parameter, no momentum encoder, and no teacher-student schedule.

### LeVLJEPA+ (`train_levljepa_plus.py`)

The multi-view variant extends LeVLJEPA with visual multi-view prediction. For each image, DINO-style global and local crops are generated. The model adds a consistency loss that encourages embeddings from all views of the same image to match the mean global-view representation.

LeVLJEPA+ uses the same cross-modal prediction and SIGReg losses as LeVLJEPA, while adding this visual multi-view term to improve frozen vision features.

## Installation

This repository uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if you do not have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Usage

Edit the paths in the relevant config file before launching:

- `output_dir`: where checkpoints and logs are written
- `cache_dir`: local dataset/cache directory
- `hf_bucket`: optional HuggingFace checkpoint upload target

Launch training with `torchrun`:

```bash
# LeVLJEPA: single-view non-contrastive VLP
torchrun --nproc_per_node=NUM_GPUS train_levljepa.py

# LeVLJEPA+: multi-view non-contrastive VLP
torchrun --nproc_per_node=NUM_GPUS train_levljepa_plus.py
```

Override config values from the command line:

```bash
torchrun --nproc_per_node=8 train_levljepa.py \
    run_name=my_run \
    batch_size=256 \
    model=small
```

For LeVLJEPA+:

```bash
torchrun --nproc_per_node=8 train_levljepa_plus.py \
    run_name=my_multiview_run \
    batch_size=256 \
    model=base \
    lambda_multi=0.2
```

## Configuration

All hyperparameters are managed with [Hydra](https://hydra.cc/).

| Config file | Script | Description |
|------------|--------|-------------|
| `configs/levljepa.yaml` | `train_levljepa.py` | Single-view LeVLJEPA |
| `configs/levljepa_plus.yaml` | `train_levljepa_plus.py` | Multi-view LeVLJEPA+ |

Model size is selected with:

```bash
model=tiny
model=small
model=base
```

The default model is `base`.

Important LeVLJEPA/LeVLJEPA+ hyperparameters include:

| Hyperparameter | Description |
|---------------|-------------|
| `lambda_vision` | SIGReg weight for vision embeddings |
| `lambda_text` | SIGReg weight for text embeddings |
| `lambda_multi` | Multi-view prediction weight, LeVLJEPA+ only |
| `projector_width` | Width of projection / prediction MLPs |
| `projector_depth` | Depth of projection / prediction MLPs |
| `global_crops_number` | Number of global image crops, LeVLJEPA+ only |
| `local_crops_number` | Number of local image crops, LeVLJEPA+ only |

## Datasets

Training uses [CC12M](https://huggingface.co/datasets/pixparse/cc12m-wds). Zero-shot evaluation uses [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Datasets are streamed from HuggingFace. Set `cache_dir` in the config to cache data locally.

## Evaluation

The training scripts include periodic ImageNet zero-shot evaluation. Additional downstream evaluations used in the paper include:

- zero-shot classification
- linear probing
- MLP probing
- ADE20K semantic segmentation
- NYUv2 monocular depth estimation
- ImageNet-9 background robustness
- FGVC-Aircraft fine-grained transfer

Evaluation scripts will be released in this repository together with the camera-ready version.

## Results

Main results from the paper, using ViT-B/16 trained on CC12M for 50k steps:

| Method | Objective | Multi-view | Avg. Linear Probe |
|--------|-----------|------------|-------------------|
| CLIP | Contrastive | No | 48.8 |
| SigLIP | Contrastive | No | 49.4 |
| LeVLJEPA | Non-contrastive | No | 47.6 |
| CLIP-Rocket | Contrastive | Yes | 57.5 |
| LeVLJEPA+ | Non-contrastive | Yes | 57.5 |

## Checkpoints

Checkpoints are not included in the repository. If released separately, links will be added here.

## Citation

```bibtex
@inproceedings{anonymous2026levljepa,
  title     = {LeVLJEPA: Non-Contrastive Joint-Embedding Prediction for Vision-Language Pretraining},
  author    = {Anonymous Authors},
  booktitle = {NeurIPS},
  year      = {2026}
}
```
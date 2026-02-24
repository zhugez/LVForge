# Neuro-Biometrics

PE Malware Detection using Custom Transformers with Deep Metric Learning.

## Project Structure

```
Neuro-Biometrics/
├── src/pe_malware/
│   ├── config/          # Training configuration
│   ├── data/            # Data loading, tokenization, preprocessing
│   ├── models/
│   │   ├── pytorch/     # PyTorch LVModel
│   │   └── flax/        # Flax models (base + metric learning variants)
│   ├── training/        # Trainers and loss functions
│   └── evaluation/      # Metrics, plotting, multi-seed analysis
├── scripts/
│   ├── train_pytorch.py # PyTorch training entry point
│   └── train_flax.py    # Flax training entry point (all loss modes)
└── pyproject.toml
```

## Supported Models

| Model | Loss | Description |
|-------|------|-------------|
| LVModel (PyTorch) | Cross-Entropy | Baseline Transformer classifier |
| FlaxLVModel | Cross-Entropy / Focal | Flax baseline with `lax.scan` optimization |
| FlaxLVModelWithArcFace | ArcFace | Angular-margin metric learning |
| FlaxLVModelWithContrastive | Contrastive | Pairwise contrastive loss |
| FlaxLVModelWithTriplet | Triplet | Batch-hard triplet mining |
| FlaxLVModelWithMultiSimilarity | Multi-Similarity | MS loss with hard pair mining |

## Usage

### PyTorch

```bash
python scripts/train_pytorch.py
```

### Flax (select loss mode)

```bash
# Baseline (cross-entropy / focal loss)
python scripts/train_flax.py --loss baseline

# Deep metric learning variants
python scripts/train_flax.py --loss arcface
python scripts/train_flax.py --loss contrastive
python scripts/train_flax.py --loss triplet
python scripts/train_flax.py --loss multi_similarity
```

## Installation

```bash
pip install -e .
```

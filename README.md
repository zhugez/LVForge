# PE Malware Metric Learning

PE Malware Detection using Custom Transformers with Deep Metric Learning (JAX/Flax).

## Project Structure

```
pe-malware-metric-learning/
├── src/pe_malware/
│   ├── config/          # Training configuration
│   ├── data/            # Data loading, tokenization, preprocessing
│   ├── models/          # Flax models (base + metric learning variants)
│   ├── training/        # Trainers and loss functions
│   └── evaluation/      # Metrics, plotting, multi-seed analysis
├── scripts/
│   └── train_flax.py    # Training entry point (all loss modes)
└── pyproject.toml
```

## Supported Models

| Model | Loss | Description |
|-------|------|-------------|
| FlaxLVModel | Cross-Entropy / Focal | Baseline Transformer classifier |
| FlaxLVModelWithArcFace | ArcFace | Angular-margin metric learning |
| FlaxLVModelWithContrastive | Contrastive | Pairwise contrastive loss |
| FlaxLVModelWithTriplet | Triplet | Batch-hard triplet mining |
| FlaxLVModelWithMultiSimilarity | Multi-Similarity | MS loss with hard pair mining |

## Usage

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

## Requirements

- Python >= 3.9
- JAX with CUDA support
- Flax, Optax
- HuggingFace Transformers (tokenizer only)
- scikit-learn, scipy, pandas, numpy, plotly

## License

MIT

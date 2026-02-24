# PE Malware Metric Learning

PE Malware Detection using Custom Transformers with Deep Metric Learning (JAX/Flax).

## Project Structure

```
LVForge/
├── src/pe_malware/
│   ├── config/          # Training configuration
│   ├── data/            # Data loading, tokenization, preprocessing
│   ├── models/          # Flax models (base + metric learning variants)
│   ├── training/        # Trainers and loss functions
│   └── evaluation/      # Metrics, plotting, multi-seed analysis
├── scripts/
│   ├── train_flax.py    # Training entry point (single loss mode)
│   └── run_all.py       # Run ALL experiments in one command
├── backup_full.py       # Backup weights + Google Drive upload (gogcli)
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
# Run ALL 5 experiments in one command
python scripts/run_all.py

# Run selected experiments only
python scripts/run_all.py --only arcface triplet

# Run all + backup weights to Google Drive
python scripts/run_all.py --backup --gdrive --account you@gmail.com

# Or run individual experiments
python scripts/train_flax.py --loss baseline
python scripts/train_flax.py --loss arcface
python scripts/train_flax.py --loss contrastive
python scripts/train_flax.py --loss triplet
python scripts/train_flax.py --loss multi_similarity
```

## Backup & Google Drive Upload

Uses [gogcli](https://github.com/steipete/gogcli) to upload experiment weights to Google Drive.

```bash
# Zip weights only
python backup_full.py

# Zip + upload to Google Drive
python backup_full.py --gdrive --account you@gmail.com

# Upload to a specific folder
python backup_full.py --gdrive --account you@gmail.com --folder-id FOLDER_ID
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

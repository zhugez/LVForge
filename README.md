# LVForge - PE Malware Metric Learning

PE Malware Detection using Custom Transformers with Deep Metric Learning (JAX/Flax).

## Quick Start (Full Pipeline)

```bash
# 1. Setup environment with uv
uv venv --python 3.12 .venv
source .venv/bin/activate      # Linux/WSL
# .venv\Scripts\activate       # Windows

# 2. Install dependencies
uv pip install -e ".[dev]"

# 3. Place dataset
# Required: finData.csv in project root
# Expected columns: "Texts" (PE string features), "label" (0=benign, 1=malware)

# 4. Run ALL 5 experiments in one command
python scripts/run_all.py

# 5. Backup results to Google Drive
python backup_full.py --gdrive --account your@gmail.com
```

## Project Structure

```
LVForge/
├── src/pe_malware/
│   ├── config/settings.py     # TrainingConfig dataclass (all hyperparams)
│   ├── data/
│   │   ├── tokenizer.py       # CustomDistilBertTokenizer
│   │   ├── preprocessor.py    # Data preprocessing & JAX conversion
│   │   └── sampling.py        # Imbalanced data subsampling
│   ├── models/
│   │   ├── transformer.py     # Base Flax Transformer blocks
│   │   ├── lv_model.py        # FlaxLVModel (baseline)
│   │   ├── arcface.py         # FlaxLVModelWithArcFace
│   │   ├── contrastive.py     # FlaxLVModelWithContrastive
│   │   ├── triplet.py         # FlaxLVModelWithTriplet
│   │   └── multi_similarity.py # FlaxLVModelWithMultiSimilarity
│   ├── training/
│   │   ├── flax_trainer.py    # Baseline trainer (CE/focal loss)
│   │   ├── flax_metric_trainer.py  # Metric learning trainer
│   │   └── losses.py          # Loss function implementations
│   └── evaluation/
│       ├── metrics.py         # evaluate_model(), evaluate_multi_seed()
│       └── plotting.py        # Plotly visualization
├── scripts/
│   ├── train_flax.py          # Train single experiment: --loss <variant>
│   └── run_all.py             # Run ALL experiments in one command
├── backup_full.py             # Zip outputs + Google Drive upload (gogcli)
├── finData.csv                # Dataset (not tracked in git)
├── client_secret.json         # Google OAuth secret (not tracked in git)
├── pyproject.toml             # Package config & dependencies
└── .wslrun.sh                 # WSL helper (clean PATH + activate venv)
```

## Dataset

- **File**: `finData.csv` (place in project root)
- **Columns**: `Texts` (PE string features), `label` (0=benign, 1=malware)
- **Source**: `E:\Test\Test\PMalwareData2.0\finData.csv`
- **Not tracked in git** (`.gitignore` excludes `*.csv`)

## Supported Models (5 Variants)

| Variant | Model Class | Loss | Description |
|---------|------------|------|-------------|
| `baseline` | FlaxLVModel | Cross-Entropy / Focal | Standard Transformer classifier |
| `arcface` | FlaxLVModelWithArcFace | ArcFace | Angular-margin metric learning (margin=0.5, scale=64) |
| `contrastive` | FlaxLVModelWithContrastive | Contrastive | Pairwise contrastive loss (margin=1.0) |
| `triplet` | FlaxLVModelWithTriplet | Triplet | Batch-hard triplet mining (margin=0.3) |
| `multi_similarity` | FlaxLVModelWithMultiSimilarity | Multi-Similarity | MS loss with hard pair mining (alpha=2.0, beta=50) |

## Training Configuration

All hyperparameters are in `src/pe_malware/config/settings.py` (TrainingConfig dataclass):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | `finData.csv` | Path to dataset |
| `embed_dim` | 256 | Transformer embedding dimension |
| `num_heads` | 8 | Multi-head attention heads |
| `ff_dim` | 512 | Feed-forward dimension |
| `num_layers` | 2 | Transformer layers |
| `num_classes` | 2 | Output classes (benign/malware) |
| `dropout_rate` | 0.1 | Dropout rate |
| `embedding_dim` | 256 | Metric learning embedding dim |
| `batch_size` | 128 | Training batch size |
| `num_epochs` | 5 | Max training epochs |
| `patience` | 2 | Early stopping patience |
| `learning_rate` | 2e-4 | Adam learning rate |
| `use_focal_loss` | True | Use focal loss for imbalanced data |
| `benign_ratio` | 0.05 | Benign subsampling ratio |

## Usage

### Run All Experiments (Recommended)

```bash
# Run all 5 experiments sequentially with summary
python scripts/run_all.py

# Run only selected variants
python scripts/run_all.py --only arcface triplet

# Run all + backup to Google Drive after completion
python scripts/run_all.py --backup --gdrive --account lyngocvuteakwondo@gmail.com

# Run all + backup to specific Drive folder
python scripts/run_all.py --backup --gdrive --account lyngocvuteakwondo@gmail.com --folder-id FOLDER_ID
```

Output: prints PASS/FAIL summary for each experiment with timing.

### Run Individual Experiment

```bash
python scripts/train_flax.py --loss baseline
python scripts/train_flax.py --loss arcface
python scripts/train_flax.py --loss contrastive
python scripts/train_flax.py --loss triplet
python scripts/train_flax.py --loss multi_similarity
```

### Evaluation Output

Each experiment automatically runs:
- `evaluate_model()` - accuracy, precision, recall, F1, confusion matrix
- `evaluate_multi_seed()` - multi-seed stability analysis
- Results saved with suffix `flax_<variant>` (e.g., `flax_arcface`)

## Backup & Google Drive Upload

Uses [gogcli](https://github.com/steipete/gogcli) to upload experiment outputs to Google Drive.

### Setup gogcli

```bash
# Linux/WSL
curl -sL https://github.com/steipete/gogcli/releases/latest/download/gogcli_0.11.0_linux_amd64.tar.gz | tar xz -C /usr/local/bin gog
```

### Backup Commands

```bash
# Zip weights/outputs only (no upload)
python backup_full.py

# Zip + upload to Google Drive
python backup_full.py --gdrive --account lyngocvuteakwondo@gmail.com

# With specific folder
python backup_full.py --gdrive --account lyngocvuteakwondo@gmail.com --folder-id FOLDER_ID

# With custom client secret path
python backup_full.py --gdrive --account lyngocvuteakwondo@gmail.com --client-secret client_secret.json
```

### What Gets Backed Up

- `checkpoints/` - model checkpoints (`.msgpack`, `.pkl`)
- `weights/` - trained weights (`.pth`, `.pt`)
- `outputs/` - evaluation results (`.json`, `.html`)
- `README.md`, `pyproject.toml`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOG_KEYRING_PASSWORD` | `neuro2024` | gogcli keyring password |

Can also be set via `.env` file in project root.

## WSL Helper

The `.wslrun.sh` script provides a clean environment for running commands in WSL (avoids Windows PATH pollution):

```bash
# From Windows terminal
wsl bash //mnt/e/Test/Test/LVForge/.wslrun.sh python scripts/run_all.py
wsl bash //mnt/e/Test/Test/LVForge/.wslrun.sh python backup_full.py --gdrive --account lyngocvuteakwondo@gmail.com
```

## Installation (Detailed)

```bash
# With uv (recommended)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# With pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Dependencies

- **Core**: jax[cuda12]>=0.4, flax>=0.8, optax>=0.2
- **Data**: transformers>=4.30, pandas>=2.0, numpy>=1.24
- **Evaluation**: scikit-learn>=1.3, scipy>=1.11, plotly>=5.15
- **Dev**: pytest>=7.0, ruff>=0.1

## GitHub Repository

- **URL**: https://github.com/zhugez/LVForge
- **Push**: `git push -u origin main` (uses HTTPS via `gh auth setup-git`)

## License

MIT

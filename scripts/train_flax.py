#!/usr/bin/env python
"""Train a Flax model for PE malware detection.

Supports multiple loss modes:
    --loss baseline    : standard cross-entropy / focal loss
    --loss arcface     : ArcFace angular-margin metric learning
    --loss contrastive : contrastive loss
    --loss triplet     : triplet loss (batch-hard mining)
    --loss multi_similarity : multi-similarity loss
"""

import argparse
import warnings
import logging

import jax
import jax.numpy as jnp
import pandas as pd

from pe_malware.config import TrainingConfig
from pe_malware.data import CustomDistilBertTokenizer, Preprocessor, subsample_imbalanced_data
from pe_malware.models import (
    FlaxLVModel,
    FlaxLVModelWithArcFace,
    FlaxLVModelWithContrastive,
    FlaxLVModelWithTriplet,
    FlaxLVModelWithMultiSimilarity,
)
from pe_malware.training.flax_trainer import FlaxTrainer
from pe_malware.training.flax_metric_trainer import FlaxMetricTrainer
from pe_malware.evaluation import evaluate_model, evaluate_multi_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'baseline': FlaxLVModel,
    'arcface': FlaxLVModelWithArcFace,
    'contrastive': FlaxLVModelWithContrastive,
    'triplet': FlaxLVModelWithTriplet,
    'multi_similarity': FlaxLVModelWithMultiSimilarity,
}


def main():
    parser = argparse.ArgumentParser(description="Train Flax PE malware detector.")
    parser.add_argument("--loss", default="baseline",
                        choices=list(MODEL_CLASSES.keys()),
                        help="Loss / model variant to train.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    jax.config.update("jax_enable_x64", False)

    devices = jax.devices()
    device_type = "GPU" if any(d.platform == "gpu" for d in devices) else "CPU"
    logger.info(f"Training on {device_type}: {devices}")

    config = TrainingConfig()

    # 1. Load & preprocess data
    data = pd.read_csv(config.data_path)
    tokenizer = CustomDistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    preprocessor = Preprocessor(tokenizer=tokenizer, data_list=data["Texts"].values, labels=data["label"].values)

    tokenized_texts, labels_np, vocab, tokenized_data_np, padding_length = preprocessor.prepare_all_data(config.percentile)
    tokenized_data_jax, labels_jax = preprocessor.to_jax(tokenized_data_np, labels_np)

    # Subsample for imbalanced dataset if needed
    if config.benign_ratio < 0.5:
        tokenized_data_jax, labels_jax = subsample_imbalanced_data(
            tokenized_data_jax, labels_jax, benign_ratio=config.benign_ratio)

    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Padding length: {padding_length}")
    logger.info(f"Data shape: {tokenized_data_jax.shape}")

    # 2. Create model
    model_cls = MODEL_CLASSES[args.loss]
    model_kwargs = dict(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        max_seq_len=padding_length,
        dropout_rate=config.dropout_rate,
    )
    if args.loss != 'baseline':
        model_kwargs['embedding_dim'] = config.embedding_dim

    if args.loss == 'arcface':
        model_kwargs.update(arcface_margin=config.arcface_margin, arcface_scale=config.arcface_scale)
    elif args.loss == 'contrastive':
        model_kwargs['contrastive_margin'] = config.contrastive_margin
    elif args.loss == 'triplet':
        model_kwargs['triplet_margin'] = config.triplet_margin
    elif args.loss == 'multi_similarity':
        model_kwargs.update(ms_alpha=config.ms_alpha, ms_beta=config.ms_beta,
                            ms_lambda=config.ms_lambda, ms_margin=config.ms_margin)

    model = model_cls(**model_kwargs)

    # 3. Init params
    key = jax.random.PRNGKey(42)
    dummy = jnp.ones((1, padding_length), dtype=jnp.int32)
    params = model.init(key, dummy, train=False)['params']
    logger.info("Model initialized successfully")

    # 4. Create trainer
    num_train = int(0.8 * len(labels_jax))

    if args.loss == 'baseline':
        trainer = FlaxTrainer(
            model=model, params=params, device=devices[0],
            num_epochs=config.num_epochs, patience=config.patience,
            lr=config.learning_rate, batch_size=config.batch_size,
            num_train_samples=num_train)
        trainer.train(tokenized_data_jax, labels_jax, vocab, tokenizer, padding_length,
                      use_focal_loss=config.use_focal_loss)
    else:
        loss_kwargs = {}
        if args.loss == 'arcface':
            loss_kwargs['use_focal_loss'] = False
        elif args.loss == 'contrastive':
            loss_kwargs.update(margin=config.contrastive_margin, alpha=0.5)
        elif args.loss == 'triplet':
            loss_kwargs.update(margin=config.triplet_margin, alpha=0.5, use_hard_mining=True)
        elif args.loss == 'multi_similarity':
            loss_kwargs.update(ms_alpha=config.ms_alpha, ms_beta=config.ms_beta,
                               ms_lambda=config.ms_lambda, ms_margin=config.ms_margin, ce_weight=0.5)

        trainer = FlaxMetricTrainer(
            model=model, params=params, device=devices[0],
            loss_mode=args.loss,
            num_epochs=config.num_epochs, patience=config.patience,
            lr=config.learning_rate, batch_size=config.batch_size,
            num_train_samples=num_train, **loss_kwargs)
        trainer.train(tokenized_data_jax, labels_jax, vocab, tokenizer, padding_length)

    # 5. Evaluate
    suffix = f"flax_{args.loss}"
    evaluate_model(trainer, tokenized_data_jax, labels_jax, batch_size=config.batch_size, suffix=suffix)
    evaluate_multi_seed(trainer, tokenized_data_jax, labels_jax, batch_size=config.batch_size, suffix=suffix)

    logger.info("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train the PyTorch LVModel for PE malware detection."""

import warnings
import logging

import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split

from pe_malware.config import TrainingConfig
from pe_malware.data import CustomDistilBertTokenizer, Preprocessor, TextDataset
from pe_malware.models.pytorch import LVModel
from pe_malware.training.pytorch_trainer import PyTorchTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    warnings.filterwarnings("ignore")
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(config.data_path)
    tokenizer = CustomDistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    preprocessor = Preprocessor(tokenizer=tokenizer, data_list=data["Texts"].values, labels=data["label"].values)

    tokenized_texts, labels, vocab, tokenized_data, padding_length = preprocessor.prepare_all_data(config.percentile)

    dataset = TextDataset(tokenized_data, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = LVModel(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2,
                            pin_memory=True, num_workers=2)

    trainer = PyTorchTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config.num_epochs,
        patience=config.patience,
    )
    trainer.train(vocab, tokenizer, padding_length)


if __name__ == "__main__":
    main()

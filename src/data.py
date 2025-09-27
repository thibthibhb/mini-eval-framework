from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import numpy as np


class IMDbDataModule:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.dataset = None
        self.train_dataloader = None
        self.eval_dataloader = None

    def prepare_data(self):
        self.dataset = load_dataset(self.config.dataset_name)

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt"
        )

    def setup(self):
        if self.dataset is None:
            self.prepare_data()

        # Apply data subset if specified (for faster training)
        train_dataset = self.dataset["train"]
        test_dataset = self.dataset["test"]

        if hasattr(self.config, 'data_subset') and self.config.data_subset < 1.0:
            subset_size = int(len(train_dataset) * self.config.data_subset)
            train_dataset = train_dataset.select(range(subset_size))

            test_subset_size = int(len(test_dataset) * self.config.data_subset)
            test_dataset = test_dataset.select(range(test_subset_size))

            print(f"Using {self.config.data_subset*100:.1f}% of data: {len(train_dataset)} train, {len(test_dataset)} test samples")

        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        test_dataset = test_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            pin_memory=True
        )

        self.eval_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            pin_memory=True
        )

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.eval_dataloader

    def get_num_classes(self):
        return 2

    def get_class_names(self):
        return ["negative", "positive"]
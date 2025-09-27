import yaml
import torch
import numpy as np
import random
from pathlib import Path
import mlflow
import mlflow.pytorch
from types import SimpleNamespace


def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Ensure numeric values are properly converted
    numeric_fields = ['learning_rate', 'weight_decay', 'warmup_ratio']
    for field in numeric_fields:
        if field in config_dict and isinstance(config_dict[field], str):
            config_dict[field] = float(config_dict[field])

    return SimpleNamespace(**config_dict)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_output_dir(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def setup_mlflow(config):
    mlflow.set_experiment("imdb_sentiment_classification")
    mlflow.start_run()

    mlflow.log_params({
        "model_name": config.model_name,
        "max_length": config.max_length,
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "seed": config.seed
    })


def log_metrics(metrics, step=None):
    if step is not None:
        mlflow.log_metrics(metrics, step=step)
    else:
        mlflow.log_metrics(metrics)


def save_model_mlflow(model, tokenizer):
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="distilbert_imdb_classifier"
    )


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
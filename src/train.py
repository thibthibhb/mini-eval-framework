import os
import argparse
from pathlib import Path
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import mlflow

from data import IMDbDataModule
from model import create_model
from utils import (
    load_config, set_seed, create_output_dir, setup_mlflow,
    log_metrics, save_model_mlflow, get_device, AverageMeter
)


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    model.train()
    losses = AverageMeter()

    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item())

        if step % config.log_steps == 0:
            progress_bar.set_postfix({'loss': f'{losses.avg:.4f}'})
            log_metrics({'train_loss': losses.avg}, step=step)

    return losses.avg


def evaluate_model(model, dataloader, device):
    model.eval()
    losses = AverageMeter()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            predictions = outputs['predictions']

            losses.update(loss.item())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)

    return {
        'eval_loss': losses.avg,
        'eval_accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels
    }


def train_model(config):
    set_seed(config.seed)
    device = get_device()
    create_output_dir(config.output_dir)

    setup_mlflow(config)

    data_module = IMDbDataModule(config)
    data_module.setup()

    model = create_model(config)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(data_module.train_dataloader) * config.num_train_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_accuracy = 0

    for epoch in range(config.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_train_epochs}")

        train_loss = train_epoch(
            model, data_module.train_dataloader, optimizer, scheduler, device, config
        )

        eval_results = evaluate_model(model, data_module.eval_dataloader, device)

        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_results['eval_loss'],
            'eval_accuracy': eval_results['eval_accuracy']
        }

        log_metrics(metrics)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Eval Loss: {eval_results['eval_loss']:.4f}")
        print(f"Eval Accuracy: {eval_results['eval_accuracy']:.4f}")

        if eval_results['eval_accuracy'] > best_accuracy:
            best_accuracy = eval_results['eval_accuracy']
            model_path = os.path.join(config.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")

    model.load_state_dict(torch.load(os.path.join(config.output_dir, 'best_model.pt')))
    save_model_mlflow(model, data_module.tokenizer)

    mlflow.end_run()

    return model, data_module


def main():
    parser = argparse.ArgumentParser(description='Train DistilBERT for IMDb sentiment classification')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')

    args = parser.parse_args()

    config = load_config(args.config)

    print("Starting training with config:")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.train_batch_size}")
    print(f"Learning rate: {config.learning_rate}")

    model, data_module = train_model(config)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
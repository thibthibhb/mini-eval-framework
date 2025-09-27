import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from pathlib import Path

from data import IMDbDataModule
from model import create_model
from utils import load_config, set_seed, get_device


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cm


def evaluate_model_detailed(model, dataloader, device, class_names, output_dir):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            predictions = outputs['predictions']
            probabilities = model.predict_proba(input_ids, attention_mask)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    cm = plot_confusion_matrix(all_labels, all_predictions, class_names, output_dir)

    print("\nConfusion Matrix:")
    print(f"True Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")

    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, labels=[i], average=None
        )
        print(f"{class_name.capitalize():>8}: Precision={class_precision[0]:.4f}, Recall={class_recall[0]:.4f}, F1={class_f1[0]:.4f}")

    # Analyze prediction confidence
    all_probabilities = np.array(all_probabilities)
    confidence_scores = np.max(all_probabilities, axis=1)

    print(f"\nPrediction Confidence:")
    print(f"Mean confidence: {np.mean(confidence_scores):.4f}")
    print(f"Std confidence:  {np.std(confidence_scores):.4f}")
    print(f"Min confidence:  {np.min(confidence_scores):.4f}")
    print(f"Max confidence:  {np.max(confidence_scores):.4f}")

    # Low confidence predictions
    low_conf_threshold = 0.6
    low_conf_indices = np.where(confidence_scores < low_conf_threshold)[0]
    print(f"\nPredictions with confidence < {low_conf_threshold}: {len(low_conf_indices)} ({len(low_conf_indices)/len(all_predictions)*100:.1f}%)")

    # Error analysis
    incorrect_predictions = np.array(all_predictions) != np.array(all_labels)
    incorrect_indices = np.where(incorrect_predictions)[0]

    if len(incorrect_indices) > 0:
        print(f"\nError Analysis:")
        print(f"Total errors: {len(incorrect_indices)} ({len(incorrect_indices)/len(all_predictions)*100:.1f}%)")

        # Errors by confidence
        error_confidences = confidence_scores[incorrect_indices]
        print(f"Mean confidence of errors: {np.mean(error_confidences):.4f}")

        high_conf_errors = np.sum(error_confidences > 0.8)
        print(f"High-confidence errors (>0.8): {high_conf_errors}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DistilBERT model with confusion matrix')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')

    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    device = get_device()

    # Setup data
    data_module = IMDbDataModule(config)
    data_module.setup()

    # Load model
    model = create_model(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    class_names = data_module.get_class_names()

    print(f"Evaluating model: {args.model_path}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Model: {config.model_name}")
    print(f"Device: {device}")

    # Run evaluation
    results = evaluate_model_detailed(
        model, data_module.eval_dataloader, device, class_names, config.output_dir
    )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
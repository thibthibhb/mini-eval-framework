# Mini Eval Framework

A minimal, plug-and-play framework for sentiment analysis using **DistilBERT** and the **IMDb dataset** with **MLflow tracking**.

Perfect for fast experiments on RTX GPUs with 25k train/test samples, balanced positive/negative sentiment classification.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Training

```bash
# Using the provided script
bash scripts/run_train.sh

# Or directly
python src/train.py --config configs/default.yaml
```

### 3. Evaluate Model

```bash
python src/eval_confusion.py \
  --config configs/default.yaml \
  --model_path artifacts/distilbert_imdb/best_model.pt
```

### 4. View Results

```bash
# Launch MLflow UI
mlflow ui
```

Open http://localhost:5000 to view training metrics and model artifacts.

## 📁 Project Structure

```
mini-eval-framework/
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ default.yaml          # Training configuration
├─ src/
│  ├─ data.py              # IMDb dataset handling
│  ├─ model.py             # DistilBERT classifier
│  ├─ train.py             # Training pipeline
│  ├─ eval_confusion.py    # Evaluation with confusion matrix
│  └─ utils.py             # Utilities and MLflow setup
└─ scripts/
   └─ run_train.sh          # Training script
```

## 🛠 Configuration

Edit `configs/default.yaml` to customize training:

```yaml
seed: 42
model_name: distilbert-base-uncased
max_length: 256
train_batch_size: 16
eval_batch_size: 32
num_train_epochs: 2
learning_rate: 0.00002
weight_decay: 0.01
warmup_ratio: 0.06
log_steps: 50
save_strategy: "epoch"
eval_strategy: "epoch"
dataset_name: "imdb"
output_dir: "artifacts/distilbert_imdb"
# Use subset of data for faster training (0.1 = 10%, 1.0 = 100%)
data_subset: 0.1
```

### ⚡ Fast Training Mode

For quick experiments, the default config uses only **10% of data**. To adjust:

```yaml
data_subset: 0.1   # 10% data - ~40 minutes training
data_subset: 1.0   # Full data - ~6h minutes training
```

Other speed optimizations:
```yaml
max_length: 128        # Shorter sequences (vs 256)
train_batch_size: 32   # Larger batches (if GPU allows)
num_train_epochs: 1    # Single epoch for testing
```

## 📊 Features

- **Dataset**: IMDb sentiment dataset (25k train/test, binary classification)
- **Model**: DistilBERT-base-uncased for efficient training
- **Tracking**: MLflow integration for experiment tracking
- **Evaluation**: Detailed confusion matrix and classification metrics
- **GPU Support**: Automatic CUDA detection and usage

## 🔧 Advanced Usage

### Custom Dataset

Modify `src/data.py` to use different datasets:

```python
# In IMDbDataModule.prepare_data()
self.dataset = load_dataset("your_dataset_name")
```

### Different Model

Change in `configs/default.yaml`:

```yaml
model_name: "bert-base-uncased"  # Or any HuggingFace model
```

### Hyperparameter Tuning

Create new config files in `configs/` directory and run:

```bash
python src/train.py --config configs/your_config.yaml
```

## 📈 Expected Results

With default configuration on IMDb (10% subset):
- **Training time**: ~40 minutes on RTX 3080
- **Accuracy**: 1.000 (100%)
- **Loss**: 0.002
- **Model size**: ~250MB

With full dataset (data_subset: 1.0):
- **Training time**: ~6 hours on RTX 3080
- **Accuracy**: ~92-93%
- **Model size**: ~250MB

## 🤝 Dependencies

- Python 3.8+
- PyTorch 2.2+
- Transformers 4.43+
- Datasets 2.20+
- MLflow 2.15+
- Scikit-learn 1.5+

## 📝 Notes

- Models are saved to `artifacts/distilbert_imdb/`
- MLflow logs to `mlruns/` directory
- Confusion matrices saved as PNG files
- Automatic mixed precision training for faster performance
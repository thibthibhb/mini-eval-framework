#!/bin/bash

# Mini Eval Framework - Training Script
# Run DistilBERT training on IMDb sentiment dataset

set -e

echo "🚀 Starting DistilBERT training on IMDb dataset..."
echo "================================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create one first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment (Unix/macOS)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Activated virtual environment (.venv/bin/activate)"
# Activate virtual environment (Windows Git Bash)
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "✅ Activated virtual environment (.venv/Scripts/activate)"
else
    echo "⚠️  Could not find activation script, assuming environment is active"
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import torch, transformers, datasets, mlflow; print('✅ All required packages found')" || {
    echo "❌ Missing dependencies. Please install:"
    echo "  pip install -r requirements.txt"
    exit 1
}

# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Create output directory
mkdir -p artifacts/distilbert_imdb

# Start training
echo ""
echo "🏋️  Starting training..."
echo "Config: configs/default.yaml"
echo "Output: artifacts/distilbert_imdb"
echo ""

python src/train.py --config configs/default.yaml

echo ""
echo "🎉 Training completed!"
echo ""
echo "To evaluate the model, run:"
echo "  python src/eval_confusion.py --config configs/default.yaml --model_path artifacts/distilbert_imdb/best_model.pt"
echo ""
echo "View MLflow results:"
echo "  mlflow ui"
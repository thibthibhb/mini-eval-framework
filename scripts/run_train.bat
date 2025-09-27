@echo off
REM Mini Eval Framework - Training Script (Windows)
REM Run DistilBERT training on IMDb sentiment dataset

echo 🚀 Starting DistilBERT training on IMDb dataset...
echo =================================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found!
    echo Please create one first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    exit /b 1
)

REM Activate virtual environment (Windows)
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo ✅ Activated virtual environment
) else (
    echo ⚠️  Could not find activation script, assuming environment is active
)

REM Check if required packages are installed
echo 🔍 Checking dependencies...
python -c "import torch, transformers, datasets, mlflow; print('✅ All required packages found')" || (
    echo ❌ Missing dependencies. Please install:
    echo   pip install -r requirements.txt
    exit /b 1
)

REM Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

REM Create output directory
if not exist "artifacts\distilbert_imdb" mkdir "artifacts\distilbert_imdb"

REM Start training
echo.
echo 🏋️  Starting training...
echo Config: configs/default.yaml
echo Output: artifacts/distilbert_imdb
echo.

python src/train.py --config configs/default.yaml

echo.
echo 🎉 Training completed!
echo.
echo To evaluate the model, run:
echo   python src/eval_confusion.py --config configs/default.yaml --model_path artifacts/distilbert_imdb/best_model.pt
echo.
echo View MLflow results:
echo   mlflow ui
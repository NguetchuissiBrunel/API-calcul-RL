# Cameroon Travel Cost RL - Complete Setup and Run Script
# This script will install dependencies, train the model, and run evaluations

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚗 Cameroon Travel Cost RL Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$venvPython = ".\.venv\bin\python.exe"

# Check if virtual environment exists
if (-not (Test-Path $venvPython)) {
    Write-Host "❌ Virtual environment not found at .venv\bin\python.exe" -ForegroundColor Red
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Step 1: Install Dependencies
Write-Host ""
Write-Host "Step 1: Installing Dependencies..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

& $venvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Warning: pip upgrade failed, continuing..." -ForegroundColor Yellow
}

Write-Host "Installing packages (this may take a few minutes)..." -ForegroundColor Cyan
& $venvPython -m pip install gymnasium numpy pandas stable-baselines3 matplotlib tensorboard --default-timeout=100

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    Write-Host "Please check your internet connection and try again" -ForegroundColor Yellow
    Write-Host "You can manually install with:" -ForegroundColor Yellow
    Write-Host "  $venvPython -m pip install -r requirements.txt" -ForegroundColor White
    exit 1
}

Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Cyan
& $venvPython -c "import gymnasium; import numpy; import stable_baselines3; import matplotlib; print('✅ All packages verified')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Package verification failed" -ForegroundColor Red
    exit 1
}

# Step 2: Train the Model
Write-Host ""
Write-Host "Step 2: Training the Model..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray
Write-Host "This will take approximately 10-20 minutes" -ForegroundColor Cyan
Write-Host "Training for 100,000 timesteps (10 iterations)..." -ForegroundColor Cyan
Write-Host ""

& $venvPython train_agent.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Training failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Training completed successfully" -ForegroundColor Green

# Step 3: Run Demo
Write-Host ""
Write-Host "Step 3: Running Demo..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

& $venvPython demo.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Demo encountered an error" -ForegroundColor Yellow
}

# Step 4: Run Evaluation
Write-Host ""
Write-Host "Step 4: Running Comprehensive Evaluation..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

& $venvPython evaluate_model.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Evaluation encountered an error" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "✅ Evaluation completed" -ForegroundColor Green
    Write-Host "📊 Check the 'evaluation_results' folder for visualizations" -ForegroundColor Cyan
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review evaluation results in 'evaluation_results/' folder" -ForegroundColor White
Write-Host "  2. Run interactive predictions:" -ForegroundColor White
Write-Host "     $venvPython predict.py" -ForegroundColor Gray
Write-Host "  3. Monitor training with TensorBoard:" -ForegroundColor White
Write-Host "     tensorboard --logdir=logs" -ForegroundColor Gray
Write-Host ""
Write-Host "Happy predicting! 🚗💨" -ForegroundColor Cyan
Write-Host ""

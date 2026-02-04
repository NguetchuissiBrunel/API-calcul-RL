# Quick Start Guide - Cameroon Travel Cost RL

## 🚀 Quick Setup (When Network is Stable)

### Step 1: Install Dependencies
```powershell
.\.venv\bin\python.exe -m pip install --upgrade pip
.\.venv\bin\python.exe -m pip install gymnasium numpy pandas stable-baselines3 matplotlib tensorboard
```

If you experience timeouts:
```powershell
.\.venv\bin\python.exe -m pip install --default-timeout=100 gymnasium numpy pandas stable-baselines3 matplotlib tensorboard
```

### Step 2: Train the Model
```powershell
.\.venv\bin\python.exe train_agent.py
```

This will:
- Train for 100,000 timesteps (10 iterations of 10,000 each)
- Save models every 10,000 steps in `models/PPO/`
- Log training metrics to `logs/` for TensorBoard
- Take approximately 10-20 minutes depending on your CPU

### Step 3: Test the Model

#### Option A: Quick Demo (5 random trips)
```powershell
.\.venv\bin\python.exe demo.py
```

#### Option B: Interactive Predictions
```powershell
.\.venv\bin\python.exe predict.py
```
Then choose:
- Mode 1: Enter your own trip details
- Mode 2: Test on predefined scenarios

#### Option C: Comprehensive Evaluation
```powershell
.\.venv\bin\python.exe evaluate_model.py
```
This generates:
- Detailed performance metrics
- 5 visualization plots in `evaluation_results/`

### Step 4: Monitor Training (Optional)
```powershell
tensorboard --logdir=logs
```
Then open: http://localhost:6006

---

## 📊 What to Expect

### After Training:
- **Mean Error**: 500-1500 CFA
- **Accuracy (within 500 CFA)**: 40-60%
- **Accuracy (within 10%)**: 50-70%

### Sample Predictions:
```
Trip: 100km, Paved, Low Traffic, No Rain, Day
→ Actual: ~10,000 CFA
→ Predicted: ~9,500-10,500 CFA

Trip: 100km, Broken, High Traffic, Rain, Night
→ Actual: ~60,000 CFA
→ Predicted: ~55,000-65,000 CFA
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'gymnasium'"
**Solution**: Use the virtual environment Python:
```powershell
.\.venv\bin\python.exe <script.py>
```

### "No models found"
**Solution**: Train the model first:
```powershell
.\.venv\bin\python.exe train_agent.py
```

### Network timeouts during pip install
**Solution**: Increase timeout or try later:
```powershell
.\.venv\bin\python.exe -m pip install --default-timeout=200 -r requirements.txt
```

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `env.py` | Custom Gymnasium environment |
| `simulation.py` | Ground truth cost calculation |
| `train_agent.py` | Train the PPO agent |
| `demo.py` | Quick 5-trip demo |
| `predict.py` | Interactive predictions |
| `evaluate_model.py` | Comprehensive evaluation |
| `requirements.txt` | Python dependencies |

---

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Train the model
3. ✅ Test predictions
4. 📈 Analyze results
5. 🔧 Fine-tune hyperparameters if needed
6. 🚀 Deploy or integrate into your application

---

**Happy Predicting! 🚗💨**

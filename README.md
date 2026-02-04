# 🚗 Cameroon Travel Cost Prediction using Reinforcement Learning

A reinforcement learning model that predicts travel costs in Cameroon based on various factors like road conditions, distance, traffic, weather, and time of day. The model learns through a self-reinforcing mechanism, receiving penalties for incorrect predictions and bonuses for accurate ones.

## 📋 Overview

This project uses **PPO (Proximal Policy Optimization)** from Stable-Baselines3 to train an agent that can predict travel costs based on:

- **Distance** (km)
- **Road Type** (Paved, Dirt, Broken)
- **Traffic Level** (Low, Medium, High)
- **Rain Intensity** (0.0 to 1.0)
- **Time of Day** (Day/Night)
- **Accidents** (Yes/No)

## 🎯 Features

- **Custom Gymnasium Environment** for travel cost prediction
- **Realistic Cost Simulation** based on Cameroon travel conditions
- **PPO Agent Training** with configurable hyperparameters
- **Comprehensive Evaluation** with detailed metrics and visualizations
- **Demo Mode** for testing trained models
- **TensorBoard Integration** for training monitoring
- **🆕 Online Learning System** - Model improves continuously after each prediction with real feedback

## 📁 Project Structure

```
cameroon_travel_cost_rl/
├── env.py                  # Custom Gymnasium environment
├── simulation.py           # Ground truth cost calculation
├── train_agent.py          # Training script
├── demo.py                 # Demo/testing script
├── predict.py              # Interactive predictions
├── evaluate_model.py       # Comprehensive evaluation with visualizations
├── compare_models.py       # Compare training checkpoints
├── online_learning.py      # 🆕 Continuous learning system
├── requirements.txt        # Python dependencies
├── models/                 # Saved trained models
│   └── PPO/
├── logs/                   # TensorBoard logs
├── evaluation_results/     # Evaluation plots and metrics
└── online_learning_data/   # 🆕 Feedback history and updated models
```

## 🚀 Installation

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

**Windows:**
```bash
.\.venv\bin\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you experience network timeouts, try:
```bash
pip install --default-timeout=100 -r requirements.txt
```

Or install packages individually:
```bash
pip install gymnasium numpy pandas stable-baselines3 matplotlib tensorboard
```

## 🎓 Usage

### 1. Train the Model

Train the PPO agent for 100,000 timesteps (saved every 10,000 steps):

```bash
.\.venv\bin\python.exe train_agent.py
```

**Training Parameters:**
- Learning Rate: 0.0003
- Batch Size: 64
- Gamma: 0.99
- Total Timesteps: 100,000

### 2. Monitor Training (Optional)

View training progress in TensorBoard:

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.

### 3. Run Demo

Test the trained model with 5 random scenarios:

```bash
.\.venv\bin\python.exe demo.py
```

**Sample Output:**
```
Trip 1: 234.5km, Broken, Traffic: High, Rain: Yes
  -> True Cost: 45678 CFA
  -> Agent Predicted: 44523.12 CFA
  -> Error: 1154.88 | Reward: 8.45
```

### 4. Comprehensive Evaluation

Evaluate the model on 200 episodes with detailed metrics and visualizations:

```bash
.\.venv\bin\python.exe evaluate_model.py
```

**Generates:**
- Overall performance metrics (MAE, median error, accuracy thresholds)
- Performance breakdown by road type, traffic, rain, and time of day
- 5 visualization plots saved in `evaluation_results/`:
  - Error distribution histogram
  - Predicted vs Actual scatter plot
  - Reward distribution
  - Error by road type
  - Error by traffic level

## 📊 Evaluation Metrics

The evaluation script provides:

### Overall Performance
- Mean Absolute Error (MAE)
- Median Absolute Error
- Standard Deviation
- Mean Percentage Error
- Accuracy within thresholds (500 CFA, 1000 CFA, 10%)

### Conditional Performance
- Error breakdown by road type (Paved, Dirt, Broken)
- Error breakdown by traffic level (Low, Medium, High)
- Error breakdown by rain conditions
- Error breakdown by time of day (Day, Night)

## 🎮 How It Works

### 1. Environment (`env.py`)

The custom Gymnasium environment:
- **Observation Space**: 6 continuous values [distance, road_type, traffic, rain, night, accident]
- **Action Space**: Single continuous value (predicted cost in CFA, 0-500,000)
- **Reward Function**:
  - Penalty: `-error/100` (proportional to prediction error)
  - Bonus: `+100` if error < 500 CFA
  - Bonus: `+20` if error < 10% of actual cost

### 2. Cost Simulation (`simulation.py`)

Calculates ground truth costs using:
- Base rate: 100 CFA/km
- Road multipliers: Paved (1.0x), Dirt (1.5x), Broken (2.5x)
- Traffic multipliers: Low (1.0x), Medium (1.3x), High (2.0x)
- Rain penalty: +20% if intensity > 0.5
- Night surcharge: +15%
- Accident penalty: +50%
- Random noise: ±10% for market variability

### 3. Training Algorithm

Uses **PPO (Proximal Policy Optimization)**:
- Policy network learns to map observations → cost predictions
- Trained on randomly generated trip scenarios
- Optimizes for maximum cumulative reward
- Saves checkpoints every 10,000 timesteps

## 📈 Expected Results

After training for 100,000 timesteps, you should see:

- **Mean Absolute Error**: ~500-1500 CFA
- **Accuracy within 500 CFA**: 40-60%
- **Accuracy within 10%**: 50-70%
- **Better performance** on paved roads vs broken roads
- **Higher errors** in high-traffic and accident scenarios

## 🔧 Customization

### Adjust Training Duration

Edit `train_agent.py`:
```python
TIMESTEPS = 10000  # Timesteps per iteration
for i in range(1, 21):  # Number of iterations (20 = 200k total)
```

### Modify Reward Function

Edit `env.py` in the `step()` method:
```python
# Example: Increase bonus for very accurate predictions
if error < 500:
    reward += 200  # Increased from 100
```

### Change Cost Calculation

Edit `simulation.py` to adjust:
- Base rates
- Multipliers for different conditions
- Add new factors (e.g., vehicle type, fuel prices)

## 🐛 Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'gymnasium'
```
**Solution**: Make sure you're using the virtual environment Python:
```bash
.\.venv\bin\python.exe <script.py>
```

### Network Timeouts During Installation
```
WARNING: Retrying... (read timeout=15)
```
**Solution**: Increase timeout:
```bash
pip install --default-timeout=100 -r requirements.txt
```

### No Models Found
```
Model not found at models/PPO/100000.zip
```
**Solution**: Train the model first with `train_agent.py`

## 📝 Future Enhancements

- [ ] Add more input features (vehicle type, fuel prices, season)
- [ ] Implement different RL algorithms (A2C, SAC, DQN)
- [ ] Create a web interface for predictions
- [ ] Train on real historical data from Cameroon
- [ ] Add multi-step trip planning
- [ ] Implement transfer learning for different regions

## 📜 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new features
- Improve the cost simulation model
- Optimize hyperparameters
- Add more evaluation metrics

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Made with ❤️ for better travel cost prediction in Cameroon**

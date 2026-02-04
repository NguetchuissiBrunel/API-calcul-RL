import gymnasium as gym
from stable_baselines3 import PPO
from env import TravelCostEnv
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_model(model_path, num_episodes=100):
    """
    Evaluate the trained model on multiple episodes and collect statistics.
    """
    print(f"Loading model from {model_path}...")
    env = TravelCostEnv()
    
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    print(f"\nEvaluating over {num_episodes} episodes...")
    
    # Metrics storage
    errors = []
    predictions = []
    actuals = []
    rewards = []
    
    # Breakdown by conditions
    by_road_type = {0: [], 1: [], 2: []}
    by_traffic = {0: [], 1: [], 2: []}
    by_rain = {"No": [], "Yes": []}
    by_night = {"Day": [], "Night": []}
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(action)
        
        error = info["error"]
        predicted = info["predicted"]
        actual = info["actual"]
        
        errors.append(error)
        predictions.append(predicted)
        actuals.append(actual)
        rewards.append(reward)
        
        # Categorize by conditions
        road_type = int(obs[1])
        traffic = int(obs[2])
        rain = "Yes" if obs[3] > 0.5 else "No"
        night = "Night" if obs[4] > 0.5 else "Day"
        
        by_road_type[road_type].append(error)
        by_traffic[traffic].append(error)
        by_rain[rain].append(error)
        by_night[night].append(error)
    
    # Calculate statistics
    errors = np.array(errors)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    rewards = np.array(rewards)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Performance:")
    print(f"  Mean Absolute Error: {errors.mean():.2f} CFA")
    print(f"  Median Absolute Error: {np.median(errors):.2f} CFA")
    print(f"  Std Dev of Error: {errors.std():.2f} CFA")
    print(f"  Min Error: {errors.min():.2f} CFA")
    print(f"  Max Error: {errors.max():.2f} CFA")
    print(f"\n  Mean Reward: {rewards.mean():.2f}")
    print(f"  Mean Percentage Error: {(errors / actuals * 100).mean():.2f}%")
    
    # Accuracy within thresholds
    within_500 = (errors < 500).sum() / len(errors) * 100
    within_1000 = (errors < 1000).sum() / len(errors) * 100
    within_10_percent = (errors < 0.1 * actuals).sum() / len(errors) * 100
    
    print(f"\nAccuracy Thresholds:")
    print(f"  Within 500 CFA: {within_500:.1f}%")
    print(f"  Within 1000 CFA: {within_1000:.1f}%")
    print(f"  Within 10% of actual: {within_10_percent:.1f}%")
    
    # Performance by conditions
    print(f"\nMean Error by Road Type:")
    road_names = {0: "Paved", 1: "Dirt", 2: "Broken"}
    for road_type, error_list in by_road_type.items():
        if error_list:
            print(f"  {road_names[road_type]}: {np.mean(error_list):.2f} CFA")
    
    print(f"\nMean Error by Traffic Level:")
    traffic_names = {0: "Low", 1: "Medium", 2: "High"}
    for traffic, error_list in by_traffic.items():
        if error_list:
            print(f"  {traffic_names[traffic]}: {np.mean(error_list):.2f} CFA")
    
    print(f"\nMean Error by Rain:")
    for condition, error_list in by_rain.items():
        if error_list:
            print(f"  {condition}: {np.mean(error_list):.2f} CFA")
    
    print(f"\nMean Error by Time of Day:")
    for condition, error_list in by_night.items():
        if error_list:
            print(f"  {condition}: {np.mean(error_list):.2f} CFA")
    
    print("="*60)
    
    # Create visualizations
    create_visualizations(errors, predictions, actuals, rewards, by_road_type, by_traffic)
    
    return {
        'errors': errors,
        'predictions': predictions,
        'actuals': actuals,
        'rewards': rewards,
        'by_road_type': by_road_type,
        'by_traffic': by_traffic,
        'by_rain': by_rain,
        'by_night': by_night
    }

def create_visualizations(errors, predictions, actuals, rewards, by_road_type, by_traffic):
    """
    Create and save visualization plots.
    """
    os.makedirs("evaluation_results", exist_ok=True)
    
    # 1. Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Absolute Error (CFA)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.axvline(errors.mean(), color='red', linestyle='--', label=f'Mean: {errors.mean():.2f}')
    plt.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.2f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation_results/error_distribution.png', dpi=300)
    print("\nSaved: evaluation_results/error_distribution.png")
    
    # 2. Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5, s=30)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Cost (CFA)', fontsize=12)
    plt.ylabel('Predicted Cost (CFA)', fontsize=12)
    plt.title('Predicted vs Actual Travel Costs', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation_results/predicted_vs_actual.png', dpi=300)
    print("Saved: evaluation_results/predicted_vs_actual.png")
    
    # 3. Reward Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=30, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel('Reward', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Rewards', fontsize=14, fontweight='bold')
    plt.axvline(rewards.mean(), color='red', linestyle='--', label=f'Mean: {rewards.mean():.2f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation_results/reward_distribution.png', dpi=300)
    print("Saved: evaluation_results/reward_distribution.png")
    
    # 4. Error by Road Type
    plt.figure(figsize=(10, 6))
    road_names = {0: "Paved", 1: "Dirt", 2: "Broken"}
    road_errors = [np.mean(by_road_type[i]) if by_road_type[i] else 0 for i in range(3)]
    bars = plt.bar([road_names[i] for i in range(3)], road_errors, 
                   color=['green', 'orange', 'red'], edgecolor='black', alpha=0.7)
    plt.ylabel('Mean Absolute Error (CFA)', fontsize=12)
    plt.title('Prediction Error by Road Type', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('evaluation_results/error_by_road_type.png', dpi=300)
    print("Saved: evaluation_results/error_by_road_type.png")
    
    # 5. Error by Traffic Level
    plt.figure(figsize=(10, 6))
    traffic_names = {0: "Low", 1: "Medium", 2: "High"}
    traffic_errors = [np.mean(by_traffic[i]) if by_traffic[i] else 0 for i in range(3)]
    bars = plt.bar([traffic_names[i] for i in range(3)], traffic_errors,
                   color=['lightblue', 'yellow', 'darkred'], edgecolor='black', alpha=0.7)
    plt.ylabel('Mean Absolute Error (CFA)', fontsize=12)
    plt.title('Prediction Error by Traffic Level', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('evaluation_results/error_by_traffic.png', dpi=300)
    print("Saved: evaluation_results/error_by_traffic.png")
    
    plt.close('all')

if __name__ == "__main__":
    # Evaluate the latest model
    model_path = "models/PPO/100000.zip"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Looking for alternative models...")
        
        # Find the latest model
        models_dir = "models/PPO"
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            if models:
                # Sort by the number in filename
                models.sort(key=lambda x: int(x.replace('.zip', '')))
                model_path = os.path.join(models_dir, models[-1])
                print(f"Using model: {model_path}")
            else:
                print("No trained models found. Please train a model first.")
                exit(1)
        else:
            print("Models directory not found. Please train a model first.")
            exit(1)
    
    evaluate_model(model_path, num_episodes=200)

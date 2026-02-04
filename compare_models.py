"""
Compare different training checkpoints to see learning progress.
"""
import os
import numpy as np
from stable_baselines3 import PPO
from env import TravelCostEnv
import matplotlib.pyplot as plt

def evaluate_checkpoint(model_path, num_episodes=100):
    """Evaluate a single checkpoint."""
    env = TravelCostEnv()
    model = PPO.load(model_path, env=env)
    
    errors = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        errors.append(info["error"])
    
    return np.array(errors)

def compare_checkpoints():
    """Compare all available checkpoints."""
    models_dir = "models/PPO"
    
    if not os.path.exists(models_dir):
        print("❌ No models directory found.")
        return
    
    # Get all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    
    if not model_files:
        print("❌ No trained models found.")
        return
    
    # Sort by timesteps
    model_files.sort(key=lambda x: int(x.replace('.zip', '')))
    
    print("="*60)
    print("📊 COMPARING TRAINING CHECKPOINTS")
    print("="*60)
    print(f"\nFound {len(model_files)} checkpoints")
    print("Evaluating each on 100 episodes...\n")
    
    timesteps = []
    mean_errors = []
    median_errors = []
    std_errors = []
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        timestep = int(model_file.replace('.zip', ''))
        
        print(f"Evaluating {timestep:,} timesteps...", end=" ")
        
        try:
            errors = evaluate_checkpoint(model_path, num_episodes=100)
            
            timesteps.append(timestep)
            mean_errors.append(errors.mean())
            median_errors.append(np.median(errors))
            std_errors.append(errors.std())
            
            print(f"✅ MAE: {errors.mean():.2f} CFA")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Display comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'Timesteps':<15} {'Mean Error':<15} {'Median Error':<15} {'Std Dev':<15}")
    print("-"*60)
    
    for i in range(len(timesteps)):
        print(f"{timesteps[i]:<15,} {mean_errors[i]:<15.2f} {median_errors[i]:<15.2f} {std_errors[i]:<15.2f}")
    
    # Calculate improvement
    if len(mean_errors) > 1:
        improvement = ((mean_errors[0] - mean_errors[-1]) / mean_errors[0]) * 100
        print("\n" + "="*60)
        print(f"📈 Overall Improvement: {improvement:.1f}%")
        print(f"   Initial MAE: {mean_errors[0]:.2f} CFA")
        print(f"   Final MAE: {mean_errors[-1]:.2f} CFA")
        print(f"   Reduction: {mean_errors[0] - mean_errors[-1]:.2f} CFA")
        print("="*60)
    
    # Create visualization
    if len(timesteps) > 1:
        create_learning_curve(timesteps, mean_errors, median_errors, std_errors)

def create_learning_curve(timesteps, mean_errors, median_errors, std_errors):
    """Create and save learning curve visualization."""
    os.makedirs("evaluation_results", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Mean and Median Error over time
    ax1.plot(timesteps, mean_errors, 'b-o', label='Mean Error', linewidth=2, markersize=8)
    ax1.plot(timesteps, median_errors, 'g-s', label='Median Error', linewidth=2, markersize=8)
    ax1.fill_between(timesteps, 
                      np.array(mean_errors) - np.array(std_errors),
                      np.array(mean_errors) + np.array(std_errors),
                      alpha=0.2, color='blue', label='±1 Std Dev')
    ax1.set_xlabel('Training Timesteps', fontsize=12)
    ax1.set_ylabel('Absolute Error (CFA)', fontsize=12)
    ax1.set_title('Learning Curve: Error vs Training Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Format x-axis to show thousands
    ax1.ticklabel_format(style='plain', axis='x')
    
    # Plot 2: Improvement percentage
    if len(mean_errors) > 1:
        improvements = [(mean_errors[0] - err) / mean_errors[0] * 100 for err in mean_errors]
        ax2.plot(timesteps, improvements, 'r-o', linewidth=2, markersize=8)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Training Timesteps', fontsize=12)
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        ax2.set_title('Cumulative Improvement Over Time', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.ticklabel_format(style='plain', axis='x')
    
    plt.tight_layout()
    plt.savefig('evaluation_results/learning_curve.png', dpi=300)
    print("\n✅ Saved learning curve: evaluation_results/learning_curve.png")
    plt.close()
    
    # Create a detailed comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(timesteps))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mean_errors, width, label='Mean Error', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, median_errors, width, label='Median Error', alpha=0.8, color='seagreen')
    
    ax.set_xlabel('Training Checkpoint', fontsize=12)
    ax.set_ylabel('Absolute Error (CFA)', fontsize=12)
    ax.set_title('Error Comparison Across Training Checkpoints', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{ts:,}' for ts in timesteps], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/checkpoint_comparison.png', dpi=300)
    print("✅ Saved checkpoint comparison: evaluation_results/checkpoint_comparison.png")
    plt.close()

if __name__ == "__main__":
    compare_checkpoints()

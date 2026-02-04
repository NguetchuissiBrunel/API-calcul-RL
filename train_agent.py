import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os
from env import TravelCostEnv

# Create directories
models_dir = "models/PPO"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train():
    print("Initializing Environment...")
    # Instantiate the env
    env = TravelCostEnv()
    
    # Reset to check if it works
    env.reset()
    
    print("Initializing PPO Agent...")
    # Initialize PPO Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )
    
    print("Starting Training...")
    # Train
    TIMESTEPS = 10000 
    for i in range(1, 11): # Train for 10 * 10000 = 100k steps
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}")
        print(f"Saved model at {TIMESTEPS*i} steps")

    print("Training Complete.")
    
    return model

if __name__ == "__main__":
    train()

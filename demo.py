import gymnasium as gym
from stable_baselines3 import PPO
from env import TravelCostEnv
import numpy as np

def run_demo():
    print("Loading Environment and Model...")
    env = TravelCostEnv()
    
    # Load the latest model (adjust path as needed after training)
    # We will assume we trained at least 10000 steps.
    # Ideally find the latest file in models/PPO
    model_path = "models/PPO/100000.zip" 
    
    try:
        model = PPO.load(model_path, env=env)
    except:
        print(f"Could not load {model_path}. Trying 10000.zip...")
        model = PPO.load("models/PPO/10000.zip", env=env)

    print("\n--- Running 5 Test Predicitons ---\n")
    
    obs, _ = env.reset()
    
    for i in range(5):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        predicted = info["predicted"]
        actual = info["actual"]
        error = info["error"]
        
        # Decode State for display
        # State: [distance, road_type, traffic, rain, night, accident]
        dist = obs[0]
        road_map = {0: "Paved", 1: "Dirt", 2: "Broken"}
        road = road_map.get(int(obs[1]), "Unknown")
        traffic_map = {0: "Low", 1: "Med", 2: "High"}
        traf = traffic_map.get(int(obs[2]), "Unknown")
        rain = "Yes" if obs[3] > 0.5 else "No"
        
        print(f"Trip {i+1}: {dist:.1f}km, {road}, Traffic: {traf}, Rain: {rain}")
        print(f"  -> True Cost: {actual} CFA")
        print(f"  -> Agent Predicted: {predicted:.2f} CFA")
        print(f"  -> Error: {error:.2f} | Reward: {reward:.2f}")
        print("-" * 30)
        
        if terminated:
            obs, _ = env.reset()

if __name__ == "__main__":
    run_demo()

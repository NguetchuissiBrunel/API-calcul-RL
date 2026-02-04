import gymnasium as gym
from stable_baselines3 import PPO
from env import TravelCostEnv
import os
import time

def accelerate_training(total_timesteps=200000, checkpoint_freq=50000):
    """
    Runs an intensive training session to quickly improve the model's accuracy
    across all new features (luggage, wide roads, etc.)
    """
    print("🚀 Starting Accelerated Self-Improvement Lab...")
    
    # Initialize Environment
    env = TravelCostEnv()
    
    # Path to latest model
    models_dir = "models/PPO"
    latest_model = None
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        if models:
            models.sort(key=lambda x: int(x.replace('.zip', '')))
            latest_model = os.path.join(models_dir, models[-1])

    if latest_model:
        print(f"📈 Loading existing model for fine-tuning: {latest_model}")
        model = PPO.load(latest_model, env=env)
    else:
        print("🆕 Creating new model from scratch...")
        model = PPO("MlpPolicy", env, verbose=1)

    # Accelerated Training Loop
    start_time = time.time()
    
    print(f"⚙️  Training for {total_timesteps} steps...")
    
    # We use a callback-like structure for frequency saving if needed, 
    # but for "fast" training we can just run it in chunks.
    steps_done = 0
    while steps_done < total_timesteps:
        chunk = min(checkpoint_freq, total_timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        
        # Save intermediate
        save_path = os.path.join(models_dir, f"improved_{steps_done}")
        model.save(save_path)
        print(f"💾 Checkpoint saved: {save_path}.zip ({steps_done}/{total_timesteps} steps)")

    end_time = time.time()
    duration = end_time - start_time
    
    print("\n✅ Accelerated Training Complete!")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"📍 Final model saved in {models_dir}")

def simulate_api_stress_test(num_requests=100):
    """
    Optional: Verifies the API can handle high volume while model is training/loaded
    """
    import requests
    import json
    import random
    
    url = "http://127.0.0.1:8000/predict"
    print(f"🧪 Stress testing API with {num_requests} random requests...")
    
    days = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    roads = ["bonne", "moyenne", "mauvaise"]
    
    success = 0
    for i in range(num_requests):
        payload = {
            "distance_km": random.uniform(1, 50),
            "etat_route": random.choice(roads),
            "heure": f"{random.randint(0,23)}:{random.randint(0,59)}",
            "jour_semaine": random.choice(days),
            "pluie": str(random.choice([0, 0.5, 1])),
            "bagages": random.choice(["oui", "non"]),
            "routes_larges": random.choice(["oui", "non"]),
            "routes_travaux": "non",
            "accident": str(random.choice([0, 1]))
        }
        try:
            r = requests.post(url, json=payload, timeout=2)
            if r.status_code == 200:
                success += 1
        except:
            pass
            
    print(f"📊 Results: {success}/{num_requests} successful requests.")

if __name__ == "__main__":
    # 1. Run the intensive training lab
    accelerate_training(total_timesteps=100000) # Fast but effective improvement
    
    # 2. (Optional) Run a quick stress test if the API is up
    # simulate_api_stress_test(20)

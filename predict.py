import gymnasium as gym
from stable_baselines3 import PPO
from env import TravelCostEnv
import numpy as np
import os

def get_latest_model():
    """Find and return the path to the latest trained model."""
    models_dir = "models/PPO"
    
    if not os.path.exists(models_dir):
        print("❌ No models directory found. Please train a model first.")
        return None

    models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]

    if not models:
        print("❌ No trained models found. Please train a model first.")
        return None

    # Sort by the number in filename to get the latest
    models.sort(key=lambda x: int(x.replace('.zip', '')))
    model_path = os.path.join(models_dir, models[-1])

    return model_path

def get_user_input():
    """Get trip parameters from user input."""
    print("\n" + "="*60)
    print("🚗 CAMEROON TRAVEL COST PREDICTOR")
    print("="*60)
    print("\nPlease enter the trip details:\n")
    
    # Distance
    while True:
        try:
            distance = float(input("📏 Distance (km): "))
            if distance > 0:
                break
            print("   ⚠️  Distance must be positive!")
        except ValueError:
            print("   ⚠️  Please enter a valid number!")
    
    # Road Type
    print("\n🛣️  Road Type:")
    print("   0 = Paved (Good condition)")
    print("   1 = Dirt (Unpaved)")
    print("   2 = Broken (Damaged/Poor condition)")
    while True:
        try:
            road_type = int(input("   Enter choice (0-2): "))
            if road_type in [0, 1, 2]:
                break
            print("   ⚠️  Please enter 0, 1, or 2!")
        except ValueError:
            print("   ⚠️  Please enter a valid number!")
    
    # Traffic Level
    print("\n🚦 Traffic Level:")
    print("   0 = Low (Free flowing)")
    print("   1 = Medium (Moderate)")
    print("   2 = High (Congested)")
    while True:
        try:
            traffic = int(input("   Enter choice (0-2): "))
            if traffic in [0, 1, 2]:
                break
            print("   ⚠️  Please enter 0, 1, or 2!")
        except ValueError:
            print("   ⚠️  Please enter a valid number!")
    
    # Rain
    print("\n🌧️  Rain Intensity:")
    print("   0.0 = No rain")
    print("   0.5 = Light rain")
    print("   1.0 = Heavy rain")
    while True:
        try:
            rain = float(input("   Enter value (0.0-1.0): "))
            if 0 <= rain <= 1:
                break
            print("   ⚠️  Please enter a value between 0.0 and 1.0!")
        except ValueError:
            print("   ⚠️  Please enter a valid number!")
    
    # Night
    print("\n🌙 Time of Day:")
    print("   0 = Day")
    print("   1 = Night")
    while True:
        try:
            night = int(input("   Enter choice (0-1): "))
            if night in [0, 1]:
                break
            print("   ⚠️  Please enter 0 or 1!")
        except ValueError:
            print("   ⚠️  Please enter a valid number!")
    
    # Accident
    print("\n🚨 Accidents Reported:")
    print("   0 = No")
    print("   1 = Yes")
    while True:
        try:
            accident = int(input("   Enter choice (0-1): "))
            if accident in [0, 1]:
                break
            print("   ⚠️  Please enter 0 or 1!")
        except ValueError:
            print("   ⚠️  Please enter a valid number!")
    # Luggage
    print("\n🧳 Luggage:")
    print("   0 = No")
    print("   1 = Yes")
    while True:
        try:
            luggage = int(input("   Enter choice (0-1): "))
            if luggage in [0, 1]:
                break
            print("   ⚠️  Please enter 0 or 1!")
        except ValueError:
            print("   ⚠️  Please enter a valid number!")

    # Wide Road
    print("\n🛣️  Wide Road:")
    print("   0 = No")
    print("   1 = Yes")
    while True:
        try:
            wide_road = int(input("   Enter choice (0-1): "))
            if wide_road in [0, 1]:
                break
            print("   ⚠️  Please enter 0 or 1!")
        except ValueError:
            print("   ⚠️  Please enter a valid number!")
    
    return np.array([distance, road_type, traffic, rain, night, accident, luggage, wide_road], dtype=np.float32)

def predict_cost(model, observation):
    """Make a prediction using the trained model."""
    action, _ = model.predict(observation, deterministic=True)
    predicted_cost = float(action[0])
    return predicted_cost

def display_prediction(observation, predicted_cost, actual_cost=None):
    """Display the prediction in a formatted way."""
    print("\n" + "="*60)
    print("📊 PREDICTION RESULTS")
    print("="*60)
    
    # Decode observation
    distance = observation[0]
    road_map = {0: "Paved (Good)", 1: "Dirt (Unpaved)", 2: "Broken (Damaged)"}
    road = road_map[int(observation[1])]
    traffic_map = {0: "Low", 1: "Medium", 2: "High"}
    traffic = traffic_map[int(observation[2])]
    rain = observation[3]
    time = "Night" if observation[4] > 0.5 else "Day"
    accident = "Yes" if observation[5] > 0.5 else "No"
    luggage = "Yes" if observation[6] > 0.5 else "No"
    wide_road = "Yes" if observation[7] > 0.5 else "No"
    
    print("\n📋 Trip Summary:")
    print(f"   Distance: {distance:.1f} km")
    print(f"   Road Type: {road}")
    print(f"   Traffic: {traffic}")
    print(f"   Rain Intensity: {rain:.1f}")
    print(f"   Time: {time}")
    print(f"   Accidents: {accident}")
    print(f"   Luggage: {luggage}")
    print(f"   Wide Road: {wide_road}")
    
    print(f"\n💰 PREDICTED COST: {predicted_cost:,.2f} CFA")
    
    if actual_cost is not None:
        error = abs(predicted_cost - actual_cost)
        error_pct = (error / actual_cost) * 100
        print(f"   Actual Cost: {actual_cost:,.2f} CFA")
        print(f"   Error: {error:,.2f} CFA ({error_pct:.1f}%)")
        
        if error < 500:
            print("   ✅ Excellent prediction!")
        elif error < 1000:
            print("   ✓ Good prediction!")
        elif error_pct < 10:
            print("   ~ Acceptable prediction")
        else:
            print("   ⚠️  High error - model may need more training")
    
    print("="*60)

def interactive_mode():
    """Run the predictor in interactive mode."""
    # Load the model
    model_path = get_latest_model()
    if model_path is None:
        return
    
    print(f"\n✅ Loading model from: {model_path}")
    env = TravelCostEnv()
    
    try:
        model = PPO.load(model_path, env=env)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    while True:
        # Get user input
        observation = get_user_input()
        
        # Make prediction
        predicted_cost = predict_cost(model, observation)
        
        # Calculate actual cost for comparison (using the simulation)
        from simulation import calculate_true_cost
        actual_cost = calculate_true_cost(
            observation[0],  # distance
            int(observation[1]),  # road_type
            int(observation[2]),  # traffic
            observation[3],  # rain
            bool(observation[4]),  # night
            bool(observation[5]),  # accident
            bool(observation[6]),  # luggage
            bool(observation[7])   # wide_road
        )
        
        # Display results
        display_prediction(observation, predicted_cost, actual_cost)
        
        # Ask if user wants to continue
        print("\n" + "-"*60)
        choice = input("\nPredict another trip? (y/n): ").strip().lower()
        if choice != 'y':
            print("\n👋 Thank you for using the Travel Cost Predictor!")
            print("="*60)
            break

def batch_mode():
    """Run predictions on predefined scenarios."""
    # Load the model
    model_path = get_latest_model()
    if model_path is None:
        return
    
    print(f"\n✅ Loading model from: {model_path}")
    env = TravelCostEnv()
    
    try:
        model = PPO.load(model_path, env=env)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Predefined scenarios
    scenarios = [
        {
            "name": "Short city trip (paved, low traffic)",
            "obs": np.array([10, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        },
        {
            "name": "Long highway trip (paved, medium traffic)",
            "obs": np.array([150, 0, 1, 0, 0, 0, 1, 1], dtype=np.float32)
        },
        {
            "name": "Rural trip (dirt road, rain)",
            "obs": np.array([50, 1, 0, 0.8, 0, 0, 0, 0], dtype=np.float32)
        },
        {
            "name": "Difficult trip (broken road, high traffic, night)",
            "obs": np.array([80, 2, 2, 0, 1, 0, 0, 0], dtype=np.float32)
        },
        {
            "name": "Emergency trip (broken road, accident, rain, night)",
            "obs": np.array([120, 2, 2, 0.9, 1, 1, 1, 0], dtype=np.float32)
        }
    ]
    
    from simulation import calculate_true_cost
    
    print("\n" + "="*60)
    print("🎯 BATCH PREDICTION MODE - COMMON SCENARIOS")
    print("="*60)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n\n{'='*60}")
        print(f"Scenario {i}: {scenario['name']}")
        print("="*60)
        
        obs = scenario['obs']
        predicted_cost = predict_cost(model, obs)
        
        actual_cost = calculate_true_cost(
            obs[0], int(obs[1]), int(obs[2]), obs[3], bool(obs[4]), bool(obs[5]), bool(obs[6]), bool(obs[7])
        )
        
        display_prediction(obs, predicted_cost, actual_cost)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚗 CAMEROON TRAVEL COST PREDICTOR")
    print("="*60)
    print("\nChoose mode:")
    print("  1 = Interactive Mode (Enter your own trip details)")
    print("  2 = Batch Mode (Test on predefined scenarios)")
    
    while True:
        try:
            choice = int(input("\nEnter choice (1-2): "))
            if choice in [1, 2]:
                break
            print("⚠️  Please enter 1 or 2!")
        except ValueError:
            print("⚠️ Please enter a valid number!")
    
    if choice == 1:
        interactive_mode()
    else:
        batch_mode()

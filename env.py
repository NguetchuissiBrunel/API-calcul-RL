import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from simulation import calculate_true_cost

class TravelCostEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The agent must predict the cost of a trip given observation parameters.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, feedback_data=None):
        super(TravelCostEnv, self).__init__()
        
        self.feedback_data = feedback_data
        self.feedback_index = 0
        
        # State Space (Observation):
        # 1. Distance (km) - 0 to 1000
        # 2. Road Type - 0, 1, 2
        # 3. Traffic - 0, 1, 2
        # 4. Rain - 0.0 to 1.0
        # 5. Night - 0 or 1
        # 6. Accident - 0 or 1
        # 7. Luggage - 0 or 1
        # 8. Wide Road - 0 or 1
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1000, 2, 2, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(low=0, high=500000, shape=(1,), dtype=np.float32)
        
        self.current_state = None
        self.actual_cost = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.feedback_data and len(self.feedback_data) > 0:
            # Use real data from feedback
            sample = self.feedback_data[self.feedback_index % len(self.feedback_data)]
            self.current_state = np.array(sample['observation'], dtype=np.float32)
            self.actual_cost = sample['actual_cost']
            self.feedback_index += 1
        else:
            # Generate a random trip scenario
            distance = np.random.uniform(1, 500) # 1km to 500km
            road_type = np.random.randint(0, 3)
            traffic = np.random.randint(0, 3)
            rain = np.random.uniform(0, 1)
            night = 1 if np.random.random() > 0.7 else 0
            accident = 1 if np.random.random() > 0.9 else 0 # 10% chance
            luggage = 1 if np.random.random() > 0.5 else 0
            wide_road = 1 if np.random.random() > 0.5 else 0
            
            self.current_state = np.array([distance, road_type, traffic, rain, night, accident, luggage, wide_road], dtype=np.float32)
            
            # Calculate Ground Truth
            self.actual_cost = calculate_true_cost(
                distance, road_type, traffic, rain, bool(night), bool(accident), bool(luggage), bool(wide_road)
            )
        
        return self.current_state, {}

    def step(self, action):
        # Action is the predicted price
        predicted_price = float(action[0])
        
        # Calculate Error
        error = abs(predicted_price - self.actual_cost)
        
        # Reward Function
        # We want to maximize Reward.
        # 1. If very close (within 5% or 500 CFA), big bonus.
        # 2. Otherwise/Also, penalty proportional to error.
        
        reward = 0
        
        # Penalty: Negative absolute error scaled down
        # e.g. Error 1000 CFA -> -10 pts
        reward -= (error / 100.0) 
        
        # Bonus for precision
        if error < 500: # Very close
            reward += 100
        elif error < 0.1 * self.actual_cost: # Within 10%
            reward += 20
            
        # Stopping condition: Each step is a new trip, so we can make this episodic with length 1?
        # OR we treat it as a continuous stream.
        # Typically for "Contextual Bandit" style (new state -> action -> reward -> done), 'terminated' is True every step.
        terminated = True 
        truncated = False
        
        info = {
            "predicted": predicted_price,
            "actual": self.actual_cost,
            "error": error
        }
        
        return self.current_state, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"State: {self.current_state} | True Cost: {self.actual_cost}")

from online_learning import demo_online_learning
import os

if __name__ == "__main__":
    # Ensure we start fresh or clear old data for this verification run
    if os.path.exists("online_learning_data/feedback_history.json"):
        os.remove("online_learning_data/feedback_history.json")
    
    demo_online_learning()

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
from env import TravelCostEnv
import os
import uvicorn
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    load_model()
    yield
    # Clean up on shutdown if needed
    print("Shutting down API...")

app = FastAPI(title="Cameroon Travel Cost Predictor API", lifespan=lifespan)

class PredictionRequest(BaseModel):
    distance_km: float
    etat_route: str  # "mvan", "bonne", "mauvaise" ? No, frontend sends: "bonne", "moyenne", "mauvaise"
    heure: str # "14:00"
    jour_semaine: str # "lundi", ...
    pluie: str # "0", "0.5", "1"
    bagages: str # "oui", "non"
    routes_larges: str # "oui", "non"
    routes_travaux: str # "oui", "non"
    accident: str # "0", "1"
    # depart_osm, destination_osm are not needed for the RL model logic unless we used them for geocoding, 
    # but the frontend sends distance_km already calculated.

class PredictionResponse(BaseModel):
    prix_estime_fcfa: float
    prix_estime_range: str
    message: str

# Global model variable
model = None

def get_latest_model():
    models_dir = "models/PPO"
    print(f"DEBUG: Checking directory {models_dir} (Absolute: {os.path.abspath(models_dir)})")
    
    if not os.path.exists(models_dir):
        print(f"DEBUG: Directory {models_dir} does NOT exist.")
        return None
    
    files = os.listdir(models_dir)
    print(f"DEBUG: Files in {models_dir}: {files}")
    
    models = [f for f in files if f.endswith('.zip')]
    if not models:
        print(f"DEBUG: No .zip files found in {models_dir}")
        return None
    
    # Improved sorting: find the one with the highest numerical value, ignoring prefixes
    def extract_number(filename):
        import re
        nums = re.findall(r'\d+', filename)
        return int(nums[0]) if nums else 0

    models.sort(key=extract_number)
    latest = models[-1]
    print(f"DEBUG: Latest model selected: {latest}")
    return os.path.join(models_dir, latest)

def load_model():
    global model
    model_path = get_latest_model()
    if model_path:
        print(f"DEBUG: Attempting to load model from {model_path}")
        try:
            env = TravelCostEnv() 
            model = PPO.load(model_path, env=env)
            print("DEBUG: Model loaded successfully into memory.")
        except Exception as e:
            print(f"DEBUG: Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("DEBUG: get_latest_model() returned None. No model to load.")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global model
    if not model:
        # Try to load again just in case
        load_model()
        if not model:
            # Fallback to simulation/heuristic if model is missing? 
            # Or just return error. The plan said "expose model".
            # Let's return error for now, or maybe a heuristic fallback.
            raise HTTPException(status_code=503, detail="Model not loaded and no trained model found.")

    # 1. Map inputs to model observation
    # Model inputs: distance, road_type, traffic, rain, night, accident, luggage, wide_road
    
    # Distance
    distance = request.distance_km
    
    # Road Type
    # "bonne" -> 0 (Paved), "moyenne" -> 1 (Dirt), "mauvaise" -> 2 (Broken)
    # Frontend sends: "bonne", "moyenne", "mauvaise"
    road_map = {"bonne": 0, "moyenne": 1, "mauvaise": 2}
    road_type = road_map.get(request.etat_route.lower(), 1) # default medium
    
    # Rain
    # Frontend sends "0", "0.5", "1" or similar strings?
    try:
        rain = float(request.pluie)
    except:
        rain = 0.0
        
    # Accident
    accident = 1 if str(request.accident) == "1" else 0
    
    # Luggage
    has_luggage = 1 if request.bagages.lower() == "oui" else 0
    
    # Wide Road
    # Wide road helps traffic flow.
    is_wide_road = 1 if request.routes_larges.lower() == "oui" else 0
    
    # Night
    # Parse hour
    try:
        hour_int = int(request.heure.split(":")[0])
    except:
        hour_int = 12
    
    is_night = 1 if (hour_int >= 19 or hour_int < 6) else 0
    
    # Traffic
    # Frontend doesn't send traffic level directly. We infer it.
    # Weekday + Rush Hour = High
    # Weekday + Normal = Medium
    # Night/Weekend = Low
    # This is a simple heuristic mapping.
    
    days_map = {
        "lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, 
        "vendredi": 4, "samedi": 5, "dimanche": 6
    }
    day_idx = days_map.get(request.jour_semaine.lower(), 0)
    
    traffic = 0 # Low
    is_weekend = (day_idx >= 5)
    
    if not is_weekend:
        # Rush hours: 7-9 and 16-19
        if (7 <= hour_int <= 9) or (16 <= hour_int <= 19):
            traffic = 2 # High
        elif (6 <= hour_int <= 20):
            traffic = 1 # Medium
        else:
            traffic = 0 # Low
    else:
        # Weekend
        if (10 <= hour_int <= 18):
            traffic = 1 # Medium
        else:
            traffic = 0 # Low
            
    # Construct Observation
    obs = np.array([distance, road_type, traffic, rain, is_night, accident, has_luggage, is_wide_road], dtype=np.float32)
    
    # Predict
    action, _ = model.predict(obs, deterministic=True)
    predicted_cost = float(action[0])
    
    # Range
    cost_min = int(predicted_cost * 0.9)
    cost_max = int(predicted_cost * 1.1)
    
    return PredictionResponse(
        prix_estime_fcfa=predicted_cost,
        prix_estime_range=f"{cost_min} - {cost_max} FCFA",
        message="Succès"
    )

if __name__ == "__main__":
    try:
        # Port is often provided by the environment in production (e.g., Render/Heroku)
        port = int(os.environ.get("PORT", 8000))
        print(f"Starting Uvicorn server on http://0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"Server failed to start: {e}")

import http.server
import json
import numpy as np
import os
from stable_baselines3 import PPO
from env import TravelCostEnv

# Configuration
PORT = 8000
MODEL_DIR = "models/PPO"

def get_latest_model():
    if not os.path.exists(MODEL_DIR):
        return None
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.zip')]
    if not models:
        return None
    models.sort(key=lambda x: int(x.replace('.zip', '')))
    return os.path.join(MODEL_DIR, models[-1])

# Global model
model = None
model_path = get_latest_model()
if model_path:
    print(f"Loading model from {model_path}...")
    env = TravelCostEnv()
    model = PPO.load(model_path, env=env)
    print("✅ Model loaded successfully!")
else:
    print("❌ No model found! API will return 503.")

class SimplePredictHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                request = json.loads(post_data.decode('utf-8'))
                
                # Input Mapping (Sync with api.py logic)
                distance = float(request.get('distance_km', 0))
                
                road_map = {"bonne": 0, "moyenne": 1, "mauvaise": 2}
                road_type = road_map.get(request.get('etat_route', 'bonne').lower(), 1)
                
                try:
                    rain = float(request.get('pluie', 0))
                except:
                    rain = 0.0
                
                accident = 1 if str(request.get('accident', '0')) == "1" else 0
                has_luggage = 1 if str(request.get('bagages', 'non')).lower() == "oui" else 0
                is_wide_road = 1 if str(request.get('routes_larges', 'non')).lower() == "oui" else 0
                
                try:
                    hour_str = request.get('heure', '12:00')
                    hour_int = int(hour_str.split(":")[0])
                except:
                    hour_int = 12
                
                is_night = 1 if (hour_int >= 19 or hour_int < 6) else 0
                
                # Traffic Heuristic
                days_map = {
                    "lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, 
                    "vendredi": 4, "samedi": 5, "dimanche": 6
                }
                day_idx = days_map.get(request.get('jour_semaine', 'lundi').lower(), 0)
                
                traffic = 0
                is_weekend = (day_idx >= 5)
                if not is_weekend:
                    if (7 <= hour_int <= 9) or (16 <= hour_int <= 19): traffic = 2
                    elif (6 <= hour_int <= 20): traffic = 1
                else:
                    if (10 <= hour_int <= 18): traffic = 1

                # Observation
                obs = np.array([distance, road_type, traffic, rain, is_night, accident, has_luggage, is_wide_road], dtype=np.float32)
                
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                    predicted_cost = float(action[0])
                    
                    cost_min = int(predicted_cost * 0.9)
                    cost_max = int(predicted_cost * 1.1)
                    
                    response = {
                        "prix_estime_fcfa": predicted_cost,
                        "prix_estime_range": f"{cost_min} - {cost_max} FCFA",
                        "message": "Succès (Standard API)"
                    }
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                else:
                    self.send_error(503, "Model not loaded")
                    
            except Exception as e:
                self.send_error(400, f"Error: {str(e)}")
        else:
            self.send_error(404)

if __name__ == "__main__":
    server_address = ('', PORT)
    httpd = http.server.HTTPServer(server_address, SimplePredictHandler)
    print(f"🚀 Server running on port {PORT}...")
    httpd.serve_forever()

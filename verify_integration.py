import json
import time
import urllib.request
import urllib.error

def test_api():
    url = "http://localhost:8000/predict"
    
    # Mock payload based on farcal frontend
    payload = {
        "distance_km": 15.5,
        "etat_route": "moyenne",
        "heure": "14:30",
        "jour_semaine": "lundi",
        "pluie": "0.5",
        "bagages": "oui",
        "routes_larges": "non",
        "routes_travaux": "non",
        "accident": "0"
    }
    
    print(f"Sending request to {url}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        start_time = time.time()
        with urllib.request.urlopen(req) as response:
            end_time = time.time()
            result = json.loads(response.read().decode('utf-8'))
            
            print(f"Status Code: {response.getcode()}")
            print(f"Time: {end_time - start_time:.4f}s")
            print("✅ Response Success:")
            print(json.dumps(result, indent=2))
            
    except urllib.error.HTTPError as e:
        print(f"❌ Request failed with status {e.code}:")
        print(e.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"❌ Connection error: {e.reason}")
        print("Make sure the API is running (uvicorn api:app --reload)")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()

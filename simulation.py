import random

def calculate_true_cost(distance_km, road_type, traffic_level, rain_intensity, is_night, accidents_reported, has_luggage, is_wide_road):
    """
    Calculates the 'ground truth' cost of a trip based on inputs.
    
    Parameters:
    - distance_km: float (km)
    - road_type: int (0=Paved, 1=Dirt, 2=Damaged/Broken)
    - traffic_level: int (0=Low, 1=Medium, 2=High)
    - rain_intensity: float (0.0 to 1.0)
    - is_night: bool
    - accidents_reported: bool
    - has_luggage: bool
    - is_wide_road: bool
    
    Returns:
    - true_cost: float (CFA Francs)
    """
    
    # Base rate per km (e.g., 100 CFA/km for standard taxi/bus average)
    base_rate = 100 
    
    # Multipliers
    road_multipliers = {0: 1.0, 1: 1.5, 2: 2.5} # Dirt is harder, Damaged is worst
    traffic_multipliers = {0: 1.0, 1: 1.3, 2: 2.0} # Traffic increases fuel/time
    
    cost = distance_km * base_rate
    
    # Apply modifiers
    cost *= road_multipliers.get(road_type, 1.0)
    cost *= traffic_multipliers.get(traffic_level, 1.0)
    
    if rain_intensity > 0.5:
        cost *= 1.2 # Rain slows down, higher risk/price
        
    if is_night:
        cost *= 1.15 # Night surcharge
        
    if accidents_reported:
        cost *= 1.5 # Massive detour or delay cost

    if has_luggage:
        cost += 500 # Fixed fee for luggage or surcharge

    if is_wide_road:
        cost *= 0.9 # 10% cheaper/faster on wide roads
            
    # Add some random noise to simulate market negotiation/variability (+/- 10%)
    noise = random.uniform(0.9, 1.1)
    cost *= noise
    
    return max(100, round(cost)) # Minimum 100 CFA

if __name__ == "__main__":
    # Test
    print(f"Test Trip (10km, Paved, Low Traffic): {calculate_true_cost(10, 0, 0, 0, False, False, False, True)} CFA")
    print(f"Test Trip (10km, Broken, High Traffic, Rain): {calculate_true_cost(10, 2, 2, 0.8, False, False, True, False)} CFA")

import pandas as pd
import numpy as np
from haversine import haversine
from datetime import date, timedelta
from collections import defaultdict

COMPATIBILITY = {
    'O-': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],  # universal donor
    'O+': ['O+', 'A+', 'B+', 'AB+'],
    'A-': ['A-', 'A+', 'AB-', 'AB+'],
    'A+': ['A+', 'AB+'],
    'B-': ['B-', 'B+', 'AB-', 'AB+'],
    'B+': ['B+', 'AB+'],
    'AB-': ['AB-', 'AB+'],
    'AB+': ['AB+']  # universal recipient (can only donate to AB+)
}

# Helper Functions
def haversine_distance(coord1, coord2):
    return haversine(coord1, coord2)

def get_compatible_donors(recipient_type):
    return COMPATIBILITY.get(recipient_type, [])

def is_eligible(donor, current_date):
    # Check donation interval (56 days minimum)
    days_since_donation = (current_date - donor['last_donation_date']).days
    if days_since_donation < 56:
        return False
    
    # Check notification cooldown (14 days minimum to avoid fatigue)
    days_since_notification = (current_date - donor['last_notification_date']).days
    if days_since_notification < 14:
        return False
    
    # Age constraint
    if not (18 <= donor['age'] <= 65):
        return False
    
    # Health status (assume pre-registered donors are healthy)
    return True

# Core Functions
def generate_synthetic_donor_data(n_donors=5000, blood_bank_location=None, city_bounds=None):
    if blood_bank_location is None:
        blood_bank_location = {'lat': 12.9716, 'lon': 77.5946} # Default to Bangalore
    if city_bounds is None:
        city_bounds = {'lat_min': 12.8, 'lat_max': 13.2, 'lon_min': 77.4, 'lon_max': 77.8}

    donors = []
    
    # Blood type distribution (India-specific)
    BLOOD_TYPE_DIST = {
        'O+': 0.36, 'O-': 0.02,
        'A+': 0.22, 'A-': 0.02,
        'B+': 0.32, 'B-': 0.02,
        'AB+': 0.04, 'AB-': 0.00 # Adjusted to make sum 1.0
    }
    
    for i in range(n_donors):
        # Blood type
        blood_type = np.random.choice(list(BLOOD_TYPE_DIST.keys()), 
                                      p=list(BLOOD_TYPE_DIST.values()))
        
        # Age (skewed toward 25-45, active donation age)
        age = int(np.random.beta(2, 3) * 47 + 18)  # Beta distribution 18-65
        
        # Geographic location (clustered around blood bank with decay)
        # Use exponential distribution for distance
        distance_km = np.random.exponential(scale=5.0)  # Mean 5km
        angle = np.random.uniform(0, 2*np.pi)
        lat_offset = distance_km * np.cos(angle) / 111  # ~111km per degree
        lon_offset = distance_km * np.sin(angle) / 111
        
        lat = blood_bank_location['lat'] + lat_offset
        lon = blood_bank_location['lon'] + lon_offset
        
        # Clip to city bounds
        lat = np.clip(lat, city_bounds['lat_min'], city_bounds['lat_max'])
        lon = np.clip(lon, city_bounds['lon_min'], city_bounds['lon_max'])
        
        # Donation history (Pareto distribution - few frequent donors, many infrequent)
        lifetime_donations = int(np.random.pareto(1.5) + 1)  # Min 1 donation
        
        # Last donation date (must be at least 56 days ago)
        days_since_last = np.random.randint(56, 365)
        
        # Response rate (correlated with lifetime donations)
        base_response = 0.15 + 0.03 * min(lifetime_donations, 10)  # 15-45%
        response_rate = np.clip(np.random.normal(base_response, 0.10), 0.05, 0.80)
        
        # No-show rate (inversely correlated with response rate)
        no_show_rate = np.clip(np.random.normal(0.20 - response_rate*0.3, 0.08), 0, 0.50)
        
        # Last notification (must be at least 14 days ago)
        days_since_notification = np.random.randint(14, 180)
        
        # Average donation interval (for regulars)
        if lifetime_donations >= 3:
            avg_interval = np.random.normal(90, 20)  # Regular donors ~3 months
        else:
            avg_interval = np.random.normal(150, 40)  # Infrequent donors
        
        donors.append({
            'id': f'D{i:06d}',
            'blood_type': blood_type,
            'age': age,
            'lat': lat,
            'lon': lon,
            'lifetime_donations': lifetime_donations,
            'last_donation_date': date.today() - timedelta(days=days_since_last),
            'response_rate': response_rate,
            'no_show_rate': no_show_rate,
            'last_notification_date': date.today() - timedelta(days=days_since_notification),
            'avg_donation_interval_days': avg_interval,
            'preferred_time': np.random.choice(['morning', 'afternoon', 'evening']),
            'contact_preference': np.random.choice(['sms', 'email', 'app'], 
                                                   p=[0.5, 0.3, 0.2])
        })
    
    return pd.DataFrame(donors)


def calculate_match_score(donor, demand_forecast, current_date, urgency_level, blood_bank_location):
    score = 0
    
    # 1. Blood type compatibility (40% weight)
    if donor['blood_type'] in get_compatible_donors(demand_forecast['needed_blood_type']):
        if donor['blood_type'] == demand_forecast['needed_blood_type']:
            score += 40  # Exact match
        else:
            score += 30  # Compatible but not exact (e.g., O- for A+)
    else:
        return 0  # Not compatible, don't match
    
    # 2. Geographic proximity (25% weight)
    distance_km = haversine_distance((donor['lat'], donor['lon']), 
                                      (blood_bank_location['lat'], blood_bank_location['lon']))
    if distance_km <= 3:
        score += 25
    elif distance_km <= 5:
        score += 20
    elif distance_km <= 10:
        score += 12
    elif distance_km <= 15:
        score += 5
    else:
        score += 1  # Beyond 15km, very low priority
    
    # 3. Historical response rate (20% weight)
    score += donor['response_rate'] * 20  # response_rate in [0, 1]
    
    # 4. Donation recency bonus (10% weight)
    days_since = (current_date - donor['last_donation_date']).days
    if 56 <= days_since <= 90:
        score += 10  # Recently eligible, likely still motivated
    elif 91 <= days_since <= 180:
        score += 8
    elif 181 <= days_since <= 365:
        score += 5
    else:
        score += 2  # Very long gap, may need re-engagement
    
    # 5. Urgency multiplier (5% weight)
    if urgency_level == 'critical':  # <24h supply
        score *= 1.3
    elif urgency_level == 'urgent':  # 24-48h supply
        score *= 1.15
    
    # Penalty for high no-show rate
    if donor['no_show_rate'] > 0.3:
        score *= 0.7
    
    return score


def match_donors_to_demand(demand_forecast_7days, all_donors, current_date, blood_bank_location):
    matches = []
    donor_quota = {}  # Track how many times each donor is contacted
    
    # Sort demand by urgency (prioritize critical shortages)
    demand_items = sorted(demand_forecast_7days.items(), 
                          key=lambda x: (x[1]['urgency'], -x[1]['units_needed']),
                          reverse=True)
    
    for blood_type, demand_info in demand_items:
        units_needed = demand_info['units_needed']
        urgency = demand_info['urgency']
        
        # Filter eligible and compatible donors
        eligible_and_compatible_donors = [
            donor for _, donor in all_donors.iterrows()
            if is_eligible(donor, current_date) and donor['blood_type'] in get_compatible_donors(blood_type)
        ]
        
        # Score all compatible donors
        scored_donors = []
        for donor in eligible_and_compatible_donors:
            score = calculate_match_score(donor, 
                                         {'needed_blood_type': blood_type},
                                         current_date,
                                         urgency,
                                         blood_bank_location)
            if score > 0:
                scored_donors.append((donor, score))
        
        # Sort by score (highest first)
        scored_donors.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N donors (aim for 3x units needed, accounting for ~30-40% response rate)
        target_notifications = min(int(units_needed * 3), len(scored_donors))
        
        for i in range(target_notifications):
            donor, score = scored_donors[i]
            
            # Constraint: don't over-contact same donor across multiple blood types
            if donor['id'] in donor_quota and donor_quota[donor['id']] >= 1:
                continue
            
            matches.append({
                'donor_id': donor['id'],
                'blood_type_needed': blood_type,
                'score': score,
                'urgency': urgency,
                'distance_km': haversine_distance((donor['lat'], donor['lon']),
                                                  (blood_bank_location['lat'], blood_bank_location['lon']))
            })
            
            donor_quota[donor['id']] = donor_quota.get(donor['id'], 0) + 1
    
    return matches


if __name__ == '__main__':
    import json
    import sys

    # Default values for command line arguments
    component = 'RBC'
    blood_group = 'O+'
    urgency = 'routine'
    units_needed = 10
    current_date = date.today()
    blood_bank_location = {'lat': 12.9716, 'lon': 77.5946} # Bangalore

    # Parse command line arguments if provided by server.js
    if len(sys.argv) > 1:
        try:
            args = json.loads(sys.argv[1])
            component = args.get('component', component)
            blood_group = args.get('blood_group', blood_group)
            urgency = args.get('urgency', urgency)
            units_needed = args.get('units_needed', units_needed)
            # current_date can be passed if needed, otherwise use today
        except json.JSONDecodeError:
            pass # Use defaults if parsing fails

    # Generate synthetic donor data (or load from CSV if available)
    # For now, we'll generate it each time for simplicity
    donors_df = generate_synthetic_donor_data(n_donors=5000, blood_bank_location=blood_bank_location)

    # Create a dummy demand forecast for the next 7 days
    demand_forecast_7days = {
        blood_group: {'units_needed': units_needed, 'urgency': urgency}
    }

    # Run the matching algorithm
    ranked_matches = match_donors_to_demand(
        demand_forecast_7days=demand_forecast_7days,
        all_donors=donors_df,
        current_date=current_date,
        blood_bank_location=blood_bank_location
    )

    # Output the ranked matches to a JSON file
    output_file_path = "ranked_donors.json"
    with open(output_file_path, 'w') as f:
        json.dump(ranked_matches, f, indent=2, default=str)
    print(f"Ranked donors saved to {output_file_path}")

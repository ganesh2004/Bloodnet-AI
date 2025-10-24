
import pandas as pd
import numpy as np
from haversine import haversine
from datetime import date, timedelta
from collections import defaultdict

# Helper Functions
def haversine_distance(coord1, coord2):
    return haversine(coord1, coord2)

def is_compatible(donor_type, recipient_type):
    # Simplified compatibility rules
    # O_neg is universal donor
    if donor_type == 'O_neg':
        return True
    # AB_pos is universal recipient
    elif recipient_type == 'AB_pos':
        return True
    # Same type is always compatible
    elif donor_type == recipient_type:
        return True
    # Specific cross-compatibilities (simplified)
    elif donor_type == 'O_pos' and recipient_type in ['A_pos', 'B_pos', 'AB_pos']:
        return True
    elif donor_type == 'A_pos' and recipient_type in ['AB_pos']:
        return True
    elif donor_type == 'B_pos' and recipient_type in ['AB_pos']:
        return True
    elif donor_type == 'A_neg' and recipient_type in ['A_pos', 'A_neg', 'AB_pos', 'AB_neg']:
        return True
    elif donor_type == 'B_neg' and recipient_type in ['B_pos', 'B_neg', 'AB_pos', 'AB_neg']:
        return True
    else:
        return False

# Global variables for demonstration
current_date = date.today()

BANK_LOCATIONS = [
    {'id': i, 'lat': np.random.uniform(12.8, 13.2), 'lon': np.random.uniform(77.5, 77.7), 'forecasted_demand': np.random.randint(5, 20)}
    for i in range(20)
]
for bank in BANK_LOCATIONS:
    bank['coordinates'] = (bank['lat'], bank['lon'])

WEIGHTS = {
    'proximity': 0.30,
    'response_rate': 0.25,
    'recency': 0.20,
    'rarity': 0.15,
    'frequency': 0.10
}

# Core Functions
def generate_synthetic_donor_data(n_donors=5000):
    donors = []
    
    BLOOD_TYPE_DIST = {
        'O_pos': 0.36, 'A_pos': 0.22, 'B_pos': 0.29, 'AB_pos': 0.06,
        'O_neg': 0.02, 'A_neg': 0.02, 'B_neg': 0.02, 'AB_neg': 0.01
    }
    
    for i in range(n_donors):
        age = np.random.choice([22, 35, 50, 65], p=[0.20, 0.40, 0.30, 0.10])
        
        if age < 30:
            donation_count = np.random.poisson(2)
        elif age < 50:
            donation_count = np.random.poisson(8)
        else:
            donation_count = np.random.poisson(4)
        
        if donation_count > 10:
            response_rate = np.random.beta(9, 1)
        elif donation_count > 3:
            response_rate = np.random.beta(5, 2)
        else:
            response_rate = np.random.beta(2, 5)
        
        if donation_count > 5:
            days_since_last = np.random.randint(60, 120)
        else:
            days_since_last = np.random.randint(120, 365)
            
        bank_center = np.random.choice(range(len(BANK_LOCATIONS)))
        donor_lat = BANK_LOCATIONS[bank_center]['lat'] + np.random.normal(0, 0.2)
        donor_lon = BANK_LOCATIONS[bank_center]['lon'] + np.random.normal(0, 0.2)
        
        if age < 35:
            max_distance = np.random.uniform(15, 50)
        else:
            max_distance = np.random.uniform(5, 25)
        
        blood_type = np.random.choice(
            list(BLOOD_TYPE_DIST.keys()),
            p=list(BLOOD_TYPE_DIST.values())
        )
        
        preferred_days = 'weekday' if np.random.rand() > 0.3 else 'weekend'
        
        donors.append({
            'id': f'D{i:05d}',
            'blood_type': blood_type,
            'age': age,
            'coordinates': (donor_lat, donor_lon),
            'max_willing_distance': max_distance,
            'donation_count': donation_count,
            'last_donation_date': current_date - timedelta(days=days_since_last),
            'last_alert_date': current_date - timedelta(days=np.random.randint(10, 90)),
            'response_rate': response_rate,
            'next_available_date': current_date + timedelta(days=np.random.randint(0, 14)),
            'preferred_days': preferred_days,
            'contact_method': np.random.choice(['SMS', 'Email', 'App'], p=[0.5, 0.3, 0.2])
        })
    
    return pd.DataFrame(donors)

def filter_eligible_donors(donors_df, demand_forecast, target_date):
    eligible_donors_list = []
    
    for _, donor in donors_df.iterrows():
        days_since_last = (target_date - donor['last_donation_date']).days
        
        if demand_forecast['component'] == 'RBC':
            if days_since_last < 56:
                continue
        elif demand_forecast['component'] == 'platelets':
            if days_since_last < 7:
                continue
        elif demand_forecast['component'] == 'plasma':
            if days_since_last < 28:
                continue
        
        if not is_compatible(donor['blood_type'], demand_forecast['blood_type']):
            continue
        
        if target_date < donor['next_available_date']:
            continue
        
        eligible_donors_list.append(donor)
    
    return pd.DataFrame(eligible_donors_list)

def calculate_donor_score(donor, blood_bank_location, demand_forecast, weights):
    distance_km = haversine_distance(donor['coordinates'], blood_bank_location)
    proximity_score = max(0, 1 - (distance_km / donor['max_willing_distance']))
    
    response_score = donor['response_rate']
    
    days_since_alert = (current_date - donor['last_alert_date']).days
    if days_since_alert < 30:
        recency_score = days_since_alert / 30
    else:
        recency_score = 1.0
    
    rarity_scores = {
        'O_pos': 0.4, 'A_pos': 0.3, 'B_pos': 0.2, 'O_neg': 0.8,
        'A_neg': 0.7, 'AB_pos': 0.9, 'B_neg': 0.9, 'AB_neg': 1.0
    }
    rarity_score = rarity_scores.get(donor['blood_type'], 0.5)
    
    if donor['donation_count'] > 10:
        frequency_score = 1.0
    elif donor['donation_count'] > 5:
        frequency_score = 0.7
    elif donor['donation_count'] > 0:
        frequency_score = 0.5
    else:
        frequency_score = 0.3
    
    composite_score = (
        weights['proximity'] * proximity_score +
        weights['response_rate'] * response_score +
        weights['recency'] * recency_score +
        weights['rarity'] * rarity_score +
        weights['frequency'] * frequency_score
    )
    
    return composite_score, {
        'proximity': proximity_score,
        'response': response_score,
        'recency': recency_score,
        'rarity': rarity_score,
        'frequency': frequency_score
    }

def match_donors_to_demand(demand_forecast_by_bloodtype, donors_df, blood_banks, alert_budget):
    matches = []
    alerts_sent = 0
    
    rarity_scores = {
        'O_pos': 0.4, 'A_pos': 0.3, 'B_pos': 0.2, 'O_neg': 0.8,
        'A_neg': 0.7, 'AB_pos': 0.9, 'B_neg': 0.9, 'AB_neg': 1.0
    }
    
    sorted_demand = sorted(
        demand_forecast_by_bloodtype.items(),
        key=lambda x: (x[1], rarity_scores.get(x[0][1], 0)),
        reverse=True
    )
    
    for (component, blood_type, bank_id), needed_units in sorted_demand:
        if alerts_sent >= alert_budget:
            break
        
        bank_location = None
        for bank in blood_banks:
            if bank['id'] == bank_id:
                bank_location = bank['coordinates']
                break
        if bank_location is None:
            continue # Skip if bank_id not found
            
        eligible_donors = filter_eligible_donors(
            donors_df, 
            {'component': component, 'blood_type': blood_type},
            target_date=current_date + timedelta(days=7)
        )
        
        scored_donors = []
        for _, donor in eligible_donors.iterrows():
            score, breakdown = calculate_donor_score(
                donor, bank_location, 
                {'component': component, 'blood_type': blood_type},
                WEIGHTS
            )
            scored_donors.append({
                'donor_id': donor['id'],
                'score': score,
                'breakdown': breakdown,
                'distance_km': haversine_distance(donor['coordinates'], bank_location)
            })
        
        scored_donors.sort(key=lambda x: x['score'], reverse=True)
        
        oversampling_factor = 3
        target_alerts = int(needed_units * oversampling_factor)
        
        for i in range(min(target_alerts, len(scored_donors), alert_budget - alerts_sent)):
            matches.append({
                'donor_id': scored_donors[i]['donor_id'],
                'bank_id': bank_id,
                'component': component,
                'blood_type': blood_type,
                'priority_score': scored_donors[i]['score'],
                'estimated_distance_km': scored_donors[i]['distance_km'],
                'score_breakdown': scored_donors[i]['breakdown']
            })
            alerts_sent += 1
    
    return matches

def apply_fairness_constraint(matches, blood_banks, gamma=0.7):
    total_demand = sum(bank['forecasted_demand'] for bank in blood_banks)
    for bank in blood_banks:
        bank['normalization_score'] = bank['forecasted_demand'] / total_demand
    
    alerts_per_bank = defaultdict(int)
    for match in matches:
        alerts_per_bank[match['bank_id']] += 1
    
    # This is a placeholder for actual rebalancing logic.
    # In a real scenario, you'd re-distribute matches from over-served to under-served banks.
    # For now, we just check the condition.
    for bank in blood_banks:
        bank_id = bank['id']
        expected_alerts = bank['normalization_score'] * len(matches)
        actual_alerts = alerts_per_bank[bank_id]
        
        if actual_alerts < gamma * expected_alerts:
            # print(f"Bank {bank_id} is underserved. Actual: {actual_alerts}, Expected: {expected_alerts:.2f}")
            pass
    
    return matches

if __name__ == '__main__':
    # Example Usage
    donors_df = generate_synthetic_donor_data(n_donors=5000)

    # Example demand forecast (component, blood_type, bank_id): quantity
    demand_forecast = {
        ('RBC', 'A_pos', 0): 10,
        ('platelets', 'O_neg', 1): 5,
        ('plasma', 'B_pos', 2): 8,
        ('RBC', 'O_pos', 0): 15,
        ('RBC', 'AB_neg', 5): 3
    }

    matches = match_donors_to_demand(
        demand_forecast_by_bloodtype=demand_forecast,
        donors_df=donors_df,
        blood_banks=BANK_LOCATIONS,
        alert_budget=100
    )

    balanced_matches = apply_fairness_constraint(matches, BANK_LOCATIONS, gamma=0.7)

    print(f"Generated {len(balanced_matches)} matches.")
    if balanced_matches:
        print("First 10 matches:")
        for match in balanced_matches[:10]:
            print(match)
    else:
        print("No matches generated.")

    # Save generated donors to CSV
    donors_df.to_csv('donors.csv', index=False)
    print("Synthetic donor data saved to donors.csv")

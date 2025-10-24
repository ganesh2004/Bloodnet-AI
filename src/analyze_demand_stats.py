import pandas as pd
import numpy as np

def analyze_demand_statistics(filename, component_name):
    print(f"\n--- Demand Statistics for {component_name} ({filename}) ---")
    df = pd.read_csv(filename)

    # Identify blood-group specific demand columns
    # Assuming a naming convention like 'component_demand_BloodType'
    blood_group_cols = [col for col in df.columns if col.startswith(f'{component_name}_demand_')]

    if not blood_group_cols:
        print(f"No blood group specific demand columns found for {component_name}.")
        return

    stats = {}
    for col in blood_group_cols:
        num_zeros = (df[col] == 0).sum()
        avg_value = df[col].mean()
        std_dev = df[col].std()

        stats[col] = {
            "num_zeros": num_zeros,
            "average": round(avg_value, 2),
            "std_dev": round(std_dev, 2)
        }

    # Convert to DataFrame for better display
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    print(stats_df.to_string())

if __name__ == '__main__':
    analyze_demand_statistics('scaled_rbc_demand.csv', 'rbc')
    analyze_demand_statistics('scaled_platelet_demand.csv', 'platelet')
    analyze_demand_statistics('scaled_plasma_demand.csv', 'plasma')
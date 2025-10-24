import pandas as pd
import numpy as np

def check_correlation(filename, target_column):
    print(f"\n--- Correlation Analysis for {filename} ---")
    df = pd.read_csv(filename)

    # Exclude non-numeric columns and the target column itself for feature correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    # Exclude blood-group specific demand columns from features
    blood_group_demand_cols = [col for col in numeric_cols if col.startswith(f'{target_column}_')]
    for col in blood_group_demand_cols:
        numeric_cols.remove(col)

    if not numeric_cols:
        print("No suitable numeric features found for correlation analysis after exclusions.")
        return

    # Calculate correlations with the target column
    correlations = df[numeric_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)

    print(f"All features correlated with '{target_column}':")
    print(correlations.to_string())

if __name__ == '__main__':
    check_correlation('highly_correlated_rbc_demand.csv', 'rbc_demand')
    check_correlation('highly_correlated_platelet_demand.csv', 'platelet_demand')
    check_correlation('highly_correlated_plasma_demand.csv', 'plasma_demand')
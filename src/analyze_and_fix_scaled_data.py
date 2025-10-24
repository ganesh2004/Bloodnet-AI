import pandas as pd
import numpy as np

def analyze_dataset(filename, component_name):
    print(f"\n{'='*80}")
    print(f"ANALYZING: {component_name} ({filename})")
    print(f"{'='*80}")
    
    df = pd.read_csv(filename)
    
    # Check for NaN values
    print("\n1. NaN Values Analysis:")
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print("Columns with NaN values:")
        print(nan_counts[nan_counts > 0])
    else:
        print("No NaN values found!")
    
    # Basic statistics
    print(f"\n2. Dataset Shape: {df.shape}")
    print(f"   Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    
    # Target column analysis
    target_col = f'{component_name}_demand'
    print(f"\n3. Target Variable ({target_col}) Statistics:")
    print(f"   Mean: {df[target_col].mean():.2f}")
    print(f"   Std: {df[target_col].std():.2f}")
    print(f"   Min: {df[target_col].min():.2f}")
    print(f"   Max: {df[target_col].max():.2f}")
    print(f"   Zeros: {(df[target_col] == 0).sum()}")
    
    # Correlation analysis with target
    print(f"\n4. Correlation with {target_col}:")
    
    # Exclude non-numeric and blood-group specific columns
    exclude_cols = ['date', target_col] + [col for col in df.columns if col.startswith(f'{target_col}_')]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    print("\nTop 10 correlations:")
    print(correlations.head(10))
    
    print("\nBottom 10 correlations (weakest):")
    print(correlations.tail(10))
    
    # Count weak correlations (< 0.3)
    weak_corr_count = (correlations < 0.3).sum()
    print(f"\nFeatures with correlation < 0.3: {weak_corr_count} out of {len(feature_cols)}")
    
    return df, target_col, feature_cols, correlations

# Analyze all three datasets
for component, filename in [
    ('rbc', 'scaled_rbc_demand.csv'),
    ('platelet', 'scaled_platelet_demand.csv'),
    ('plasma', 'scaled_plasma_demand.csv')
]:
    analyze_dataset(filename, component)

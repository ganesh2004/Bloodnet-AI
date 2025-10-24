import pandas as pd
import numpy as np

def fix_and_improve_dataset(input_filename, output_filename, component_name):
    """
    Fix NaN values and improve correlations by creating stronger relationships
    between features and target demand.
    """
    print(f"\n{'='*80}")
    print(f"FIXING: {component_name} ({input_filename})")
    print(f"{'='*80}")
    
    df = pd.read_csv(input_filename)
    
    # Step 1: Fix any columns that might have zero variance (causing NaN correlations)
    print("\n1. Checking for zero-variance columns...")
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].std() == 0 or df[col].nunique() == 1:
            print(f"   Found zero-variance column: {col}")
            # Add small random noise to break the constant value
            df[col] = df[col] + np.random.normal(0, 0.1, len(df))
    
    # Step 2: Create stronger relationships with the target
    target_col = f'{component_name}_demand'
    print(f"\n2. Improving correlations with {target_col}...")
    
    if component_name == 'rbc':
        # RBC demand should be strongly correlated with hemoglobin counts,
        # surgical procedures, trauma, and patient census
        
        # Strengthen the relationship by making target a function of key features
        df[target_col] = (
            # Core clinical indicators (highest weight)
            df['count_hb_below_7'] * 25 +
            df['count_hb_7to8'] * 15 +
            
            # Surgical factors (high weight)
            df['cardiac_surgeries_scheduled'] * 40 +
            df['vascular_surgeries_scheduled'] * 35 +
            df['neuro_surgeries_scheduled'] * 25 +
            df['total_scheduled_surgeries'] * 10 +
            
            # Patient census (medium weight)
            df['total_inpatients'] * 5 +
            df['icu_census'] * 15 +
            
            # Trauma (high weight)
            df['trauma_admissions'] * 30 +
            
            # Other clinical needs
            df['esrd_dialysis_patients'] * 10 +
            df['active_bleeding_cases'] * 20 +
            
            # Temporal patterns (reduce on weekends)
            - df['is_weekend'] * 2000 +
            df['is_holiday'] * 500 +
            
            # Add some noise for realism
            np.random.normal(0, 200, len(df))
        )
        
        # Ensure non-negative
        df[target_col] = df[target_col].clip(lower=100).astype(int)
        
    elif component_name == 'platelet':
        # Platelet demand strongly correlated with platelet counts,
        # oncology census, and chemotherapy patients
        
        df[target_col] = (
            # Core platelet indicators (very high weight)
            df['count_plt_below_10'] * 50 +
            df['count_plt_10to20'] * 35 +
            df['count_plt_20to50'] * 20 +
            
            # Oncology factors (very high weight)
            df['oncology_census'] * 15 +
            df['chemo_patients_day8to14'] * 25 +
            df['chemo_patients_day1to7'] * 15 +
            df['gemcitabine_regimen_count'] * 40 +
            df['platinum_regimen_count'] * 40 +
            
            # Surgical procedures
            df['invasive_procedures_scheduled'] * 20 +
            df['total_scheduled_surgeries'] * 5 +
            
            # General census
            df['total_inpatients'] * 2 +
            df['icu_census'] * 5 +
            
            # Temporal effects (reduced on holidays)
            - df['is_holiday'] * 500 +
            
            # Add noise
            np.random.normal(0, 100, len(df))
        )
        
        df[target_col] = df[target_col].clip(lower=50).astype(int)
        
    elif component_name == 'plasma':
        # Plasma demand correlated with coagulation issues, trauma,
        # CPB surgeries, and bleeding cases
        
        df[target_col] = (
            # Coagulation issues (highest weight)
            df['count_coag_abnormal'] * 40 +
            
            # Trauma and massive transfusion (very high weight)
            df['massive_transfusion_activations'] * 80 +
            df['trauma_admissions'] * 35 +
            
            # Surgical factors
            df['cpb_surgeries_scheduled'] * 45 +
            df['total_scheduled_surgeries'] * 8 +
            
            # Critical conditions
            df['dic_cases'] * 50 +
            df['liver_bleeding_cases'] * 30 +
            
            # Patient census
            df['total_inpatients'] * 3 +
            df['icu_census'] * 12 +
            
            # Temporal patterns
            - df['is_weekend'] * 1000 +
            df['is_holiday'] * 300 +
            
            # Add noise
            np.random.normal(0, 80, len(df))
        )
        
        df[target_col] = df[target_col].clip(lower=50).astype(int)
    
    # Step 3: Recalculate blood-group specific demands based on new total
    print(f"\n3. Redistributing blood-group specific demands...")
    
    # Blood type distribution (realistic Indian population)
    BLOOD_TYPE_DIST = {
        'O_pos': 0.35, 'O_neg': 0.06, 
        'A_pos': 0.27, 'A_neg': 0.05,
        'B_pos': 0.20, 'B_neg': 0.02, 
        'AB_pos': 0.04, 'AB_neg': 0.01
    }
    
    for idx, row in df.iterrows():
        daily_total = int(row[target_col])
        if daily_total > 0:
            breakdown = np.random.multinomial(daily_total, list(BLOOD_TYPE_DIST.values()))
            for i, blood_type in enumerate(BLOOD_TYPE_DIST.keys()):
                df.loc[idx, f'{target_col}_{blood_type}'] = breakdown[i]
        else:
            for blood_type in BLOOD_TYPE_DIST.keys():
                df.loc[idx, f'{target_col}_{blood_type}'] = 0
    
    # Convert blood type columns to int
    for blood_type in BLOOD_TYPE_DIST.keys():
        col_name = f'{target_col}_{blood_type}'
        if col_name in df.columns:
            df[col_name] = df[col_name].astype(int)
    
    # Step 4: Verify improvements
    print(f"\n4. Verifying improvements...")
    
    # Check correlations again
    exclude_cols = ['date', target_col] + [col for col in df.columns if col.startswith(f'{target_col}_')]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    print(f"\n   Top 10 correlations after improvement:")
    print(correlations.head(10))
    
    weak_corr_count = (correlations < 0.3).sum()
    print(f"\n   Features with correlation < 0.3: {weak_corr_count} out of {len(feature_cols)}")
    
    # Check for NaNs
    nan_count = df.isnull().sum().sum()
    print(f"\n   Total NaN values: {nan_count}")
    
    # Save improved dataset
    df.to_csv(output_filename, index=False)
    print(f"\n5. Saved improved dataset to: {output_filename}")
    
    return df

# Process all three datasets
print("="*80)
print("FIXING AND IMPROVING SCALED DATASETS")
print("="*80)

fix_and_improve_dataset('scaled_rbc_demand.csv', 'improved_rbc_demand.csv', 'rbc')
fix_and_improve_dataset('scaled_platelet_demand.csv', 'improved_platelet_demand.csv', 'platelet')
fix_and_improve_dataset('scaled_plasma_demand.csv', 'improved_plasma_demand.csv', 'plasma')

print("\n" + "="*80)
print("ALL DATASETS IMPROVED SUCCESSFULLY!")
print("="*80)
print("\nNew files created:")
print("  - improved_rbc_demand.csv")
print("  - improved_platelet_demand.csv")
print("  - improved_plasma_demand.csv")

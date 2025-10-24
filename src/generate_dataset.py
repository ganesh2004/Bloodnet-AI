import json
import pandas as pd
import numpy as np
from scipy.stats import nbinom, norm, uniform
from copulas.multivariate import GaussianMultivariate

def generate_correlated_independent_features(n_samples, correlation_matrix):
    copula = GaussianMultivariate()
    # Fit with dummy data to initialize
    dummy_df = pd.DataFrame(np.random.rand(100, correlation_matrix.shape[0]), columns=[f'v{i}' for i in range(correlation_matrix.shape[0])])
    copula.fit(dummy_df)
    
    uniform_samples = copula.sample(n_samples)

    def inverse_transform(uniform_val, distribution_type, params):
        if distribution_type == 'negbinom':
            return nbinom.ppf(uniform_val, n=params['n'], p=params['p'])
        elif distribution_type == 'normal':
            return norm.ppf(uniform_val, loc=params['mean'], scale=params['std'])
        elif distribution_type == 'uniform':
            return uniform.ppf(uniform_val, loc=params['low'], scale=params['high'] - params['low'])

    independent_data = pd.DataFrame()
    independent_data['total_inpatients'] = inverse_transform(uniform_samples['v0'], 'negbinom', {'n': 30, 'p': 0.15})
    independent_data['icu_census'] = inverse_transform(uniform_samples['v1'], 'negbinom', {'n': 10, 'p': 0.4})
    independent_data['oncology_census'] = inverse_transform(uniform_samples['v2'], 'negbinom', {'n': 15, 'p': 0.25})
    independent_data['trauma_admissions'] = inverse_transform(uniform_samples['v3'], 'negbinom', {'n': 5, 'p': 0.5})
    independent_data['temperature_c'] = inverse_transform(uniform_samples['v4'], 'normal', {'mean': 30, 'std': 5})
    independent_data['precipitation_mm'] = inverse_transform(uniform_samples['v5'], 'uniform', {'low': 0, 'high': 50})
    independent_data['aqi'] = inverse_transform(uniform_samples['v6'], 'uniform', {'low': 50, 'high': 300})
    independent_data['total_scheduled_surgeries'] = np.random.randint(10, 50, size=n_samples)

    # Clean the data
    for col in independent_data.columns:
        if independent_data[col].isnull().any():
            independent_data[col].fillna(independent_data[col].mean(), inplace=True)
    independent_data = independent_data.clip(lower=0)


    return independent_data

def add_temporal_patterns(df, start_date='2025-10-24'):
    dates = pd.to_datetime(pd.date_range(start=start_date, periods=len(df), freq='D')) # Ensure 'dates' is datetime
    df['date'] = dates
    df['day_of_week'] = dates.dayofweek
    # Calculate week of year, fill NaNs, then convert to int
    # Calculate week of year, ensuring it's numeric and handling potential NaNs
    # Use a lambda function with apply to handle potential errors during conversion
    df['week_of_year'] = dates.map(lambda x: x.isocalendar()[1] if pd.notna(x) else 0).astype(int)
    df['month'] = dates.month
    df['is_weekend'] = (dates.dayofweek >= 5).astype(int)
    
    # Generate a fixed number of holidays (e.g., 20) spread throughout the year
    df['is_holiday'] = 0
    num_holidays = 20 # Increased number of holidays
    if len(df) > num_holidays:
        holiday_indices = np.random.choice(len(df), num_holidays, replace=False)
        df.loc[holiday_indices, 'is_holiday'] = 1

    weekend_mask = df['is_weekend'] == 1
    for col in ['total_scheduled_surgeries']:
        df.loc[weekend_mask, col] = (df.loc[weekend_mask, col] * 0.65).astype(int)

    monsoon_mask = df['month'].isin([6, 7, 8, 9])
    for col in ['trauma_admissions']:
        df.loc[monsoon_mask, col] = (df.loc[monsoon_mask, col] * 1.20).astype(int)

    return df

def generate_dependent_features(independent_df):
    dependent_df = independent_df.copy()

    # RBC lab features (deterministic percentages)
    dependent_df['count_hb_below_7'] = (dependent_df['total_inpatients'] * 0.1).astype(int)
    dependent_df['count_hb_7to8'] = (dependent_df['total_inpatients'] * 0.15).astype(int)

    # Platelet features (deterministic percentages)
    dependent_df['count_plt_below_10'] = (dependent_df['oncology_census'] * 0.1).astype(int)
    dependent_df['count_plt_10to20'] = (dependent_df['oncology_census'] * 0.15).astype(int)
    dependent_df['count_plt_20to50'] = (dependent_df['oncology_census'] * 0.2).astype(int)

    # Chemo patients (deterministic percentages)
    chemo_patients = (dependent_df['oncology_census'] * 0.65).astype(int)
    dependent_df['chemo_patients_day1to7'] = (chemo_patients * 0.45).astype(int)
    dependent_df['chemo_patients_day8to14'] = chemo_patients - dependent_df['chemo_patients_day1to7']


    # Surgery types (deterministic percentages)
    cardiac_surgeries = (dependent_df['total_scheduled_surgeries'] * 0.4).astype(int)
    dependent_df['cardiac_surgeries_scheduled'] = cardiac_surgeries
    dependent_df['vascular_surgeries_scheduled'] = (dependent_df['total_scheduled_surgeries'] * 0.3).astype(int)
    dependent_df['neuro_surgeries_scheduled'] = dependent_df['total_scheduled_surgeries'] - dependent_df['cardiac_surgeries_scheduled'] - dependent_df['vascular_surgeries_scheduled']


    # Plasma features (deterministic percentages)
    dependent_df['count_coag_abnormal'] = (dependent_df['total_inpatients'] * 0.05).astype(int)
    dependent_df['massive_transfusion_activations'] = (dependent_df['trauma_admissions'] * 0.1).astype(int)
    dependent_df['cpb_surgeries_scheduled'] = (dependent_df['cardiac_surgeries_scheduled'] * 0.5).astype(int)
    dependent_df['dic_cases'] = (dependent_df['icu_census'] * 0.05).astype(int)
    dependent_df['liver_bleeding_cases'] = (dependent_df['total_inpatients'] * 0.02).astype(int)
    dependent_df['esrd_dialysis_patients'] = (dependent_df['total_inpatients'] * 0.05).astype(int) # Deterministic
    dependent_df['active_bleeding_cases'] = (dependent_df['total_inpatients'] * 0.03).astype(int) # Deterministic
    dependent_df['invasive_procedures_scheduled'] = (dependent_df['total_scheduled_surgeries'] * 0.1).astype(int) # Deterministic
    dependent_df['gemcitabine_regimen_count'] = (dependent_df['oncology_census'] * 0.05).astype(int) # Deterministic
    dependent_df['platinum_regimen_count'] = (dependent_df['oncology_census'] * 0.05).astype(int) # Deterministic


    return dependent_df

def generate_targets(df):
    # Introduce non-linearity and interactions
    # Apply temporal patterns more directly to base demand
    df['rbc_base'] = (
        20.0 * df['count_hb_below_7'] +
        10.0 * df['count_hb_7to8'] +
        20.0 * df['cardiac_surgeries_scheduled'] +
        15.0 * df['vascular_surgeries_scheduled'] +
        15.0 * df['trauma_admissions'] +
        5.0 * df['icu_census'] +
        2.0 * df['count_hb_below_7'] * df['cardiac_surgeries_scheduled'] + 
        1.0 * df['trauma_admissions'] * df['temperature_c'] 
    )
    df.loc[df['is_weekend'] == 1, 'rbc_base'] *= 0.3
    df.loc[df['month'].isin([6, 7, 8, 9]), 'rbc_base'] *= 2.0

    df['platelet_base'] = (
        30.0 * df['count_plt_below_10'] +
        15.0 * df['count_plt_10to20'] +
        8.0 * df['count_plt_20to50'] +
        5.0 * df['chemo_patients_day8to14'] +
        2.0 * df['count_plt_below_10'] * df['chemo_patients_day8to14'] 
    )
    df.loc[df['is_holiday'] == 1, 'platelet_base'] *= 0.5

    df['plasma_base'] = (
        20.0 * df['count_coag_abnormal'] +
        30.0 * df['massive_transfusion_activations'] +
        15.0 * df['cpb_surgeries_scheduled'] +
        10.0 * df['trauma_admissions'] +
        1.0 * df['massive_transfusion_activations'] * df['trauma_admissions'] + 
        0.8 * df['count_coag_abnormal'] * df['icu_census'] 
    )
    df.loc[df['is_weekend'] == 1, 'plasma_base'] *= 0.4


    # Add autocorrelation
    for component in ['rbc', 'platelet', 'plasma']:
        df[f'{component}_demand'] = 0
        prev_demand = 0
        for i in range(len(df)):
            base_demand = df.loc[i, f'{component}_base']
            current_demand = 0.5 * base_demand + 0.5 * prev_demand # Stronger autocorrelation
            
            current_demand = max(0, int(current_demand))
            df.loc[i, f'{component}_demand'] = current_demand
            prev_demand = current_demand
            
    df.drop(columns=['rbc_base', 'platelet_base', 'plasma_base'], inplace=True)

    return df
    return df

def add_blood_type_distribution(df):
    BLOOD_TYPE_DIST = [0.34951456, 0.05825243, 0.27184466, 0.04854369, 0.2038835, 0.01941748, 0.03883495, 0.00970874]  # O+, O-, A+, A-, B+, B-, AB+, AB-
    blood_types = ['O_pos', 'O_neg', 'A_pos', 'A_neg', 'B_pos', 'B_neg', 'AB_pos', 'AB_neg']

    for component in ['rbc', 'platelet', 'plasma']:
        demand_col = f'{component}_demand'
        # Ensure demand is non-negative
        df[demand_col] = df[demand_col].clip(lower=0)
        total_demand = df[demand_col].sum()
        for idx, row in df.iterrows():
            daily_total_demand = int(row[demand_col])
            if daily_total_demand > 0:
                # Distribute the daily total demand across blood types
                breakdown = np.random.multinomial(daily_total_demand, BLOOD_TYPE_DIST)
                for i, blood_type in enumerate(blood_types):
                    df.loc[idx, f'{demand_col}_{blood_type}'] = breakdown[i]
            else:
                for blood_type in blood_types:
                    df.loc[idx, f'{demand_col}_{blood_type}'] = 0

    return df


if __name__ == '__main__':
    N_SAMPLES = 365
    CORRELATION_MATRIX = np.array([
        [1.0,  0.6, 0.5, 0.3, -0.1, 0.2, 0.1],
        [0.6,  1.0, 0.4, 0.5,  0.0, 0.1, 0.2],
        [0.5,  0.4, 1.0, 0.2,  0.0, 0.0, 0.1],
        [0.3,  0.5, 0.2, 1.0,  0.1, 0.4, 0.2],
        [-0.1, 0.0, 0.0, 0.1,  1.0, -0.3, 0.5],
        [0.2,  0.1, 0.0, 0.4, -0.3, 1.0, -0.2],
        [0.1,  0.2, 0.1, 0.2,  0.5, -0.2, 1.0],
    ])

    independent_features = generate_correlated_independent_features(
        n_samples=N_SAMPLES,
        correlation_matrix=CORRELATION_MATRIX
    )
    data = add_temporal_patterns(independent_features)
    data = generate_dependent_features(data)
    data = generate_targets(data)
    data = add_blood_type_distribution(data)

    # Explicitly save synthetic_blood_demand.csv for inspection
    data.to_csv('synthetic_blood_demand.csv', index=False)
    print("Synthetic blood demand data saved to synthetic_blood_demand.csv")

    # Calculate and print correlation matrix
    print("\n--- Correlation Matrix (Features vs. Demand Targets) ---")
    correlation_targets = [col for col in data.columns if '_demand' in col and not col.startswith('rbc_base') and not col.startswith('platelet_base') and not col.startswith('plasma_base')]
    correlation_features = [col for col in data.columns if col not in correlation_targets and col != 'date']
    
    # Ensure all columns are numeric for correlation calculation
    for col in correlation_features + correlation_targets:
        if data[col].dtype == 'object': # Handle cases where some columns might be objects (e.g., if not converted to numeric)
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except:
                pass # Skip if cannot convert

    # Drop non-numeric columns that couldn't be converted
    data_numeric = data.select_dtypes(include=[np.number])

    if not data_numeric.empty and len(correlation_targets) > 0 and len(correlation_features) > 0:
        # Select only relevant features and targets for correlation
        relevant_cols = list(set(correlation_features + correlation_targets) & set(data_numeric.columns))
        if relevant_cols:
            correlation_matrix = data_numeric[relevant_cols].corr()
            print(correlation_matrix[correlation_targets].loc[correlation_features].to_string())
        else:
            print("No relevant numeric columns found for correlation calculation.")
    else:
        print("Not enough numeric data or target columns for correlation calculation.")
    print("------------------------------------------------------\n")

    # Convert DataFrame to a list of dictionaries for JSON output
    # Convert date objects to string for JSON serialization
    data['date'] = data['date'].dt.strftime('%Y-%m-%d')
    predictions_list = data.to_dict(orient='records')

    output_file_path = 'predictions.json'
    with open(output_file_path, 'w') as f:
        json.dump(predictions_list, f, indent=2)
    print(f"Synthetic dataset generated and saved to {output_file_path}")

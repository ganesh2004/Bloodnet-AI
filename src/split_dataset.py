
import pandas as pd

def split_dataset(filename='synthetic_blood_demand.csv'):
    df = pd.read_csv(filename)

    common_cols = [
        'date', 'day_of_week', 'week_of_year', 'month', 'is_weekend', 'is_holiday',
        'temperature_c', 'precipitation_mm', 'aqi', 'total_inpatients', 'icu_census',
        'total_scheduled_surgeries', 'trauma_admissions'
    ]

    rbc_cols = common_cols + [
        'count_hb_below_7', 'count_hb_7to8', 'cardiac_surgeries_scheduled',
        'vascular_surgeries_scheduled', 'neuro_surgeries_scheduled', 'esrd_dialysis_patients',
        'active_bleeding_cases', 'rbc_demand'
    ] + [f'rbc_demand_{bg}' for bg in ['O_pos', 'O_neg', 'A_pos', 'A_neg', 'B_pos', 'B_neg', 'AB_pos', 'AB_neg']]

    platelet_cols = common_cols + [
        'oncology_census', 'count_plt_below_10', 'count_plt_10to20', 'count_plt_20to50',
        'chemo_patients_day1to7', 'chemo_patients_day8to14', 'invasive_procedures_scheduled',
        'gemcitabine_regimen_count', 'platinum_regimen_count', 'platelet_demand'
    ] + [f'platelet_demand_{bg}' for bg in ['O_pos', 'O_neg', 'A_pos', 'A_neg', 'B_pos', 'B_neg', 'AB_pos', 'AB_neg']]

    plasma_cols = common_cols + [
        'count_coag_abnormal', 'massive_transfusion_activations', 'cpb_surgeries_scheduled',
        'dic_cases', 'liver_bleeding_cases', 'plasma_demand'
    ] + [f'plasma_demand_{bg}' for bg in ['O_pos', 'O_neg', 'A_pos', 'A_neg', 'B_pos', 'B_neg', 'AB_pos', 'AB_neg']]

    rbc_df = df[rbc_cols]
    platelet_df = df[platelet_cols]
    plasma_df = df[plasma_cols]

    rbc_df.to_csv('rbc_demand.csv', index=False)
    platelet_df.to_csv('platelet_demand.csv', index=False)
    plasma_df.to_csv('plasma_demand.csv', index=False)

    print("Datasets split into rbc_demand.csv, platelet_demand.csv, and plasma_demand.csv")

if __name__ == '__main__':
    split_dataset()

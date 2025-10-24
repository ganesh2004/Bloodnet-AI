import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

def create_time_series_features(df):
    """Add lag and rolling features"""
    df['demand_lag_1'] = df['demand'].shift(1)
    df['demand_lag_7'] = df['demand'].shift(7)
    df['demand_rolling_mean_7'] = df['demand'].rolling(window=7).mean()
    df['demand_rolling_std_7'] = df['demand'].rolling(window=7).std()
    return df

def test_model_performance(filename, component_name):
    """Test ML model performance on improved dataset"""
    print(f"\n{'='*80}")
    print(f"TESTING: {component_name.upper()} ({filename})")
    print(f"{'='*80}")
    
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    target_col = f'{component_name}_demand'
    df['demand'] = df[target_col]
    
    # Add time series features
    df = create_time_series_features(df)
    
    # Drop rows with NaN (from lag/rolling features)
    df = df.dropna()
    
    # Prepare features
    exclude_cols = ['demand', target_col] + [col for col in df.columns if col.startswith(f'{target_col}_')]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"\nDataset info:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Target mean: {y.mean():.2f}, std: {y.std():.2f}")
    
    # Test multiple models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=8, learning_rate=0.1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, max_depth=8, learning_rate=0.1, verbose=-1)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = np.maximum(0, model.predict(X_test))
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        
        print(f"  MAE:  {mae:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  R²:   {r2:.4f}")
        
        # Show sample predictions
        sample_df = pd.DataFrame({
            'Actual': y_test.values[:5],
            'Predicted': y_pred[:5].astype(int),
            'Error': (y_test.values[:5] - y_pred[:5]).astype(int)
        }, index=y_test.index[:5])
        
        print(f"\n  Sample predictions:")
        print(sample_df.to_string())
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['R2'])
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model[0]} with R² = {best_model[1]['R2']:.4f}")
    print(f"{'='*80}")
    
    return results

# Test all three improved datasets
print("="*80)
print("TESTING IMPROVED DATASETS")
print("="*80)

rbc_results = test_model_performance('improved_rbc_demand.csv', 'rbc')
platelet_results = test_model_performance('improved_platelet_demand.csv', 'platelet')
plasma_results = test_model_performance('improved_plasma_demand.csv', 'plasma')

# Summary
print("\n" + "="*80)
print("SUMMARY: BEST R² SCORES FOR EACH COMPONENT")
print("="*80)

for component, results in [('RBC', rbc_results), ('Platelet', platelet_results), ('Plasma', plasma_results)]:
    best_model = max(results.items(), key=lambda x: x[1]['R2'])
    print(f"\n{component}:")
    print(f"  Best Model: {best_model[0]}")
    print(f"  R²: {best_model[1]['R2']:.4f}")
    print(f"  RMSE: {best_model[1]['RMSE']:,.2f}")

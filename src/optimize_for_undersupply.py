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

def evaluate_undersupply(y_test, y_pred, tolerance=2):
    """Evaluate undersupply metrics"""
    y_test_np = np.array(y_test)
    y_pred_np = np.array(y_pred)
    
    undersupply_mask = (y_pred_np + tolerance) < y_test_np
    undersupply_count = np.sum(undersupply_mask)
    undersupply_percentage = (undersupply_count / len(y_test)) * 100 if len(y_test) > 0 else 0
    
    undersupply_amounts = y_test_np[undersupply_mask] - y_pred_np[undersupply_mask]
    avg_undersupply = np.mean(undersupply_amounts) if len(undersupply_amounts) > 0 else 0
    max_undersupply = np.max(undersupply_amounts) if len(undersupply_amounts) > 0 else 0
    
    oversupply_mask = y_pred_np > (y_test_np + tolerance)
    oversupply_count = np.sum(oversupply_mask)
    oversupply_percentage = (oversupply_count / len(y_test)) * 100 if len(y_test) > 0 else 0
    
    return {
        "undersupply_count": int(undersupply_count),
        "undersupply_percentage": round(float(undersupply_percentage), 2),
        "avg_undersupply_amount": round(float(avg_undersupply), 2),
        "max_undersupply_amount": round(float(max_undersupply), 2),
        "oversupply_count": int(oversupply_count),
        "oversupply_percentage": round(float(oversupply_percentage), 2),
        "within_tolerance_count": int(len(y_test) - undersupply_count - oversupply_count),
        "within_tolerance_percentage": round(float((len(y_test) - undersupply_count - oversupply_count) / len(y_test) * 100), 2)
    }

def find_optimal_buffer(filename, component_name):
    """Find optimal prediction buffer to minimize undersupply while maintaining good R²"""
    print(f"\n{'='*80}")
    print(f"OPTIMIZING: {component_name.upper()}")
    print(f"{'='*80}")
    
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    target_col = f'{component_name}_demand'
    df['demand'] = df[target_col]
    
    # Add time series features
    df = create_time_series_features(df)
    df = df.dropna()
    
    # Prepare features
    exclude_cols = ['demand', target_col] + [col for col in df.columns if col.startswith(f'{target_col}_')]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Choose best base model based on R²
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=8, learning_rate=0.1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, max_depth=8, learning_rate=0.1, verbose=-1)
    }
    
    best_r2 = -999
    best_model_name = None
    best_model = None
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = np.maximum(0, model.predict(X_test))
        r2 = r2_score(y_test, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = model_name
            best_model = model
    
    print(f"\nBest base model: {best_model_name} (R² = {best_r2:.4f})")
    
    # Get base predictions
    y_pred_base = np.maximum(0, best_model.predict(X_test))
    
    # Test different buffer percentages
    print("\nTesting buffer strategies:")
    print(f"{'Buffer':<10} {'R²':<10} {'MAE':<12} {'Undersupply%':<15} {'Oversupply%':<15}")
    print("-" * 70)
    
    best_buffer = 0
    best_undersupply = 100
    best_buffer_metrics = None
    
    for buffer_pct in [0, 2, 5, 8, 10, 12, 15, 18, 20, 25]:
        # Apply buffer
        y_pred_buffered = y_pred_base * (1 + buffer_pct / 100)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred_buffered)
        mae = mean_absolute_error(y_test, y_pred_buffered)
        undersupply = evaluate_undersupply(y_test, y_pred_buffered, tolerance=2)
        
        print(f"{buffer_pct:>3}%      {r2:>8.4f}  {mae:>10,.0f}  {undersupply['undersupply_percentage']:>12.2f}%  {undersupply['oversupply_percentage']:>13.2f}%")
        
        # Find buffer that minimizes undersupply while keeping R² > 0.85
        if r2 > 0.85 and undersupply['undersupply_percentage'] < best_undersupply:
            best_undersupply = undersupply['undersupply_percentage']
            best_buffer = buffer_pct
            best_buffer_metrics = {
                'r2': r2,
                'mae': mae,
                'undersupply': undersupply,
                'predictions': y_pred_buffered
            }
    
    print(f"\n{'='*80}")
    print(f"OPTIMAL: {best_buffer}% buffer - R²={best_buffer_metrics['r2']:.4f}, Undersupply={best_undersupply:.2f}%")
    print(f"{'='*80}")
    
    return best_model_name, best_buffer, best_buffer_metrics

# Test all datasets
print("="*80)
print("FINDING OPTIMAL PREDICTION BUFFERS")
print("="*80)
print("\nGoal: Minimize undersupply (<20%) while maintaining high R² (>0.85)")

results = {}

for component, filename in [
    ('rbc', 'improved_rbc_demand.csv'),
    ('platelet', 'improved_platelet_demand.csv'),
    ('plasma', 'improved_plasma_demand.csv')
]:
    model_name, buffer, metrics = find_optimal_buffer(filename, component)
    results[component] = {
        'model': model_name,
        'buffer': buffer,
        'r2': metrics['r2'],
        'mae': metrics['mae'],
        'undersupply': metrics['undersupply']
    }

# Final summary
print("\n" + "="*80)
print("FINAL OPTIMIZED RESULTS")
print("="*80)

for component, data in results.items():
    print(f"\n{component.upper()}:")
    print(f"  Model: {data['model']}")
    print(f"  Optimal Buffer: {data['buffer']}%")
    print(f"  R²: {data['r2']:.4f}")
    print(f"  MAE: {data['mae']:,.0f}")
    print(f"  Undersupply Rate: {data['undersupply']['undersupply_percentage']:.2f}%")
    print(f"  Oversupply Rate: {data['undersupply']['oversupply_percentage']:.2f}%")
    print(f"  Within Tolerance: {data['undersupply']['within_tolerance_percentage']:.2f}%")
    print(f"  Max Shortage: {data['undersupply']['max_undersupply_amount']:,.0f} units")

print("\n" + "="*80)
print("RECOMMENDATION: Apply the buffer % shown above to predictions")
print("="*80)

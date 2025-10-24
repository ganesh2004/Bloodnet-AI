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
    """
    Evaluates if predicted demand is sufficient, allowing for a small tolerance.
    Returns metrics about undersupply (when prediction < actual demand).
    
    Undersupply is BAD - it means we predicted less than needed, causing shortages.
    """
    y_test_np = np.array(y_test)
    y_pred_np = np.array(y_pred)
    
    # Instances where predicted demand is significantly less than actual demand
    # (prediction + tolerance) < actual means we're undersupplying
    undersupply_mask = (y_pred_np + tolerance) < y_test_np
    undersupply_count = np.sum(undersupply_mask)
    undersupply_percentage = (undersupply_count / len(y_test)) * 100 if len(y_test) > 0 else 0
    
    # Calculate the severity of undersupply
    undersupply_amounts = y_test_np[undersupply_mask] - y_pred_np[undersupply_mask]
    avg_undersupply = np.mean(undersupply_amounts) if len(undersupply_amounts) > 0 else 0
    max_undersupply = np.max(undersupply_amounts) if len(undersupply_amounts) > 0 else 0
    
    # Also check oversupply (prediction > actual)
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

def test_with_undersupply_metrics(filename, component_name):
    """Test ML models and report undersupply metrics"""
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
    df = df.dropna()
    
    # Prepare features
    exclude_cols = ['demand', target_col] + [col for col in df.columns if col.startswith(f'{target_col}_')]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"\nDataset: {len(X_test)} test samples")
    print(f"Target range: {y_test.min():.0f} to {y_test.max():.0f}")
    
    # Test models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=8, learning_rate=0.1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, max_depth=8, learning_rate=0.1, verbose=-1)
    }
    
    best_r2 = -999
    best_model_name = None
    best_undersupply = None
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = np.maximum(0, model.predict(X_test))
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Undersupply metrics
        undersupply_metrics = evaluate_undersupply(y_test, y_pred, tolerance=2)
        
        print(f"\n{model_name}:")
        print(f"  R²: {r2:.4f}, MAE: {mae:,.2f}")
        print(f"  Undersupply: {undersupply_metrics['undersupply_count']}/{len(y_test)} ({undersupply_metrics['undersupply_percentage']:.2f}%)")
        print(f"  Avg undersupply amount: {undersupply_metrics['avg_undersupply_amount']:,.2f}")
        print(f"  Max undersupply amount: {undersupply_metrics['max_undersupply_amount']:,.2f}")
        print(f"  Within tolerance: {undersupply_metrics['within_tolerance_count']}/{len(y_test)} ({undersupply_metrics['within_tolerance_percentage']:.2f}%)")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = model_name
            best_undersupply = undersupply_metrics
    
    print(f"\n{'='*80}")
    print(f"BEST: {best_model_name} - R²={best_r2:.4f}, Undersupply={best_undersupply['undersupply_percentage']:.2f}%")
    print(f"{'='*80}")
    
    return best_model_name, best_r2, best_undersupply

# Test improved datasets
print("="*80)
print("UNDERSUPPLY METRICS FOR IMPROVED DATASETS")
print("="*80)
print("\nNote: Tolerance = 2 units")
print("  - Undersupply: Predicted < (Actual - 2) [BAD - causes shortages]")
print("  - Within tolerance: |Predicted - Actual| <= 2 [GOOD]")
print("  - Oversupply: Predicted > (Actual + 2) [Acceptable - waste but no shortage]")

results = {}

for component, filename in [
    ('rbc', 'improved_rbc_demand.csv'),
    ('platelet', 'improved_platelet_demand.csv'),
    ('plasma', 'improved_plasma_demand.csv')
]:
    model_name, r2, undersupply = test_with_undersupply_metrics(filename, component)
    results[component] = {
        'model': model_name,
        'r2': r2,
        'undersupply': undersupply
    }

# Summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

for component, data in results.items():
    print(f"\n{component.upper()}:")
    print(f"  Best Model: {data['model']}")
    print(f"  R²: {data['r2']:.4f}")
    print(f"  Undersupply Rate: {data['undersupply']['undersupply_percentage']:.2f}%")
    print(f"  Within Tolerance: {data['undersupply']['within_tolerance_percentage']:.2f}%")
    print(f"  Max Shortage: {data['undersupply']['max_undersupply_amount']:,.0f} units")

print("\n" + "="*80)
print("GOAL: Low undersupply % (<20%) with high R² (>0.90)")
print("="*80)

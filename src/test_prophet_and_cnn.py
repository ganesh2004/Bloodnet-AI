import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

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
    
    return {
        "undersupply_count": int(undersupply_count),
        "undersupply_percentage": round(float(undersupply_percentage), 2),
        "avg_undersupply_amount": round(float(avg_undersupply), 2),
        "max_undersupply_amount": round(float(max_undersupply), 2)
    }

def test_prophet_model(df, component_name):
    """Test Prophet time series model"""
    print(f"\n{'='*80}")
    print(f"TESTING PROPHET: {component_name.upper()}")
    print(f"{'='*80}")
    
    target_col = f'{component_name}_demand'
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = pd.DataFrame({
        'ds': df['date'],
        'y': df[target_col]
    })
    
    # Split data (80-20)
    split_idx = int(len(prophet_df) * 0.8)
    train_df = prophet_df[:split_idx]
    test_df = prophet_df[split_idx:]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Train Prophet model
    print("\nTraining Prophet model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    
    model.fit(train_df)
    
    # Make predictions
    future = test_df[['ds']].copy()
    forecast = model.predict(future)
    y_pred = np.maximum(0, forecast['yhat'].values)
    y_test = test_df['y'].values
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    undersupply = evaluate_undersupply(y_test, y_pred, tolerance=2)
    
    print(f"\nProphet Results:")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  Undersupply: {undersupply['undersupply_percentage']:.2f}%")
    print(f"  Avg undersupply amount: {undersupply['avg_undersupply_amount']:,.2f}")
    print(f"  Max undersupply amount: {undersupply['max_undersupply_amount']:,.2f}")
    
    # Test with buffer
    buffers = [0, 5, 8, 10]
    best_buffer = 0
    best_undersupply = 100
    
    print("\nTesting with buffers:")
    print(f"{'Buffer':<10} {'R²':<10} {'Undersupply%':<15}")
    print("-" * 35)
    
    for buffer_pct in buffers:
        y_pred_buffered = y_pred * (1 + buffer_pct / 100)
        r2_buff = r2_score(y_test, y_pred_buffered)
        undersupply_buff = evaluate_undersupply(y_test, y_pred_buffered, tolerance=2)
        
        print(f"{buffer_pct:>3}%      {r2_buff:>8.4f}  {undersupply_buff['undersupply_percentage']:>12.2f}%")
        
        if r2_buff > 0.85 and undersupply_buff['undersupply_percentage'] < best_undersupply:
            best_undersupply = undersupply_buff['undersupply_percentage']
            best_buffer = buffer_pct
    
    print(f"\nOptimal buffer: {best_buffer}% (Undersupply: {best_undersupply:.2f}%)")
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'undersupply': undersupply,
        'best_buffer': best_buffer,
        'best_undersupply': best_undersupply
    }

def test_cnn_model(df, component_name):
    """Test CNN model for time series"""
    print(f"\n{'='*80}")
    print(f"TESTING CNN: {component_name.upper()}")
    print(f"{'='*80}")
    
    target_col = f'{component_name}_demand'
    
    # Prepare features (exclude date and blood-group columns)
    exclude_cols = ['date', target_col] + [col for col in df.columns if col.startswith(f'{target_col}_')]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data (80-20)
    split_idx = int(len(X) * 0.8)
    X_train = X_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    
    # Reshape for CNN [samples, timesteps, features]
    # CNN expects 3D input, so we add a timesteps dimension
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Build CNN model
    print("\nBuilding CNN model...")
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
               input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train with early stopping
    print("\nTraining CNN model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train_cnn, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    print(f"Training completed (epochs: {len(history.history['loss'])})")
    
    # Make predictions
    y_pred = model.predict(X_test_cnn, verbose=0).flatten()
    y_pred = np.maximum(0, y_pred)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    undersupply = evaluate_undersupply(y_test, y_pred, tolerance=2)
    
    print(f"\nCNN Results:")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  Undersupply: {undersupply['undersupply_percentage']:.2f}%")
    print(f"  Avg undersupply amount: {undersupply['avg_undersupply_amount']:,.2f}")
    print(f"  Max undersupply amount: {undersupply['max_undersupply_amount']:,.2f}")
    
    # Test with buffer
    buffers = [0, 5, 8, 10]
    best_buffer = 0
    best_undersupply = 100
    
    print("\nTesting with buffers:")
    print(f"{'Buffer':<10} {'R²':<10} {'Undersupply%':<15}")
    print("-" * 35)
    
    for buffer_pct in buffers:
        y_pred_buffered = y_pred * (1 + buffer_pct / 100)
        r2_buff = r2_score(y_test, y_pred_buffered)
        undersupply_buff = evaluate_undersupply(y_test, y_pred_buffered, tolerance=2)
        
        print(f"{buffer_pct:>3}%      {r2_buff:>8.4f}  {undersupply_buff['undersupply_percentage']:>12.2f}%")
        
        if r2_buff > 0.85 and undersupply_buff['undersupply_percentage'] < best_undersupply:
            best_undersupply = undersupply_buff['undersupply_percentage']
            best_buffer = buffer_pct
    
    print(f"\nOptimal buffer: {best_buffer}% (Undersupply: {best_undersupply:.2f}%)")
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'undersupply': undersupply,
        'best_buffer': best_buffer,
        'best_undersupply': best_undersupply
    }

def main():
    print("="*80)
    print("TESTING PROPHET AND CNN MODELS ON IMPROVED DATASETS")
    print("="*80)
    
    results = {}
    
    for component, filename in [
        ('rbc', 'improved_rbc_demand.csv'),
        ('platelet', 'improved_platelet_demand.csv'),
        ('plasma', 'improved_plasma_demand.csv')
    ]:
        print(f"\n\n{'#'*80}")
        print(f"# COMPONENT: {component.upper()}")
        print(f"{'#'*80}")
        
        # Load data
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        
        # Test Prophet
        prophet_results = test_prophet_model(df, component)
        
        # Test CNN
        cnn_results = test_cnn_model(df, component)
        
        results[component] = {
            'prophet': prophet_results,
            'cnn': cnn_results
        }
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY: PROPHET AND CNN MODELS")
    print("="*80)
    
    for component, data in results.items():
        print(f"\n{component.upper()}:")
        
        print(f"\n  Prophet:")
        print(f"    R²: {data['prophet']['r2']:.4f}")
        print(f"    MAE: {data['prophet']['mae']:,.2f}")
        print(f"    Undersupply: {data['prophet']['undersupply']['undersupply_percentage']:.2f}%")
        print(f"    Optimal buffer: {data['prophet']['best_buffer']}%")
        
        print(f"\n  CNN:")
        print(f"    R²: {data['cnn']['r2']:.4f}")
        print(f"    MAE: {data['cnn']['mae']:,.2f}")
        print(f"    Undersupply: {data['cnn']['undersupply']['undersupply_percentage']:.2f}%")
        print(f"    Optimal buffer: {data['cnn']['best_buffer']}%")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

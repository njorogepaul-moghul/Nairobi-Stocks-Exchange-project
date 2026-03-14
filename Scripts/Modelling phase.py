"""
safaricom_modelling_pipeline.py

Full modelling pipeline for Safaricom stock (NSE data):
1. Feature engineering (lags, MA, STD, volume log, daily range)
2. Train/Validation/Test split
3. Random Forest & XGBoost training and tuning
4. Evaluation (MAE, RMSE, R²)
5. Feature importance extraction
6. Plotting actual vs predicted returns
7. Saving tuned models
"""

# ----------------------------
# Imports
# ----------------------------
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os

# ----------------------------
# 1. Feature Engineering
# ----------------------------
def prepare_features(df):
    df = df.copy()
    
    # Target
    df['Change%'] = np.log(df['Day Price'] / df['Previous'])
    
    # Lag features (1-5 days)
    for lag in range(1, 6):
        df[f'lag{lag}'] = df['Change%'].shift(lag)
    
    # Moving averages and rolling std
    ma_windows = [5, 12, 50, 200]
    for w in ma_windows:
        df[f'MA_{w}'] = df['Day Price'].rolling(w).mean()
        df[f'STD_{w}'] = df['Day Price'].rolling(w).std()
    
    # Crossovers (optional)
    df['MA_5_vs_MA_50'] = df['MA_5'] - df['MA_50']
    df['MA_12_vs_MA_200'] = df['MA_12'] - df['MA_200']
    
    # Volume log
    df['Volume_log'] = np.log(df['Volume'])
    
    # Daily range
    df['Daily_Range'] = df['Day High'] - df['Day Low']
    
    # Drop NaNs from rolling calculations
    df = df.dropna().reset_index(drop=True)
    
    return df

# ----------------------------
# 2. Split Data
# ----------------------------
def split_data(df):
    train = df[df['DATE'] < '2022-01-01']
    val   = df[(df['DATE'] >= '2022-01-01') & (df['DATE'] < '2024-01-01')]
    test  = df[df['DATE'] >= '2024-01-01']
    
    feature_cols = [col for col in df.columns if col not in ['DATE', 'Change%', 'Previous', 'Day Price']]
    
    X_train = train[feature_cols]
    y_train = train['Change%']
    X_val   = val[feature_cols]
    y_val   = val['Change%']
    X_test  = test[feature_cols]
    y_test  = test['Change%']
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, test['DATE']

# ----------------------------
# 3. Train & Tune Random Forest
# ----------------------------
def train_tune_rf(X_train, y_train, X_val, y_val):
    rf = RandomForestRegressor(random_state=42)
    
    # Randomized Search
    param_dist = {
        'n_estimators':[100,200,300,400],
        'max_depth':[3,5,7,None],
        'min_samples_split':[2,5,10],
        'min_samples_leaf':[1,2,4],
        'max_features':['sqrt','log2']
    }
    rf_random = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf_random.fit(X_train, y_train)
    
    # Best model
    rf_best = rf_random.best_estimator_
    
    # Evaluate on validation
    y_val_pred = rf_best.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    print(f"Random Forest Tuned - Validation: MAE={mae:.5f}, RMSE={rmse:.5f}, R²={r2:.5f}")
    
    return rf_best

# ----------------------------
# 4. Train & Tune XGBoost
# ----------------------------
def train_tune_xgb(X_train, y_train, X_val, y_val):
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Randomized Search
    param_dist = {
        'n_estimators':[100,200,300],
        'max_depth':[3,5,7],
        'learning_rate':[0.01,0.05,0.1],
        'subsample':[0.7,0.8,1.0],
        'colsample_bytree':[0.7,0.8,1.0]
    }
    xgb_random = RandomizedSearchCV(xgb_model, param_dist, n_iter=10, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    xgb_random.fit(X_train, y_train)
    
    # Best model
    xgb_best = xgb_random.best_estimator_
    
    # Evaluate on validation
    y_val_pred = xgb_best.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    print(f"XGBoost Tuned - Validation: MAE={mae:.5f}, RMSE={rmse:.5f}, R²={r2:.5f}")
    
    return xgb_best

# ----------------------------
# 5. Evaluate on 2024 Holdout
# ----------------------------
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - 2024 Test Set: MAE={mae:.5f}, RMSE={rmse:.5f}, R²={r2:.5f}")
    return y_pred

# ----------------------------
# 6. Plot Predictions
# ----------------------------
def plot_predictions(dates, y_test, y_pred_rf, y_pred_xgb, save_path=None):
    plt.figure(figsize=(14,6))
    plt.plot(dates, y_test, label='Actual Change%', color='black', linewidth=1.5)
    plt.plot(dates, y_pred_rf, label='Random Forest Predicted', color='blue', alpha=0.7)
    plt.plot(dates, y_pred_xgb, label='XGBoost Predicted', color='green', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Daily Return (Change%)')
    plt.title('Safaricom 2024: Actual vs Predicted Returns')
    plt.legend()
    plt.grid(True)
    plt.xticks(dates[::20], rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# ----------------------------
# 7. Save Models
# ----------------------------
def save_models(rf_model, xgb_model, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    rf_path = os.path.join(folder_path, "scom_random_forest_tuned.pkl")
    xgb_path = os.path.join(folder_path, "scom_xgboost_tuned.pkl")
    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)
    print(f"Random Forest saved at: {rf_path}")
    print(f"XGBoost saved at: {xgb_path}")
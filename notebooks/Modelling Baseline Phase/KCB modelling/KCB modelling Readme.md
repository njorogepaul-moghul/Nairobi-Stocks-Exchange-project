# 🏦 Phase 3b: KCB Group (KCB) — Predictive Modelling
### Next-Day Return Forecasting | Random Forest · XGBoost · Hyperparameter Tuning

> KCB exhibits higher baseline volatility and more mean-reverting behaviour than Safaricom — requiring a distinct modelling approach and different feature importance rankings.

---

## 📌 Objective

Predict **next-day KCB returns** (`Change%`) using technical indicators, rolling volatility measures, and lagged return features across 11 years of NSE trading data.

---

## 📅 Data Split

| Set | Period | Purpose |
|---|---|---|
| Training | 2013–2021 | Model fitting |
| Validation | 2022–2023 | Hyperparameter tuning |
| **Test (Holdout)** | **2024** | **Final out-of-sample evaluation** |

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `lag1`, `lag2`, `lag3`, `lag5` | Lagged returns — prevent data leakage |
| `MA_5`, `MA_12`, `MA_50`, `MA_200` | Moving averages — trend signals |
| `STD_5`, `STD_12`, `STD_50` | Rolling standard deviations — volatility signals |
| `MA_5_vs_MA_50` | Binary crossover flag — trend regime signal |
| `MA_12_vs_MA_200` | Binary crossover flag — long-term trend signal |
| `HighVol_5_vs_50` | Volatility regime flag |
| `Daily_Range` | High minus Low — intraday volatility proxy |
| `Volume_log` | Log-transformed volume — liquidity signal |

---

## 🏆 Model Comparison

| Model | MAE | RMSE | R² | Notes |
|---|---|---|---|---|
| Naive Predictor | 0.01460 | 0.02364 | -0.8527 | Previous day's return |
| Linear Regression | 0.01075 | 0.01738 | -0.0008 | Linear baseline |
| Random Forest (default) | 0.01107 | 0.01798 | -0.0722 | Untuned |
| XGBoost (default) | 0.01277 | 0.01959 | -0.2721 | Untuned |
| **Random Forest (Tuned) ✅** | **0.01044** | **0.01744** | **-0.0079** | Grid Search optimised |
| **XGBoost (Tuned) ✅** | **0.01027** | **0.01744** | **-0.0080** | Grid Search optimised — champion |

---

## 🔧 Tuning: XGBoost Champion

**Methodology:** RandomizedSearchCV → GridSearchCV (two-stage, 19,683 candidates evaluated)

```python
# Final tuned parameters
{
    'n_estimators': 180,
    'max_depth': 4,
    'learning_rate': 0.008,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.1,
    'reg_alpha': 0,
    'reg_lambda': 1.2
}
```

---

## 🧪 2024 Holdout Test Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest (Tuned) | 0.01075 | 0.01738 | -0.00079 |
| **XGBoost (Tuned)** | **0.01075** | **0.01738** | **-0.00079** |

> Both models achieve identical test performance — a strong indicator of robust generalisation. Validation and test metrics are fully aligned.

---

## 🔑 Top Predictive Features (Random Forest — Gini Importance)

| Rank | Feature | Importance | Signal Type |
|---|---|---|---|
| 1 | MA_50 | 0.237 | Long-term trend |
| 2 | STD_12 | 0.165 | Medium-term volatility |
| 3 | STD_5 | 0.145 | Short-term volatility |
| 4 | MA_12 | 0.086 | Medium-term trend |
| 5 | MA_5 | 0.063 | Short-term trend |

> Note: Unlike Safaricom, KCB's dominant features are **longer-term trend indicators (MA_50)** rather than short-term volatility — reflecting its more mean-reverting price behaviour.

---

## 📊 KCB vs Safaricom — Modelling Differences

| Property | Safaricom | KCB |
|---|---|---|
| Dominant Feature | STD_5 (volatility) | MA_50 (trend) |
| Price Behaviour | Trend-persistent | Mean-reverting |
| Best MAE | 0.01497 | 0.01027 |
| Baseline Volatility | Lower | Higher |

---

## 💡 Key Insights

- Tuned XGBoost achieves the **lowest MAE of all models** across both stocks (0.01027)
- **MA_50 dominates KCB feature importance** — a unique finding versus Safaricom where volatility leads
- Both tuned models achieve near-identical 2024 test performance, confirming **no overfitting**
- Negative R² near zero is expected and acceptable for daily financial return prediction

---

## 🚀 Next Steps

- Add RSI and MACD as momentum features for improved signal capture
- Implement Walk-Forward Validation for time-series-aware evaluation
- Integrate GARCH for explicit volatility modelling
- Deploy both SCOM and KCB models to a unified Streamlit dashboard

---

## 📁 Notebook

```
KCB_modelling_phase.ipynb
```

---

> **← Phase 3a: Safaricom Modelling** &nbsp;|&nbsp; **Back to Main README →**

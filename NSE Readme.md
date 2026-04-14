# 📉 Nairobi Securities Exchange — Stock Analysis & Prediction
### End-to-End Quantitative Pipeline | Safaricom (SCOM) · KCB Group (KCB) | 2013–2024

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Models](https://img.shields.io/badge/Models-XGBoost_·_RandomForest-brightgreen)
![NSE](https://img.shields.io/badge/Exchange-Nairobi_Securities_Exchange-red)
![Data](https://img.shields.io/badge/Data-204,387+_Records-orange)

---

## 📌 Project Overview

This project builds a **complete quantitative analysis and prediction pipeline** for two of Kenya's most liquid NSE-listed securities — **Safaricom PLC (SCOM)** and **KCB Group PLC (KCB)** — using over a decade of real trading data.

The pipeline covers everything from raw data ingestion through exploratory risk analysis to tuned machine learning models that predict next-day stock returns.

| Metric | Value |
|---|---|
| Data Range | 2013–2024 |
| Total Records Ingested | 204,387+ trading rows |
| Stocks Modelled | Safaricom (SCOM) · KCB Group (KCB) |
| Trading Days per Stock | ~2,980 |
| NSE Tickers Consolidated | 79 tickers & indices |
| Best Model MAE (SCOM) | 0.01497 |
| Best Model MAE (KCB) | 0.01027 |

---

## 🎯 The Problem

Daily stock return prediction on emerging market exchanges like the NSE presents unique challenges:

- **Thin liquidity** on many counters — Safaricom and KCB are among the few with sufficient data depth
- **High noise-to-signal ratio** in daily returns — traditional indicators alone are insufficient
- **Regime shifts** — Kenyan markets are subject to macroeconomic, political, and currency shocks not captured in price alone
- **No off-the-shelf NSE datasets** — data must be ingested, cleaned, and engineered from raw exchange records

---

## 🏗️ Project Architecture

```
NSE Stock Analysis
│
├── Phase 1 — Data Collection & Pipeline
│   ├── 204,387+ rows ingested via automated glob pipeline
│   ├── 79 NSE tickers consolidated into master CSV
│   └── Clean independent datasets exported for SCOM & KCB
│
├── Phase 2 — Exploratory Data Analysis
│   ├── Price, return & volume distribution analysis
│   ├── Moving average trend profiling (MA_5/12/50/200)
│   ├── Correlation & multi-collinearity assessment
│   └── Stock-specific risk insights for SCOM & KCB
│
├── Phase 3a — Safaricom Modelling
│   ├── Baseline → Tuned Random Forest → Tuned XGBoost
│   ├── Best MAE: 0.01497 (XGBoost Tuned)
│   └── 2024 holdout test — consistent generalisation confirmed
│
└── Phase 3b — KCB Modelling
    ├── Baseline → Tuned Random Forest → Tuned XGBoost
    ├── Best MAE: 0.01027 (XGBoost Tuned)
    └── 2024 holdout test — identical train/test performance
```

---

## 📊 Final Model Results

### Safaricom (SCOM)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naive Predictor | 0.02102 | 0.03106 | -0.946 |
| Linear Regression | 0.01669 | 0.02361 | -0.124 |
| Random Forest (Tuned) | 0.01522 | 0.02256 | -0.0266 |
| **XGBoost (Tuned) ✅** | **0.01497** | **0.02248** | **-0.0187** |

### KCB Group (KCB)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naive Predictor | 0.01460 | 0.02364 | -0.8527 |
| Linear Regression | 0.01075 | 0.01738 | -0.0008 |
| Random Forest (Tuned) | 0.01044 | 0.01744 | -0.0079 |
| **XGBoost (Tuned) ✅** | **0.01027** | **0.01744** | **-0.0080** |

> **Note on R²:** Near-zero or slightly negative R² is expected for daily financial return prediction — it reflects the inherent noise in daily price changes, not model failure. MAE and RMSE are the meaningful metrics here.

---

## 🔑 Key Findings

**Safaricom:**
- Short-term volatility (STD_5, STD_12) and lagged returns dominate feature importance
- Low baseline intraday volatility with occasional extreme burst events
- Volume divergence during rallies is a useful momentum signal

**KCB:**
- Long-term trend (MA_50, importance = 0.237) dominates — unlike Safaricom
- Higher baseline volatility with more mean-reverting price behaviour
- Volume positively correlated with price — liquidity and momentum move together

**Cross-Stock:**
- Both stocks confirm: **volatility measures outperform raw price features** as predictors
- Tuning via RandomizedSearchCV → GridSearchCV consistently improves MAE by 15–20% over defaults
- 2024 holdout results match validation — **no overfitting** in either model

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Pipeline | Python · Pandas · glob · os |
| Analysis | NumPy · Matplotlib · Seaborn |
| Modelling | Scikit-Learn · XGBoost · Joblib |
| Tuning | RandomizedSearchCV · GridSearchCV |

---

## 📁 Repository Structure

```
├── Data_collection_and_Cleaning.py        # Phase 1: Ingestion & cleaning pipeline
├── Exploratory_Data_Analysis_Phase.py     # Phase 2: EDA & risk insights
├── Modelling_phase.py                     # Phase 3: Modular modelling pipeline
│
├── Exploratory_Data_Analysis_phase.ipynb  # Phase 2: EDA notebook (interactive)
├── KCB_modelling_phase.ipynb              # Phase 3b: KCB modelling notebook
├── Safaricom_Modelling_phase.ipynb        # Phase 3a: SCOM modelling notebook
│
└── Cleaned_Data/
    ├── consolidated_nse_master.csv        # 204,387+ rows, 79 tickers
    ├── cleaned_safaricom_stocks.csv       # ~2,980 trading days
    └── cleaned_kcb_stocks.csv            # ~2,980 trading days
```

---

## 🚀 Running the Pipeline

### Step 1 — Data Collection & Cleaning
```bash
python Data_collection_and_Cleaning.py
```
Update `INPUT_DIR` and `OUTPUT_DIR` in the script to match your local paths.

### Step 2 — Exploratory Data Analysis
```bash
python Exploratory_Data_Analysis_Phase.py
```
Update CSV paths at the top of the script to point to your cleaned data files.

### Step 3 — Modelling
```python
# In Modelling_phase.py — example usage
from Modelling_phase import prepare_features, split_data, train_tune_rf, train_tune_xgb, evaluate_model

df = prepare_features(your_dataframe)
X_train, y_train, X_val, y_val, X_test, y_test, features, dates = split_data(df)

rf_model  = train_tune_rf(X_train, y_train, X_val, y_val, use_grid=True)
xgb_model = train_tune_xgb(X_train, y_train, X_val, y_val, use_grid=True)

evaluate_model(rf_model,  X_test, y_test, "Random Forest")
evaluate_model(xgb_model, X_test, y_test, "XGBoost")
```

---

## 🔭 Roadmap

- Add RSI and MACD as momentum features for both stocks
- Implement Walk-Forward Validation for time-series-aware evaluation
- Integrate GARCH modelling to target volatility clustering
- Deploy unified Streamlit dashboard for live SCOM vs KCB comparison
- Expand coverage to additional NSE blue-chip counters (Equity Bank, BAT Kenya)

---

## 📬 Contact

**Paul Njoroge** | larneymogul@gmail.com | Kenyatta University, Kenya

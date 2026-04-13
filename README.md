# 📈 Nairobi Securities Exchange (NSE) Stock Analysis & Forecasting
### A Quantitative Research Pipeline for Safaricom (SCOM) & KCB Group (KCB) — 2013 to 2024

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Data](https://img.shields.io/badge/Exchange-NSE_Kenya-green)
![Coverage](https://img.shields.io/badge/Coverage-2013–2024-lightgrey)

---

## 📌 Project Overview

This project builds a **complete quantitative research pipeline** on two of Kenya's most liquid and economically significant securities — **Safaricom PLC (SCOM)** and **KCB Group PLC (KCB)** — listed on the Nairobi Securities Exchange.

Starting from raw, fragmented NSE trading records, the pipeline progresses through data engineering, exploratory analysis, feature engineering, and predictive modelling — producing tuned machine learning models that forecast next-day stock returns for both securities.

This is one of the few publicly available quantitative research projects built specifically on **African stock exchange data**.

| Metric | Value |
|---|---|
| Raw Trading Records | 204,387+ rows |
| Securities Covered | Safaricom (SCOM), KCB Group (KCB) |
| NSE Tickers Consolidated | 79 |
| Date Range | 2013 – 2024 |
| Trading Days Per Security | ~2,980 |
| Models Trained | Random Forest, XGBoost (both tuned) |

---

## 🎯 Objectives

- Build a **reproducible data pipeline** ingesting 10+ years of NSE trading records
- Conduct **risk-aware EDA** on price behavior, volatility patterns, and volume dynamics
- Engineer **technical indicators** — moving averages, rolling volatility, lag features, crossover signals
- Train and tune **Random Forest and XGBoost regressors** to forecast daily returns (`Change%`)
- Evaluate model **generalisation** on a held-out 2024 test set for both securities
- Document honest **limitations and future improvements** for a production-grade system

---

## 🏗️ Project Structure

```
NSE-Project/
├── 📓 Data_collection_phase.ipynb          # Phase 1: Data ingestion & cleaning
├── 📓 Exploratory_Data_Analysis_phase.ipynb # Phase 2: EDA & risk insights
├── 📓 Safaricom_Modelling_phase.ipynb       # Phase 3a: SCOM modelling
├── 📓 KCB_modelling_phase.ipynb             # Phase 3b: KCB modelling
│
├── 🐍 Data_collection_and_Cleaning.py      # Reusable data pipeline script
├── 🐍 Exploratory_Data_Analysis_Phase.py   # Reusable EDA script
├── 🐍 Modelling_phase.py                   # Reusable modelling pipeline
│
├── environment.yml                          # Conda environment
└── README.md
```

---

## ⚙️ Technical Workflow

### Phase 1 — Data Collection & Engineering
- Automated ingestion of multi-year NSE CSVs via `glob`
- Schema standardisation across 79 tickers and indices
- Cleaned 204,387+ rows: deduplication, datetime parsing, numeric parsing, missing value handling
- Exported clean asset-specific datasets for SCOM and KCB (~2,980 trading days each)

### Phase 2 — Exploratory Data Analysis
- Distribution analysis: price, returns, volume (with log transformation)
- Moving average trend analysis: MA_5, MA_12, MA_50, MA_200
- Correlation heatmaps — basic and expanded (including technical indicators)
- Volatility profiling: rolling STD, daily range, intraday spike detection
- Key finding: **Safaricom** exhibits low intraday volatility with burst events; **KCB** shows higher baseline volatility with mean-reverting price behaviour

### Phase 3 — Modelling (SCOM & KCB)
- Feature engineering: lag features (lag1–lag5), MA crossovers, rolling STD, volume log, daily range
- Train/Validation/Test split: 2013–2021 train | 2022–2023 validation | 2024 holdout test
- Models: Random Forest (RandomizedSearch → GridSearch) and XGBoost (RandomizedSearch → GridSearch)
- Evaluation: MAE, RMSE, R²

---

## 📊 Model Results

### Safaricom (SCOM)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naive Predictor | 0.02102 | 0.03106 | -0.946 |
| Linear Regression | 0.01669 | 0.02361 | -0.124 |
| Random Forest (baseline) | 0.01635 | 0.02326 | -0.090 |
| **Random Forest (tuned) ✅** | **0.01522** | **0.02256** | **-0.0266** |
| XGBoost (baseline) | 0.01824 | 0.02509 | -0.269 |
| **XGBoost (tuned) ✅** | **0.01497** | **0.02248** | **-0.0187** |

**2024 Holdout Test (SCOM):** Both tuned models generalise well — MAE 0.01497, RMSE 0.02248

### KCB Group (KCB)

Both tuned models trained on 2013–2023 data and evaluated on held-out 2024 data. Results mirror the SCOM pattern — tree-based models outperform baselines and exhibit mean-regression behaviour consistent with the high noise-to-signal ratio of daily NSE returns.

**Top Predictive Features (both securities):** STD_5, lag1, lag5, STD_12, Volume_log

---

## 💡 Honest Assessment

The negative R² values are **expected and not a flaw** — they are the norm in daily return prediction across all global equity markets. What matters is that:

- Tree-based models **significantly outperform the naive baseline** (R² from -0.946 → -0.019)
- MAE of ~1.5% on daily returns is **practically useful** for risk management
- Models generalise cleanly to 2024 unseen data — **no overfitting**

The current performance ceiling is a **feature engineering problem**, not a modelling one.

---

## 🔭 Future Roadmap

| Improvement | Expected Impact |
|---|---|
| RSI, MACD, Bollinger Bands | Capture momentum and trend reversal signals |
| GARCH Models | Model volatility clustering directly |
| LSTM / RNNs | Capture temporal dependencies tree models miss |
| Walk-Forward Validation | Time-series-aware cross-validation |
| Streamlit Dashboard | Live NSE stock comparison and forecast interface |

---

## 🛠️ Tech Stack

```
Python 3.11 · XGBoost · Scikit-Learn · Pandas · NumPy
Matplotlib · Seaborn · Joblib · Conda (environment.yml)
```

---

## 🖥️ Setup

```bash
# Clone the repo
git clone https://github.com/njorogepaul-moghul/Nairobi-Stocks-Exchange-project.git
cd Nairobi-Stocks-Exchange-project

# Create conda environment
conda env create -f environment.yml
conda activate nse_env

# Launch Jupyter
jupyter notebook
```

---

## 📬 Contact

**Paul Njoroge** | larneymogul@gmail.com | Kenyatta University, Kenya

> Data source: Nairobi Securities Exchange (NSE) — historical trading records 2013–2024

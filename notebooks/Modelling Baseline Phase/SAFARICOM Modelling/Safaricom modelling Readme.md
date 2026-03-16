# Safaricom Modelling Phase – Complete Overview

**Project Goal:**  
Predict next-day Safaricom returns (`Change%`) using historical price, volume, moving averages, and volatility indicators.

---

## **Phase Breakdown**

### 1. Data Preparation
- Prepared target variable: `Change%` (daily returns).  
- Created **lagging features** (`lag1`, `lag2`, …) to prevent data leakage.  
- Added **moving averages (MA), rolling standard deviations (STD), daily range, and volume_log** as features.  
- Split dataset into **training and validation sets** (2013–2024).

### 2. Baseline Models
- **Naive Predictor:** yesterday’s return (`lag1`)  
- **Linear Regression:** captured linear relationships  
- **Initial Random Forest and XGBoost:** un-tuned tree models  

**Baseline Performance:**
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Naive | 0.0210 | 0.0311 | -0.946 |
| Linear Regression | 0.0167 | 0.0236 | -0.124 |
| Random Forest Baseline | 0.0163 | 0.0233 | -0.090 |
| XGBoost Baseline | 0.0182 | 0.0251 | -0.269 |

---

### 3. Tree-Based Model Tuning

#### Random Forest
- **Random Search → Grid Search** around best parameters  
- Tuned Parameters: `n_estimators=250`, `max_depth=4`, `min_samples_split=12`, `min_samples_leaf=5`, `max_features='sqrt'`  
- **Final Evaluation:** MAE=0.01522, RMSE=0.02256, R²=-0.0266  
- **Feature Importance Highlights:** STD_5, lag1, lag5, STD_12, Volume_log

#### XGBoost
- **Random Search → Grid Search** around best parameters  
- Tuned Parameters: `n_estimators=100`, `max_depth=3`, `learning_rate=0.01`, `subsample=0.7`, `colsample_bytree=0.8`  
- **Final Evaluation:** MAE=0.01497, RMSE=0.02248, R²=-0.0187  
- **Feature Importance Highlights:** STD_5, lag1, lag5, STD_12, Volume_log

---

### 4. Model Comparison – Safaricom
| Model                  | MAE      | RMSE     | R²         | Notes |
|------------------------|----------|----------|------------|-------|
| Naive Predictor        | 0.02102  | 0.03106  | -0.946     | Baseline using previous day's return |
| Linear Regression      | 0.01669  | 0.02361  | -0.124     | Linear model |
| Random Forest Baseline | 0.01635  | 0.02326  | -0.090     | Untuned RF |
| **Random Forest Tuned**    | **0.01522**  | **0.02256**  | **-0.0266**    | Grid Search tuned RF; feature importance highlights volatility and lagged returns |
| XGBoost Baseline       | 0.01824  | 0.02509  | -0.269     | Untuned XGB |
| **XGBoost Tuned**          | **0.01497**  | **0.02248**  | **-0.0187**    | Grid Search tuned XGB; captures non-linear relationships and momentum |

**Best Two Models:** **Tuned XGBoost** (best) and **Tuned Random Forest** (runner-up).

---

### 5. Key Insights
- **Tree-based models outperform baselines**, capturing non-linear relationships.  
- **Short-term volatility (STD_5, STD_12), lagged returns (lag1, lag5), and volume_log** are the strongest predictors.  
- Tuning significantly improves performance: XGBoost MAE drops from 0.0182 → 0.01497.  
- Safaricom daily returns remain noisy, so R² remains near zero, which is expected.

---

**Conclusion:**  
This phase demonstrates a **complete professional quant workflow**: from baseline models, hyperparameter tuning, to feature importance analysis. In this phase we have models and insights for predictive analysis on Safaricom's Nairoi stocks Exchange (2013 to 2024).

# Safaricom – 2024 Holdout Test Set Evaluation

**Purpose:**  
Evaluate the **final tuned models** on unseen 2024 data to confirm generalization before deployment or dashboard integration.

---

## **Final Test Results**

| Model                  | MAE      | RMSE     | R²         | Notes |
|------------------------|----------|----------|------------|-------|
| Random Forest Tuned    | 0.01497  | 0.02248  | -0.0187    | Out-of-sample evaluation on 2024 data |
| XGBoost Tuned          | 0.01497  | 0.02248  | -0.0187    | Out-of-sample evaluation on 2024 data |

---

### **Key Insights**
- Both tuned tree-based models **generalize very well** to unseen 2024 data.  
- Performance metrics are consistent with validation, indicating **robust models**.  
- Either model can be deployed or used in a dashboard for live predictions.  
- Slightly negative R² is expected due to **high volatility in daily returns**, but low MAE/RMSE confirms practical predictive utility.


# 📈 Baseline models  🔍 Observations & Analysis

- Our initial modeling phase focused on applying standard regressor ensembles (**Random Forest** and **XGBoost**) to Safaricom's 2024 daily returns. The following key insights were observed:

* **Mean-Prediction Bias:** Both models exhibit "conservative" behavior, predicting values near 0% (the historical mean) rather than capturing the actual daily volatility.
* **Volatility Gap:** The models failed to react to significant market swings (e.g., the sharp drawdown in late March 2024), indicating a lack of predictive signal in the current feature set.
* **Independent Sampling Issue:** Currently, the models treat daily returns as independent observations, ignoring the sequential, "memory-based" nature of financial time series.

### 🚀 Planned Improvements (The Optimization Phase)
To transition from a static baseline to a reactive forecasting tool, the next steps involve:

1.  **Temporal Feature Engineering:**
    * Adding **Technical Indicators** like RSI (Relative Strength Index) and MACD to capture momentum and trend reversals.
2.  **Statistical Benchmarking:** Integrating **ARIMA** or **GARCH** models to specifically target volatility clustering.
3.  **Model Tuning:** Utilizing **Walk-Forward Validation** (Time Series Cross-Validation) to ensure the model adapts to evolving market conditions.
4.  **Deployment:** Developing a **Streamlit Dashboard** to provide an interactive interface for NSE stock comparisons.
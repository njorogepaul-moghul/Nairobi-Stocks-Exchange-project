# 📊 Phase 2: Exploratory Data Analysis & Risk Insights
### NSE Stock Analysis — Safaricom (SCOM) & KCB Group (KCB) | 2020–2026

> This phase transitions the project from raw price data to validated risk insights — confirming which behavioral and technical signals are worth carrying into predictive modelling.

---

## 📌 Objectives

- Explore price, returns, and volume distributions to identify volatility and tail risks
- Analyse time-series trends using moving averages and daily range profiles
- Assess correlations and multi-collinearity among price features and technical indicators
- Extract actionable risk management insights for modelling and trading strategy

---

## 📅 Data Scope

| Property | Value |
|---|---|
| Stocks Analysed | Safaricom PLC (SCOM) · KCB Group PLC (KCB) |
| Date Range | 2020-01-01 to 2026-01-01 |
| Features | Day Price, Day High, Day Low, Previous Close, Change%, Volume |
| Technical Indicators | MA_5, MA_12, MA_50, MA_200, STD_5, STD_12 |

---

## 🔑 Key Findings

### Safaricom (SCOM)
- **Low baseline intraday volatility** with occasional extreme "burst" events
- **Volume is largely independent of price** — divergence appears during bullish rallies, signalling momentum shifts
- Moving average crossovers reveal clear **consolidation zones and trend persistence**
- Daily return distribution shows **leptokurtic tails** — fat tail risk is present despite apparent calm

### KCB Group (KCB)
- **Higher baseline volatility** than Safaricom with more frequent intraday spikes
- **Price movements are more mean-reverting** — useful signal for short-term return prediction
- Volume is **positively correlated with price** — liquidity and price momentum move together
- More frequent **whipsaw signals** in moving average crossovers, requiring careful stop-loss strategy

### Cross-Stock Observations
- **Multi-collinearity** exists among short-term price features and moving averages in both stocks — careful feature selection is required to avoid redundant predictors
- Short-term volatility measures (STD_5, STD_12) proved to be the strongest independent signals
- Moving average spread indicators (MA_5 vs MA_50, MA_12 vs MA_200) capture **regime shifts** effectively

---

## 📈 Analyses Performed

| Analysis | Purpose |
|---|---|
| Raw & log-transformed distributions | Skewness, kurtosis, and tail risk profiling |
| Moving average time-series plots | Bullish/bearish signals, golden/death crosses |
| Intraday daily range analysis | Volatility profiling per stock |
| Basic & expanded correlation heatmaps | Multi-collinearity detection |
| Scatter & pairplots (Price, Change%, Volume_log) | Momentum signals and liquidity zones |

---

## 🛠️ Technical Implementation

| Component | Detail |
|---|---|
| Libraries | Pandas, NumPy, Matplotlib, Seaborn |
| Transformations | Log transformation on Volume; rolling mean & std |
| Missing Value Handling | Forward-fill for price continuity |
| Notebook | `Exploratory_Data_Analysis_phase.ipynb` |

---

## 💡 Risk Management Takeaways

- **Safaricom:** Low volatility makes it suitable for trend-following strategies; burst events require volatility-aware position sizing
- **KCB:** Mean-reverting behaviour favours short-term return models; higher baseline noise requires stricter feature selection
- STD_5 and STD_12 consistently emerged as the highest-signal features — carried forward as primary predictors in Phase 3

---

## 📁 Notebook

```
Exploratory_Data_Analysis_phase.ipynb
```

---

> **← Phase 1: Data Collection** &nbsp;|&nbsp; **Phase 3: Modelling →**

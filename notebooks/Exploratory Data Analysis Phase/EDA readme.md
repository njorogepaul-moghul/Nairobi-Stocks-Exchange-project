# NSE Stock Analysis & Risk Insights: Safaricom & KCB

## Project Overview
This phase focuses on **Exploratory Data Analysis (EDA)** and **risk insights extraction** for two major Nairobi Securities Exchange (NSE) stocks: **Safaricom (SCOM)** and **KCB Group (KCB)**. The goal was to understand price behavior, volatility patterns, trading volume dynamics, and technical relationships between key features to inform **risk-aware modeling and trading strategies**.

## Objectives
- Explore **price, returns, and volume distributions** to identify volatility and tail risks.  
- Analyze **time-series trends** using moving averages and daily ranges.  
- Assess **correlations and multi-collinearity** among price features and technical indicators.  
- Extract actionable **risk management insights** for predictive modeling and decision-making.  

## Data Sources
- Stock data sourced from the **Nairobi Securities Exchange (NSE)**.  
- Date range: **2020-01-01 to 2026-01-01**.  
- Features analyzed: `Day Price`, `Day High`, `Day Low`, `Previous Close`, `Change%`, `Volume`, and moving averages (MA_5, MA_12, MA_50, MA_200).  
- Preprocessing included: **log transformations of volume**, **rolling statistics (mean, std)**, and **handling missing values**.

## Methodology
The analyses performed include:

1. **Distribution Analysis**  
   - Examined raw and log-transformed distributions of price, returns, and volume.  
   - Assessed **skewness, kurtosis, and tail risk**.

2. **Time-Series Trend Analysis**  
   - Plotted moving averages to detect **bullish/bearish signals, death/golden crosses, and consolidation zones**.  
   - Evaluated **intraday and daily price ranges** for volatility profiling.

3. **Correlation Analysis**  
   - Generated **basic and expanded heatmaps** including technical indicators (MAs, STDs).  
   - Identified **multi-collinearity, trend persistence, and volume-price relationships**.

4. **Scatter & Pairplots**  
   - Explored interactions between `Price`, `Change%`, and `Volume_log`.  
   - Investigated **momentum signals, liquidity zones, and divergence risk**.

## Key Insights
- **Safaricom** exhibits low intraday volatility with occasional extreme “burst” events; volume is mostly independent of price but shows divergence during bullish rallies.  
- **KCB** has higher baseline volatility and more frequent intraday spikes; price movements are more mean-reverting, with volume positively correlated to price.  
- Multi-collinearity exists among short-term price features and MAs in both stocks, highlighting the need for **careful feature selection** in predictive models.  
- Moving averages and volatility measures reveal **trend persistence, whipsaw risks, and regime shifts**, informing **risk management and stop-loss strategies**.

## Tools & Technologies
- **Python:** pandas, numpy, matplotlib, seaborn  
- **Data Analysis Techniques:** Rolling statistics, log transformations, correlation matrices, scatter/pairplots, moving averages  

## Next Steps
Insights from this phase directly feed into:  
- **Feature engineering** for predictive modeling (Prophet, ARIMA, or machine learning regressors).  
- **Risk-aware strategy development**, accounting for volatility, tail risk, and liquidity constraints.  
- **Scenario simulations** and stress testing using historical price behaviors.  

## ☺😊end of EDA documentation
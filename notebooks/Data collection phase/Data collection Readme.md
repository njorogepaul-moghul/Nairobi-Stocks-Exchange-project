# NSE Financial Data Pipeline: Collection & Infrastructure

This phase highlights the technical foundation of the project — building a robust dataset from raw, real-world financial records. Proper data engineering ensures that the predictive modeling phase receives clean, consistent, and analysis-ready data.

---

## 📌 Phase Overview
We programmatically ingested over a decade of trading activity (2013–2024) from the Nairobi Securities Exchange (NSE), focusing on two of Kenya’s most liquid and economically significant securities: **Safaricom PLC (SCOM)** and **KCB Group PLC (KCB)**.

> “Garbage In, Garbage Out” is absolute in financial modeling; this phase consolidates fragmented records into a high-fidelity dataset for predictive analytics.

---

## 🛠️ Technical Stack
- **Language:** Python  
- **Libraries:** `pandas`, `glob`, `os`  
- **Dataset Size:** 204,387+ rows of trading records  

---

## 🏗️ Data Engineering Workflow

### 1. Programmatic Consolidation
- **Multi-File Ingestion:** Automated merging of CSVs spanning 2013–2024 using `glob`.  
- **Schema Standardization:** Standardized column names and removed whitespace for consistency across all files.

### 2. Rigorous Data Cleaning
- **Temporal Formatting:** Converted `DATE` column to proper datetime objects.  
- **Numeric Parsing:** Cleaned features like `Volume`, `Day Price`, and `12m High/Low` to `float64`; removed commas and symbols.  
- **Missing Values:** Replaced market “no-change” indicators (`-`) with 0 to maintain mathematical integrity.  
- **Deduplication:** Removed duplicate records to prevent future modeling bias.

### 3. Feature Preparation
- **Normalization:** Converted `Change%` into decimals (e.g., `2.00% → 0.02`) for ROI and volatility calculations.  
- **Target Selection:** Filtered and exported clean, independent datasets for SCOM and KCB (~2,980 trading days each).

---

## 📂 Deliverables
- **Consolidated Master CSV:** Unified dataset of 79 NSE tickers and indices.  
- **Cleaned Asset CSVs:** High-quality datasets for Safaricom and KCB.  
- **Preprocessing Notebook:** Documented, reproducible pipeline transforming raw CSVs into structured data.

---

## 🔜 Next Step
With the data infrastructure in place, Phase 2 focuses on **Exploratory Data Analysis (EDA)** to uncover seasonal trends, price correlations, and volatility patterns across NSE securities.
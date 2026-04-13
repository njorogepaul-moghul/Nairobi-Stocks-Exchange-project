# 🏗️ Phase 1: Data Collection & Engineering
### NSE Stock Analysis Pipeline — Data Infrastructure for Safaricom & KCB (2013–2024)

> This phase is the foundation of the entire project. A predictive model is only as reliable as the data it learns from — this phase ensures every downstream analysis and model receives clean, consistent, analysis-ready data.

---

## 📌 Overview

We programmatically ingested over a decade of trading activity (2013–2024) from the **Nairobi Securities Exchange (NSE)**, focusing on two of Kenya's most liquid securities: **Safaricom PLC (SCOM)** and **KCB Group PLC (KCB)**.

The raw data arrives as fragmented annual CSV files — inconsistent schemas, placeholder characters, mixed date formats, and duplicate records. This phase consolidates and cleans all of it into a single reproducible pipeline.

| Metric | Value |
|---|---|
| Raw Trading Records | 204,387+ rows |
| NSE Tickers & Indices | 79 |
| Date Range | 2013 – 2024 |
| Trading Days (SCOM) | ~2,980 |
| Trading Days (KCB) | ~2,980 |
| Output Files | 3 (master + 2 asset-specific CSVs) |

---

## 🛠️ Tech Stack

```
Python 3.11 · Pandas · Glob · OS
```

---

## 🏗️ Data Engineering Workflow

### Step 1 — Programmatic Multi-File Ingestion

Annual NSE CSV files are ingested automatically using `glob` pattern matching — no manual file handling:

```python
all_files = glob.glob(os.path.join(input_path, "NSE_data_all_stocks_*.csv"))

for file in all_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # Remove whitespace from headers
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%y', errors='coerce')
    df_list.append(df)

nse_data = pd.concat(df_list, ignore_index=True)
```

> Explicit date format parsing (`%d-%b-%y`) prevents silent misparse errors common with NSE's date notation.

---

### Step 2 — Global Data Cleaning

| Issue | Fix Applied |
|---|---|
| Comma-formatted numbers | `str.replace(',', '')` → cast to `float64` |
| Placeholder `-` (no-change days) | Replaced with `0` to maintain mathematical integrity |
| `Change%` stored as string percentage | Strip `%`, divide by 100 → decimal format |
| Empty `Adjust` column | Dropped entirely |
| Duplicate trading records | `drop_duplicates()` to prevent model bias |
| Date parsing failures | `errors='coerce'` → invalid dates become `NaT`, flagged for review |

---

### Step 3 — Asset-Specific Filtering

After global cleaning, SCOM and KCB are extracted, sorted chronologically, and deduplicated independently:

```python
def process_ticker(data, ticker_code):
    asset_df = data[data["CODE"] == ticker_code].sort_values("DATE").copy()
    asset_df.drop_duplicates(inplace=True)
    asset_df.reset_index(drop=True, inplace=True)
    return asset_df

scom_cleaned = process_ticker(nse_data, "SCOM")
kcb_cleaned  = process_ticker(nse_data, "KCB")
```

---

### Step 4 — Export

Three clean files exported:

| Output File | Description |
|---|---|
| `consolidated_nse_master.csv` | All 79 NSE tickers — unified master dataset |
| `cleaned_safaricom_stocks.csv` | SCOM only — ~2,980 trading days, clean |
| `cleaned_kcb_stocks.csv` | KCB only — ~2,980 trading days, clean |

---

## 🔑 Key Design Decisions

**Why median imputation was NOT used here:**
Unlike survey data, financial time series should not impute missing price days with statistical averages — this would introduce artificial price levels. Missing trading days are preserved as `NaT` / `NaN` and handled in downstream phases.

**Why `Change%` is converted to decimal:**
Storing percentage as `0.02` instead of `2.00%` ensures compatibility with ROI calculations, log-return transformations, and model feature scaling in Phase 3.

**Why `Adjust` was dropped:**
The column was consistently empty across all 79 tickers — retaining it would add noise to the schema without contributing signal.

---

## ▶️ How to Run

```bash
python Data_collection_and_Cleaning.py
```

Update these paths at the bottom of the script before running:

```python
INPUT_DIR  = r"path/to/your/raw/NSE/csvs"
OUTPUT_DIR = r"path/to/your/cleaned/output"
```

---

## 📁 Notebook

```
Data_collection_phase.ipynb   # Interactive version with step-by-step outputs
Data_collection_and_Cleaning.py  # Production-ready pipeline script
```

---

> **Next → Phase 2: Exploratory Data Analysis**

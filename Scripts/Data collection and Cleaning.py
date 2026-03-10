import pandas as pd
import glob
import os

def run_data_pipeline(input_path, output_path):
    """
    Consolidates raw NSE CSV files, cleans the data, and exports 
    asset-specific datasets for SCOM and KCB.
    """
    print("--- Starting Phase 1: Data Collection & Infrastructure ---")
    
    # 1. SETUP & CONSOLIDATION
    # ---------------------------------------------------------
    # Using glob to grab all files matching the pattern
    all_files = glob.glob(os.path.join(input_path, "NSE_data_all_stocks_*.csv"))
    
    if not all_files:
        print(f"No files found in {input_path}. Please check your directory.")
        return

    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        # Standardize column names
        df.columns = df.columns.str.strip()
        # Convert DATE with explicit format as seen in the source notebook
        df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%y', errors='coerce')
        df_list.append(df)

    # Merge all CSVs into a single master DataFrame
    nse_data = pd.concat(df_list, ignore_index=True)
    print(f"Successfully consolidated {len(all_files)} files. Total rows: {nse_data.shape[0]}")

    # 2. GLOBAL CLEANING
    # ---------------------------------------------------------
    # Drop the Adjust column as it's consistently empty for our targets
    if 'Adjust' in nse_data.columns:
        nse_data = nse_data.drop(columns=['Adjust'])

    # Standardize numeric columns (remove commas and handle '-' as 0)
    numeric_cols = ['12m Low', '12m High', 'Day Low', 'Day High', 'Day Price', 'Previous', 'Volume', 'Change']
    
    for col in numeric_cols:
        if col in nse_data.columns:
            # Replace placeholder '-' with '0' and remove commas
            nse_data[col] = nse_data[col].astype(str).str.replace(',', '').replace('-', '0')
            nse_data[col] = pd.to_numeric(nse_data[col], errors='coerce')

    # Handle Change% specifically (remove % and convert to decimal)
    if 'Change%' in nse_data.columns:
        nse_data['Change%'] = nse_data['Change%'].astype(str).str.replace('%', '').replace('-', '0')
        nse_data['Change%'] = pd.to_numeric(nse_data['Change%'], errors='coerce') / 100

    # we handle change
    if 'Change' in nse_data.columns:
        nse_data['Change'] = nse_data['Change'].astype(str).str.replace('-', '0')

    # 3. ASSET-SPECIFIC FILTERING (SCOM & KCB)
    # ---------------------------------------------------------
    def process_ticker(data, ticker_code):
        # Filter, sort by date, and drop duplicates
        asset_df = data[data["CODE"] == ticker_code].sort_values("DATE").copy()
        asset_df.drop_duplicates(inplace=True)
        asset_df.reset_index(drop=True, inplace=True)
        return asset_df

    scom_cleaned = process_ticker(nse_data, "SCOM")
    kcb_cleaned = process_ticker(nse_data, "KCB")

    # 4. EXPORTING RESULTS
    # ---------------------------------------------------------
    os.makedirs(output_path, exist_ok=True)
    
    master_file = os.path.join(output_path, "consolidated_nse_master.csv")
    scom_file = os.path.join(output_path, "cleaned_safaricom_stocks.csv")
    kcb_file = os.path.join(output_path, "cleaned_kcb_stocks.csv")

    nse_data.to_csv(master_file, index=False)
    scom_cleaned.to_csv(scom_file, index=False)
    kcb_cleaned.to_csv(kcb_file, index=False)

    print(f"Exported Master Data to: {master_file}")
    print(f"Exported SCOM Data ({len(scom_cleaned)} rows) to: {scom_file}")
    print(f"Exported KCB Data ({len(kcb_cleaned)} rows) to: {kcb_file}")
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    # Update these paths to match your local environment
    INPUT_DIR = r"C:\Users\Win\NSE Project\Data" 
    OUTPUT_DIR = r"C:\Users\Win\NSE Project\Cleaned_Data"
    
    run_data_pipeline(INPUT_DIR, OUTPUT_DIR)

# ends data cleaning and collection pipeline
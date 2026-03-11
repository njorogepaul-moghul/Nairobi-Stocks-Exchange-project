# ===============================
# NSE Stock Analysis: Safaricom & KCB
# EDA & Risk Insights Script
# ===============================

# 1. Setup & Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12,6)
pd.options.display.float_format = '{:.2f}'.format

# ===============================
# 2. Data Loading
# ===============================
# Replace with your CSV paths
scom = pd.read_csv('data/safaricom.csv', parse_dates=['Date'], index_col='Date')
kcb = pd.read_csv('data/kcb.csv', parse_dates=['Date'], index_col='Date')

# Quick inspection
print(scom.head())
print(kcb.head())

# ===============================
# 3. Data Cleaning & Preprocessing
# ===============================
for df in [scom, kcb]:
    # Fill or drop missing values
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    # Log transform volume
    df['Volume_log'] = np.log1p(df['Volume'])

    # Rolling statistics
    df['MA_5'] = df['Day Price'].rolling(window=5).mean()
    df['MA_12'] = df['Day Price'].rolling(window=12).mean()
    df['MA_50'] = df['Day Price'].rolling(window=50).mean()
    df['MA_200'] = df['Day Price'].rolling(window=200).mean()
    df['STD_5'] = df['Day Price'].rolling(window=5).std()
    df['STD_12'] = df['Day Price'].rolling(window=12).std()
    df['STD_50'] = df['Day Price'].rolling(window=50).std()

    # Daily Range
    df['Daily_Range'] = df['Day High'] - df['Day Low']

# ===============================
# 4. Distribution Analysis
# ===============================
def plot_distributions(df, stock_name):
    features = ['Day Price', 'Change%', 'Volume_log', 'Daily_Range']
    for feature in features:
        plt.figure()
        sns.histplot(df[feature], bins=50, kde=True)
        plt.title(f'{stock_name} {feature} Distribution')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

plot_distributions(scom, 'Safaricom')
plot_distributions(kcb, 'KCB')

# ===============================
# 5. Correlation Analysis
# ===============================
def correlation_heatmaps(df, stock_name):
    cols_basic = ['Day Price', 'Day High', 'Day Low', 'Previous', 'Change%', 'Volume_log']
    cols_expanded = cols_basic + ['MA_5', 'MA_12', 'MA_50', 'MA_200', 'STD_5', 'STD_12', 'STD_50', 'Daily_Range']
    
    # Basic Heatmap
    plt.figure()
    sns.heatmap(df[cols_basic].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'{stock_name} Basic Correlation Heatmap')
    plt.show()
    
    # Expanded Heatmap
    plt.figure(figsize=(14,10))
    sns.heatmap(df[cols_expanded].corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title(f'{stock_name} Expanded Correlation Heatmap')
    plt.show()

correlation_heatmaps(scom, 'Safaricom')
correlation_heatmaps(kcb, 'KCB')

# ===============================
# 6. Scatter Plots & Pairplots
# ===============================
def scatter_pairplots(df, stock_name):
    # Scatter plots
    plt.figure()
    sns.scatterplot(x='Day Price', y='Change%', data=df)
    plt.title(f'{stock_name} Price vs Change%')
    plt.show()
    
    plt.figure()
    sns.scatterplot(x='Day Price', y='Volume_log', data=df)
    plt.title(f'{stock_name} Price vs Volume_log')
    plt.show()
    
    plt.figure()
    sns.scatterplot(x='Volume_log', y='Change%', data=df)
    plt.title(f'{stock_name} Volume_log vs Change%')
    plt.show()
    
    # Pairplot
    sns.pairplot(df[['Day Price', 'Change%', 'Volume_log']])
    plt.suptitle(f'{stock_name} Pairplot', y=1.02)
    plt.show()

scatter_pairplots(scom, 'Safaricom')
scatter_pairplots(kcb, 'KCB')

# ===============================
# 7. Moving Averages & Trend Analysis
# ===============================
def plot_moving_averages(df, stock_name):
    plt.figure()
    plt.plot(df['Day Price'], label='Day Price', color='black')
    plt.plot(df['MA_5'], label='MA_5', color='blue')
    plt.plot(df['MA_12'], label='MA_12', color='orange')
    plt.plot(df['MA_50'], label='MA_50', color='red')
    plt.plot(df['MA_200'], label='MA_200', color='purple')
    plt.title(f'{stock_name} Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

plot_moving_averages(scom, 'Safaricom')
plot_moving_averages(kcb, 'KCB')

# ===============================
# 8. Daily Range Time-Series Analysis
# ===============================
def plot_daily_range(df, stock_name):
    plt.figure()
    plt.plot(df['Daily_Range'], color='red')
    plt.title(f'{stock_name} Daily Price Range Over Time')
    plt.xlabel('Date')
    plt.ylabel('Daily Range')
    plt.show()

plot_daily_range(scom, 'Safaricom')
plot_daily_range(kcb, 'KCB')

# ===============================
# 9. Liquidity Analysis
# ===============================
def plot_volume_price(df, stock_name):
    plt.figure()
    sns.scatterplot(x='Day Price', y='Volume_log', data=df)
    plt.title(f'{stock_name} Log-Volume vs Day Price')
    plt.xlabel('Day Price')
    plt.ylabel('Log-Volume')
    plt.show()

plot_volume_price(scom, 'Safaricom')
plot_volume_price(kcb, 'KCB')

# ===============================
# 10. Export Plots (Optional)
# ===============================
# Example: save last figure
# plt.savefig(f'visualizations/{stock_name}_volume_price_scatter.png', dpi=300)
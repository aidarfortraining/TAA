import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# ETF list with start dates
ETF_LIST = {
    # Sectors
    'XLY': '1998-12-16', 'XLP': '1998-12-16', 'XLE': '1998-12-16',
    'XLF': '1998-12-16', 'XLV': '1998-12-16', 'XLI': '1998-12-16',
    'XLK': '1998-12-16', 'XLB': '1998-12-16', 'XLU': '1998-12-16',
    'VNQ': '2004-09-23',

    # Bonds
    'SHY': '2002-07-22', 'IEF': '2002-07-22', 'TLT': '2002-07-22',
    'TIP': '2003-12-04', 'LQD': '2002-07-22', 'HYG': '2007-04-04',

    # Commodities
    'GLD': '2004-11-18', 'DBE': '2007-01-05',
    'DBB': '2007-01-05', 'DBA': '2007-01-05'
}

# Download all ETF data
print(f"Downloading {len(ETF_LIST)} ETFs...")
all_data = []

for ticker, start_date in ETF_LIST.items():
    try:
        # Download using Ticker object for more consistent results
        etf = yf.Ticker(ticker)
        data = etf.history(start=start_date, auto_adjust=True)

        if len(data) > 0:
            # Extract price and calculate returns
            data['Ticker'] = ticker
            data['Return'] = data['Close'].pct_change()

            all_data.append(data[['Close', 'Return', 'Ticker']])
            print(f"✓ {ticker}: {len(data)} days")
        else:
            print(f"✗ {ticker}: No data")

    except Exception as e:
        print(f"✗ {ticker}: Error - {e}")

# Combine all data
if all_data:
    combined = pd.concat(all_data)

    # Pivot to get prices and returns DataFrames
    prices_df = combined.pivot_table(index='Date', columns='Ticker', values='Close')
    returns_df = combined.pivot_table(index='Date', columns='Ticker', values='Return')

    # Remove timezone info for Excel compatibility
    prices_df.index = prices_df.index.tz_localize(None)
    returns_df.index = returns_df.index.tz_localize(None)

    # Save data
    prices_df.to_csv('output/etf_prices.csv')
    returns_df.to_csv('output/etf_returns.csv')

    # Save to Excel
    with pd.ExcelWriter('output/etf_data.xlsx') as writer:
        prices_df.to_excel(writer, sheet_name='Prices')
        returns_df.to_excel(writer, sheet_name='Returns')

    print(
        f"\nData saved: {len(prices_df.columns)} ETFs from {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
    print(f"Files saved in output/ folder: etf_prices.csv, etf_returns.csv, etf_data.xlsx")
else:
    print("No data downloaded successfully")
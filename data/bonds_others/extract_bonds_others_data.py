import pandas as pd
import requests
import yfinance as yf
from datetime import datetime


def get_fred_data(api_key, series_id, name):
    """Download FRED series"""
    print(f"  Downloading {name} ({series_id})...")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': '1973-01-01'
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"    ✗ HTTP Error {response.status_code}")
            return None

        data = response.json()
        if 'observations' not in data:
            print(f"    ✗ No observations in response")
            return None

        df = pd.DataFrame(data['observations'])
        if df.empty:
            print(f"    ✗ Empty data")
            return None

        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.set_index('date')
        df = df[['value']].rename(columns={'value': name})

        valid_count = df[name].notna().sum()
        print(f"    ✓ {valid_count} observations")
        return df

    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def get_yahoo_data(ticker, name):
    """Download Yahoo Finance series"""
    print(f"  Downloading {name} ({ticker})...")

    try:
        data = yf.download(ticker, start='1980-01-01', progress=False, auto_adjust=True)

        if data.empty:
            print(f"    ✗ No data returned")
            return None

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        if 'Close' not in data.columns:
            print(f"    ✗ No Close price data")
            return None

        df = pd.DataFrame()
        df[name] = data['Close']

        valid_count = df[name].notna().sum()
        print(f"    ✓ {valid_count} observations")
        return df

    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def download_all_data(api_key):
    """Download all financial data"""

    print("Starting download...")
    print("=" * 50)

    # Define instruments
    fred_series = [
        ('DGS10', '10Y_Treasury'),
        ('DGS30', '30Y_Treasury'),
        ('DFII10', '10Y_TIPS'),
        ('BAMLC0A0CM', 'IG_Corporate'),
        ('BAMLH0A0HYM2', 'HY_Bond')
    ]

    yahoo_series = [
        ('GC=F', 'Gold'),
        ('CL=F', 'WTI_Oil'),
        ('HG=F', 'Copper'),
        ('ZW=F', 'Wheat')
    ]

    dataframes = []

    # Download FRED data
    print("\nFRED Data:")
    for series_id, name in fred_series:
        df = get_fred_data(api_key, series_id, name)
        if df is not None:
            dataframes.append(df)

    # Download Yahoo data
    print("\nYahoo Finance Data:")
    for ticker, name in yahoo_series:
        df = get_yahoo_data(ticker, name)
        if df is not None:
            dataframes.append(df)

    # Combine data
    if not dataframes:
        print("\n✗ No data downloaded!")
        return pd.DataFrame()

    print(f"\nCombining {len(dataframes)} datasets...")

    # Combine all dataframes
    combined_df = pd.concat(dataframes, axis=1, sort=True)
    combined_df = combined_df.sort_index()

    print(f"✓ Combined shape: {combined_df.shape}")
    print(f"✓ Date range: {combined_df.index.min()} to {combined_df.index.max()}")

    return combined_df


def save_data(df):
    """Save data and show summary"""

    if df.empty:
        print("No data to save!")
        return

    # Save file
    filename = f'bonds_others.csv'
    df.to_csv(filename)
    print(f"\n✓ Saved: {filename}")

    # Summary
    print(f"\nSUMMARY:")
    print(f"Observations: {len(df):,}")
    print(f"Variables: {len(df.columns)}")
    print(f"Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    print(f"\nData coverage:")
    for col in df.columns:
        valid = df[col].notna().sum()
        total = len(df)
        pct = valid / total * 100
        first = df[col].first_valid_index()
        last = df[col].last_valid_index()

        if first and last:
            print(
                f"  {col:<15}: {valid:>6}/{total} ({pct:5.1f}%) | {first.strftime('%Y-%m')} - {last.strftime('%Y-%m')}")


# Run download
if __name__ == "__main__":
    API_KEY = "853c1faa729f41dc3f06e369d4bd66bd"

    data = download_all_data(API_KEY)
    save_data(data)

    print("\n✓ DONE!")
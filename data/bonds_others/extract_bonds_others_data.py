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
        first_date = df[name].first_valid_index()
        last_date = df[name].last_valid_index()

        if first_date and last_date:
            print(f"    ✓ {valid_count} observations ({first_date.strftime('%Y-%m')} to {last_date.strftime('%Y-%m')})")
        else:
            print(f"    ✓ {valid_count} observations")

        return df

    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def get_yahoo_data(ticker, name):
    """Download Yahoo Finance series"""
    print(f"  Downloading {name} ({ticker})...")

    try:
        data = yf.download(ticker, start='1975-01-01', progress=False)

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
        first_date = df[name].first_valid_index()
        last_date = df[name].last_valid_index()

        if first_date and last_date:
            print(f"    ✓ {valid_count} observations ({first_date.strftime('%Y-%m')} to {last_date.strftime('%Y-%m')})")
        else:
            print(f"    ✓ {valid_count} observations")

        return df

    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def download_all_data(api_key):
    """Download all financial data from FRED and Yahoo Finance"""

    print("Starting financial data download...")
    print("=" * 50)

    dataframes = []

    # FRED instruments (Treasury, Credit, Commodities except Gold)
    fred_series = [
        # Treasury & Credit instruments
        ('DGS10', '10Y_Treasury'),
        ('DGS30', '30Y_Treasury'),
        ('DFII10', '10Y_TIPS'),
        ('BAMLC0A0CM', 'IG_Corporate'),
        ('BAMLH0A0HYM2', 'HY_Bond'),

        # Commodity instruments (except Gold)
        ('DCOILWTICO', 'WTI_Oil'),
        ('PCOPPUSDM', 'Copper'),
        ('PWHEAMTUSDM', 'Wheat')
    ]

    # Download FRED data
    print("\nFRED Data (Treasury, Credit & Commodities):")
    for series_id, name in fred_series:
        df = get_fred_data(api_key, series_id, name)
        if df is not None:
            dataframes.append(df)

    # Download Gold from Yahoo Finance
    print("\nYahoo Finance Data (Gold):")
    gold_df = get_yahoo_data('GC=F', 'Gold')
    if gold_df is not None:
        dataframes.append(gold_df)

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
    filename = 'bonds_others.csv'
    df.to_csv(filename)
    print(f"\n✓ Saved: {filename}")

    # Summary
    print(f"\nSUMMARY:")
    print(f"Observations: {len(df):,}")
    print(f"Variables: {len(df.columns)}")
    print(f"Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    # Categorize instruments
    treasury_credit = ['10Y_Treasury', '30Y_Treasury', '10Y_TIPS', 'IG_Corporate', 'HY_Bond']
    commodities = ['Gold', 'WTI_Oil', 'Copper', 'Wheat']

    print(f"\nTreasury & Credit Instruments (FRED):")
    for col in treasury_credit:
        if col in df.columns:
            valid = df[col].notna().sum()
            total = len(df)
            pct = valid / total * 100
            first = df[col].first_valid_index()
            last = df[col].last_valid_index()

            if first and last:
                print(
                    f"  {col:<15}: {valid:>6}/{total} ({pct:5.1f}%) | {first.strftime('%Y-%m')} - {last.strftime('%Y-%m')}")

    print(f"\nCommodity Instruments:")
    for col in commodities:
        if col in df.columns:
            valid = df[col].notna().sum()
            total = len(df)
            pct = valid / total * 100
            first = df[col].first_valid_index()
            last = df[col].last_valid_index()

            source = "Yahoo" if col == "Gold" else "FRED"
            if first and last:
                print(
                    f"  {col:<15}: {valid:>6}/{total} ({pct:5.1f}%) | {first.strftime('%Y-%m')} - {last.strftime('%Y-%m')} ({source})")

    # Data quality assessment
    print(f"\nData Quality Assessment:")
    excellent_coverage = (df.count() / len(df) >= 0.9).sum()
    good_coverage = ((df.count() / len(df) >= 0.7) & (df.count() / len(df) < 0.9)).sum()
    fair_coverage = ((df.count() / len(df) >= 0.5) & (df.count() / len(df) < 0.7)).sum()
    poor_coverage = (df.count() / len(df) < 0.5).sum()

    print(f"  Excellent (90%+ coverage): {excellent_coverage} instruments")
    print(f"  Good (70-90% coverage):     {good_coverage} instruments")
    print(f"  Fair (50-70% coverage):     {fair_coverage} instruments")
    print(f"  Poor (<50% coverage):       {poor_coverage} instruments")

    # Data sources summary
    print(f"\nData Sources:")
    print(f"  FRED API: Treasury yields, Credit spreads, Oil, Copper, Wheat")
    print(f"  Yahoo Finance: Gold (GC=F futures)")


# Run download
if __name__ == "__main__":
    API_KEY = "853c1faa729f41dc3f06e369d4bd66bd"

    data = download_all_data(API_KEY)
    save_data(data)

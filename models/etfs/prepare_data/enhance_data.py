import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def prepare_taa_data(filepath='output/merged_monthly_data.csv',
                     regime_data_path='output/regime_classification/regime_history.csv'):
    print("Loading data...")
    # Load main data
    df = pd.read_csv(filepath)

    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Split into asset groups
    # 1. Kenneth French factors
    ff_factors = ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems',
                  'BusEq', 'Telcm', 'Utils', 'Shops', 'Hlth', 'Money', 'Other']

    # 2. Traditional assets (require return calculation)
    traditional_assets = ['10Y_Treasury', '30Y_Treasury', '10Y_TIPS',
                          'IG_Corporate', 'HY_Bond']

    # 3. Commodities (require return calculation)
    commodities = ['WTI_Oil', 'Copper', 'Wheat', 'Gold']

    # 4. Commodity ETFs
    commodity_etfs = ['DBA', 'DBB', 'DBE', 'GLD']

    # 5. Bond ETFs
    bond_etfs = ['HYG', 'IEF', 'LQD', 'SHY', 'TIP', 'TLT']

    # 6. Sector ETFs
    sector_etfs = ['VNQ', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK',
                   'XLP', 'XLU', 'XLV', 'XLY']

    # All assets requiring return calculation
    price_based_assets = (traditional_assets + commodities +
                          commodity_etfs + bond_etfs + sector_etfs)

    print("Calculating returns...")
    # Create DataFrame for returns
    returns_df = pd.DataFrame(index=df.index)

    # Copy FF factors (already returns)
    for factor in ff_factors:
        if factor in df.columns:
            returns_df[f'{factor}_return'] = df[factor]

    # Calculate returns for other assets
    for asset in price_based_assets:
        if asset in df.columns:
            # Simple return: (P_t - P_{t-1}) / P_{t-1}
            returns_df[f'{asset}_return'] = df[asset].pct_change()

    print("Filling missing values...")
    # Filling strategy:
    # 1. For ETFs before their inception - use proxies from similar assets

    # ETF to proxy mapping
    etf_proxy_mapping = {
        # Bond ETFs
        'IEF': '10Y_Treasury',  # iShares 7-10 Year Treasury
        'TLT': '30Y_Treasury',  # iShares 20+ Year Treasury
        'TIP': '10Y_TIPS',  # iShares TIPS Bond
        'LQD': 'IG_Corporate',  # iShares Investment Grade Corporate
        'HYG': 'HY_Bond',  # iShares High Yield Corporate
        'SHY': '10Y_Treasury',  # iShares 1-3 Year Treasury (using 10Y as proxy)

        # Commodity ETFs
        'GLD': 'Gold',  # SPDR Gold Shares
        'DBE': 'WTI_Oil',  # Energy ETF
        'DBB': 'Copper',  # Base metals ETF (copper as main component)
        'DBA': 'Wheat',  # Agriculture ETF (wheat as proxy)
    }

    # Fill missing ETF values with proxy returns
    for etf, proxy in etf_proxy_mapping.items():
        etf_col = f'{etf}_return'
        proxy_col = f'{proxy}_return'

        if etf_col in returns_df.columns and proxy_col in returns_df.columns:
            # Fill only values where ETF = NaN and proxy != NaN
            mask = returns_df[etf_col].isna() & returns_df[proxy_col].notna()
            returns_df.loc[mask, etf_col] = returns_df.loc[mask, proxy_col]

    # For sector ETFs use FF factors as proxies
    sector_proxy_mapping = {
        'XLB': 'Manuf',  # Materials ≈ Manufacturing
        'XLE': 'Enrgy',  # Energy
        'XLF': 'Money',  # Financials
        'XLI': 'Manuf',  # Industrials ≈ Manufacturing
        'XLK': 'BusEq',  # Technology ≈ Business Equipment
        'XLP': 'NoDur',  # Consumer Staples
        'XLU': 'Utils',  # Utilities
        'XLV': 'Hlth',  # Healthcare
        'XLY': 'Shops',  # Consumer Discretionary ≈ Shops
        'VNQ': 'Other',  # Real Estate (using Other as proxy)
    }

    for etf, ff_factor in sector_proxy_mapping.items():
        etf_col = f'{etf}_return'
        ff_col = f'{ff_factor}_return'

        if etf_col in returns_df.columns and ff_col in returns_df.columns:
            mask = returns_df[etf_col].isna() & returns_df[ff_col].notna()
            returns_df.loc[mask, etf_col] = returns_df.loc[mask, ff_col]

    print("Adding statistical features...")
    # Add rolling features for each asset
    features_df = returns_df.copy()

    # List of all assets for analysis
    all_assets = [col.replace('_return', '') for col in returns_df.columns]

    for asset in all_assets:
        return_col = f'{asset}_return'
        if return_col in features_df.columns:
            # 3-month rolling volatility
            features_df[f'{asset}_vol_3m'] = features_df[return_col].rolling(
                window=3, min_periods=2).std()

            # 6-month rolling volatility
            features_df[f'{asset}_vol_6m'] = features_df[return_col].rolling(
                window=6, min_periods=3).std()

            # Momentum (3-month cumulative return)
            features_df[f'{asset}_momentum_3m'] = features_df[return_col].rolling(
                window=3, min_periods=2).apply(lambda x: (1 + x).prod() - 1)

            # Momentum (6-month cumulative return)
            features_df[f'{asset}_momentum_6m'] = features_df[return_col].rolling(
                window=6, min_periods=3).apply(lambda x: (1 + x).prod() - 1)

            # Moving average of returns
            features_df[f'{asset}_ma_3m'] = features_df[return_col].rolling(
                window=3, min_periods=2).mean()

    print("Adding cross-asset correlations...")
    # Calculate rolling correlations between asset classes
    # Stocks vs Bonds (using XLK vs IEF as proxy)
    if 'XLK_return' in features_df.columns and 'IEF_return' in features_df.columns:
        features_df['stock_bond_corr'] = features_df['XLK_return'].rolling(
            window=12, min_periods=6).corr(features_df['IEF_return'])

    # Commodities vs Stocks (using DBC/GLD vs XLK)
    if 'GLD_return' in features_df.columns and 'XLK_return' in features_df.columns:
        features_df['commodity_stock_corr'] = features_df['GLD_return'].rolling(
            window=12, min_periods=6).corr(features_df['XLK_return'])

    print("Loading regime data...")
    # Load economic regimes if file exists
    try:
        regime_df = pd.read_csv(regime_data_path, index_col=0, parse_dates=True)
        # Rename column if needed
        if 'regime' not in regime_df.columns and len(regime_df.columns) == 1:
            regime_df.columns = ['regime']

        # Merge with main data
        features_df = features_df.merge(regime_df, left_index=True, right_index=True, how='left')

        # Create one-hot encoding for regimes
        for regime_id in [1, 2, 3, 4]:
            features_df[f'regime_{regime_id}'] = (features_df['regime'] == regime_id).astype(int)

        print("Regimes successfully added")
    except:
        print("Warning: Regime file not found, continuing without regimes")

    print("Final data cleaning...")
    # Remove rows with too many missing values (first months)
    # Keep rows where at least 50% of data is available
    threshold = len(features_df.columns) * 0.5
    features_df = features_df.dropna(thresh=threshold)

    # Fill remaining missing values with zeros (assume zero return)
    features_df = features_df.fillna(0)

    print(f"\nFinal data shape: {features_df.shape}")
    print(f"Period: {features_df.index[0]} - {features_df.index[-1]}")
    print(f"Number of features: {len(features_df.columns)}")

    # Save prepared data
    output_path = 'output/taa_prepared_data.csv'
    features_df.to_csv(output_path)
    print(f"\nData saved to: {output_path}")

    # Also create a simplified version with returns only
    returns_only = returns_df.dropna(thresh=len(returns_df.columns) * 0.5).fillna(0)
    returns_only.to_csv('output/taa_returns_data.csv')
    print(f"Returns only saved to: output/taa_returns_data.csv")

    return features_df


def create_asset_groups_mapping():
    """
    Creates asset groups mapping for use in the model
    """
    asset_groups = {
        'Equities_FF': ['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems',
                        'BusEq', 'Telcm', 'Utils', 'Shops', 'Hlth', 'Money', 'Other'],

        'Equities_ETF': ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP',
                         'XLU', 'XLV', 'XLY', 'VNQ'],

        'Bonds_Traditional': ['10Y_Treasury', '30Y_Treasury', '10Y_TIPS',
                              'IG_Corporate', 'HY_Bond'],

        'Bonds_ETF': ['IEF', 'TLT', 'TIP', 'LQD', 'HYG', 'SHY'],

        'Commodities': ['WTI_Oil', 'Copper', 'Wheat', 'Gold'],

        'Commodities_ETF': ['DBA', 'DBB', 'DBE', 'GLD']
    }

    # Save mapping
    import json
    with open('output/asset_groups_mapping.json', 'w') as f:
        json.dump(asset_groups, f, indent=2)

    print("Asset groups mapping saved to: output/asset_groups_mapping.json")

    return asset_groups


if __name__ == "__main__":
    # Prepare data
    prepared_data = prepare_taa_data()

    # Create asset groups mapping
    asset_groups = create_asset_groups_mapping()

    # Display statistics by groups
    print("\n" + "=" * 60)
    print("ASSET GROUP STATISTICS")
    print("=" * 60)

    for group_name, assets in asset_groups.items():
        available_assets = [asset for asset in assets
                            if f'{asset}_return' in prepared_data.columns]
        print(f"\n{group_name}:")
        print(f"  Available: {len(available_assets)} out of {len(assets)}")
        print(f"  Assets: {', '.join(available_assets[:5])}" +
              (" ..." if len(available_assets) > 5 else ""))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings

warnings.filterwarnings('ignore')


def analyze_prepared_data(data_path='output/taa_prepared_data.csv',
                          returns_path='output/taa_returns_data.csv'):

    print("Loading prepared data...")
    features_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    # Load asset groups mapping
    with open('output/asset_groups_mapping.json', 'r') as f:
        asset_groups = json.load(f)

    # Period statistics
    print("\n" + "=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)
    print(f"Data period: {features_df.index[0].date()} - {features_df.index[-1].date()}")
    print(f"Number of months: {len(features_df)}")
    print(f"Number of features: {len(features_df.columns)}")

    # Data availability analysis by periods
    print("\nData availability by decades:")
    decades = {
        '1970s': ('1970-01-01', '1979-12-31'),
        '1980s': ('1980-01-01', '1989-12-31'),
        '1990s': ('1990-01-01', '1999-12-31'),
        '2000s': ('2000-01-01', '2009-12-31'),
        '2010s': ('2010-01-01', '2019-12-31'),
        '2020s': ('2020-01-01', '2025-12-31')
    }

    for decade, (start, end) in decades.items():
        decade_data = returns_df[start:end]
        if len(decade_data) > 0:
            non_zero_cols = (decade_data != 0).sum()
            available_assets = (non_zero_cols > len(decade_data) * 0.5).sum()
            print(f"  {decade}: {available_assets} assets with data")

    # Return statistics by groups
    print("\n" + "=" * 60)
    print("RETURN STATISTICS BY GROUPS")
    print("=" * 60)

    group_stats = {}

    for group_name, assets in asset_groups.items():
        return_cols = [f'{asset}_return' for asset in assets
                       if f'{asset}_return' in returns_df.columns]

        if return_cols:
            group_returns = returns_df[return_cols]

            # Calculate statistics
            stats_dict = {
                'mean_return': group_returns.mean().mean() * 12,  # Annual
                'volatility': group_returns.std().mean() * np.sqrt(12),  # Annual
                'sharpe': (group_returns.mean().mean() * 12) / (group_returns.std().mean() * np.sqrt(12)),
                'min_return': group_returns.min().min(),
                'max_return': group_returns.max().max(),
                'skewness': group_returns.apply(lambda x: stats.skew(x.dropna())).mean(),
                'kurtosis': group_returns.apply(lambda x: stats.kurtosis(x.dropna())).mean()
            }

            group_stats[group_name] = stats_dict

            print(f"\n{group_name}:")
            print(f"  Average annual return: {stats_dict['mean_return']:.2%}")
            print(f"  Annual volatility: {stats_dict['volatility']:.2%}")
            print(f"  Sharpe ratio: {stats_dict['sharpe']:.3f}")
            print(f"  Skewness: {stats_dict['skewness']:.3f}")
            print(f"  Kurtosis: {stats_dict['kurtosis']:.3f}")

    # Correlation analysis between groups
    print("\n" + "=" * 60)
    print("CORRELATIONS BETWEEN ASSET GROUPS")
    print("=" * 60)

    # Create average returns by groups
    group_returns_df = pd.DataFrame(index=returns_df.index)

    for group_name, assets in asset_groups.items():
        return_cols = [f'{asset}_return' for asset in assets
                       if f'{asset}_return' in returns_df.columns]
        if return_cols:
            group_returns_df[group_name] = returns_df[return_cols].mean(axis=1)

    # Correlation matrix
    corr_matrix = group_returns_df.corr()

    print("\nAsset group correlation matrix:")
    print(corr_matrix.round(3))

    # Visualizations
    create_visualizations(returns_df, features_df, group_returns_df,
                          group_stats, corr_matrix, asset_groups)

    # Regime analysis (if available)
    if 'regime' in features_df.columns:
        analyze_regimes(returns_df, features_df, asset_groups)

    return features_df, returns_df, group_stats


def create_visualizations(returns_df, features_df, group_returns_df,
                          group_stats, corr_matrix, asset_groups):
    """
    Creates visualizations for data analysis
    """

    # Style setup
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Group correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title('Correlations between asset groups', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('output/taa_group_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Risk-return by groups
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']

    for i, (group_name, stats) in enumerate(group_stats.items()):
        ax.scatter(stats['volatility'], stats['mean_return'],
                   s=200, c=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   label=group_name, alpha=0.7, edgecolors='black')

        # Add labels
        ax.annotate(group_name,
                    (stats['volatility'], stats['mean_return']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.7)

    ax.set_xlabel('Volatility (annual)', fontsize=12)
    ax.set_ylabel('Return (annual)', fontsize=12)
    ax.set_title('Risk-return profile by asset groups', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add Sharpe = 0.5 line
    x_range = np.array(ax.get_xlim())
    ax.plot(x_range, 0.5 * x_range, 'k--', alpha=0.3, label='Sharpe = 0.5')

    plt.tight_layout()
    plt.savefig('output/taa_risk_return_profile.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Cumulative returns time series by groups
    fig, ax = plt.subplots(figsize=(14, 8))

    cumulative_returns = (1 + group_returns_df).cumprod()

    for i, column in enumerate(cumulative_returns.columns):
        ax.plot(cumulative_returns.index, cumulative_returns[column],
                label=column, linewidth=2, alpha=0.8)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative return', fontsize=12)
    ax.set_title('Cumulative returns of asset groups', fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('output/taa_cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Rolling correlations between stocks and bonds
    if 'Equities_ETF' in group_returns_df.columns and 'Bonds_ETF' in group_returns_df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))

        rolling_corr = group_returns_df['Equities_ETF'].rolling(
            window=36, min_periods=12).corr(group_returns_df['Bonds_ETF'])

        ax.plot(rolling_corr.index, rolling_corr, linewidth=2, color='darkblue')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(rolling_corr.index, 0, rolling_corr,
                        where=(rolling_corr < 0), alpha=0.3, color='red',
                        label='Negative correlation')
        ax.fill_between(rolling_corr.index, 0, rolling_corr,
                        where=(rolling_corr >= 0), alpha=0.3, color='green',
                        label='Positive correlation')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title('36-month rolling correlation: Stocks vs Bonds',
                     fontsize=14, pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 1)

        plt.tight_layout()
        plt.savefig('output/taa_rolling_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("\nVisualizations saved:")
    print("  - taa_group_correlations.png")
    print("  - taa_risk_return_profile.png")
    print("  - taa_cumulative_returns.png")
    print("  - taa_rolling_correlation.png")


def analyze_regimes(returns_df, features_df, asset_groups):
    """
    Analyzes returns by economic regimes
    """

    print("\n" + "=" * 60)
    print("RETURN ANALYSIS BY ECONOMIC REGIMES")
    print("=" * 60)

    regime_names = {
        1: 'Goldilocks',
        2: 'Reflation',
        3: 'Deflation',
        4: 'Stagflation'
    }

    # Merge regimes with returns
    regime_returns = returns_df.copy()
    regime_returns['regime'] = features_df['regime']

    # Analysis by groups and regimes
    results = []

    for group_name, assets in asset_groups.items():
        return_cols = [f'{asset}_return' for asset in assets
                       if f'{asset}_return' in returns_df.columns]

        if return_cols:
            for regime_id, regime_name in regime_names.items():
                regime_data = regime_returns[regime_returns['regime'] == regime_id][return_cols]

                if len(regime_data) > 0:
                    results.append({
                        'Group': group_name,
                        'Regime': regime_name,
                        'Avg_Return': regime_data.mean().mean() * 12,
                        'Volatility': regime_data.std().mean() * np.sqrt(12),
                        'Sharpe': (regime_data.mean().mean() * 12) /
                                  (regime_data.std().mean() * np.sqrt(12) + 1e-6),
                        'Observations': len(regime_data)
                    })

    results_df = pd.DataFrame(results)

    # Display pivot tables
    pivot_return = results_df.pivot(index='Group', columns='Regime', values='Avg_Return')
    pivot_volatility = results_df.pivot(index='Group', columns='Regime', values='Volatility')
    pivot_sharpe = results_df.pivot(index='Group', columns='Regime', values='Sharpe')

    print("\nAverage annual return by regime (%):")
    print(pivot_return.round(2))

    print("\nAnnual volatility by regime (%):")
    print(pivot_volatility.round(2))

    print("\nSharpe ratio by regime:")
    print(pivot_sharpe.round(3))

    # Visualization: heatmap of returns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Returns
    sns.heatmap(pivot_return, annot=True, fmt='.1f', cmap='RdYlGn',
                center=0, ax=ax1, cbar_kws={'label': 'Annual Return (%)'})
    ax1.set_title('Average annual return by groups and regimes', fontsize=12)
    ax1.set_xlabel('Economic regime')
    ax1.set_ylabel('Asset group')

    # Sharpe ratios
    sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax2, cbar_kws={'label': 'Sharpe Ratio'})
    ax2.set_title('Sharpe ratio by groups and regimes', fontsize=12)
    ax2.set_xlabel('Economic regime')
    ax2.set_ylabel('Asset group')

    plt.tight_layout()
    plt.savefig('output/taa_regime_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nVisualization saved: taa_regime_performance.png")

    # Save results
    results_df.to_csv('output/taa_regime_analysis.csv', index=False)
    print("Detailed analysis saved: taa_regime_analysis.csv")


if __name__ == "__main__":
    # Analyze prepared data
    features_df, returns_df, group_stats = analyze_prepared_data()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nData is ready for TAA model building!")
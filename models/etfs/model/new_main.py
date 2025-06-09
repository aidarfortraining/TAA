"""
Robust TAA Portfolio Allocation Model
With improved constraints and risk management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import os


class RobustTAAAllocator:
    """
    Robust TAA Portfolio Allocation System with improved risk controls
    """

    def __init__(self, returns_file: str, regimes_file: str):
        """Initialize TAA system"""
        # Load data
        self.returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        self.regimes_df = pd.read_csv(regimes_file, index_col=0, parse_dates=True)

        # Define historical assets
        self.historical_assets = {
            'Equity_Sectors': [
                'NoDur', 'Durbl', 'Manuf', 'Enrgy', 'Chems',
                'BusEq', 'Telcm', 'Utils', 'Shops', 'Hlth',
                'Money', 'Other'
            ],
            'Bonds': [
                '10Y_Treasury', '30Y_Treasury', '10Y_TIPS',
                'IG_Corporate', 'HY_Bond'
            ],
            'Commodities': [
                'WTI_Oil', 'Copper', 'Wheat', 'Gold'
            ]
        }

        # ETF mapping
        self.etf_mapping = {
            'XLP': 'NoDur',
            'XLY': 'Durbl',
            'XLI': 'Manuf',
            'XLE': 'Enrgy',
            'XLB': 'Chems',
            'XLK': 'BusEq',
            'XLC': 'Telcm',
            'XLU': 'Utils',
            'XLV': 'Hlth',
            'XLF': 'Money',
            'VNQ': 'Other',
            'IEF': '10Y_Treasury',
            'TLT': '30Y_Treasury',
            'TIP': '10Y_TIPS',
            'LQD': 'IG_Corporate',
            'HYG': 'HY_Bond',
            'GLD': 'Gold',
            'DBA': 'Wheat',
            'DBB': 'Copper',
            'DBE': 'WTI_Oil'
        }

        # Regime names
        self.regime_names = {
            1: 'Goldilocks',
            2: 'Reflation',
            3: 'Deflation',
            4: 'Stagflation'
        }

        # Risk constraints
        self.max_return_cap = 0.30  # Cap monthly returns at 30%
        self.min_return_cap = -0.30  # Cap monthly losses at -30%

    def prepare_and_clean_data(self) -> pd.DataFrame:
        """Prepare and clean historical data"""
        # Get all historical assets
        all_historical = (self.historical_assets['Equity_Sectors'] +
                          self.historical_assets['Bonds'] +
                          self.historical_assets['Commodities'])

        # Create return columns
        return_cols = [f"{asset}_return" for asset in all_historical]
        available_cols = [col for col in return_cols if col in self.returns_df.columns]

        # Combine returns with regimes
        data = pd.concat([
            self.returns_df[available_cols],
            self.regimes_df['regime']
        ], axis=1)

        # Remove rows with NaN
        data = data.dropna()

        # Cap extreme returns
        print("Capping extreme returns...")
        extreme_count = 0
        for col in available_cols:
            original = data[col].copy()
            data[col] = data[col].clip(lower=self.min_return_cap, upper=self.max_return_cap)
            capped = (original != data[col]).sum()
            if capped > 0:
                extreme_count += capped
                print(f"  {col}: capped {capped} extreme values")

        print(f"Total extreme values capped: {extreme_count}")
        print(f"Clean data prepared: {len(data)} observations")

        return data

    def calculate_robust_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate robust statistics using trimmed means and winsorization"""
        regime_stats = {}

        for regime in range(1, 5):
            regime_data = data[data['regime'] == regime]

            if len(regime_data) < 12:
                print(f"Warning: Regime {regime} has only {len(regime_data)} observations")
                continue

            # Calculate correlation matrix for this regime
            returns_data = pd.DataFrame()
            stats = {}

            for col in regime_data.columns:
                if col.endswith('_return'):
                    asset = col.replace('_return', '')
                    returns = regime_data[col]
                    returns_data[asset] = returns

                    # Use robust statistics
                    # Trim top and bottom 5% for mean calculation
                    trimmed_mean = returns.iloc[int(len(returns) * 0.05):int(len(returns) * 0.95)].mean()

                    stats[asset] = {
                        'mean_return': trimmed_mean * 12,  # Annualized
                        'volatility': returns.std() * np.sqrt(12),  # Annualized
                        'median_return': returns.median() * 12,
                        'iqr': (returns.quantile(0.75) - returns.quantile(0.25)) * np.sqrt(12),
                        'observations': len(returns)
                    }
                    stats[asset]['sharpe'] = (stats[asset]['mean_return'] /
                                              stats[asset]['volatility']
                                              if stats[asset]['volatility'] > 0 else 0)

            # Store correlation matrix
            self.correlation_matrices = getattr(self, 'correlation_matrices', {})
            if len(returns_data.columns) > 1:
                self.correlation_matrices[regime] = returns_data.corr()

            regime_stats[regime] = {
                'name': self.regime_names[regime],
                'asset_stats': stats,
                'observations': len(regime_data),
                'percentage': len(regime_data) / len(data) * 100
            }

        return regime_stats

    def create_regime_specific_portfolios(self) -> Dict:
        """Create predefined robust portfolios for each regime"""

        portfolios = {}

        # Goldilocks: Growth-oriented but diversified
        portfolios[1] = {
            'regime_name': 'Goldilocks',
            'strategy': 'Growth with moderate risk',
            'asset_allocation': {
                # Equities 70%
                'BusEq': 0.15,  # Technology
                'Hlth': 0.12,  # Healthcare
                'Durbl': 0.10,  # Consumer Discretionary
                'Money': 0.08,  # Financials
                'NoDur': 0.08,  # Consumer Staples
                'Manuf': 0.08,  # Manufacturing
                'Shops': 0.05,  # Retail
                'Utils': 0.04,  # Utilities
                # Bonds 25%
                'IG_Corporate': 0.10,
                '10Y_Treasury': 0.10,
                '10Y_TIPS': 0.05,
                # Commodities 5%
                'Gold': 0.05
            }
        }

        # Reflation: Balanced with inflation protection
        portfolios[2] = {
            'regime_name': 'Reflation',
            'strategy': 'Inflation beneficiaries',
            'asset_allocation': {
                # Equities 55%
                'Enrgy': 0.12,  # Energy
                'Chems': 0.10,  # Materials
                'Manuf': 0.08,  # Industrials
                'Money': 0.08,  # Financials
                'NoDur': 0.05,  # Staples
                'Utils': 0.05,  # Utilities
                'BusEq': 0.05,  # Technology
                'Hlth': 0.02,  # Healthcare
                # Bonds 22%
                '10Y_TIPS': 0.15,
                'HY_Bond': 0.05,
                'IG_Corporate': 0.02,
                # Commodities 23%
                'WTI_Oil': 0.08,
                'Copper': 0.06,
                'Gold': 0.06,
                'Wheat': 0.03
            }
        }

        # Deflation: Defensive positioning
        portfolios[3] = {
            'regime_name': 'Deflation',
            'strategy': 'Capital preservation',
            'asset_allocation': {
                # Equities 30%
                'Utils': 0.08,  # Utilities
                'NoDur': 0.08,  # Consumer Staples
                'Hlth': 0.06,  # Healthcare
                'Telcm': 0.04,  # Telecom
                'Money': 0.04,  # Financials
                # Bonds 65%
                '30Y_Treasury': 0.25,
                '10Y_Treasury': 0.20,
                'IG_Corporate': 0.15,
                '10Y_TIPS': 0.05,
                # Commodities 5%
                'Gold': 0.05
            }
        }

        # Stagflation: Real assets focus
        portfolios[4] = {
            'regime_name': 'Stagflation',
            'strategy': 'Real asset protection',
            'asset_allocation': {
                # Equities 30%
                'Enrgy': 0.08,  # Energy
                'Utils': 0.06,  # Utilities
                'NoDur': 0.06,  # Consumer Staples
                'Hlth': 0.05,  # Healthcare
                'Chems': 0.05,  # Materials
                # Bonds 35%
                '10Y_TIPS': 0.25,
                '10Y_Treasury': 0.05,
                'HY_Bond': 0.05,
                # Commodities 35%
                'Gold': 0.15,
                'WTI_Oil': 0.08,
                'Wheat': 0.07,
                'Copper': 0.05
            }
        }

        # Normalize and validate allocations
        for regime, portfolio in portfolios.items():
            weights = portfolio['asset_allocation']
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                print(f"Warning: Regime {regime} weights sum to {total}, normalizing...")
                portfolio['asset_allocation'] = {k: v / total for k, v in weights.items()}

        return portfolios

    def backtest_robust_strategy(self, data: pd.DataFrame, portfolios: Dict) -> Dict:
        """Backtest the robust TAA strategy"""
        print("\nRunning robust backtest...")

        portfolio_returns = []
        portfolio_values = [1.0]
        regime_changes = []
        monthly_weights = []

        current_regime = None
        current_weights = {}

        for i in range(len(data)):
            date = data.index[i]
            regime = int(data.iloc[i]['regime'])

            # Check for regime change
            if current_regime != regime:
                current_regime = regime
                if regime in portfolios:
                    current_weights = portfolios[regime]['asset_allocation']
                    regime_changes.append({
                        'date': date,
                        'regime': regime,
                        'regime_name': self.regime_names[regime]
                    })

            # Calculate portfolio return
            period_return = 0
            active_weight = 0

            for asset, weight in current_weights.items():
                asset_col = f"{asset}_return"
                if asset_col in data.columns and weight > 0:
                    asset_return = data.iloc[i][asset_col]
                    if not np.isnan(asset_return):
                        period_return += weight * asset_return
                        active_weight += weight

            # Normalize by active weight
            if active_weight > 0:
                period_return = period_return / active_weight

            portfolio_returns.append(period_return)
            new_value = portfolio_values[-1] * (1 + period_return)
            portfolio_values.append(new_value)

            # Track monthly weights
            monthly_weights.append({
                'date': date,
                'regime': regime,
                'weights': current_weights.copy()
            })

        # Create results
        results = pd.DataFrame({
            'date': data.index,
            'portfolio_return': portfolio_returns,
            'portfolio_value': portfolio_values[1:],
            'regime': data['regime']
        }).set_index('date')

        # Calculate performance metrics
        annual_return = results['portfolio_return'].mean() * 12
        annual_vol = results['portfolio_return'].std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

        # Maximum drawdown
        running_max = results['portfolio_value'].expanding().max()
        drawdown = (results['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        win_rate = (results['portfolio_return'] > 0).mean()

        # Best and worst months
        best_month = results['portfolio_return'].max()
        worst_month = results['portfolio_return'].min()

        return {
            'results': results,
            'metrics': {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'best_month': best_month,
                'worst_month': worst_month,
                'total_return': portfolio_values[-1] - 1,
                'regime_changes': len(regime_changes)
            },
            'regime_changes': regime_changes,
            'monthly_weights': monthly_weights
        }

    def create_analysis_plots(self, backtest_results: Dict, save_path: str = 'robust_taa_analysis.png'):
        """Create comprehensive analysis plots"""
        results = backtest_results['results']
        metrics = backtest_results['metrics']

        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Robust TAA Portfolio Analysis', fontsize=16)

        # 1. Cumulative performance
        ax = axes[0, 0]
        ax.plot(results.index, results['portfolio_value'], 'b-', linewidth=2, label='TAA Portfolio')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Cumulative Performance')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')

        # Add regime shading
        regime_colors = {1: 'green', 2: 'yellow', 3: 'red', 4: 'purple'}
        for regime in range(1, 5):
            mask = results['regime'] == regime
            if mask.any():
                ax.fill_between(results.index,
                                results['portfolio_value'].min() * 0.9,
                                results['portfolio_value'].max() * 1.1,
                                where=mask, alpha=0.1, color=regime_colors[regime])

        # 2. Rolling performance metrics
        ax = axes[0, 1]
        rolling_returns = results['portfolio_return'].rolling(12).mean() * 12
        rolling_vol = results['portfolio_return'].rolling(12).std() * np.sqrt(12)
        rolling_sharpe = rolling_returns / rolling_vol

        ax.plot(rolling_sharpe.index, rolling_sharpe, 'g-', label='Rolling Sharpe (12M)')
        ax.axhline(y=metrics['sharpe_ratio'], color='r', linestyle='--',
                   label=f'Full Period Sharpe: {metrics["sharpe_ratio"]:.2f}')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Rolling Risk-Adjusted Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Monthly returns distribution
        ax = axes[1, 0]
        returns_by_regime = []
        labels = []

        for regime in range(1, 5):
            regime_returns = results[results['regime'] == regime]['portfolio_return']
            if len(regime_returns) > 0:
                returns_by_regime.append(regime_returns.values)
                labels.append(self.regime_names[regime])

        bp = ax.boxplot(returns_by_regime, labels=labels, patch_artist=True)
        for patch, regime in zip(bp['boxes'], range(1, len(labels) + 1)):
            patch.set_facecolor(regime_colors[regime])

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_ylabel('Monthly Return')
        ax.set_title('Return Distribution by Regime')
        ax.grid(True, alpha=0.3)

        # 4. Drawdown chart
        ax = axes[1, 1]
        running_max = results['portfolio_value'].expanding().max()
        drawdown = (results['portfolio_value'] - running_max) / running_max

        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown, 'r-', linewidth=1)
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(f'Portfolio Drawdown (Max: {metrics["max_drawdown"]:.1%})')
        ax.grid(True, alpha=0.3)

        # 5. Asset allocation over time
        ax = axes[2, 0]

        # Sample monthly to avoid overcrowding
        sample_dates = results.index[::12]  # Every 12 months
        equity_weights = []
        bond_weights = []
        commodity_weights = []

        for date in sample_dates:
            # Find closest weight entry
            closest_idx = np.argmin(np.abs(results.index - date))
            regime = results.iloc[closest_idx]['regime']

            if regime in self.portfolios:
                weights = self.portfolios[regime]['asset_allocation']

                eq_weight = sum(w for a, w in weights.items()
                                if a in self.historical_assets['Equity_Sectors'])
                bond_weight = sum(w for a, w in weights.items()
                                  if a in self.historical_assets['Bonds'])
                comm_weight = sum(w for a, w in weights.items()
                                  if a in self.historical_assets['Commodities'])

                equity_weights.append(eq_weight)
                bond_weights.append(bond_weight)
                commodity_weights.append(comm_weight)

        ax.stackplot(sample_dates, equity_weights, bond_weights, commodity_weights,
                     labels=['Equity', 'Bonds', 'Commodities'],
                     colors=['blue', 'green', 'orange'], alpha=0.7)
        ax.set_ylabel('Allocation (%)')
        ax.set_title('Asset Class Allocation Over Time')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)

        # 6. Performance summary
        ax = axes[2, 1]
        summary_text = f"""
Performance Summary
{'=' * 30}
Annual Return: {metrics['annual_return']:.2%}
Annual Volatility: {metrics['annual_volatility']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']:.1%}
Calmar Ratio: {metrics['calmar_ratio']:.2f}

Win Rate: {metrics['win_rate']:.1%}
Best Month: {metrics['best_month']:.1%}
Worst Month: {metrics['worst_month']:.1%}

Total Return: {metrics['total_return'] * 100:.1f}%
Regime Changes: {metrics['regime_changes']}
        """

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {save_path}")

    def save_etf_allocations(self, portfolios: Dict, output_dir: str = 'output'):
        """Save ETF allocations based on the mapping"""
        os.makedirs(output_dir, exist_ok=True)

        etf_allocations = {}

        for regime, portfolio in portfolios.items():
            regime_name = portfolio['regime_name']
            etf_weights = {}

            # Map historical assets to ETFs
            for etf, historical_asset in self.etf_mapping.items():
                if historical_asset in portfolio['asset_allocation']:
                    etf_weights[etf] = portfolio['asset_allocation'][historical_asset]
                else:
                    etf_weights[etf] = 0.0

            etf_allocations[regime_name] = etf_weights

        # Create DataFrame
        etf_df = pd.DataFrame(etf_allocations).round(4)
        etf_df = etf_df[etf_df.sum(axis=1) > 0]  # Remove zero rows

        # Save to CSV
        etf_df.to_csv(f'{output_dir}/robust_etf_allocations.csv')

        # Save detailed allocations as JSON
        with open(f'{output_dir}/robust_allocations.json', 'w') as f:
            json.dump({
                'historical_allocations': {p['regime_name']: p['asset_allocation']
                                           for p in portfolios.values()},
                'etf_allocations': etf_allocations,
                'strategy_descriptions': {p['regime_name']: p['strategy']
                                          for p in portfolios.values()}
            }, f, indent=2)

        print(f"\nETF Allocations saved to {output_dir}/")
        print("\nETF Allocation Summary:")
        print(etf_df)

        return etf_df

    def run_robust_analysis(self):
        """Run complete robust TAA analysis"""
        print("=" * 60)
        print("ROBUST TAA PORTFOLIO ANALYSIS")
        print("=" * 60)

        # 1. Prepare and clean data
        clean_data = self.prepare_and_clean_data()

        # 2. Calculate robust statistics
        self.regime_statistics = self.calculate_robust_statistics(clean_data)

        # 3. Create regime-specific portfolios
        self.portfolios = self.create_regime_specific_portfolios()

        # 4. Run backtest
        backtest_results = self.backtest_robust_strategy(clean_data, self.portfolios)

        # 5. Display results
        print("\n" + "=" * 40)
        print("BACKTEST RESULTS")
        print("=" * 40)
        metrics = backtest_results['metrics']
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'return' in key or 'rate' in key or 'month' in key:
                    print(f"{key.replace('_', ' ').title()}: {value:.2%}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")

        # 6. Create visualizations
        self.create_analysis_plots(backtest_results)

        # 7. Save ETF allocations
        self.save_etf_allocations(self.portfolios)

        return {
            'backtest_results': backtest_results,
            'portfolios': self.portfolios,
            'regime_statistics': self.regime_statistics
        }


if __name__ == "__main__":
    # File paths
    RETURNS_FILE = 'input/taa_returns_aligned.csv'
    REGIMES_FILE = 'input/regime_history_aligned.csv'

    # Run robust TAA analysis
    taa = RobustTAAAllocator(RETURNS_FILE, REGIMES_FILE)
    results = taa.run_robust_analysis()

    print("\n" + "=" * 60)
    print("ROBUST TAA ANALYSIS COMPLETE")
    print("=" * 60)
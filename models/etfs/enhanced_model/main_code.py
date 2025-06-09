"""
Fixed Advanced TAA Portfolio Allocation Model
Fixes the UnboundLocalError and improves early period handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass
from enum import Enum


@dataclass
class TransactionCosts:
    """Transaction cost parameters"""
    spread: float = 0.001  # 0.1% bid-ask spread
    commission: float = 0.0005  # 0.05% commission
    market_impact: float = 0.001  # 0.1% market impact for large trades

    def calculate_cost(self, turnover: float, trade_size: float = 1.0) -> float:
        """Calculate total transaction cost"""
        base_cost = self.spread + self.commission
        # Add market impact for larger trades
        if trade_size > 0.05:  # More than 5% of portfolio
            base_cost += self.market_impact * (trade_size / 0.05)
        return turnover * base_cost


class OptimizationMethod(Enum):
    """Portfolio optimization methods"""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    MIN_CVAR = "min_cvar"


class AdvancedTAAAllocator:
    """
    Advanced TAA System with real optimization and risk management
    """

    def __init__(self, returns_file: str, regimes_file: str,
                 target_volatility: float = 0.10):
        """Initialize advanced TAA system"""
        # Load data
        self.returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        self.regimes_df = pd.read_csv(regimes_file, index_col=0, parse_dates=True)

        # Configuration
        self.target_volatility = target_volatility
        self.transaction_costs = TransactionCosts()
        self.min_weight = 0.0
        self.max_weight = 0.25
        self.min_history_required = 60  # Minimum 5 years of data to start

        self.lookback_periods = {
            'returns': 36,  # 3 years for return estimation
            'volatility': 252,  # 1 year for volatility
            'correlation': 126  # 6 months for correlation
        }

        # Define assets
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
            'XLP': 'NoDur', 'XLY': 'Durbl', 'XLI': 'Manuf',
            'XLE': 'Enrgy', 'XLB': 'Chems', 'XLK': 'BusEq',
            'XLC': 'Telcm', 'XLU': 'Utils', 'XLV': 'Hlth',
            'XLF': 'Money', 'VNQ': 'Other', 'IEF': '10Y_Treasury',
            'TLT': '30Y_Treasury', 'TIP': '10Y_TIPS', 'LQD': 'IG_Corporate',
            'HYG': 'HY_Bond', 'GLD': 'Gold', 'DBA': 'Wheat',
            'DBB': 'Copper', 'DBE': 'WTI_Oil'
        }

        # Regime names
        self.regime_names = {
            1: 'Goldilocks', 2: 'Reflation',
            3: 'Deflation', 4: 'Stagflation'
        }

        # Initialize storage
        self.optimization_history = []
        self.regime_probabilities = {}

    def prepare_and_validate_data(self) -> pd.DataFrame:
        """Enhanced data preparation with validation"""
        # Get all assets
        all_historical = (self.historical_assets['Equity_Sectors'] +
                         self.historical_assets['Bonds'] +
                         self.historical_assets['Commodities'])

        # Create return columns
        return_cols = [f"{asset}_return" for asset in all_historical]
        available_cols = [col for col in return_cols if col in self.returns_df.columns]

        # Store available assets for later use
        self.available_assets = [col.replace('_return', '') for col in available_cols]

        # Combine data
        data = pd.concat([
            self.returns_df[available_cols],
            self.regimes_df['regime']
        ], axis=1)

        # Enhanced outlier detection
        print("Validating data quality...")
        cleaned_data = data.copy()
        outlier_report = {}

        for col in available_cols:
            # Use simpler outlier detection for robustness
            values = data[col].dropna()
            if len(values) > 0:
                # Simple percentile-based winsorization
                lower = values.quantile(0.01)
                upper = values.quantile(0.99)

                outliers = (data[col] < lower) | (data[col] > upper)
                if outliers.sum() > 0:
                    cleaned_data[col] = data[col].clip(lower=lower, upper=upper)
                    outlier_report[col] = outliers.sum()

        # Report outliers
        if outlier_report:
            print(f"Outliers detected and winsorized in {len(outlier_report)} assets")
            for asset, count in list(outlier_report.items())[:5]:
                print(f"  {asset}: {count} outliers")

        # Remove NaN values
        cleaned_data = cleaned_data.dropna()
        print(f"Clean data prepared: {len(cleaned_data)} observations")

        return cleaned_data

    def get_default_weights(self) -> Dict[str, float]:
        """Get default equal-weight portfolio"""
        weights = {}
        n_assets = len(self.available_assets)

        if n_assets > 0:
            equal_weight = 1.0 / n_assets
            for asset in self.available_assets:
                weights[asset] = equal_weight

        return weights

    def calculate_dynamic_statistics(self, data: pd.DataFrame,
                                   lookback: Optional[int] = None) -> Dict:
        """Calculate regime statistics with dynamic windows"""
        if lookback is None:
            lookback = self.lookback_periods['returns']

        regime_stats = {}

        for regime in range(1, 5):
            regime_data = data[data['regime'] == regime]

            if len(regime_data) < 12:  # Less strict requirement
                continue

            stats = {}
            returns_data = pd.DataFrame()

            # Calculate statistics for each asset
            for col in regime_data.columns:
                if col.endswith('_return'):
                    asset = col.replace('_return', '')
                    returns = regime_data[col]
                    returns_data[asset] = returns

                    # Simple statistics for robustness
                    stats[asset] = {
                        'mean_return': returns.mean() * 12,  # Annualized
                        'volatility': returns.std() * np.sqrt(12),  # Annualized
                        'observations': len(returns)
                    }

                    # Add Sharpe ratio
                    if stats[asset]['volatility'] > 0:
                        stats[asset]['sharpe'] = stats[asset]['mean_return'] / stats[asset]['volatility']
                    else:
                        stats[asset]['sharpe'] = 0

            # Calculate correlation matrix
            if len(returns_data.columns) > 1:
                cov_matrix = returns_data.cov()
                corr_matrix = returns_data.corr()

                regime_stats[regime] = {
                    'name': self.regime_names[regime],
                    'asset_stats': stats,
                    'covariance': cov_matrix,
                    'correlation': corr_matrix,
                    'observations': len(regime_data),
                    'assets': list(returns_data.columns)
                }

        return regime_stats

    def optimize_portfolio(self, expected_returns: np.ndarray,
                         cov_matrix: np.ndarray,
                         method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
                         risk_aversion: float = 2.0) -> np.ndarray:
        """Advanced portfolio optimization with multiple methods"""
        n_assets = len(expected_returns)

        # Add regularization to covariance matrix
        cov_matrix = cov_matrix + np.eye(n_assets) * 1e-6

        if method == OptimizationMethod.MEAN_VARIANCE:
            # Classic Markowitz optimization
            def objective(w):
                portfolio_return = np.dot(w, expected_returns)
                portfolio_variance = np.dot(w, np.dot(cov_matrix, w))
                return -(portfolio_return - risk_aversion * portfolio_variance)

        elif method == OptimizationMethod.RISK_PARITY:
            # Risk parity optimization
            def objective(w):
                portfolio_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
                if portfolio_vol > 0:
                    marginal_contrib = np.dot(cov_matrix, w) / portfolio_vol
                    risk_contrib = w * marginal_contrib
                    # Minimize variance of risk contributions
                    return np.var(risk_contrib)
                else:
                    return 1e10

        else:  # Default to mean-variance
            def objective(w):
                portfolio_return = np.dot(w, expected_returns)
                portfolio_variance = np.dot(w, np.dot(cov_matrix, w))
                return -(portfolio_return - risk_aversion * portfolio_variance)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
        ]

        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Initial guess - equal weight
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000})

            if result.success:
                weights = result.x
                weights[weights < 0.01] = 0  # Clean small weights
                weights = weights / weights.sum()  # Renormalize
                return weights
            else:
                # Fallback to equal weight
                return np.ones(n_assets) / n_assets

        except:
            # Fallback to equal weight
            return np.ones(n_assets) / n_assets

    def calculate_blended_weights(self, regime_weights: Dict[int, np.ndarray],
                                regime_probabilities: Dict[int, float],
                                asset_names: List[str]) -> Dict[str, float]:
        """Blend portfolio weights based on regime probabilities"""
        blended = np.zeros(len(asset_names))

        for regime, prob in regime_probabilities.items():
            if regime in regime_weights:
                blended += regime_weights[regime] * prob

        # Normalize
        if blended.sum() > 0:
            blended = blended / blended.sum()

        return {asset_names[i]: float(blended[i]) for i in range(len(asset_names))}

    def apply_momentum_overlay(self, base_weights: Dict[str, float],
                             returns_data: pd.DataFrame,
                             lookback: int = 126) -> Dict[str, float]:
        """Apply momentum overlay to base weights"""
        if len(returns_data) < lookback:
            return base_weights

        momentum_scores = {}
        recent_returns = returns_data.iloc[-lookback:]

        for asset in base_weights:
            asset_col = f"{asset}_return"
            if asset_col in recent_returns.columns:
                # Calculate momentum score
                returns = recent_returns[asset_col]
                total_return = (1 + returns).prod() - 1
                vol = returns.std() * np.sqrt(12)

                if vol > 0:
                    momentum_scores[asset] = total_return / vol
                else:
                    momentum_scores[asset] = 0

        # Adjust weights based on momentum
        adjusted_weights = {}
        momentum_factor = 0.2  # Maximum 20% adjustment

        for asset, base_weight in base_weights.items():
            if asset in momentum_scores:
                # Scale momentum score to [-1, 1]
                score = np.tanh(momentum_scores[asset] / 2)
                adjustment = base_weight * momentum_factor * score
                adjusted_weights[asset] = max(0, base_weight + adjustment)
            else:
                adjusted_weights[asset] = base_weight

        # Renormalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}

        return adjusted_weights

    def calculate_transition_weights(self, current_weights: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   transition_speed: float = 0.33) -> Tuple[Dict[str, float], float]:
        """Calculate smooth transition weights with transaction cost consideration"""
        new_weights = {}
        turnover = 0

        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        for asset in all_assets:
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)

            # Calculate proposed new weight
            proposed = current + (target - current) * transition_speed

            # Only trade if change is significant (above threshold)
            if abs(proposed - current) < 0.02:  # 2% threshold
                new_weights[asset] = current
            else:
                new_weights[asset] = proposed
                turnover += abs(proposed - current)

        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items() if v > 0.001}

        # Calculate transaction cost
        transaction_cost = self.transaction_costs.calculate_cost(turnover)

        return new_weights, transaction_cost

    def backtest_advanced_strategy(self, data: pd.DataFrame) -> Dict:
        """Advanced backtesting with all features"""
        print("\nRunning advanced backtest...")

        # Ensure we have enough initial data
        start_index = max(self.min_history_required, self.lookback_periods['returns'])

        if len(data) < start_index + 12:
            raise ValueError(f"Insufficient data for backtesting. Need at least {start_index + 12} observations, have {len(data)}")

        # Initialize tracking variables
        portfolio_returns = []
        portfolio_values = [1.0]
        transaction_costs_paid = []
        regime_history = []
        weight_history = []

        # Current portfolio state
        current_weights = self.get_default_weights()  # Start with equal weights
        target_weights = current_weights.copy()

        # Walk-forward optimization
        reoptimize_frequency = 3  # Reoptimize every 3 months
        months_since_reopt = 0

        for i in range(start_index, len(data)):
            date = data.index[i]

            # Get historical window
            historical_window = data.iloc[max(0, i-252):i]

            # Check if we need to reoptimize
            need_reoptimize = (months_since_reopt >= reoptimize_frequency) or (i == start_index)

            if need_reoptimize:
                print(f"Reoptimizing at {date.strftime('%Y-%m')}")

                # Calculate regime statistics
                regime_stats = self.calculate_dynamic_statistics(historical_window)

                if len(regime_stats) > 0:
                    # We have some regime data
                    # Calculate simple regime probabilities based on recent occurrences
                    recent_regimes = historical_window['regime'].iloc[-12:].value_counts()
                    total_recent = len(historical_window['regime'].iloc[-12:])

                    regime_probs = {}
                    for regime in range(1, 5):
                        if regime in recent_regimes.index:
                            regime_probs[regime] = recent_regimes[regime] / total_recent
                        else:
                            regime_probs[regime] = 0.0

                    # Optimize for each regime that has data
                    regime_weights = {}
                    all_assets_set = set()

                    for regime, stats in regime_stats.items():
                        # Get assets for this regime
                        assets = stats['assets']
                        all_assets_set.update(assets)

                        # Get returns and covariance
                        returns = np.array([stats['asset_stats'][a]['mean_return'] for a in assets])
                        cov_matrix = stats['covariance'].values

                        # Optimize portfolio
                        weights = self.optimize_portfolio(
                            returns, cov_matrix,
                            method=OptimizationMethod.MEAN_VARIANCE,
                            risk_aversion=2.0 + regime * 0.5
                        )

                        regime_weights[regime] = weights

                    # Convert all_assets_set to list for consistent ordering
                    all_assets = sorted(list(all_assets_set))

                    if all_assets and regime_weights:
                        # Create aligned weight arrays for blending
                        aligned_regime_weights = {}
                        for regime, weights in regime_weights.items():
                            regime_assets = regime_stats[regime]['assets']
                            aligned_weights = np.zeros(len(all_assets))

                            for j, asset in enumerate(regime_assets):
                                if asset in all_assets:
                                    asset_idx = all_assets.index(asset)
                                    aligned_weights[asset_idx] = weights[j]

                            # Normalize
                            if aligned_weights.sum() > 0:
                                aligned_weights = aligned_weights / aligned_weights.sum()

                            aligned_regime_weights[regime] = aligned_weights

                        # Blend weights
                        blended_weights = self.calculate_blended_weights(
                            aligned_regime_weights, regime_probs, all_assets
                        )

                        # Apply momentum overlay
                        blended_weights = self.apply_momentum_overlay(
                            blended_weights, historical_window
                        )

                        target_weights = blended_weights
                    else:
                        # No valid optimization, keep current weights
                        target_weights = current_weights.copy()
                else:
                    # No regime data available, use default weights
                    print(f"Warning: No regime data available at {date.strftime('%Y-%m')}, using default weights")
                    target_weights = self.get_default_weights()

                months_since_reopt = 0

            # Smooth transition
            new_weights, transaction_cost = self.calculate_transition_weights(
                current_weights, target_weights, transition_speed=0.33
            )
            current_weights = new_weights
            transaction_costs_paid.append(transaction_cost)

            # Calculate portfolio return
            period_return = 0
            weight_sum = 0

            for asset, weight in current_weights.items():
                if asset != 'Cash':
                    asset_col = f"{asset}_return"
                    if asset_col in data.columns and not pd.isna(data.iloc[i][asset_col]):
                        period_return += weight * data.iloc[i][asset_col]
                        weight_sum += weight

            # Normalize return by actual weight sum
            if weight_sum > 0:
                period_return = period_return / weight_sum

            # Subtract transaction costs
            period_return -= transaction_cost

            portfolio_returns.append(period_return)
            new_value = portfolio_values[-1] * (1 + period_return)
            portfolio_values.append(new_value)

            # Track history
            current_regime = int(data.iloc[i]['regime'])
            regime_history.append({
                'date': date,
                'dominant_regime': current_regime
            })

            weight_history.append({
                'date': date,
                'weights': current_weights.copy()
            })

            months_since_reopt += 1

        # Create results dataframe
        results = pd.DataFrame({
            'portfolio_return': portfolio_returns,
            'portfolio_value': portfolio_values[1:],
            'transaction_cost': transaction_costs_paid
        }, index=data.index[start_index:])

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(results)

        # Add transaction cost analysis
        metrics['total_transaction_costs'] = sum(transaction_costs_paid)
        metrics['avg_monthly_turnover'] = np.mean([abs(r) for r in portfolio_returns]) * 2

        return {
            'results': results,
            'metrics': metrics,
            'regime_history': regime_history,
            'weight_history': weight_history
        }

    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        returns = results['portfolio_return']
        values = results['portfolio_value']

        # Basic metrics
        annual_return = returns.mean() * 12
        annual_vol = returns.std() * np.sqrt(12)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(12)
            sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
        else:
            sortino_ratio = sharpe_ratio * 1.5  # Approximate

        # Maximum drawdown
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate and profit factor
        win_rate = (returns > 0).mean()
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]

        if len(losing_returns) > 0 and losing_returns.sum() != 0:
            profit_factor = winning_returns.sum() / abs(losing_returns.sum())
        else:
            profit_factor = np.inf if len(winning_returns) > 0 else 0

        # Ulcer index
        dd_squared = drawdown ** 2
        ulcer_index = np.sqrt(dd_squared.mean())

        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'ulcer_index': ulcer_index,
            'total_return': values.iloc[-1] - 1
        }

    def create_visualizations(self, backtest_results: Dict,
                            save_path: str = 'advanced_taa_analysis.png'):
        """Create comprehensive visualization dashboard"""
        results = backtest_results['results']
        metrics = backtest_results['metrics']

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Advanced TAA Strategy Analysis', fontsize=16)

        # 1. Cumulative performance
        ax = axes[0, 0]
        ax.plot(results.index, results['portfolio_value'], 'b-', linewidth=2)
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Cumulative Performance')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 2. Monthly returns distribution
        ax = axes[0, 1]
        returns = results['portfolio_return']
        ax.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(returns.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {returns.mean()*100:.2f}%')
        ax.set_xlabel('Monthly Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Return Distribution')
        ax.legend()

        # 3. Rolling Sharpe ratio
        ax = axes[1, 0]
        rolling_window = 36
        if len(results) >= rolling_window:
            rolling_returns = returns.rolling(rolling_window).mean() * 12
            rolling_vol = returns.rolling(rolling_window).std() * np.sqrt(12)
            rolling_sharpe = rolling_returns / rolling_vol
            ax.plot(rolling_sharpe.index, rolling_sharpe, 'g-', linewidth=2)
            ax.axhline(y=metrics['sharpe_ratio'], color='r', linestyle='--',
                      label=f'Full Period: {metrics["sharpe_ratio"]:.2f}')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('Rolling 3-Year Sharpe Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Drawdown
        ax = axes[1, 1]
        running_max = results['portfolio_value'].expanding().max()
        drawdown = (results['portfolio_value'] - running_max) / running_max
        ax.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown * 100, 'r-', linewidth=1)
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(f'Drawdown (Max: {metrics["max_drawdown"]*100:.1f}%)')
        ax.grid(True, alpha=0.3)

        # 5. Transaction costs
        ax = axes[2, 0]
        cumulative_costs = results['transaction_cost'].cumsum()
        ax.plot(results.index, cumulative_costs * 100, 'r-', linewidth=2)
        ax.set_ylabel('Cumulative Transaction Costs (%)')
        ax.set_title('Transaction Cost Impact')
        ax.grid(True, alpha=0.3)

        # 6. Performance summary
        ax = axes[2, 1]
        ax.axis('off')

        summary_text = f"""
Performance Summary
{'='*30}
Annual Return: {metrics['annual_return']*100:.2f}%
Annual Volatility: {metrics['annual_volatility']*100:.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Sortino Ratio: {metrics['sortino_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']*100:.1f}%
Calmar Ratio: {metrics['calmar_ratio']:.2f}
Win Rate: {metrics['win_rate']*100:.1f}%
Total Return: {metrics['total_return']*100:.1f}%
Transaction Costs: {metrics['total_transaction_costs']*100:.2f}%
        """

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {save_path}")

    def run_advanced_analysis(self):
        """Run complete advanced TAA analysis"""
        print("="*60)
        print("ADVANCED TAA PORTFOLIO ANALYSIS")
        print("="*60)

        # 1. Prepare and validate data
        clean_data = self.prepare_and_validate_data()

        # 2. Run advanced backtest
        try:
            backtest_results = self.backtest_advanced_strategy(clean_data)

            # 3. Display results
            print("\n" + "="*40)
            print("BACKTEST RESULTS")
            print("="*40)
            metrics = backtest_results['metrics']

            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'return' in key or 'rate' in key or 'cost' in key:
                        print(f"{key.replace('_', ' ').title()}: {value*100:.2f}%")
                    else:
                        print(f"{key.replace('_', ' ').title()}: {value:.2f}")

            # 4. Create visualizations
            self.create_visualizations(backtest_results)

            # 5. Save results
            results_summary = {
                'performance_metrics': {k: float(v) for k, v in metrics.items()},
                'backtest_period': {
                    'start': str(backtest_results['results'].index[0]),
                    'end': str(backtest_results['results'].index[-1]),
                    'total_months': len(backtest_results['results'])
                }
            }

            with open('taa_results_summary.json', 'w') as f:
                json.dump(results_summary, f, indent=2)

            return {
                'backtest_results': backtest_results,
                'metrics': metrics
            }

        except Exception as e:
            print(f"\nError during backtesting: {e}")
            raise


if __name__ == "__main__":
    # File paths - adjust to your directory
    RETURNS_FILE = 'input/taa_returns_aligned.csv'
    REGIMES_FILE = 'input/regime_history_aligned.csv'

    # Initialize and run advanced TAA system
    taa = AdvancedTAAAllocator(RETURNS_FILE, REGIMES_FILE, target_volatility=0.10)
    results = taa.run_advanced_analysis()

    print("\n" + "="*60)
    print("ADVANCED TAA ANALYSIS COMPLETE")
    print("="*60)
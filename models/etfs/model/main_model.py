"""
Enhanced Tactical Asset Allocation Model
Fixed critical issues and improved robustness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import joblib
import json
import warnings
from datetime import datetime
import os
import logging
from typing import Dict, Tuple, Optional, List

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TacticalAssetAllocation:
    def __init__(self, regime_model_path: str, data_path: str,
                 asset_groups_path: str = 'input/asset_groups_mapping.json'):
        """Initialize TAA model with improved error handling"""

        try:
            # Load regime classifier
            logger.info("Loading regime classification model...")
            self.regime_model = joblib.load(regime_model_path)
            self.scaler = self.regime_model['scaler']
            self.feature_names = self.regime_model['feature_names']
            self.best_model = self.regime_model['best_model']
        except FileNotFoundError:
            logger.error(f"Regime model file not found: {regime_model_path}")
            raise

        try:
            # Load data
            logger.info("Loading asset returns data...")
            self.returns_df = pd.read_csv(data_path, index_col=0, parse_dates=True)

            # CRITICAL FIX: Proper data cleaning
            self.returns_df = self.returns_df.replace([np.inf, -np.inf], np.nan)
            # More conservative clipping for monthly returns
            self.returns_df = self.returns_df.clip(-0.30, 0.30)  # ±30% monthly max

        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            raise

        try:
            # Load asset groups
            with open(asset_groups_path, 'r') as f:
                self.asset_groups = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Asset groups file not found: {asset_groups_path}")
            self.asset_groups = self._create_default_asset_groups()

        # Initialize portfolio components
        self.regime_portfolios = {}
        self.historical_assets = []
        self.modern_assets = []
        self.transition_matrix = self.regime_model.get('transition_matrix')

        # IMPROVED Risk parameters
        self.risk_free_rate = 0.02  # Annual risk-free rate
        self.max_position_size = 0.25  # Reduced from 40% to 25% for better diversification
        self.min_position_size = 0.02  # Minimum 2% to avoid tiny positions
        self.min_assets = 5  # Minimum number of assets in portfolio
        self.rebalance_threshold = 0.05

        # Transaction costs
        self.transaction_cost = 0.001  # 0.1% per trade

        # Regime names
        self.regime_names = {
            1: 'Goldilocks',
            2: 'Reflation',
            3: 'Deflation',
            4: 'Stagflation'
        }

    def _create_default_asset_groups(self):
        """Create default asset groups based on asset names"""
        return {
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

    def calculate_regime_statistics(self, returns: pd.DataFrame, regimes: pd.Series) -> Dict:
        """Calculate return statistics for each regime with improved robustness"""

        regime_stats = {}

        for regime_id in range(1, 5):
            regime_mask = regimes == regime_id
            regime_returns = returns[regime_mask]

            if len(regime_returns) > 12:  # Need at least 12 months
                # Clean data more carefully
                regime_returns_clean = regime_returns.copy()
                regime_returns_clean = regime_returns_clean.replace([np.inf, -np.inf], np.nan)

                # Remove columns with too many NaNs
                valid_cols = regime_returns_clean.columns[
                    regime_returns_clean.isna().sum() < len(regime_returns_clean) * 0.5
                ]
                regime_returns_clean = regime_returns_clean[valid_cols]
                regime_returns_clean = regime_returns_clean.fillna(0)

                # Calculate statistics
                mean_returns = regime_returns_clean.mean()

                # FIXED: Proper LedoitWolf usage
                try:
                    if len(regime_returns_clean) > regime_returns_clean.shape[1]:
                        lw = LedoitWolf()
                        lw.fit(regime_returns_clean)
                        cov_matrix = lw.covariance_
                    else:
                        # Not enough samples for LedoitWolf
                        cov_matrix = regime_returns_clean.cov().values
                        # Add regularization
                        cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * 0.001
                except Exception as e:
                    logger.warning(f"Covariance estimation failed for regime {regime_id}: {e}")
                    cov_matrix = regime_returns_clean.cov().values
                    cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * 0.001

                regime_stats[regime_id] = {
                    'mean_returns': mean_returns,
                    'volatility': regime_returns_clean.std(),
                    'cov_matrix': pd.DataFrame(cov_matrix,
                                              index=regime_returns_clean.columns,
                                              columns=regime_returns_clean.columns),
                    'observations': len(regime_returns_clean),
                    'regime_name': self.regime_names[regime_id]
                }

        return regime_stats

    def optimize_portfolio(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                          regime_id: int) -> pd.Series:
        """Optimize portfolio weights with improved constraints and robustness"""

        # Filter assets based on data quality
        valid_mask = ~(mean_returns.isna() | (mean_returns == 0))

        # Additional filtering based on regime
        if regime_id == 3:  # Deflation - prefer defensive assets
            # Keep assets with positive or slightly negative returns
            valid_mask = valid_mask & (mean_returns > -0.01)
        else:
            # Remove extreme negative returns
            valid_mask = valid_mask & (mean_returns > -0.10)

        valid_assets = mean_returns[valid_mask].index

        # Ensure minimum number of assets
        if len(valid_assets) < self.min_assets:
            logger.warning(f"Too few valid assets ({len(valid_assets)}) for regime {regime_id}")
            # Take top assets by Sharpe ratio
            asset_sharpe = mean_returns / (np.sqrt(np.diag(cov_matrix)) + 1e-6)
            valid_assets = asset_sharpe.nlargest(max(self.min_assets, 10)).index

        mean_returns_valid = mean_returns[valid_assets]
        cov_matrix_valid = cov_matrix.loc[valid_assets, valid_assets]

        # Ensure positive definite covariance matrix
        min_eigenvalue = np.min(np.linalg.eigvals(cov_matrix_valid))
        if min_eigenvalue < 0:
            cov_matrix_valid = cov_matrix_valid + np.eye(len(cov_matrix_valid)) * (-min_eigenvalue + 1e-6)

        n_assets = len(mean_returns_valid)

        # Initial guess - equal weight
        x0 = np.ones(n_assets) / n_assets

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.min_position_size, self.max_position_size)
                  for _ in range(n_assets)]

        # Regime-specific risk aversion
        risk_aversion = {
            1: 1.0,   # Goldilocks - balanced
            2: 1.5,   # Reflation - moderate risk
            3: 3.0,   # Deflation - high risk aversion
            4: 2.0    # Stagflation - elevated risk aversion
        }

        gamma = risk_aversion.get(regime_id, 1.5)

        # Objective function - Mean-Variance optimization
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns_valid)
            portfolio_var = np.dot(weights, np.dot(cov_matrix_valid, weights))

            # Mean-variance utility function
            utility = portfolio_return - (gamma / 2) * portfolio_var
            return -utility  # Negative because we minimize

        # Add regime-specific constraints
        if regime_id == 3:  # Deflation
            # Ensure significant allocation to bonds
            bond_indices = [i for i, asset in enumerate(mean_returns_valid.index)
                           if any(bond in asset for bond in
                                 ['Treasury', 'TIPS', 'Bond', 'IEF', 'TLT', 'LQD', 'SHY', 'TIP'])]

            if bond_indices:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: np.sum([x[i] for i in bond_indices]) - 0.4
                })

        elif regime_id == 4:  # Stagflation
            # Ensure allocation to real assets
            real_asset_indices = [i for i, asset in enumerate(mean_returns_valid.index)
                                 if any(comm in asset for comm in
                                       ['Oil', 'Copper', 'Gold', 'DBA', 'DBB', 'DBE', 'GLD', 'TIPS', 'TIP'])]

            if real_asset_indices:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: np.sum([x[i] for i in real_asset_indices]) - 0.25
                })

        # Optimize
        try:
            result = minimize(objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9})

            if result.success:
                # Create full weight vector
                weights = pd.Series(0, index=mean_returns.index)
                weights[valid_assets] = result.x

                # Clean up small weights
                weights[weights < self.min_position_size] = 0

                # Renormalize
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    # Fallback to equal weight on valid assets
                    weights[valid_assets] = 1.0 / len(valid_assets)

                return weights
            else:
                logger.warning(f"Optimization failed for regime {regime_id}: {result.message}")
                return self._get_fallback_weights(mean_returns, valid_assets)

        except Exception as e:
            logger.error(f"Optimization error for regime {regime_id}: {e}")
            return self._get_fallback_weights(mean_returns, valid_assets)

    def _get_fallback_weights(self, mean_returns: pd.Series, valid_assets: pd.Index) -> pd.Series:
        """Get fallback weights when optimization fails"""
        weights = pd.Series(0, index=mean_returns.index)

        # Use risk parity or equal weight
        if len(valid_assets) > 0:
            weights[valid_assets] = 1.0 / len(valid_assets)
        else:
            # Last resort - equal weight top 10 assets
            top_assets = mean_returns.nlargest(10).index
            weights[top_assets] = 0.1

        return weights

    def backtest_strategy(self, start_date: str = '2001-01-01',
                         end_date: str = None,
                         initial_capital: float = 100000) -> Dict:
        """Backtest the TAA strategy with transaction costs and improved realism"""

        logger.info(f"Backtesting TAA strategy from {start_date}...")

        # Get returns and regimes
        returns = self.returns_df[start_date:end_date].copy()

        # Clean returns
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.fillna(0)

        # Load regime history
        regime_history = self._load_regime_history()

        # Align dates
        common_dates = returns.index.intersection(regime_history.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates for backtesting")

        returns = returns.loc[common_dates]
        regimes = regime_history.loc[common_dates]

        # Initialize tracking
        portfolio_values = [initial_capital]
        portfolio_returns = []
        turnover = []
        regime_changes = []
        weights_history = {}

        current_weights = None
        previous_weights = None
        current_regime = None

        logger.info(f"Backtesting over {len(returns)} periods")

        # Process each period
        for i, (date, regime) in enumerate(regimes['regime'].items()):
            # Check for regime change or rebalancing need
            needs_rebalance = False

            if regime != current_regime:
                needs_rebalance = True
                current_regime = regime
                regime_changes.append(date)
            elif current_weights is not None and i % 3 == 0:  # Check quarterly
                # Calculate weight drift
                if previous_weights is not None:
                    weight_drift = np.abs(current_weights - previous_weights).sum()
                    if weight_drift > self.rebalance_threshold:
                        needs_rebalance = True

            if needs_rebalance:
                # Get new target weights
                use_modern = date >= pd.Timestamp('2000-01-01')
                portfolio_key = f"{'modern' if use_modern else 'historical'}_{regime}"

                if portfolio_key in self.regime_portfolios:
                    optimal_weights = self.regime_portfolios[portfolio_key]['weights']

                    # Align weights with current available assets
                    new_weights = pd.Series(0, index=returns.columns)
                    for asset_col in optimal_weights.index:
                        if asset_col in returns.columns:
                            new_weights[asset_col] = optimal_weights[asset_col]

                    # Normalize weights
                    if new_weights.sum() > 0:
                        new_weights = new_weights / new_weights.sum()
                    else:
                        # Equal weight fallback
                        new_weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)

                    # Calculate turnover and transaction costs
                    if current_weights is not None:
                        trade_amounts = np.abs(new_weights - current_weights)
                        turnover_pct = trade_amounts.sum()
                        transaction_cost = turnover_pct * self.transaction_cost
                        turnover.append(turnover_pct)
                    else:
                        transaction_cost = self.transaction_cost  # Initial purchase cost
                        turnover.append(1.0)

                    previous_weights = current_weights
                    current_weights = new_weights
                    weights_history[date] = current_weights.copy()
                else:
                    logger.warning(f"No portfolio found for {portfolio_key}")
                    transaction_cost = 0

            else:
                transaction_cost = 0

            # Calculate return if we have weights
            if current_weights is not None and current_weights.sum() > 0:
                period_returns = returns.loc[date]
                gross_return = (current_weights * period_returns).sum()

                # Apply transaction costs
                net_return = gross_return - transaction_cost

                # Ensure return is finite
                if not np.isfinite(net_return):
                    net_return = 0

                portfolio_returns.append(net_return)
                new_value = portfolio_values[-1] * (1 + net_return)
                portfolio_values.append(new_value)

                # Update weights for drift (assuming no rebalancing)
                if not needs_rebalance and current_weights is not None:
                    # Adjust weights based on returns
                    current_weights = current_weights * (1 + period_returns)
                    if current_weights.sum() > 0:
                        current_weights = current_weights / current_weights.sum()

        # Create results
        if len(portfolio_returns) > 0:
            results = pd.DataFrame({
                'portfolio_value': portfolio_values[1:],
                'returns': portfolio_returns
            }, index=returns.index[:len(portfolio_returns)])

            # Calculate metrics (corrected for monthly data)
            total_return = (portfolio_values[-1] / initial_capital - 1) * 100

            # Annualized metrics
            n_years = len(results) / 12
            annual_return = ((portfolio_values[-1] / initial_capital) ** (1/n_years) - 1) * 100
            annual_vol = results['returns'].std() * np.sqrt(12) * 100

            # Risk-free rate adjustment
            excess_return = annual_return - self.risk_free_rate * 100
            sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0

            # Maximum drawdown
            cumulative = (1 + results['returns']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100

            # Other metrics
            win_rate = (results['returns'] > 0).mean() * 100
            avg_turnover = np.mean(turnover) if turnover else 0

        else:
            results = pd.DataFrame()
            total_return = annual_return = annual_vol = sharpe_ratio = max_drawdown = win_rate = avg_turnover = 0

        # Log results
        logger.info("\nBacktest Results:")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Annual Return: {annual_return:.2f}%")
        logger.info(f"Annual Volatility: {annual_vol:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Average Turnover: {avg_turnover:.1%}")
        logger.info(f"Number of regime changes: {len(regime_changes)}")

        return {
            'results': results,
            'metrics': {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_turnover': avg_turnover
            },
            'regime_changes': regime_changes,
            'weights_history': weights_history
        }

    def train_historical_model(self, start_date: str = '1973-01-01',
                              end_date: str = '2000-01-01'):
        """Train model on historical assets"""

        logger.info(f"Training historical model ({start_date} to {end_date})...")

        # Identify assets
        if not self.historical_assets:
            self.identify_available_assets()

        # Get historical returns
        historical_returns = self.returns_df[start_date:end_date]
        historical_asset_cols = [f'{asset}_return' for asset in self.historical_assets]
        existing_cols = [col for col in historical_asset_cols if col in historical_returns.columns]
        historical_returns = historical_returns[existing_cols]

        # Clean data
        historical_returns = historical_returns.dropna(how='all')

        # Get regimes
        regime_history = self._load_regime_history()

        # Align data
        common_dates = historical_returns.index.intersection(regime_history.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates between returns and regime history")

        historical_returns = historical_returns.loc[common_dates]
        regimes = regime_history.loc[common_dates]

        # Calculate regime statistics
        regime_stats = self.calculate_regime_statistics(historical_returns, regimes['regime'])

        # Optimize portfolio for each regime
        logger.info("Optimizing portfolios for each regime...")

        for regime_id, stats in regime_stats.items():
            if stats['observations'] < 12:
                logger.warning(f"Skipping regime {regime_id} - insufficient data")
                continue

            logger.info(f"\nOptimizing for {stats['regime_name']} ({stats['observations']} observations)")

            # Optimize with monthly returns
            weights = self.optimize_portfolio(stats['mean_returns'], stats['cov_matrix'], regime_id)

            # Calculate annual metrics for reporting
            annual_returns = stats['mean_returns'] * 12
            annual_cov = stats['cov_matrix'] * 12

            expected_return = (weights * annual_returns).sum()
            expected_vol = np.sqrt(np.dot(weights, np.dot(annual_cov, weights)))

            self.regime_portfolios[f'historical_{regime_id}'] = {
                'weights': weights,
                'expected_return': expected_return,
                'expected_vol': expected_vol,
                'sharpe': (expected_return - self.risk_free_rate) / expected_vol if expected_vol > 0 else 0,
                'regime_name': stats['regime_name'],
                'n_assets': (weights > self.min_position_size).sum()
            }

            # Log top holdings
            top_holdings = weights.nlargest(5)
            logger.info(f"Top holdings: {dict(top_holdings.apply(lambda x: float(f'{x:.3f}')))}")

    def train_modern_model(self, start_date: str = '2000-01-01'):
        """Train model on modern assets with all available data"""

        logger.info(f"Training modern model (from {start_date})...")

        # Get all available assets
        all_assets = self.historical_assets + self.modern_assets

        # Get returns
        modern_returns = self.returns_df[start_date:]
        asset_cols = [f'{asset}_return' for asset in all_assets
                     if f'{asset}_return' in modern_returns.columns]
        modern_returns = modern_returns[asset_cols]

        # Clean data
        modern_returns = modern_returns.dropna(how='all')

        # Get regimes
        regime_history = self._load_regime_history()

        # Align data
        common_dates = modern_returns.index.intersection(regime_history.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates between returns and regime history")

        modern_returns = modern_returns.loc[common_dates]
        regimes = regime_history.loc[common_dates]

        # Calculate regime statistics
        regime_stats = self.calculate_regime_statistics(modern_returns, regimes['regime'])

        # Optimize portfolio for each regime
        logger.info("Optimizing modern portfolios for each regime...")

        for regime_id, stats in regime_stats.items():
            if stats['observations'] < 12:
                logger.warning(f"Skipping regime {regime_id} - insufficient data")
                continue

            logger.info(f"\nOptimizing for {stats['regime_name']} ({stats['observations']} observations)")

            # Optimize with monthly returns
            weights = self.optimize_portfolio(stats['mean_returns'], stats['cov_matrix'], regime_id)

            # Calculate annual metrics for reporting
            annual_returns = stats['mean_returns'] * 12
            annual_cov = stats['cov_matrix'] * 12

            expected_return = (weights * annual_returns).sum()
            expected_vol = np.sqrt(np.dot(weights, np.dot(annual_cov, weights)))

            self.regime_portfolios[f'modern_{regime_id}'] = {
                'weights': weights,
                'expected_return': expected_return,
                'expected_vol': expected_vol,
                'sharpe': (expected_return - self.risk_free_rate) / expected_vol if expected_vol > 0 else 0,
                'regime_name': stats['regime_name'],
                'n_assets': (weights > self.min_position_size).sum()
            }

            # Log top holdings
            top_holdings = weights.nlargest(5)
            logger.info(f"Top holdings: {dict(top_holdings.apply(lambda x: float(f'{x:.3f}')))}")

    def identify_available_assets(self):
        """Identify historical vs modern assets based on data availability"""

        historical_cutoff = '2000-01-01'
        return_cols = [col for col in self.returns_df.columns if col.endswith('_return')]

        for col in return_cols:
            asset = col.replace('_return', '')
            early_data = self.returns_df[col][:historical_cutoff]

            # Check if asset has meaningful data before 2000
            non_zero_ratio = (early_data != 0).sum() / len(early_data) if len(early_data) > 0 else 0

            if non_zero_ratio > 0.5:
                self.historical_assets.append(asset)
            else:
                self.modern_assets.append(asset)

        logger.info(f"Historical assets (pre-2000): {len(self.historical_assets)}")
        logger.info(f"Modern assets (post-2000): {len(self.modern_assets)}")

        return self.historical_assets, self.modern_assets

    def _load_regime_history(self):
        """Load regime history from various possible locations"""
        possible_paths = [
            'input/regime_history_aligned.csv',
            'input/regime_history.csv',
            'output/regime_classification/regime_history.csv'
        ]

        for path in possible_paths:
            try:
                regime_history = pd.read_csv(path, index_col=0, parse_dates=True)
                logger.info(f"Loaded regime history from: {path}")
                return regime_history
            except FileNotFoundError:
                continue

        raise FileNotFoundError("Could not find regime history file in any expected location")

    def get_current_regime(self):
        """Get current economic regime"""
        try:
            regime_history = self._load_regime_history()
            return int(regime_history.iloc[-1]['regime'])
        except:
            return 1

    def get_portfolio_weights(self, regime: int = None, use_modern: bool = True):
        """Get portfolio weights for a given regime"""

        if regime is None:
            regime = self.get_current_regime()

        portfolio_key = f"{'modern' if use_modern else 'historical'}_{regime}"

        if portfolio_key in self.regime_portfolios:
            return self.regime_portfolios[portfolio_key]['weights']
        else:
            logger.warning(f"No portfolio found for {portfolio_key}")
            return None

    def save_model(self, filepath: str = 'taa_model.pkl'):
        """Save the TAA model"""

        model_data = {
            'regime_portfolios': self.regime_portfolios,
            'historical_assets': self.historical_assets,
            'modern_assets': self.modern_assets,
            'asset_groups': self.asset_groups,
            'risk_parameters': {
                'risk_free_rate': self.risk_free_rate,
                'max_position_size': self.max_position_size,
                'min_position_size': self.min_position_size,
                'min_assets': self.min_assets,
                'rebalance_threshold': self.rebalance_threshold,
                'transaction_cost': self.transaction_cost
            },
            'training_date': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

        # Save summary
        self.save_portfolio_summary(filepath.replace('.pkl', '_summary.txt'))

    def save_portfolio_summary(self, filepath: str):
        """Save human-readable portfolio summary"""

        with open(filepath, 'w') as f:
            f.write("TACTICAL ASSET ALLOCATION MODEL SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Historical Assets: {len(self.historical_assets)}\n")
            f.write(f"Modern Assets: {len(self.modern_assets)}\n\n")

            # Risk parameters
            f.write("Risk Parameters:\n")
            f.write(f"  Max Position Size: {self.max_position_size:.1%}\n")
            f.write(f"  Min Position Size: {self.min_position_size:.1%}\n")
            f.write(f"  Min Assets: {self.min_assets}\n")
            f.write(f"  Transaction Cost: {self.transaction_cost:.2%}\n\n")

            # Portfolio summaries
            for key, portfolio in self.regime_portfolios.items():
                f.write(f"\n{key.upper()}:\n")
                f.write(f"Regime: {portfolio['regime_name']}\n")
                f.write(f"Expected Return: {portfolio['expected_return']:.2%}\n")
                f.write(f"Expected Volatility: {portfolio['expected_vol']:.2%}\n")
                f.write(f"Sharpe Ratio: {portfolio['sharpe']:.3f}\n")
                f.write(f"Number of Assets: {portfolio['n_assets']}\n")

                # Top holdings
                f.write("\nTop Holdings:\n")
                top_holdings = portfolio['weights'].nlargest(10)
                for asset, weight in top_holdings.items():
                    if weight > 0.01:
                        f.write(f"  {asset.replace('_return', '')}: {weight:.1%}\n")

    def plot_regime_allocations(self, save_path: str = 'regime_allocations.png'):
        """Create visualization of allocations across regimes"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio Allocations by Economic Regime', fontsize=16)

        for idx, regime_id in enumerate(range(1, 5)):
            ax = axes[idx // 2, idx % 2]

            # Get modern portfolio
            portfolio_key = f'modern_{regime_id}'
            if portfolio_key in self.regime_portfolios:
                weights = self.regime_portfolios[portfolio_key]['weights']

                # Group by asset class
                asset_class_weights = {}
                for asset, weight in weights.items():
                    if weight > 0.01:
                        asset_name = asset.replace('_return', '')

                        # Determine asset class
                        if any(eq in asset_name for eq in ['XL', 'VNQ'] + self.asset_groups['Equities_FF']):
                            asset_class = 'Equities'
                        elif any(bond in asset_name for bond in ['Treasury', 'TIPS', 'Bond', 'IEF', 'TLT', 'LQD', 'HYG', 'SHY', 'TIP']):
                            asset_class = 'Bonds'
                        elif any(comm in asset_name for comm in ['Oil', 'Copper', 'Wheat', 'Gold', 'DBA', 'DBB', 'DBE', 'GLD']):
                            asset_class = 'Commodities'
                        else:
                            asset_class = 'Other'

                        if asset_class not in asset_class_weights:
                            asset_class_weights[asset_class] = 0
                        asset_class_weights[asset_class] += weight

                if asset_class_weights:
                    # Plot pie chart
                    colors = {'Equities': '#1f77b4', 'Bonds': '#ff7f0e',
                             'Commodities': '#2ca02c', 'Other': '#d62728'}
                    pie_colors = [colors.get(ac, '#9467bd') for ac in asset_class_weights.keys()]

                    wedges, texts, autotexts = ax.pie(
                        asset_class_weights.values(),
                        labels=asset_class_weights.keys(),
                        autopct='%1.1f%%',
                        colors=pie_colors,
                        startangle=90
                    )

                    # Format text
                    for text in texts:
                        text.set_fontsize(10)
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(10)
                        autotext.set_weight('bold')

                portfolio_info = self.regime_portfolios[portfolio_key]
                ax.set_title(f"{self.regime_names[regime_id]}\n"
                           f"Return: {portfolio_info['expected_return']:.1%}, "
                           f"Vol: {portfolio_info['expected_vol']:.1%}, "
                           f"Sharpe: {portfolio_info['sharpe']:.2f}")
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(self.regime_names[regime_id])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Allocation plot saved to {save_path}")


def main():
    """Main execution function"""

    # Configuration
    REGIME_MODEL_PATH = 'input/regime_classifier.pkl'
    DATA_PATH = 'input/taa_returns_aligned.csv'
    OUTPUT_DIR = 'output/taa_model_enhanced'

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Initialize TAA model
        logger.info("Initializing Enhanced TAA model...")
        taa = TacticalAssetAllocation(
            regime_model_path=REGIME_MODEL_PATH,
            data_path=DATA_PATH
        )

        # Train historical model
        taa.train_historical_model()

        # Train modern model
        taa.train_modern_model()

        # Backtest strategy
        backtest_results = taa.backtest_strategy(start_date='2001-01-01')

        # Save model
        model_path = os.path.join(OUTPUT_DIR, 'taa_model_enhanced.pkl')
        taa.save_model(model_path)

        # Create visualizations
        taa.plot_regime_allocations(os.path.join(OUTPUT_DIR, 'regime_allocations.png'))

        # Plot backtest results
        if backtest_results and len(backtest_results['results']) > 0:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

            results = backtest_results['results']

            # Portfolio value
            ax1.plot(results.index,
                    results['portfolio_value'] / 100000,
                    label='TAA Strategy', linewidth=2, color='darkblue')
            ax1.set_ylabel('Portfolio Value (Normalized)')
            ax1.set_title('Enhanced TAA Strategy Performance')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Add regime changes
            for change_date in backtest_results['regime_changes'][:20]:  # Limit to first 20
                if change_date in results.index:
                    ax1.axvline(x=change_date, color='red', alpha=0.3, linestyle='--')

            # Monthly returns
            ax2.bar(results.index, results['returns'] * 100,
                   color=['green' if r > 0 else 'red' for r in results['returns']])
            ax2.set_ylabel('Monthly Return (%)')
            ax2.set_title('Monthly Returns')
            ax2.grid(True, alpha=0.3)

            # Drawdown
            cumulative = (1 + results['returns']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100

            ax3.fill_between(drawdown.index, 0, drawdown,
                           color='red', alpha=0.3, label='Drawdown')
            ax3.set_ylabel('Drawdown (%)')
            ax3.set_xlabel('Date')
            ax3.set_title('Strategy Drawdown')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'backtest_results.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

        # Save backtest metrics
        with open(os.path.join(OUTPUT_DIR, 'backtest_metrics.txt'), 'w') as f:
            f.write("ENHANCED TAA STRATEGY BACKTEST RESULTS\n")
            f.write("=" * 40 + "\n\n")
            for metric, value in backtest_results['metrics'].items():
                f.write(f"{metric.replace('_', ' ').title()}: {value:.2f}\n")

        logger.info("\nEnhanced TAA model training complete!")
        logger.info(f"Results saved to: {OUTPUT_DIR}/")

        return taa, backtest_results

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    taa_model, results = main()
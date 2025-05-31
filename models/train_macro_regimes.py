"""
Economic Regime Classification System for TAA Portfolio Management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import joblib
import json
from typing import Dict, List, Optional
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing HMM
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    logger.warning("hmmlearn not installed. HMM models will not be available.")
    HMM_AVAILABLE = False


class RegimeDefinition:
    """Definition of economic regime with quantitative criteria"""
    def __init__(self, name: str, growth_threshold: float, inflation_threshold: float,
                 description: str, historical_examples: List[str]):
        self.name = name
        self.growth_threshold = growth_threshold
        self.inflation_threshold = inflation_threshold
        self.description = description
        self.historical_examples = historical_examples


class EconomicRegimeClassifier:
    """
    Economic regime classification system

    Four main regimes:
    1. Goldilocks (Growth + Low Inflation)
    2. Reflation (Growth + High Inflation)
    3. Deflation (Slowdown + Low Inflation)
    4. Stagflation (Slowdown + High Inflation)
    """

    START_YEAR = '1973-01-01'

    def __init__(self, data_path: str):
        """Initialize regime classifier"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load and filter data
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.df = self.df[self.df.index >= self.START_YEAR]

        logger.info(f"Loaded {len(self.df)} observations from {self.df.index[0]} to {self.df.index[-1]}")

        # Validate data
        self._validate_data()

        # Define regimes
        self.regimes = {
            1: RegimeDefinition("Goldilocks", 0.0, 0.0,
                              "Economic Growth + Low Inflation",
                              ["1995-1999", "2013-2017"]),
            2: RegimeDefinition("Reflation", 0.0, 0.0,
                              "Economic Growth + High Inflation",
                              ["2003-2007", "2021-2022"]),
            3: RegimeDefinition("Deflation", 0.0, 0.0,
                              "Economic Slowdown + Low Inflation",
                              ["2008-2009", "2020-Q2"]),
            4: RegimeDefinition("Stagflation", 0.0, 0.0,
                              "Economic Slowdown + High Inflation",
                              ["1973-1975", "1979-1982"])
        }

        # Initialize model components
        self.models = {}
        self.feature_importance = {}
        self.scaler = None
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        self.regime_history = None
        self.transition_matrix = None
        self.preserve_data_mode = False

    def _validate_data(self):
        """Validate data has required indicators"""
        # Check for growth indicators
        growth_indicators = ['Composite_Growth', 'Real GDP_YOY', 'Industrial Production Index_YOY']
        has_growth = any(col in self.df.columns for col in growth_indicators)

        # Check for inflation indicators
        inflation_indicators = ['Composite_Inflation', 'Core CPI_YOY', 'CPI All Items_YOY']
        has_inflation = any(col in self.df.columns for col in inflation_indicators)

        if not has_growth:
            raise ValueError(f"No growth indicators found. Expected one of: {growth_indicators}")
        if not has_inflation:
            raise ValueError(f"No inflation indicators found. Expected one of: {inflation_indicators}")

        logger.info("Data validation passed")

    def prepare_regime_indicators(self) -> pd.DataFrame:
        """Prepare key indicators for regime determination"""
        logger.info("Preparing regime indicators...")

        regime_data = pd.DataFrame(index=self.df.index)

        # 1. Growth score
        if 'Composite_Growth' in self.df.columns:
            regime_data['growth_score'] = self.df['Composite_Growth']
        else:
            # Create composite from available indicators
            growth_cols = self._find_indicators(['Real GDP_YOY', 'GDPC1_YOY',
                                               'Industrial Production Index_YOY', 'INDPRO_YOY',
                                               'Retail Sales Ex Food Services_YOY', 'RSXFS_YOY',
                                               'Nonfarm Payrolls_MOM', 'PAYEMS_MOM'])
            regime_data['growth_score'] = self._create_composite_score(growth_cols)

        # 2. Inflation score
        if 'Composite_Inflation' in self.df.columns:
            regime_data['inflation_score'] = self.df['Composite_Inflation']
        else:
            inflation_cols = self._find_indicators(['Core CPI_YOY', 'CPILFESL_YOY',
                                                  'CPI All Items_YOY', 'CPIAUCSL_YOY',
                                                  'PPI All Commodities_YOY', 'PPIACO_YOY'])
            regime_data['inflation_score'] = self._create_composite_score(inflation_cols)

        # 3. Additional indicators
        regime_data['growth_momentum'] = regime_data['growth_score'].diff(3)
        regime_data['inflation_momentum'] = regime_data['inflation_score'].diff(3)
        regime_data['growth_inflation_diff'] = regime_data['growth_score'] - regime_data['inflation_score']

        # 4. Optional indicators
        self._add_optional_indicators(regime_data)

        # Clean data
        regime_data_clean = regime_data.dropna(subset=['growth_score', 'inflation_score'])
        regime_data_clean = regime_data_clean.ffill().fillna(0)

        logger.info(f"Prepared {len(regime_data_clean)} observations with {len(regime_data_clean.columns)} features")

        return regime_data_clean

    def _find_indicators(self, candidates: List[str]) -> List[str]:
        """Find available indicators from candidates list"""
        return [col for col in candidates if col in self.df.columns]

    def _create_composite_score(self, columns: List[str]) -> pd.Series:
        """Create standardized composite score from multiple columns"""
        if not columns:
            raise ValueError("No columns provided for composite score")

        standardized = pd.DataFrame()
        for col in columns:
            if '_MOM' in col:
                values = self.df[col] * 12
            else:
                values = self.df[col]

            if values.std() > 0:
                standardized[col] = (values - values.mean()) / values.std()
            else:
                standardized[col] = 0

        return standardized.mean(axis=1)

    def _add_optional_indicators(self, regime_data: pd.DataFrame):
        """Add optional indicators if available"""
        # Financial stress
        if 'Financial_Stress_Index' in self.df.columns:
            regime_data['financial_stress'] = self.df['Financial_Stress_Index']
        else:
            stress_components = []
            for col in ['VIX', 'VIXCLS']:
                if col in self.df.columns:
                    stress_components.append(self.df[col])
            for col in ['High Yield Spread', 'BAMLH0A0HYM2']:
                if col in self.df.columns:
                    stress_components.append(self.df[col])

            if stress_components:
                stress_df = pd.DataFrame()
                for i, comp in enumerate(stress_components):
                    if comp.std() > 0:
                        stress_df[f'stress_{i}'] = (comp - comp.mean()) / comp.std()
                regime_data['financial_stress'] = stress_df.mean(axis=1)

        # Yield curve
        for col in ['10Y-2Y Spread', 'T10Y2Y', '10Y_2Y_Spread']:
            if col in self.df.columns:
                regime_data['yield_curve'] = self.df[col]
                break

        # VIX
        for col in ['VIX', 'VIXCLS']:
            if col in self.df.columns:
                regime_data['vix'] = self.df[col]
                regime_data['vix_ma'] = regime_data['vix'].rolling(window=21).mean()
                regime_data['vix_regime'] = (regime_data['vix'] > regime_data['vix_ma']).astype(int)
                break

    def classify_regimes_rule_based(self, regime_data: pd.DataFrame) -> pd.Series:
        """Rule-based regime classification"""
        logger.info("Classifying regimes using rule-based approach...")

        regimes = pd.Series(index=regime_data.index, dtype=int)

        # Apply classification rules
        growth_positive = regime_data['growth_score'] > 0
        inflation_low = regime_data['inflation_score'] < 0

        regimes[growth_positive & inflation_low] = 1  # Goldilocks
        regimes[growth_positive & ~inflation_low] = 2  # Reflation
        regimes[~growth_positive & inflation_low] = 3  # Deflation
        regimes[~growth_positive & ~inflation_low] = 4  # Stagflation

        # Smooth transitions
        return self._smooth_regime_transitions(regimes, min_duration=2)

    def _smooth_regime_transitions(self, regimes: pd.Series, min_duration: int = 2) -> pd.Series:
        """Remove regime periods shorter than min_duration"""
        regimes_smooth = regimes.copy()

        # Find regime changes
        changes = regimes.diff().fillna(0) != 0
        change_indices = regimes.index[changes].tolist()

        # Check duration of each regime period
        for i in range(len(change_indices) - 1):
            start_idx = regimes.index.get_loc(change_indices[i])
            end_idx = regimes.index.get_loc(change_indices[i + 1])
            duration = end_idx - start_idx

            if duration < min_duration and i > 0:
                # Replace short regime with previous
                prev_regime = regimes.iloc[start_idx - 1]
                regimes_smooth.iloc[start_idx:end_idx] = prev_regime

        return regimes_smooth

    def create_advanced_features(self, regime_data: pd.DataFrame,
                               preserve_data: bool = False) -> pd.DataFrame:
        """Create features for ML models"""
        logger.info("Creating advanced features...")

        features = regime_data.copy()

        # Determine if we need to preserve data
        if len(regime_data) < 252:
            preserve_data = True

        # Moving averages
        for window in [3, 6, 12]:
            min_periods = max(1, window // 2) if preserve_data else window
            features[f'growth_ma_{window}'] = features['growth_score'].rolling(
                window=window, min_periods=min_periods).mean()
            features[f'inflation_ma_{window}'] = features['inflation_score'].rolling(
                window=window, min_periods=min_periods).mean()

        # Volatility
        vol_window = 12
        vol_min_periods = 6 if preserve_data else 12
        features['growth_volatility'] = features['growth_score'].rolling(
            window=vol_window, min_periods=vol_min_periods).std()
        features['inflation_volatility'] = features['inflation_score'].rolling(
            window=vol_window, min_periods=vol_min_periods).std()

        # Trends
        if 'growth_ma_12' in features.columns:
            features['growth_trend'] = np.where(features['growth_score'] > features['growth_ma_12'], 1, -1)
            features['inflation_trend'] = np.where(features['inflation_score'] > features['inflation_ma_12'], 1, -1)
        else:
            features['growth_trend'] = np.where(features['growth_score'] > features['growth_ma_3'], 1, -1)
            features['inflation_trend'] = np.where(features['inflation_score'] > features['inflation_ma_3'], 1, -1)

        # Time features
        features['month'] = features.index.month
        features['quarter'] = features.index.quarter
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

        # Interactions
        features['growth_x_inflation'] = features['growth_score'] * features['inflation_score']
        if 'financial_stress' in features.columns:
            features['growth_x_stress'] = features['growth_score'] * features['financial_stress']

        # Lags
        lag_values = [1, 3] if preserve_data else [1, 3, 6]
        for lag in lag_values:
            features[f'growth_lag_{lag}'] = features['growth_score'].shift(lag)
            features[f'inflation_lag_{lag}'] = features['inflation_score'].shift(lag)

        # Remove NaN
        features = features.dropna()
        logger.info(f"Created {len(features.columns)} features with {len(features)} observations")

        return features

    def build_ml_models(self, features: pd.DataFrame, labels: pd.Series) -> Dict:
        """Build and train ML models for regime classification"""
        logger.info("Building ML models...")

        # Align data
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]

        logger.info(f"Total samples for modeling: {len(X)}")

        if len(X) < 100:
            logger.warning("Limited data for modeling. Results may be less reliable.")

        # Time-based train/test split
        # Use last 20% of data for testing
        split_point = int(len(X) * 0.8)

        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Train period: {X_train.index[0]} to {X_train.index[-1]}")
        logger.info(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scaler = scaler
        self.feature_names = X.columns.tolist()

        results = {}

        # 1. Logistic Regression
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )

        # Time series cross-validation
        n_splits = min(3, len(X_train) // 60)
        if n_splits > 1:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_train_scaled):
                lr_temp = LogisticRegression(
                    multi_class='multinomial',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                )
                lr_temp.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
                val_pred = lr_temp.predict(X_train_scaled[val_idx])
                cv_scores.append(accuracy_score(y_train.iloc[val_idx], val_pred))

            logger.info(f"LR CV mean accuracy: {np.mean(cv_scores):.3f}")

        # Train final model
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_accuracy = accuracy_score(y_test, lr_pred)

        results['logistic_regression'] = {
            'model': lr_model,
            'accuracy': lr_accuracy,
            'predictions': lr_pred,
            'classification_report': classification_report(y_test, lr_pred)
        }

        # Feature importance
        self.feature_importance['logistic_regression'] = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(lr_model.coef_).mean(axis=0)
        }).sort_values('importance', ascending=False)

        logger.info(f"Logistic Regression Test Accuracy: {lr_accuracy:.3f}")

        # 2. Random Forest
        logger.info("Training Random Forest...")

        # Simple hyperparameter selection
        if len(X_train) < 200:
            rf_params = {'n_estimators': 50, 'max_depth': 5}
        else:
            rf_params = {'n_estimators': 100, 'max_depth': 10}

        rf_model = RandomForestClassifier(
            **rf_params,
            min_samples_split=max(2, len(X_train) // 50),
            min_samples_leaf=max(1, len(X_train) // 100),
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)

        results['random_forest'] = {
            'model': rf_model,
            'accuracy': rf_accuracy,
            'predictions': rf_pred,
            'classification_report': classification_report(y_test, rf_pred)
        }

        self.feature_importance['random_forest'] = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"Random Forest Test Accuracy: {rf_accuracy:.3f}")

        # 3. Hidden Markov Model (if available)
        if HMM_AVAILABLE and len(X_train) > 100:
            logger.info("Training Hidden Markov Model...")

            # Use key features for HMM
            hmm_features = ['growth_score', 'inflation_score', 'growth_momentum', 'inflation_momentum']
            hmm_features = [f for f in hmm_features if f in X.columns]

            if len(hmm_features) >= 2:
                X_hmm_train = X_train[hmm_features].values
                X_hmm_test = X_test[hmm_features].values

                # Train HMM
                best_hmm = None
                best_score = -np.inf

                for attempt in range(3):
                    try:
                        hmm_model = hmm.GaussianHMM(
                            n_components=4,
                            covariance_type="full",
                            n_iter=100,
                            random_state=42 + attempt
                        )
                        hmm_model.fit(X_hmm_train)
                        score = hmm_model.score(X_hmm_train)

                        if score > best_score:
                            best_score = score
                            best_hmm = hmm_model
                    except:
                        continue

                if best_hmm is not None:
                    # Predict states
                    hmm_states_train = best_hmm.predict(X_hmm_train)
                    hmm_states_test = best_hmm.predict(X_hmm_test)

                    # Map HMM states to regimes
                    state_to_regime = {}
                    for state in range(4):
                        mask = hmm_states_train == state
                        if mask.sum() > 0:
                            most_common = pd.Series(y_train[mask]).mode()
                            if len(most_common) > 0:
                                state_to_regime[state] = most_common[0]

                    # Apply mapping
                    hmm_pred = pd.Series(hmm_states_test).map(state_to_regime).fillna(1).astype(int).values
                    hmm_accuracy = accuracy_score(y_test, hmm_pred)

                    results['hmm'] = {
                        'model': best_hmm,
                        'accuracy': hmm_accuracy,
                        'predictions': hmm_pred,
                        'transition_matrix': best_hmm.transmat_,
                        'classification_report': classification_report(y_test, hmm_pred)
                    }

                    logger.info(f"HMM Accuracy: {hmm_accuracy:.3f}")

                    # Save transition matrix
                    self.transition_matrix = pd.DataFrame(
                        best_hmm.transmat_,
                        index=[f"From_{self.regimes[state_to_regime.get(i, 1)].name}" for i in range(4)],
                        columns=[f"To_{self.regimes[state_to_regime.get(i, 1)].name}" for i in range(4)]
                    )

        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        self.models = {name: result['model'] for name, result in results.items()}

        logger.info(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.3f}")

        return results

    def predict_regime(self, current_data: pd.DataFrame,
                      model_name: Optional[str] = None) -> Dict:
        """Predict current regime"""
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")

        # Prepare features - use same preserve_data setting as during training
        preserve_data = getattr(self, 'preserve_data_mode', len(current_data) < 400)
        features = self.create_advanced_features(current_data, preserve_data=preserve_data)

        if len(features) == 0:
            raise ValueError("No valid features after preprocessing")

        # Get last observation and only use features that were used in training
        last_features = features.iloc[-1:]

        # Filter to only features that exist in both training and current data
        available_features = [f for f in self.feature_names if f in last_features.columns]
        if len(available_features) != len(self.feature_names):
            logger.warning(f"Some features missing. Using {len(available_features)} of {len(self.feature_names)} features")
            # Fill missing features with zeros
            for missing_feature in set(self.feature_names) - set(available_features):
                last_features[missing_feature] = 0

        last_features = last_features[self.feature_names]
        last_features_scaled = self.scaler.transform(last_features)

        # Predict
        current_regime = model.predict(last_features_scaled)[0]

        # Get probabilities
        if hasattr(model, 'predict_proba'):
            regime_probabilities = model.predict_proba(last_features_scaled)[0]
        else:
            regime_probabilities = np.zeros(4)
            regime_probabilities[current_regime - 1] = 1.0

        return {
            'current_regime': int(current_regime),
            'regime_name': self.regimes[current_regime].name,
            'regime_description': self.regimes[current_regime].description,
            'probabilities': {
                self.regimes[i+1].name: float(prob)
                for i, prob in enumerate(regime_probabilities)
            },
            'model_used': model_name,
            'prediction_date': features.index[-1].strftime('%Y-%m-%d')
        }

    def analyze_regime_characteristics(self, features: pd.DataFrame,
                                     regimes: pd.Series) -> pd.DataFrame:
        """Analyze characteristics of each regime"""
        analysis_df = features.copy()
        analysis_df['regime'] = regimes

        # Key indicators to analyze
        key_indicators = [
            'growth_score', 'inflation_score', 'growth_momentum',
            'inflation_momentum', 'financial_stress', 'yield_curve', 'vix'
        ]

        available_indicators = [col for col in key_indicators if col in analysis_df.columns]

        # Calculate statistics by regime
        regime_stats = []

        for regime_id in range(1, 5):
            regime_data = analysis_df[analysis_df['regime'] == regime_id]

            if len(regime_data) > 0:
                stats = {
                    'regime_id': regime_id,
                    'regime_name': self.regimes[regime_id].name,
                    'observations': len(regime_data),
                    'percentage': len(regime_data) / len(analysis_df) * 100,
                    'avg_duration_months': self._calculate_avg_duration(regimes, regime_id)
                }

                # Add mean values for indicators
                for indicator in available_indicators:
                    stats[f'{indicator}_mean'] = regime_data[indicator].mean()
                    stats[f'{indicator}_std'] = regime_data[indicator].std()

                regime_stats.append(stats)

        return pd.DataFrame(regime_stats)

    def _calculate_avg_duration(self, regimes: pd.Series, regime_id: int) -> float:
        """Calculate average duration of regime periods"""
        # Find all periods of this regime
        is_regime = (regimes == regime_id).astype(int)
        regime_changes = is_regime.diff().fillna(0)

        # Start and end points
        starts = regimes.index[regime_changes == 1].tolist()
        ends = regimes.index[regime_changes == -1].tolist()

        if not starts:
            return 0

        # If last period hasn't ended
        if len(ends) < len(starts):
            ends.append(regimes.index[-1])

        # Match starts with ends properly
        # Some ends might come before starts due to the data starting in a regime
        valid_durations = []
        for i, start in enumerate(starts):
            # Find the first end after this start
            matching_ends = [e for e in ends if e > start]
            if matching_ends:
                end = matching_ends[0]
                duration = (end - start).days / 30  # In months
                if duration > 0:  # Only positive durations
                    valid_durations.append(duration)

        return np.mean(valid_durations) if valid_durations else 0

    def validate_historical_periods(self, regimes: pd.Series) -> Dict:
        """Validate classification against known historical periods"""
        logger.info("Validating against historical periods...")

        # Known historical periods (adjusted for data availability)
        historical_periods = {
            'Stagflation_1970s': ('1973-11-01', '1975-12-31', 4),
            'Volcker_Disinflation': ('1979-10-01', '1982-12-31', 4),
            'Great_Moderation': ('1995-01-01', '1999-12-31', 1),
            'Dot_Com_Bust': ('2001-03-01', '2001-11-30', 3),
            'Housing_Bubble': ('2004-01-01', '2007-06-30', 2),
            'Financial_Crisis': ('2008-09-01', '2009-06-30', 3),
            'COVID_Shock': ('2020-03-01', '2020-06-30', 3),
            'Post_COVID_Inflation': ('2021-06-01', '2022-12-31', 2)
        }

        validation_results = {}
        data_start = regimes.index[0]
        data_end = regimes.index[-1]

        for period_name, (start, end, expected_regime) in historical_periods.items():
            period_start = pd.Timestamp(start)
            period_end = pd.Timestamp(end)

            # Check if period is in our data range
            if period_start < data_start or period_end > data_end:
                validation_results[period_name] = {
                    'error': 'Period outside data range'
                }
                continue

            # Get regimes for this period
            period_regimes = regimes[start:end]

            if len(period_regimes) > 0:
                most_common_regime = period_regimes.mode()[0]
                regime_percentage = (period_regimes == most_common_regime).sum() / len(period_regimes) * 100

                validation_results[period_name] = {
                    'expected': self.regimes[expected_regime].name,
                    'actual': self.regimes[most_common_regime].name,
                    'accuracy': regime_percentage,
                    'correct': most_common_regime == expected_regime
                }

        return validation_results

    def plot_regime_analysis(self, features: pd.DataFrame, regimes: pd.Series,
                           save_path: str = 'regime_analysis.png'):
        """Create comprehensive regime analysis visualization"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Economic Regime Analysis', fontsize=16)

        regime_colors = {1: 'green', 2: 'yellow', 3: 'red', 4: 'darkred'}

        # 1. Regimes over time
        ax = axes[0, 0]
        for regime_id in range(1, 5):
            mask = regimes == regime_id
            if mask.sum() > 0:
                ax.scatter(regimes.index[mask], [regime_id] * mask.sum(),
                          color=regime_colors[regime_id],
                          label=self.regimes[regime_id].name,
                          alpha=0.6, s=10)

        ax.set_ylabel('Regime')
        ax.set_title('Economic Regimes Over Time')
        ax.legend()
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels([self.regimes[i].name for i in range(1, 5)])

        # 2. Growth vs Inflation scatter
        ax = axes[0, 1]
        if 'growth_score' in features.columns and 'inflation_score' in features.columns:
            for regime_id in range(1, 5):
                mask = regimes == regime_id
                if mask.sum() > 0:
                    ax.scatter(features.loc[mask, 'growth_score'],
                              features.loc[mask, 'inflation_score'],
                              color=regime_colors[regime_id],
                              label=self.regimes[regime_id].name,
                              alpha=0.5)

            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Growth Score')
            ax.set_ylabel('Inflation Score')
            ax.set_title('Regime Distribution in Growth-Inflation Space')
            ax.legend()

        # 3. Regime distribution
        ax = axes[1, 0]
        regime_counts = regimes.value_counts().sort_index()
        colors = [regime_colors[i] for i in regime_counts.index]
        bars = ax.bar(range(len(regime_counts)), regime_counts.values, color=colors)
        ax.set_xticks(range(len(regime_counts)))
        ax.set_xticklabels([self.regimes[i].name for i in regime_counts.index], rotation=45)
        ax.set_ylabel('Number of Months')
        ax.set_title('Regime Distribution')

        # Add percentages
        total = regime_counts.sum()
        for bar, count in zip(bars, regime_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count/total*100:.1f}%',
                   ha='center', va='bottom')

        # 4. Transition matrix
        ax = axes[1, 1]
        if self.transition_matrix is not None:
            sns.heatmap(self.transition_matrix, annot=True, fmt='.2f',
                       cmap='Blues', ax=ax, cbar_kws={'label': 'Transition Probability'})
            ax.set_title('Regime Transition Probabilities')

        # 5. Average characteristics by regime
        ax = axes[2, 0]
        regime_chars = self.analyze_regime_characteristics(features, regimes)

        if 'growth_score_mean' in regime_chars.columns:
            x = np.arange(len(regime_chars))
            width = 0.35

            ax.bar(x - width/2, regime_chars['growth_score_mean'],
                  width, label='Growth', color='green', alpha=0.7)
            ax.bar(x + width/2, regime_chars['inflation_score_mean'],
                  width, label='Inflation', color='red', alpha=0.7)

            ax.set_xlabel('Regime')
            ax.set_ylabel('Average Score')
            ax.set_title('Average Growth and Inflation by Regime')
            ax.set_xticks(x)
            ax.set_xticklabels(regime_chars['regime_name'], rotation=45)
            ax.legend()
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 6. Feature importance
        ax = axes[2, 1]
        if self.feature_importance:
            top_features = self.feature_importance.get(
                self.best_model_name,
                list(self.feature_importance.values())[0]
            ).head(10)

            ax.barh(top_features['feature'], top_features['importance'])
            ax.set_xlabel('Importance')
            ax.set_title(f'Top 10 Feature Importance ({self.best_model_name})')
            ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, filepath: str = 'regime_classifier.pkl'):
        """Save trained model and components"""
        logger.info(f"Saving model to {filepath}...")

        model_package = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'regimes': {k: vars(v) for k, v in self.regimes.items()},
            'transition_matrix': self.transition_matrix,
            'feature_importance': self.feature_importance,
            'models': self.models,
            'preserve_data_mode': self.preserve_data_mode
        }

        joblib.dump(model_package, filepath)

        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'features_count': len(self.feature_names) if self.feature_names else 0,
            'preserve_data_mode': self.preserve_data_mode,
            'regimes': {
                str(k): {
                    'name': v.name,
                    'description': v.description,
                    'historical_examples': v.historical_examples
                }
                for k, v in self.regimes.items()
            }
        }

        with open(filepath.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model saved successfully")

    def run_complete_analysis(self, output_dir: str = None):
        """Run complete regime classification analysis"""
        logger.info("="*60)
        logger.info("Starting Economic Regime Classification Analysis")
        logger.info("="*60)

        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 1. Prepare indicators
        regime_data = self.prepare_regime_indicators()

        # Check minimum data
        MIN_OBSERVATIONS = 100
        if len(regime_data) < MIN_OBSERVATIONS:
            raise ValueError(f"Insufficient data: {len(regime_data)} observations. "
                           f"Need at least {MIN_OBSERVATIONS}.")

        # 2. Classify regimes
        regimes = self.classify_regimes_rule_based(regime_data)
        self.regime_history = regimes

        # 3. Create features
        preserve_data = len(regime_data) < 400
        if preserve_data:
            logger.info("Using data preservation mode due to limited observations")
        features = self.create_advanced_features(regime_data, preserve_data=preserve_data)

        # Save preserve_data mode for predictions
        self.preserve_data_mode = preserve_data

        # 4. Build models
        model_results = self.build_ml_models(features, regimes)

        # 5. Validate on historical periods
        validation_results = self.validate_historical_periods(regimes)

        logger.info("\nHistorical Validation Results:")
        for period, result in validation_results.items():
            if 'correct' in result:
                status = "✓" if result['correct'] else "✗"
                logger.info(f"{status} {period}: Expected {result['expected']}, "
                          f"Got {result['actual']} ({result['accuracy']:.1f}% of period)")

        # 6. Analyze regime characteristics
        regime_characteristics = self.analyze_regime_characteristics(features, regimes)

        logger.info("\nRegime Characteristics:")
        print(regime_characteristics[['regime_name', 'observations', 'percentage',
                                    'avg_duration_months']].round(2))

        # 7. Create visualizations
        plot_path = os.path.join(output_dir, 'regime_analysis.png') if output_dir else 'regime_analysis.png'
        self.plot_regime_analysis(features, regimes, save_path=plot_path)

        # 8. Save model
        model_path = os.path.join(output_dir, 'regime_classifier.pkl') if output_dir else 'regime_classifier.pkl'
        self.save_model(model_path)

        # 9. Test prediction
        logger.info("\nTesting prediction on latest data:")
        # Use prepared regime_data for prediction
        latest_prediction = self.predict_regime(regime_data.iloc[-30:])

        logger.info(f"Current regime: {latest_prediction['regime_name']}")
        logger.info("Regime probabilities:")
        for regime, prob in latest_prediction['probabilities'].items():
            logger.info(f"  {regime}: {prob:.2%}")

        logger.info("\nAnalysis complete!")

        return {
            'regime_history': regimes,
            'model_results': model_results,
            'validation_results': validation_results,
            'regime_characteristics': regime_characteristics,
            'latest_prediction': latest_prediction
        }


# Main execution
if __name__ == "__main__":
    # Configuration
    DATA_PATH = 'input/economic_indicators_lagged.csv'
    OUTPUT_DIR = 'output/regime_classification'

    # Check HMM availability
    if not HMM_AVAILABLE:
        print("WARNING: hmmlearn not installed. HMM models will be skipped.")
        print("To install: pip install hmmlearn")
        print()

    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Initialize and run classifier
        classifier = EconomicRegimeClassifier(data_path=DATA_PATH)
        results = classifier.run_complete_analysis(output_dir=OUTPUT_DIR)

        # Save regime history
        regime_history_path = os.path.join(OUTPUT_DIR, 'regime_history.csv')
        results['regime_history'].to_csv(regime_history_path, header=['regime'])

        # Save report
        report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("ECONOMIC REGIME CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Data file: {DATA_PATH}\n")
            f.write(f"Date range: {classifier.df.index[0]} to {classifier.df.index[-1]}\n")
            f.write(f"Total observations: {len(results['regime_history'])}\n\n")

            f.write("Model Performance:\n")
            for model_name, model_result in results['model_results'].items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"Accuracy: {model_result['accuracy']:.3f}\n")
                if 'classification_report' in model_result:
                    f.write(f"\nClassification Report:\n{model_result['classification_report']}\n")
                else:
                    f.write("(No detailed classification report available)\n")

            f.write("\nRegime Characteristics:\n")
            f.write(results['regime_characteristics'].to_string())

            f.write("\n\nLatest Prediction:\n")
            f.write(f"Current regime: {results['latest_prediction']['regime_name']}\n")
            f.write("Probabilities:\n")
            for regime, prob in results['latest_prediction']['probabilities'].items():
                f.write(f"  {regime}: {prob:.2%}\n")

        print("\n" + "="*60)
        print("ECONOMIC REGIME CLASSIFICATION COMPLETE")
        print("="*60)
        print(f"Best model: {classifier.best_model_name}")
        print(f"Current regime: {results['latest_prediction']['regime_name']}")
        print(f"\nResults saved to: {OUTPUT_DIR}/")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

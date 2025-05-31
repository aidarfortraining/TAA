"""
Economic Data Collection System for TAA (Tactical Asset Allocation)
"""

import pandas as pd
from fredapi import Fred
import warnings
import os
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Configuration parameters
START_DATE = '1973-01-01'  # Optimal point - complete data for most indicators
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')

# Create output directories if they don't exist
os.makedirs('output', exist_ok=True)
os.makedirs('output/plots', exist_ok=True)


class ImprovedEconomicDataCollector:
    def __init__(self, api_key: str, start_date: str = START_DATE, end_date: str = END_DATE):
        """
        Initialize with API key as parameter for security
        """
        self.start_date = start_date
        self.end_date = end_date
        self.fred = Fred(api_key=api_key)
        self.data_quality_report = {}  # For data quality report

        # Extended dictionary of indicators with additional parameters
        self.indicators = {
            # ECONOMIC GROWTH INDICATORS
            'growth_indicators': {
                'GDPC1': {
                    'name': 'Real GDP',
                    'description': 'US Real GDP (inflation-adjusted)',
                    'frequency': 'quarterly',
                    'transform': 'yoy_growth',
                    'publication_lag': 30,  # Published with ~30 day lag
                    'start_date_override': None  # Use common start date
                },
                'INDPRO': {
                    'name': 'Industrial Production Index',
                    'description': 'Industrial production index',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 15
                },
                'PAYEMS': {
                    'name': 'Nonfarm Payrolls',
                    'description': 'Nonfarm employment',
                    'frequency': 'monthly',
                    'transform': 'mom_change',
                    'publication_lag': 5
                },
                'UNRATE': {
                    'name': 'Unemployment Rate',
                    'description': 'Unemployment rate',
                    'frequency': 'monthly',
                    'transform': 'level',
                    'publication_lag': 5
                },
                'ICSA': {
                    'name': 'Initial Jobless Claims',
                    'description': 'Initial jobless claims',
                    'frequency': 'weekly',
                    'transform': '4week_ma',
                    'publication_lag': 5
                },
                'HOUST': {
                    'name': 'Housing Starts',
                    'description': 'Housing starts',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 18
                },
                'RSXFS': {
                    'name': 'Retail Sales Ex Food Services',
                    'description': 'Retail sales excluding food services',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 14
                }
            },

            # INFLATION INDICATORS
            'inflation_indicators': {
                'CPILFESL': {
                    'name': 'Core CPI',
                    'description': 'Core inflation (excluding food and energy)',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 12
                },
                'CPIAUCSL': {
                    'name': 'CPI All Items',
                    'description': 'Consumer price index all items',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 12
                },
                'PPIACO': {
                    'name': 'PPI All Commodities',
                    'description': 'Producer price index',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 14
                },
                'DCOILWTICO': {
                    'name': 'WTI Oil Price',
                    'description': 'WTI oil price',
                    'frequency': 'daily',
                    'transform': 'monthly_avg_yoy',
                    'publication_lag': 1
                },
                'T5YIE': {
                    'name': '5-Year Breakeven Inflation',
                    'description': '5-year inflation expectations',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1,
                    'start_date_override': '2003-01-01'  # Data available from 2003
                },
                'T10YIE': {
                    'name': '10-Year Breakeven Inflation',
                    'description': '10-year inflation expectations',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1,
                    'start_date_override': '2003-01-01'
                }
            },

            # MONETARY AND FINANCIAL INDICATORS
            'monetary_indicators': {
                'DFF': {
                    'name': 'Federal Funds Rate',
                    'description': 'Fed funds rate',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1
                },
                'DGS10': {
                    'name': '10-Year Treasury Rate',
                    'description': '10-year Treasury yield',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1
                },
                'DGS2': {
                    'name': '2-Year Treasury Rate',
                    'description': '2-year Treasury yield',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1
                },
                'T10Y2Y': {
                    'name': '10Y-2Y Spread',
                    'description': 'Yield spread (yield curve)',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1
                },
                'M2SL': {
                    'name': 'M2 Money Supply',
                    'description': 'M2 money supply',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 14
                }
            },

            # MARKET STRESS AND SENTIMENT INDICATORS
            'market_indicators': {
                'VIXCLS': {
                    'name': 'VIX',
                    'description': 'Volatility index (fear index)',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1,
                    'start_date_override': '1990-01-01'  # VIX starts from 1990
                },
                'DEXUSEU': {
                    'name': 'USD/EUR Exchange Rate',
                    'description': 'USD to EUR exchange rate',
                    'frequency': 'daily',
                    'transform': 'monthly_avg_yoy',
                    'publication_lag': 1,
                    'start_date_override': '1999-01-01'  # EUR appeared in 1999
                },
                'BAMLH0A0HYM2': {
                    'name': 'High Yield Spread',
                    'description': 'High yield bond spread',
                    'frequency': 'daily',
                    'transform': 'monthly_avg',
                    'publication_lag': 1,
                    'start_date_override': '1996-12-31'
                },
                'UMCSENT': {
                    'name': 'Consumer Sentiment',
                    'description': 'Consumer sentiment',
                    'frequency': 'monthly',
                    'transform': 'level',
                    'publication_lag': 0
                }
            },

            # LEADING INDICATORS
            'leading_indicators': {
                'AWHMAN': {
                    'name': 'Average Weekly Hours Manufacturing',
                    'description': 'Average weekly hours in manufacturing',
                    'frequency': 'monthly',
                    'transform': 'yoy_change',
                    'publication_lag': 5
                },
                'NEWORDER': {
                    'name': 'Manufacturers New Orders',
                    'description': 'Manufacturers new orders',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 35
                },
                'PERMIT': {
                    'name': 'Building Permits',
                    'description': 'Building permits',
                    'frequency': 'monthly',
                    'transform': 'yoy_growth',
                    'publication_lag': 18
                }
            }
        }

    def fetch_single_indicator(self, series_id: str, indicator_info: Dict) -> Optional[pd.DataFrame]:
        """
        Indicator loading with special case handling
        """
        try:
            print(f"Loading {indicator_info['name']} ({series_id})...")

            # Determine start date for specific indicator
            start_date = indicator_info.get('start_date_override', self.start_date)

            # Get raw data
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=self.end_date
            )

            if data is None or len(data) == 0:
                print(f"  ⚠️  No data for {series_id}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[series_id])
            df.index = pd.to_datetime(df.index)

            # Save data quality information
            self.data_quality_report[series_id] = {
                'name': indicator_info['name'],
                'raw_count': len(df),
                'first_date': df.index[0],
                'last_date': df.index[-1],
                'frequency': indicator_info['frequency']
            }

            # Apply transformations
            df_transformed = self.apply_transformation(
                df,
                series_id,
                indicator_info['transform'],
                indicator_info['frequency']
            )

            print(f"  ✓ Loaded: {len(df_transformed)} observations")

            return df_transformed

        except Exception as e:
            print(f"  ✗ Error loading {series_id}: {str(e)}")
            return None

    def apply_transformation(self, df: pd.DataFrame, series_id: str,
                           transform_type: str, frequency: str) -> pd.DataFrame:
        """
        Transformation application considering data frequency
        """
        # First apply basic transformations
        if transform_type == 'level':
            result = df

        elif transform_type == 'yoy_growth':
            if frequency == 'quarterly':
                # For quarterly data - growth vs corresponding quarter last year
                df[f'{series_id}_YOY'] = df[series_id].pct_change(periods=4) * 100
            else:
                # For monthly data - vs corresponding month
                df[f'{series_id}_YOY'] = df[series_id].pct_change(periods=12) * 100
            result = df[[f'{series_id}_YOY']]

        elif transform_type == 'mom_change':
            df[f'{series_id}_MOM'] = df[series_id].diff()
            result = df[[f'{series_id}_MOM']]

        elif transform_type == '4week_ma':
            # Apply 4-week moving average
            df_ma = df.copy()
            df_ma[f'{series_id}_4WMA'] = df[series_id].rolling(window=4, min_periods=1).mean()
            # For weekly data first resample to monthly frequency
            # take monthly average
            df_monthly = df_ma[[f'{series_id}_4WMA']].resample('M').mean()
            result = df_monthly

        elif transform_type == 'monthly_avg':
            df_monthly = df.resample('M').mean()
            df_monthly.columns = [f'{series_id}_MAVG']
            result = df_monthly

        elif transform_type == 'monthly_avg_yoy':
            df_monthly = df.resample('M').mean()
            df_monthly[f'{series_id}_MAVG_YOY'] = df_monthly[series_id].pct_change(periods=12) * 100
            result = df_monthly[[f'{series_id}_MAVG_YOY']]

        elif transform_type == 'yoy_change':
            df[f'{series_id}_YOY_CHG'] = df[series_id].diff(periods=12)
            result = df[[f'{series_id}_YOY_CHG']]

        else:
            result = df

        return result

    def standardize_frequency_improved(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Frequency standardization with intelligent smoothing

        Key principles:
        1. All data is converted to monthly frequency through averaging
        2. High-frequency data is smoothed before aggregation
        3. Quarterly data is distributed correctly
        4. Minimize information loss
        """
        print("\nStandardizing data frequency...")

        # Determine time boundaries
        first_valid = df.first_valid_index()
        last_valid = df.last_valid_index()

        if first_valid is None or last_valid is None:
            return df

        # Create unified monthly index
        monthly_index = pd.date_range(
            start=first_valid.replace(day=1),
            end=last_valid,
            freq='M'
        )

        # Resulting DataFrame
        df_monthly = pd.DataFrame(index=monthly_index)

        # Process each column individually
        for col in df.columns:
            print(f"  Processing {col}...", end="")
            series = df[col].dropna()

            if len(series) == 0:
                print(" skipped (no data)")
                continue

            # Analyze data frequency
            freq_info = self._analyze_frequency(series)

            if freq_info['type'] == 'quarterly':
                # Special handling for quarterly data
                monthly_data = self._quarterly_to_monthly_proper(series)
                df_monthly[col] = monthly_data
                print(f" quarterly → monthly (distribution method)")

            elif freq_info['type'] == 'monthly':
                # Monthly data just align to index
                series_aligned = series.resample('M').last()
                df_monthly[col] = series_aligned
                print(f" monthly (alignment)")

            elif freq_info['type'] == 'weekly':
                # Weekly data - average by month
                # First apply light smoothing
                series_smooth = series.rolling(
                    window=2,  # 2-week window
                    min_periods=1,
                    center=True
                ).mean()
                series_monthly = series_smooth.resample('M').mean()
                df_monthly[col] = series_monthly
                print(f" weekly → monthly (average)")

            elif freq_info['type'] == 'daily':
                # Daily data - apply adaptive smoothing
                # Determine optimal smoothing window based on volatility
                volatility = series.pct_change().std()

                if volatility > 0.05:  # High volatility (e.g., VIX, currencies)
                    window = 10
                    print_suffix = "high volatility"
                elif volatility > 0.02:  # Medium volatility
                    window = 7
                    print_suffix = "medium volatility"
                else:  # Low volatility (e.g., interest rates)
                    window = 5
                    print_suffix = "low volatility"

                # Apply adaptive smoothing
                series_smooth = self._adaptive_smooth(series, window=window)
                series_monthly = series_smooth.resample('M').mean()
                df_monthly[col] = series_monthly
                print(f" daily → monthly ({print_suffix}, window={window})")

            else:
                # Undefined frequency - use safe approach
                series_monthly = series.resample('M').mean()
                df_monthly[col] = series_monthly
                print(f" undefined frequency → monthly (average)")

        # Intelligent missing value filling
        print("\nFilling missing values...")
        df_monthly = self._smart_fill_missing(df_monthly)

        print(f"\nStandardization complete:")
        print(f"  Final shape: {df_monthly.shape}")
        print(f"  Period: {df_monthly.index.min()} - {df_monthly.index.max()}")

        return df_monthly

    def _analyze_frequency(self, series: pd.Series) -> dict:
        """
        Analyzes time series frequency
        """
        if len(series) < 2:
            return {'type': 'unknown', 'days': None}

        # Try to determine frequency through pandas
        freq_guess = pd.infer_freq(series.index)

        if freq_guess:
            if 'Q' in freq_guess:
                return {'type': 'quarterly', 'freq': freq_guess}
            elif 'M' in freq_guess:
                return {'type': 'monthly', 'freq': freq_guess}
            elif 'W' in freq_guess:
                return {'type': 'weekly', 'freq': freq_guess}
            elif 'D' in freq_guess or 'B' in freq_guess:
                return {'type': 'daily', 'freq': freq_guess}

        # If pandas couldn't determine, analyze manually
        # Take median difference between observations
        time_diffs = series.index.to_series().diff().dt.days.dropna()
        median_days = time_diffs.median()

        if 85 <= median_days <= 95:
            return {'type': 'quarterly', 'days': median_days}
        elif 28 <= median_days <= 32:
            return {'type': 'monthly', 'days': median_days}
        elif 6 <= median_days <= 8:
            return {'type': 'weekly', 'days': median_days}
        elif median_days <= 5:
            return {'type': 'daily', 'days': median_days}
        else:
            return {'type': 'unknown', 'days': median_days}

    def _adaptive_smooth(self, series: pd.Series, window: int) -> pd.Series:
        """
        Adaptive smoothing with outlier handling
        """
        # Step 1: Identify and handle outliers
        # Use IQR (interquartile range) method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Create copy for processing
        series_clean = series.copy()

        # Replace outliers with boundaries (winsorization)
        series_clean = series_clean.clip(lower=lower_bound, upper=upper_bound)

        # Step 2: Apply moving average with different weights
        # Use exponentially weighted average for smoother results
        series_smooth = series_clean.ewm(
            span=window,
            min_periods=int(window * 0.6),  # Minimum 60% of window for valid value
            adjust=True  # Adjustment for series beginning
        ).mean()

        return series_smooth

    def _smart_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligent missing value filling
        """
        df_filled = df.copy()

        for col in df.columns:
            missing_count = df[col].isna().sum()

            if missing_count == 0:
                continue

            # Determine missing pattern
            total_count = len(df[col])
            missing_pct = missing_count / total_count

            if missing_pct > 0.5:
                # If more than half data is missing, don't fill
                print(f"  {col}: too many missing values ({missing_pct:.1%}), leaving as is")
                continue

            # Determine missing pattern
            is_start_missing = df[col].iloc[:12].isna().sum() > 6
            is_end_missing = df[col].iloc[-12:].isna().sum() > 6

            if is_start_missing and not is_end_missing:
                # Missing at start - use backward fill with limit
                df_filled[col] = df[col].fillna(method='bfill', limit=6)
                print(f"  {col}: filling missing values at start (bfill)")

            elif is_end_missing and not is_start_missing:
                # Missing at end - use forward fill with limit
                df_filled[col] = df[col].fillna(method='ffill', limit=6)
                print(f"  {col}: filling missing values at end (ffill)")

            else:
                # Missing in middle or distributed - use interpolation
                # First fill edge values
                df_filled[col] = df[col].fillna(method='ffill', limit=3)
                df_filled[col] = df_filled[col].fillna(method='bfill', limit=3)

                # Then interpolate remaining
                if df_filled[col].isna().sum() > 0:
                    df_filled[col] = df_filled[col].interpolate(
                        method='polynomial',
                        order=2,  # Quadratic interpolation
                        limit=6,  # Maximum 6 consecutive months
                        limit_direction='both'
                    )
                print(f"  {col}: interpolating internal missing values")

        return df_filled

    def _quarterly_to_monthly_proper(self, quarterly_series: pd.Series) -> pd.Series:
        """
        Proper conversion of quarterly data to monthly
        Solves First_Valid offset issue
        """
        # Create resulting series
        result = pd.Series(dtype=float)

        for date, value in quarterly_series.items():
            # Determine quarter and year
            year = date.year
            quarter = date.quarter

            # Determine months for quarter
            if quarter == 1:
                months = [1, 2, 3]
            elif quarter == 2:
                months = [4, 5, 6]
            elif quarter == 3:
                months = [7, 8, 9]
            else:
                months = [10, 11, 12]

            # Assign value to all months of quarter
            for month in months:
                # Use last day of month for consistency
                month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                result[month_end] = value

        return result

    def add_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creation of composite indicators with validation
        """
        print("\nCreating composite indicators...")

        # Standardize data before creating composites
        df_standardized = df.copy()

        # Z-score normalization for correct averaging
        for col in df_standardized.columns:
            if df_standardized[col].std() > 0:
                df_standardized[col] = (df_standardized[col] - df_standardized[col].mean()) / df_standardized[col].std()

        # Financial stress indicator
        stress_components = []
        if 'VIX' in df.columns:
            stress_components.append(df_standardized['VIX'])
        if 'High Yield Spread' in df.columns:
            stress_components.append(df_standardized['High Yield Spread'])
        if '10Y-2Y Spread' in df.columns:
            # Invert since negative spread = stress
            stress_components.append(-df_standardized['10Y-2Y Spread'])

        if stress_components:
            df['Financial_Stress_Index'] = pd.concat(stress_components, axis=1).mean(axis=1)
            print("  ✓ Created Financial_Stress_Index")

        # Composite growth indicator
        growth_cols = ['Real GDP', 'Industrial Production Index', 'Retail Sales Ex Food Services',
                      'Nonfarm Payrolls']
        available_growth = [col for col in growth_cols if col in df.columns]
        if len(available_growth) >= 2:
            df['Composite_Growth'] = df_standardized[available_growth].mean(axis=1)
            print(f"  ✓ Created Composite_Growth from {len(available_growth)} components")

        # Composite inflation indicator
        inflation_cols = ['Core CPI', 'CPI All Items', 'PPI All Commodities']
        available_inflation = [col for col in inflation_cols if col in df.columns]
        if len(available_inflation) >= 2:
            df['Composite_Inflation'] = df_standardized[available_inflation].mean(axis=1)
            print(f"  ✓ Created Composite_Inflation from {len(available_inflation)} components")

        # Economic regime indicator (simple version)
        if 'Composite_Growth' in df.columns and 'Composite_Inflation' in df.columns:
            df['Economic_Regime_Score'] = df['Composite_Growth'] - 0.5 * df['Composite_Inflation']
            print("  ✓ Created Economic_Regime_Score")

        return df

    def apply_publication_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply realistic publication lags to data
        """
        print("\nApplying publication lags...")

        df_lagged = df.copy()

        # Collect lag information
        lag_info = {}
        for category in self.indicators.values():
            for series_id, info in category.items():
                if info['name'] in df.columns:
                    lag_days = info.get('publication_lag', 0)
                    if lag_days > 0:
                        lag_info[info['name']] = lag_days

        # Apply lags
        for col, lag_days in lag_info.items():
            if col in df_lagged.columns:
                # Shift data by publication lag days
                df_lagged[col] = df_lagged[col].shift(periods=lag_days // 30)  # Approximately in months
                print(f"  Applied {lag_days} day lag to {col}")

        return df_lagged

    def create_data_quality_plots(self, df: pd.DataFrame):
        """
        Create visualizations for data quality assessment
        """
        print("\nCreating data quality visualizations...")

        # 1. Data availability heatmap
        plt.figure(figsize=(20, 12))

        # Create binary availability matrix
        availability = (~df.isna()).astype(int)

        # Aggregate by year for better visualization
        yearly_availability = availability.resample('Y').mean()

        sns.heatmap(yearly_availability.T, cmap='RdYlGn', cbar_kws={'label': 'Data Availability Rate'},
                   xticklabels=[str(d.year) for d in yearly_availability.index[::5]],
                   yticklabels=yearly_availability.columns)

        plt.title('Data Availability by Year', fontsize=16, pad=20)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Indicator', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('output/plots/data_availability_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Data coverage timeline
        plt.figure(figsize=(15, 8))

        coverage = availability.sum(axis=1) / len(df.columns) * 100
        coverage_smooth = coverage.rolling(window=12).mean()

        plt.plot(coverage.index, coverage, alpha=0.3, label='Monthly coverage')
        plt.plot(coverage_smooth.index, coverage_smooth, linewidth=2, label='12-month average')

        # Add important economic events
        events = {
            '1973-10': 'Oil Crisis',
            '1979-07': 'Second Oil Shock',
            '1987-10': 'Black Monday',
            '1990-07': 'Early 90s Recession',
            '2000-03': 'Dot-com Crash',
            '2008-09': 'Financial Crisis',
            '2020-03': 'COVID-19'
        }

        for date, event in events.items():
            event_date = pd.to_datetime(date)
            if event_date in coverage.index:
                plt.axvline(x=event_date, color='red', alpha=0.3, linestyle='--')
                plt.text(event_date, plt.ylim()[1]*0.95, event, rotation=90,
                        verticalalignment='top', fontsize=9)

        plt.title('Data Coverage Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Percentage of Available Indicators', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/plots/data_coverage_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Visualizations saved to output/plots/")

    def generate_quality_report(self, df: pd.DataFrame):
        """
        Generate detailed data quality report
        """
        report = []
        report.append("="*80)
        report.append("DATA QUALITY REPORT")
        report.append("="*80)
        report.append(f"\nData period: {df.index.min()} - {df.index.max()}")
        report.append(f"Total indicators: {len(df.columns)}")
        report.append(f"Total observations: {len(df)}")

        # Coverage statistics
        report.append("\n" + "-"*60)
        report.append("COVERAGE STATISTICS")
        report.append("-"*60)

        for col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                coverage_pct = (len(valid_data) / len(df)) * 100
                report.append(f"\n{col}:")
                report.append(f"  First value: {valid_data.index[0]}")
                report.append(f"  Last value: {valid_data.index[-1]}")
                report.append(f"  Coverage: {coverage_pct:.1f}%")
                report.append(f"  Missing values: {df[col].isna().sum()}")

        # Save report
        with open('output/data_quality_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print("\n  ✓ Data quality report saved to output/data_quality_report.txt")

    def collect_all_data(self) -> pd.DataFrame:
        """
        Main method to collect all data with improvements
        """
        all_data = {}
        failed_indicators = []

        # Process all indicator categories
        for category, indicators in self.indicators.items():
            print(f"\n{'='*60}")
            print(f"Loading category: {category}")
            print(f"{'='*60}")

            for series_id, info in indicators.items():
                df = self.fetch_single_indicator(series_id, info)
                if df is not None and len(df) > 0:
                    # Save with readable name
                    column_name = df.columns[0]
                    all_data[info['name']] = df[column_name]
                else:
                    failed_indicators.append((series_id, info['name']))

        if failed_indicators:
            print(f"\n⚠️  Failed to load {len(failed_indicators)} indicators:")
            for series_id, name in failed_indicators:
                print(f"   - {name} ({series_id})")

        # Combine all data into one DataFrame
        combined_df = pd.DataFrame(all_data)

        # Apply improved frequency standardization
        combined_df = self.standardize_frequency_improved(combined_df)

        print(f"\n✓ Data collection complete!")
        print(f"  Final dataframe shape: {combined_df.shape}")
        print(f"  Period: {combined_df.index.min()} - {combined_df.index.max()}")

        return combined_df

    def save_data(self, df: pd.DataFrame, df_lagged: Optional[pd.DataFrame] = None):
        """
        Save data with improved organization
        """
        # Save raw data
        df.to_csv('output/economic_indicators_raw.csv')
        print(f"\n✓ Raw data saved to: output/economic_indicators_raw.csv")

        # Save lagged data if provided
        if df_lagged is not None:
            df_lagged.to_csv('output/economic_indicators_lagged.csv')
            print(f"✓ Lagged data saved to: output/economic_indicators_lagged.csv")

        # Create detailed data description
        description = pd.DataFrame({
            'Indicator': df.columns,
            'First_Valid': df.apply(lambda x: x.first_valid_index()),
            'Last_Valid': df.apply(lambda x: x.last_valid_index()),
            'Count': df.apply(lambda x: x.count()),
            'Missing_Count': df.apply(lambda x: x.isna().sum()),
            'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2),
            'Mean': df.mean().round(2),
            'Std': df.std().round(2),
            'Min': df.min().round(2),
            'Max': df.max().round(2)
        })

        description.to_csv('output/data_description_detailed.csv')
        print(f"✓ Detailed description saved to: output/data_description_detailed.csv")

        # Save data quality report in JSON
        with open('output/data_quality_report.json', 'w') as f:
            json.dump(self.data_quality_report, f, indent=2, default=str)
        print(f"✓ Quality report in JSON: output/data_quality_report.json")


def collect_economic_data_improved(api_key: str,
                                 create_plots: bool = True,
                                 apply_lags: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function for collecting economic data

    Parameters:
    -----------
    api_key : str
        Your FRED API key
    create_plots : bool
        Whether to create data quality visualizations
    apply_lags : bool
        Whether to apply publication lags

    Returns:
    --------
    tuple : (df_raw, df_lagged)
        df_raw - raw data without lags
        df_lagged - data with publication lags applied
    """
    print("="*80)
    print("ECONOMIC DATA COLLECTION SYSTEM")
    print("="*80)

    # Create collector instance
    collector = ImprovedEconomicDataCollector(api_key=api_key)

    # Collect all base indicators
    df = collector.collect_all_data()

    # Add composite indicators
    df = collector.add_composite_indicators(df)

    # Create version with publication lags
    df_lagged = None
    if apply_lags:
        df_lagged = collector.apply_publication_lag(df)

    # Create visualizations
    if create_plots:
        collector.create_data_quality_plots(df)

    # Generate quality report
    collector.generate_quality_report(df)

    # Save results
    collector.save_data(df, df_lagged)

    # Output final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"\nTotal indicators collected: {len(df.columns)}")
    print(f"Data period: {df.index.min()} - {df.index.max()}")
    print(f"Total monthly observations: {len(df)}")

    # Coverage statistics by decade
    print("\nData coverage by decade:")
    for decade in range(1970, 2030, 10):
        decade_start = pd.Timestamp(f'{decade}-01-01')
        decade_end = pd.Timestamp(f'{decade+9}-12-31')
        decade_data = df[(df.index >= decade_start) & (df.index <= decade_end)]
        if len(decade_data) > 0:
            coverage = (decade_data.count() / len(decade_data)).mean() * 100
            print(f"  {decade}s: {coverage:.1f}%")

    return df, df_lagged


# Example usage
if __name__ == "__main__":
    # IMPORTANT: Replace with your real API key!
    API_KEY = '853c1faa729f41dc3f06e369d4bd66bd'

    # Run improved data collection
    df_raw, df_lagged = collect_economic_data_improved(
        api_key=API_KEY,
        create_plots=True,
        apply_lags=True
    )

    print("\n✓ Process complete! Check the output/ folder for results.")

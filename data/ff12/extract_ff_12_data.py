import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime
import os
import warnings

# Suppress FutureWarning from pandas_datareader
warnings.filterwarnings('ignore', category=FutureWarning)

# Create output directory
os.makedirs('output', exist_ok=True)

# Configuration
START_DATE = '1973-01-01'

# Industry name mapping
INDUSTRY_NAMES = {
    'NoDur': 'Consumer NonDurables',
    'Durbl': 'Consumer Durables',
    'Manuf': 'Manufacturing',
    'Enrgy': 'Energy',
    'Chems': 'Chemicals',
    'BusEq': 'Business Equipment',
    'Telcm': 'Telecommunications',
    'Utils': 'Utilities',
    'Shops': 'Retail',
    'Hlth': 'Healthcare',
    'Money': 'Finance',
    'Other': 'Other'
}


def download_ff12_data(start_date=START_DATE):
    """Download FF12 industry returns from Kenneth French Data Library"""
    print("Downloading FF12 data...")

    # Get FF12 data (value-weighted monthly returns)
    industries = pdr.get_data_famafrench('12_Industry_Portfolios', start=start_date)[0]

    # Convert from percentage to decimal
    industries = industries / 100

    print(f"✓ Downloaded {len(industries)} monthly observations")
    print(f"✓ Period: {industries.index[0]} to {industries.index[-1]}")

    return industries


def calculate_metrics(returns_df):
    """Calculate additional metrics for the industry portfolios"""

    # Calculate cumulative returns
    cumulative_returns = (1 + returns_df).cumprod()

    # Calculate rolling metrics (12-month)
    rolling_vol = returns_df.rolling(window=12).std() * np.sqrt(12)
    rolling_sharpe = (returns_df.rolling(window=12).mean() * 12) / (returns_df.rolling(window=12).std() * np.sqrt(12))

    # Combine metrics
    metrics_df = pd.concat({
        'returns': returns_df,
        'cumulative': cumulative_returns,
        'rolling_vol_12m': rolling_vol,
        'rolling_sharpe_12m': rolling_sharpe
    }, axis=1)

    return metrics_df


def create_summary_statistics(returns_df):
    """Create summary statistics for each industry"""

    summary = pd.DataFrame({
        'Industry': returns_df.columns,
        'Industry_Name': [INDUSTRY_NAMES[col] for col in returns_df.columns],
        'Start_Date': [returns_df[col].first_valid_index() for col in returns_df.columns],
        'End_Date': [returns_df[col].last_valid_index() for col in returns_df.columns],
        'Count': returns_df.count(),
        'Mean_Monthly': returns_df.mean(),
        'Mean_Annual': returns_df.mean() * 12,
        'Vol_Monthly': returns_df.std(),
        'Vol_Annual': returns_df.std() * np.sqrt(12),
        'Sharpe': (returns_df.mean() * 12) / (returns_df.std() * np.sqrt(12)),
        'Min_Monthly': returns_df.min(),
        'Max_Monthly': returns_df.max(),
        'Skewness': returns_df.skew(),
        'Kurtosis': returns_df.kurtosis(),
        'Total_Return': [(1 + returns_df[col]).prod() - 1 for col in returns_df.columns]
    }).round(4)

    return summary


def save_data(returns_df, metrics_df, summary_df):
    """Save all data in multiple formats"""

    # Save returns
    returns_df.to_csv('output/ff12_returns.csv')
    print(f"\n✓ Saved returns to: output/ff12_returns.csv")

    # Save metrics
    metrics_df.to_csv('output/ff12_metrics.csv')
    print(f"✓ Saved metrics to: output/ff12_metrics.csv")

    # Save summary
    summary_df.to_csv('output/ff12_summary_stats.csv', index=False)
    print(f"✓ Saved summary to: output/ff12_summary_stats.csv")

    # Save correlation matrix
    correlation_df = returns_df.corr()
    correlation_df.to_csv('output/ff12_correlations.csv')
    print(f"✓ Saved correlations to: output/ff12_correlations.csv")

    # Save to Excel
    with pd.ExcelWriter('output/ff12_data.xlsx') as writer:
        returns_df.to_excel(writer, sheet_name='Monthly_Returns')
        summary_df.to_excel(writer, sheet_name='Summary_Stats', index=False)
        correlation_df.to_excel(writer, sheet_name='Correlations')

        # Add metadata
        metadata = pd.DataFrame({
            'Item': ['Data Source', 'Start Date', 'End Date', 'Total Months',
                    'Industries Count', 'Download Date'],
            'Value': ['Kenneth French Data Library',
                     returns_df.index[0].strftime('%Y-%m-%d'),
                     returns_df.index[-1].strftime('%Y-%m-%d'),
                     len(returns_df),
                     len(returns_df.columns),
                     datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)

    print(f"✓ Saved Excel file to: output/ff12_data.xlsx")


def main():
    """Main execution function"""
    print("="*60)
    print("Kenneth French FF12 Data Extraction")
    print("="*60)

    try:
        # Download data
        returns_df = download_ff12_data()

        # Calculate metrics
        metrics_df = calculate_metrics(returns_df)

        # Create summary statistics
        summary_df = create_summary_statistics(returns_df)

        # Save all data
        save_data(returns_df, metrics_df, summary_df)

        # Display summary
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)

        print(f"\nData Summary:")
        print(f"- Period: {returns_df.index[0].strftime('%Y-%m-%d')} to {returns_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"- Total months: {len(returns_df)}")
        print(f"- Industries: {len(returns_df.columns)}")

        print("\nIndustry Performance (Annualized):")
        print(f"{'Industry':<15} {'Name':<25} {'Return':>10} {'Volatility':>12} {'Sharpe':>10}")
        print("-" * 75)

        for _, row in summary_df.iterrows():
            print(f"{row['Industry']:<15} {row['Industry_Name']:<25} "
                  f"{row['Mean_Annual']:>9.1%} {row['Vol_Annual']:>11.1%} "
                  f"{row['Sharpe']:>10.2f}")

        # Top/Bottom performers
        print("\nTop 3 Industries by Sharpe Ratio:")
        top_sharpe = summary_df.nlargest(3, 'Sharpe')[['Industry_Name', 'Sharpe', 'Mean_Annual', 'Vol_Annual']]
        for _, row in top_sharpe.iterrows():
            print(f"  {row['Industry_Name']:<25} Sharpe: {row['Sharpe']:.2f}, "
                  f"Return: {row['Mean_Annual']:.1%}, Vol: {row['Vol_Annual']:.1%}")

        print("\nHighest Risk Industries (by volatility):")
        high_vol = summary_df.nlargest(3, 'Vol_Annual')[['Industry_Name', 'Vol_Annual', 'Mean_Annual']]
        for _, row in high_vol.iterrows():
            print(f"  {row['Industry_Name']:<25} Vol: {row['Vol_Annual']:.1%}, Return: {row['Mean_Annual']:.1%}")

        # Correlation insights
        corr = returns_df.corr()
        print("\nLowest Industry Correlations (best for diversification):")
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append({
                    'Industry1': corr.columns[i],
                    'Industry2': corr.columns[j],
                    'Correlation': corr.iloc[i, j]
                })

        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation')
        for _, row in corr_df.head(3).iterrows():
            print(f"  {INDUSTRY_NAMES[row['Industry1']]} <-> {INDUSTRY_NAMES[row['Industry2']]}: "
                  f"{row['Correlation']:.3f}")

        print("\n✓ ALL DONE! Check the output/ directory for detailed files.")

        return returns_df, summary_df

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure pandas_datareader is installed:")
        print("pip install pandas-datareader")
        raise


if __name__ == "__main__":
    returns_df, summary_df = main()
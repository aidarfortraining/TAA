"""
Quick fix script for TAA date alignment issue
Run this to immediately fix the KeyError problem
"""

import pandas as pd
import os


def quick_fix():
    """Quick fix for date alignment issue"""

    print("QUICK FIX FOR TAA DATE ALIGNMENT")
    print("=" * 60)

    # Check if files exist
    files_to_check = [
        'input/taa_prepared_data.csv',
        'input/regime_history.csv'
    ]

    for file in files_to_check:
        if not os.path.exists(file):
            print(f"ERROR: Missing file: {file}")
            return False

    print("\n1. Loading data files...")

    # Load data
    returns_df = pd.read_csv('input/taa_prepared_data.csv', index_col=0, parse_dates=True)
    regime_history = pd.read_csv('input/regime_history.csv',
                                 index_col=0, parse_dates=True)

    print(f"Returns: {len(returns_df)} rows, from {returns_df.index[0]} to {returns_df.index[-1]}")
    print(f"Regimes: {len(regime_history)} rows, from {regime_history.index[0]} to {regime_history.index[-1]}")

    # Fix dates
    print("\n2. Standardizing dates to month-end...")

    # Convert to month-end
    returns_df.index = returns_df.index.to_period('M').to_timestamp('M')
    regime_history.index = regime_history.index.to_period('M').to_timestamp('M')

    # Find overlap
    common_dates = returns_df.index.intersection(regime_history.index)
    print(f"\n3. Found {len(common_dates)} overlapping months")

    if len(common_dates) == 0:
        print("ERROR: No overlapping dates!")
        return False

    print(f"Common period: {common_dates[0]} to {common_dates[-1]}")

    # Create aligned versions
    print("\n4. Creating aligned datasets...")

    aligned_returns = returns_df.loc[common_dates].copy()
    aligned_regimes = regime_history.loc[common_dates].copy()

    # Save
    aligned_returns.to_csv('input/taa_returns_aligned.csv')
    aligned_regimes.to_csv('input/regime_history_aligned.csv')

    print("\n5. Saved aligned files:")
    print("   - taa_returns_aligned.csv")
    print("   - regime_history_aligned.csv")

    # Create a modified main script
    print("\n6. Creating modified TAA script...")

    modified_script = '''"""
TAA Model with Fixed Date Alignment
"""

import sys
sys.path.append('.')

# Import the original model
from taa_portfolio_model import *

# Override the main function to use aligned data
def main_fixed():
    """Main execution with aligned data"""

    # Configuration
    REGIME_MODEL_PATH = 'output/regime_classification/regime_classifier.pkl'
    DATA_PATH = 'output/taa_prepared_data.csv'
    OUTPUT_DIR = 'output/taa_model'

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Force use of aligned data
    logger.info("Using aligned data files for compatibility")

    # Initialize TAA model with aligned data
    logger.info("Initializing TAA model...")
    taa = TacticalAssetAllocation(
        regime_model_path=REGIME_MODEL_PATH,
        data_path=DATA_PATH,
        use_aligned_data=True  # Force aligned data
    )

    # Train historical model
    taa.train_historical_model()

    # Train modern model
    taa.train_modern_model()

    # Backtest strategy
    backtest_results = taa.backtest_strategy(start_date='2001-01-01')

    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'taa_model.pkl')
    taa.save_model(model_path)

    # Create visualizations
    taa.plot_regime_allocations(os.path.join(OUTPUT_DIR, 'regime_allocations.png'))

    # Plot backtest results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Portfolio value
    ax1.plot(backtest_results['results'].index, 
             backtest_results['results']['portfolio_value'] / 100000,
             label='TAA Strategy', linewidth=2)
    ax1.set_ylabel('Portfolio Value (Normalized)')
    ax1.set_title('TAA Strategy Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add regime changes
    for change_date in backtest_results['regime_changes']:
        if change_date in backtest_results['results'].index:
            ax1.axvline(x=change_date, color='red', alpha=0.3, linestyle='--')

    # Drawdown
    cumulative = (1 + backtest_results['results']['returns']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100

    ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('Strategy Drawdown')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'backtest_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save backtest metrics
    with open(os.path.join(OUTPUT_DIR, 'backtest_metrics.txt'), 'w') as f:
        f.write("TAA STRATEGY BACKTEST RESULTS\\n")
        f.write("="*40 + "\\n\\n")
        for metric, value in backtest_results['metrics'].items():
            f.write(f"{metric.replace('_', ' ').title()}: {value:.2f}\\n")

    logger.info("\\nTAA model training complete!")
    logger.info(f"Results saved to: {OUTPUT_DIR}/")

    return taa, backtest_results


if __name__ == "__main__":
    taa_model, results = main_fixed()
'''

    with open('taa_model_fixed.py', 'w') as f:
        f.write(modified_script)

    print("\n7. Created taa_model_fixed.py")

    print("\n" + "=" * 60)
    print("QUICK FIX COMPLETE!")
    print("=" * 60)
    print("\nNow run:")
    print("  python taa_model_fixed.py")
    print("\nThis will use the aligned data automatically.")

    return True


if __name__ == "__main__":
    success = quick_fix()

    if not success:
        print("\nQuick fix failed. Please check the error messages above.")
        print("Make sure all required files exist in the output directory.")
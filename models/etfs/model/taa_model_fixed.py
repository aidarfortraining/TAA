"""
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
        f.write("TAA STRATEGY BACKTEST RESULTS\n")
        f.write("="*40 + "\n\n")
        for metric, value in backtest_results['metrics'].items():
            f.write(f"{metric.replace('_', ' ').title()}: {value:.2f}\n")

    logger.info("\nTAA model training complete!")
    logger.info(f"Results saved to: {OUTPUT_DIR}/")

    return taa, backtest_results


if __name__ == "__main__":
    taa_model, results = main_fixed()

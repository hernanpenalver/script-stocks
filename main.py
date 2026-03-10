import os
import sys
import webbrowser

import config
from data_loader import download_prices
from backtester import run_all
from report import generate_report
from strategies import (
    BuyAndHold,
    SMACrossover,
    RSIStrategy,
    RSIWithStopLoss,
    RSIPercentile,
    MACDStrategy,
    MACDWithStopLoss,
    MACDOptimized,
    BollingerBands,
    RSIBollinger,
)


def build_strategies():
    return [
        BuyAndHold(),
        SMACrossover(fast=20, slow=50),
        SMACrossover(fast=50, slow=200),
        RSIStrategy(period=14, oversold=30, overbought=70),
        RSIWithStopLoss(period=14, oversold=30, overbought=70),
        RSIPercentile(period=14, low_pct=0.10, high_pct=0.90),
        MACDStrategy(fast=12, slow=26, signal=9),
        MACDWithStopLoss(fast=12, slow=26, signal=9, sl_threshold=0.005),
        MACDOptimized(train_ratio=0.70),
        BollingerBands(period=20, num_std=2.0),
        RSIBollinger(bb_period=20, bb_std=2.0, rsi_period=14, oversold=30, overbought=70),
    ]


def main():
    print("=" * 60)
    print("  Stock Backtesting Framework")
    print("=" * 60)

    # 1. Download data
    all_symbols = config.SYMBOLS_USA + config.SYMBOLS_ARG
    prices = download_prices(all_symbols, config.START_DATE, config.END_DATE)

    if not prices:
        print("ERROR: No price data was downloaded. Check your internet connection.")
        sys.exit(1)

    # 2. Download S&P 500 benchmark
    print("Downloading S&P 500 benchmark (^GSPC) ...")
    from backtester import run_backtest
    sp500_prices = download_prices(["^GSPC"], config.START_DATE, config.END_DATE)
    sp500_series = sp500_prices.get("^GSPC")
    sp500_benchmark = None
    if sp500_series is not None:
        sp500_benchmark = run_backtest(sp500_series, BuyAndHold(), config.INITIAL_CAPITAL)
        print(f"  S&P 500 CAGR: {sp500_benchmark['metrics']['CAGR']}%  "
              f"Total Return: {sp500_benchmark['metrics']['Total Return']}%")
    else:
        print("  [WARN] Could not download S&P 500 data.")

    # 3. Build strategies
    strategies = build_strategies()
    print(f"\nStrategies: {[s.name for s in strategies]}")
    print(f"Symbols:    {list(prices.keys())}")
    print(f"Capital:    ${config.INITIAL_CAPITAL:,.0f}\n")

    # 4. Run backtests
    print("Running backtests ...")
    results = run_all(prices, strategies, config.INITIAL_CAPITAL)

    # 5. Generate report
    output_path = os.path.join(os.path.dirname(__file__), "reporte.html")
    abs_path = generate_report(results, output_path, sp500_benchmark=sp500_benchmark)

    # 5. Open in browser
    print(f"\nOpening report in browser ...")
    webbrowser.open(f"file://{abs_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

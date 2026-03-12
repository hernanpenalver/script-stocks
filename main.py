import os
import sys
import webbrowser

import config
from data_loader import download_prices, download_ohlcv
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
    Momentum,
    DonchianBreakout,
    ADXTrend,
    MeanReversion,
    PairsTrading,
    VolumeProfile,
)

# Pairs to trade: (asset_A, asset_B) — strategy runs on A, using B as reference
PAIRS = [
    ("AAPL", "MSFT"),    # Tech giants, highly correlated
    ("GOOGL", "META"),   # Digital advertising peers
    ("JPM", "BMA"),      # Banks: US vs Argentina
    ("GGAL", "BMA"),     # Argentine banks
    ("YPF", "XOM"),      # Energy: Argentina vs US
]


def build_strategies(prices: dict = None, ohlcv: dict = None):
    base = [
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
        BollingerBands(period=20, num_std=2.5, exit_at_mid=True),
        BollingerBands(period=20, num_std=3.0, exit_at_mid=True),
        RSIBollinger(bb_period=20, bb_std=2.0, rsi_period=14, oversold=30, overbought=70),
        Momentum(lookback=252, skip=21),
        DonchianBreakout(entry_period=20, exit_period=10),
        DonchianBreakout(entry_period=55, exit_period=20),
        ADXTrend(period=14, threshold=25),
        MeanReversion(period=20, entry_std=1.5),
        MeanReversion(period=50, entry_std=2.0),
        MeanReversion(period=20, entry_std=2.5, exit_std=0.5),
        MeanReversion(period=50, entry_std=3.0, exit_std=0.5),
    ]

    # Build pairs trading strategies (only for symbols that have a pair)
    pairs_strategies = {}
    if prices:
        for sym_a, sym_b in PAIRS:
            if sym_a in prices and sym_b in prices:
                pairs_strategies[sym_a] = PairsTrading(
                    pair_prices=prices[sym_b], pair_name=sym_b,
                )
                pairs_strategies[sym_b] = PairsTrading(
                    pair_prices=prices[sym_a], pair_name=sym_a,
                )

    # Build volume profile strategies (per symbol, needs volume data)
    vol_strategies = {}
    if ohlcv:
        for sym, df in ohlcv.items():
            vol_strategies[sym] = [
                VolumeProfile(volume=df["Volume"], lookback=60, exit_at="poc"),
                VolumeProfile(volume=df["Volume"], lookback=120, exit_at="vah"),
            ]

    return base, pairs_strategies, vol_strategies


def main():
    print("=" * 60)
    print("  Stock Backtesting Framework")
    print("=" * 60)

    # 1. Download data
    all_symbols = config.SYMBOLS_USA + config.SYMBOLS_ARG + config.SYMBOLS_ETF
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

    # 2b. Download OHLCV data (for volume profile strategies)
    ohlcv = download_ohlcv(all_symbols, config.START_DATE, config.END_DATE)

    # 3. Build strategies
    base_strategies, pairs_strategies, vol_strategies = build_strategies(prices, ohlcv)
    print(f"\nStrategies: {[s.name for s in base_strategies]}")
    if pairs_strategies:
        print(f"Pairs:      {[(s, ps.name) for s, ps in pairs_strategies.items()]}")
    if vol_strategies:
        print(f"VolProfile: {len(vol_strategies)} symbols with volume strategies")
    print(f"Symbols:    {list(prices.keys())}")
    print(f"Capital:    ${config.INITIAL_CAPITAL:,.0f}\n")

    # 4. Run backtests
    print("Running backtests ...")
    results = run_all(prices, base_strategies, config.INITIAL_CAPITAL)

    # 4b. Run pairs trading backtests
    if pairs_strategies:
        from backtester import run_backtest
        print("\nRunning pairs trading backtests ...")
        for symbol, pair_strat in pairs_strategies.items():
            if symbol in prices:
                result = run_backtest(prices[symbol], pair_strat, config.INITIAL_CAPITAL)
                results[symbol][pair_strat.name] = result
                print(f"  {symbol} | {pair_strat.name} | "
                      f"Return={result['metrics']['Total Return']}% "
                      f"Sharpe={result['metrics']['Sharpe']}")

    # 4c. Run volume profile backtests
    if vol_strategies:
        from backtester import run_backtest
        print("\nRunning volume profile backtests ...")
        for symbol, vol_strats in vol_strategies.items():
            if symbol in prices:
                for vs in vol_strats:
                    result = run_backtest(prices[symbol], vs, config.INITIAL_CAPITAL)
                    results[symbol][vs.name] = result
                    print(f"  {symbol} | {vs.name} | "
                          f"Return={result['metrics']['Total Return']}% "
                          f"Sharpe={result['metrics']['Sharpe']}")

    # 5. Generate report
    output_path = os.path.join(os.path.dirname(__file__), "reporte.html")
    abs_path = generate_report(results, output_path, sp500_benchmark=sp500_benchmark)

    # 5. Open in browser
    print(f"\nOpening report in browser ...")
    webbrowser.open(f"file://{abs_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

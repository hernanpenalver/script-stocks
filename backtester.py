import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Return the running drawdown (negative values) relative to the rolling peak."""
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def run_backtest(
    prices: pd.Series,
    strategy: BaseStrategy,
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Run a backtest for a single strategy on a single price series.

    Shift(1) is applied to signals to avoid look-ahead bias.

    Returns a dict with:
        equity      : pd.Series  — portfolio value over time
        metrics     : dict       — computed performance metrics
    """
    raw_signals = strategy.generate_signals(prices)
    # Shift signals by 1 day to avoid look-ahead bias
    signals = raw_signals.shift(1).fillna(0)

    daily_returns = prices.pct_change().fillna(0)
    strategy_returns = signals * daily_returns

    equity = initial_capital * (1 + strategy_returns).cumprod()
    equity.iloc[0] = initial_capital  # anchor starting capital

    metrics = _compute_metrics(equity, strategy_returns, signals, prices)
    yearly = _compute_yearly_returns(equity)
    result = {"equity": equity, "signals": signals, "metrics": metrics, "yearly": yearly}
    if hasattr(strategy, "_last_params") and strategy._last_params is not None:
        result["best_params"] = strategy._last_params
    return result


def _compute_yearly_returns(equity: pd.Series) -> pd.Series:
    """Annual return for each calendar year present in the equity curve."""
    yearly = equity.resample("YE").last()
    yearly_start = equity.resample("YE").first()
    returns = (yearly / yearly_start) - 1
    returns.index = returns.index.year
    return returns


def _compute_metrics(
    equity: pd.Series,
    strategy_returns: pd.Series,
    signals: pd.Series,
    prices: pd.Series,
) -> dict:
    total_days = len(equity)
    years = total_days / 252.0

    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    # CAGR
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    # Sharpe ratio (annualised, rf = 0)
    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Sortino ratio (downside deviation)
    downside = strategy_returns[strategy_returns < 0]
    downside_std = downside.std()
    sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    # Max drawdown
    dd = _drawdown_series(equity)
    max_drawdown = dd.min()  # most negative value

    # Calmar ratio
    calmar = (cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # Trade statistics
    # Reconstruct individual trade P&L using the position signal (not returns,
    # which can be 0 even while in a trade if the daily price didn't change).
    entry_idx = None
    trade_returns_list = []
    for i, (ret, sig) in enumerate(
        zip(strategy_returns, signals)
    ):
        if sig == 1 and entry_idx is None:
            entry_idx = i
        elif sig == 0 and entry_idx is not None:
            # Trade closed: compute compound return over the period
            trade_slice = strategy_returns.iloc[entry_idx:i]
            trade_ret = (1 + trade_slice).prod() - 1
            trade_returns_list.append(trade_ret)
            entry_idx = None
    # Close any open trade at end
    if entry_idx is not None:
        trade_slice = strategy_returns.iloc[entry_idx:]
        trade_ret = (1 + trade_slice).prod() - 1
        trade_returns_list.append(trade_ret)

    num_trades = len(trade_returns_list)
    win_rate = (
        sum(1 for r in trade_returns_list if r > 0) / num_trades
        if num_trades > 0
        else 0.0
    )

    return {
        "Total Return": round(total_return * 100, 2),
        "CAGR": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Calmar": round(calmar, 3),
        "Win Rate": round(win_rate * 100, 2),
        "Num Trades": num_trades,
    }


def run_all(
    prices_dict: dict[str, pd.Series],
    strategies: list[BaseStrategy],
    initial_capital: float = 10_000.0,
) -> dict[str, dict[str, dict]]:
    """
    Run all strategies over all symbols.

    Returns nested dict: results[symbol][strategy_name] = {equity, signals, metrics}
    """
    results: dict[str, dict[str, dict]] = {}
    total = len(prices_dict) * len(strategies)
    done = 0
    for symbol, prices in prices_dict.items():
        results[symbol] = {}
        for strat in strategies:
            result = run_backtest(prices, strat, initial_capital)
            results[symbol][strat.name] = result
            done += 1
            print(f"  [{done}/{total}] {symbol} | {strat.name} | "
                  f"Return={result['metrics']['Total Return']}% "
                  f"Sharpe={result['metrics']['Sharpe']}")
    return results

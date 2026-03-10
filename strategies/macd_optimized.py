import numpy as np
import pandas as pd
from itertools import product
from .base import BaseStrategy

# Grid of parameter combinations to search
_FAST_VALUES   = [8, 12, 16, 20]
_SLOW_VALUES   = [21, 26, 30, 35]
_SIGNAL_VALUES = [7, 9, 12]


def _macd_signals(prices: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    ema_fast    = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow    = prices.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line > signal_line).astype(int).fillna(0)


def _sharpe(prices: pd.Series, fast: int, slow: int, signal: int) -> float:
    signals        = _macd_signals(prices, fast, slow, signal).shift(1).fillna(0)
    daily_returns  = prices.pct_change().fillna(0)
    strat_returns  = signals * daily_returns
    std = strat_returns.std()
    return (strat_returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0


class MACDOptimized(BaseStrategy):
    """
    MACD with per-symbol parameter optimisation.

    Methodology (avoids look-ahead bias):
      1. Use the first `train_ratio` fraction of the price series as a
         training window.
      2. Grid-search all combinations of (fast, slow, signal) and select
         the set that maximises the annualised Sharpe ratio on the
         training window.
      3. Generate signals for the *full* price series using the best params.

    Grid searched:
        fast   : 8, 12, 16, 20
        slow   : 21, 26, 30, 35
        signal : 7, 9, 12
    """

    def __init__(self, train_ratio: float = 0.70):
        self.train_ratio = train_ratio

    @property
    def name(self) -> str:
        return "MACD-Opt"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        train_end    = max(int(len(prices) * self.train_ratio), 100)
        train_prices = prices.iloc[:train_end]

        best_sharpe = -np.inf
        best_params = (12, 26, 9)  # safe fallback

        for fast, slow, signal in product(_FAST_VALUES, _SLOW_VALUES, _SIGNAL_VALUES):
            if fast >= slow:
                continue
            sh = _sharpe(train_prices, fast, slow, signal)
            if sh > best_sharpe:
                best_sharpe = sh
                best_params = (fast, slow, signal)

        fast, slow, signal = best_params
        self._last_params = best_params
        print(f"    MACD-Opt best params: fast={fast}, slow={slow}, signal={signal} "
              f"(train Sharpe={best_sharpe:.3f})")

        return _macd_signals(prices, fast, slow, signal)

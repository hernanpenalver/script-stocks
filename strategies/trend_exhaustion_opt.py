import numpy as np
import pandas as pd
from .base import BaseStrategy
from .trend_exhaustion import TrendExhaustion

# MA periods to search for the optimal exit
_EXIT_MA_PERIODS = [5, 8, 10, 13, 15, 20, 30, 40, 50, 65, 80, 100, 150, 200]
# MA types to try
_EXIT_MA_TYPES = ["ema", "sma"]


def _generate_signals_with_ma_exit(
    trend_dir: pd.Series,
    score: pd.Series,
    prices: pd.Series,
    exit_ma: pd.Series,
    entry_score: int,
) -> pd.Series:
    """
    Same entry logic as TrendExhaustion but exit when price crosses
    below the moving average (instead of using the exhaustion score).

    Entry:
      - Bullish trend (trend_dir == 1) regardless of score
      - Bearish trend with exhaustion score >= entry_score (reversal bet)

    Exit:
      - Price closes below the exit moving average
    """
    position = pd.Series(0, index=prices.index, dtype=int)
    in_trade = False

    for i in range(len(prices)):
        td = trend_dir.iloc[i]
        sc = score.iloc[i]
        p = prices.iloc[i]
        ma = exit_ma.iloc[i]

        if np.isnan(sc) or np.isnan(ma):
            position.iloc[i] = 0
            continue

        if not in_trade:
            # Enter: bullish trend OR bearish exhaustion reversal
            if td == 1:
                in_trade = True
            elif td == -1 and sc >= entry_score:
                in_trade = True
        else:
            # Exit: price crosses below the moving average
            if p < ma:
                in_trade = False

        position.iloc[i] = 1 if in_trade else 0

    return position


def _max_drawdown(prices: pd.Series, signals: pd.Series, initial: float = 10_000.0) -> float:
    """Compute max drawdown for a given signal series (more negative = worse)."""
    shifted = signals.shift(1).fillna(0)
    daily_ret = prices.pct_change().fillna(0)
    equity = initial * (1 + shifted * daily_ret).cumprod()
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    return dd.min()


class TrendExhaustionOpt(BaseStrategy):
    """
    Trend Exhaustion with optimised MA exit to minimise drawdown.

    Same entry logic as TrendExhaustion (bullish trend follow + bearish
    reversal on exhaustion), but exit is triggered when price crosses
    below a moving average instead of using the exhaustion score.

    The MA period and type (SMA/EMA) are optimised on a training window
    (first `train_ratio` of data) by selecting the combination that
    minimises maximum drawdown.

    Grid searched:
        MA period : 5, 8, 10, 13, 15, 20, 30, 40, 50, 65, 80, 100, 150, 200
        MA type   : EMA, SMA

    Parameters
    ----------
    ohlcv : pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume.
    entry_score : int
        Min exhaustion score to enter on bearish reversal (default 3).
    train_ratio : float
        Fraction of data used for optimisation (default 0.70).
    """

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        entry_score: int = 3,
        train_ratio: float = 0.70,
    ):
        self.ohlcv = ohlcv
        self.entry_score = entry_score
        self.train_ratio = train_ratio
        self._last_params = None

    @property
    def name(self) -> str:
        return f"TrendExh-Opt(entry>={self.entry_score})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # Build a base TrendExhaustion instance to reuse its indicator logic
        base = TrendExhaustion(ohlcv=self.ohlcv, entry_score=self.entry_score)

        df = self.ohlcv.reindex(prices.index).dropna()
        idx = prices.index
        close = df["Close"].reindex(idx)
        high = df["High"].reindex(idx)
        low = df["Low"].reindex(idx)
        volume = df["Volume"].reindex(idx).fillna(0)

        # Compute trend and scores once (expensive)
        trend_dir = base._compute_trend(close, high, low)
        score = base._compute_scores(close, high, low, volume, trend_dir)

        # Train/test split
        train_end = max(int(len(prices) * self.train_ratio), 100)
        train_prices = prices.iloc[:train_end]
        train_trend = trend_dir.iloc[:train_end]
        train_score = score.iloc[:train_end]

        # Grid search: find MA period + type that minimises max drawdown
        best_dd = -np.inf  # least negative = best
        best_period = 20
        best_type = "ema"

        for ma_type in _EXIT_MA_TYPES:
            for period in _EXIT_MA_PERIODS:
                if period >= train_end * 0.5:
                    continue  # skip if MA period is too large for training data

                if ma_type == "ema":
                    exit_ma = train_prices.ewm(span=period, adjust=False).mean()
                else:
                    exit_ma = train_prices.rolling(period).mean()

                sigs = _generate_signals_with_ma_exit(
                    train_trend, train_score, train_prices, exit_ma,
                    self.entry_score,
                )
                dd = _max_drawdown(train_prices, sigs)

                if dd > best_dd:
                    best_dd = dd
                    best_period = period
                    best_type = ma_type

        self._last_params = (best_type, best_period, best_dd)
        print(f"    TrendExh-Opt best exit: {best_type.upper()}({best_period}) "
              f"(train MaxDD={best_dd * 100:.2f}%)")

        # Generate signals for full series with the optimal MA
        if best_type == "ema":
            full_exit_ma = prices.ewm(span=best_period, adjust=False).mean()
        else:
            full_exit_ma = prices.rolling(best_period).mean()

        return _generate_signals_with_ma_exit(
            trend_dir, score, prices, full_exit_ma,
            self.entry_score,
        )

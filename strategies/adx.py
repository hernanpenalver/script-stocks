import pandas as pd
import numpy as np
from .base import BaseStrategy


def _compute_adx(prices: pd.Series, period: int = 14):
    """
    Compute ADX, +DI and -DI from a price series.

    Since we only have close prices (no high/low), we approximate:
    - True Range ≈ abs(close - prev_close)
    - +DM = max(close - prev_close, 0)
    - -DM = max(prev_close - close, 0)
    """
    diff = prices.diff()
    plus_dm = diff.clip(lower=0)
    minus_dm = (-diff).clip(lower=0)
    tr = diff.abs()

    # Wilder's smoothing (exponential with alpha = 1/period)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * smooth_plus / atr
    minus_di = 100 * smooth_minus / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx, plus_di, minus_di


class ADXTrend(BaseStrategy):
    """
    ADX Trend-following strategy.

    Go long when ADX > threshold (strong trend) AND +DI > -DI (uptrend).
    Exit when ADX drops below threshold or -DI crosses above +DI.

    Parameters
    ----------
    period : int
        ADX calculation period (default 14).
    threshold : float
        Minimum ADX value to consider a trend strong (default 25).
    """

    def __init__(self, period: int = 14, threshold: float = 25.0):
        self.period = period
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"ADX({self.period},{int(self.threshold)})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        adx, plus_di, minus_di = _compute_adx(prices, self.period)

        signal = ((adx > self.threshold) & (plus_di > minus_di)).astype(int)
        return signal.fillna(0)

import pandas as pd
from .base import BaseStrategy


class Momentum(BaseStrategy):
    """
    Time-series momentum (12-1).

    Go long when the asset's return over the past `lookback` months
    (excluding the most recent `skip` month) is positive.

    Based on Jegadeesh & Titman (1993) and Moskowitz, Ooi & Pedersen (2012).
    The 1-month skip avoids the well-documented short-term reversal effect.

    Parameters
    ----------
    lookback : int
        Total lookback window in trading days (default 252 ≈ 12 months).
    skip : int
        Recent days to skip (default 21 ≈ 1 month).
    """

    def __init__(self, lookback: int = 252, skip: int = 21):
        self.lookback = lookback
        self.skip = skip

    @property
    def name(self) -> str:
        months_look = self.lookback // 21
        months_skip = self.skip // 21
        return f"Momentum {months_look}-{months_skip}"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # Return from (t - lookback) to (t - skip)
        past_price = prices.shift(self.lookback)
        recent_price = prices.shift(self.skip)
        momentum_return = (recent_price - past_price) / past_price

        signal = (momentum_return > 0).astype(int)
        return signal.fillna(0)

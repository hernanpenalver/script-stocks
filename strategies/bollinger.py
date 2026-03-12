import pandas as pd
from .base import BaseStrategy


class BollingerBands(BaseStrategy):
    """
    Mean-reversion via Bollinger Bands.

    Enter long when price falls below the lower band (oversold),
    exit when price rises above the upper band (overbought).

    Parameters
    ----------
    period : int
        Lookback window for the moving average and std dev (default 20).
    num_std : float
        Number of std devs for entry band (default 2.0).
    exit_at_mid : bool
        If True, exit when price reverts to the moving average instead of
        waiting for the upper band. Increases win rate at the cost of
        shorter trades (default False).
    """

    def __init__(self, period: int = 20, num_std: float = 2.0,
                 exit_at_mid: bool = False):
        self.period = period
        self.num_std = num_std
        self.exit_at_mid = exit_at_mid

    @property
    def name(self) -> str:
        tag = "mid" if self.exit_at_mid else "band"
        return f"Bollinger({self.period},{self.num_std}std,{tag})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        rolling = prices.rolling(self.period)
        mid = rolling.mean()
        std = rolling.std()
        lower = mid - self.num_std * std
        upper = mid + self.num_std * std

        position = pd.Series(0, index=prices.index, dtype=int)
        in_trade = False
        for i in range(len(prices)):
            p = prices.iloc[i]
            lo = lower.iloc[i]
            if pd.isna(lo):
                position.iloc[i] = 0
                continue
            exit_level = mid.iloc[i] if self.exit_at_mid else upper.iloc[i]
            if not in_trade and p < lo:
                in_trade = True
            elif in_trade and p > exit_level:
                in_trade = False
            position.iloc[i] = 1 if in_trade else 0
        return position

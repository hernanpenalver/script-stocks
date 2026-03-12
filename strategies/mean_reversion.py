import pandas as pd
from .base import BaseStrategy


class MeanReversion(BaseStrategy):
    """
    Mean Reversion strategy.

    Enter long when price falls more than `entry_std` standard deviations
    below its moving average (oversold / stretched too far down).
    Exit when price reverts back above the moving average (offset by
    `exit_std` standard deviations below the mean).

    Parameters
    ----------
    period : int
        Lookback window for the moving average and std dev (default 20).
    entry_std : float
        Number of std devs below the mean to trigger entry (default 1.5).
    exit_std : float
        Number of std devs below the mean for exit level (default 0.0,
        i.e. exit at the mean). Use a small positive value like 0.5 to
        exit before reaching the mean (locks profit earlier, higher win rate).
    """

    def __init__(self, period: int = 20, entry_std: float = 1.5,
                 exit_std: float = 0.0):
        self.period = period
        self.entry_std = entry_std
        self.exit_std = exit_std

    @property
    def name(self) -> str:
        if self.exit_std > 0:
            return f"MeanRev({self.period},{self.entry_std},exit{self.exit_std})"
        return f"MeanRev({self.period},{self.entry_std})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        ma = prices.rolling(self.period).mean()
        std = prices.rolling(self.period).std()
        lower = ma - self.entry_std * std
        exit_level = ma - self.exit_std * std

        position = pd.Series(0, index=prices.index, dtype=int)
        in_trade = False
        for i in range(len(prices)):
            p = prices.iloc[i]
            lo = lower.iloc[i]
            ex = exit_level.iloc[i]
            if pd.isna(lo):
                continue
            if not in_trade and p < lo:
                in_trade = True
            elif in_trade and p > ex:
                in_trade = False
            position.iloc[i] = 1 if in_trade else 0
        return position

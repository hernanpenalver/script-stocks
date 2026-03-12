import pandas as pd
from .base import BaseStrategy


class DonchianBreakout(BaseStrategy):
    """
    Donchian Channel Breakout (Turtle Traders system).

    Enter long when price breaks above the highest high of the last
    `entry_period` days. Exit when price breaks below the lowest low
    of the last `exit_period` days.

    The original Turtle system used entry=20, exit=10 (System 1)
    or entry=55, exit=20 (System 2).

    Parameters
    ----------
    entry_period : int
        Lookback days for the entry channel (upper band).
    exit_period : int
        Lookback days for the exit channel (lower band).
    """

    def __init__(self, entry_period: int = 20, exit_period: int = 10):
        self.entry_period = entry_period
        self.exit_period = exit_period

    @property
    def name(self) -> str:
        return f"Donchian({self.entry_period}/{self.exit_period})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # Upper channel: highest high over entry_period (shifted to avoid look-ahead)
        upper = prices.rolling(self.entry_period).max().shift(1)
        # Lower channel: lowest low over exit_period (shifted to avoid look-ahead)
        lower = prices.rolling(self.exit_period).min().shift(1)

        position = pd.Series(0, index=prices.index, dtype=int)
        in_trade = False
        for i in range(len(prices)):
            p = prices.iloc[i]
            hi = upper.iloc[i]
            lo = lower.iloc[i]
            if pd.isna(hi) or pd.isna(lo):
                continue
            if not in_trade and p > hi:
                in_trade = True
            elif in_trade and p < lo:
                in_trade = False
            position.iloc[i] = 1 if in_trade else 0
        return position

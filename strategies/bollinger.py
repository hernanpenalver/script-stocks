import pandas as pd
from .base import BaseStrategy


class BollingerBands(BaseStrategy):
    """
    Mean-reversion via Bollinger Bands.

    Enter long when price falls below the lower band (oversold),
    exit when price rises above the upper band (overbought).
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std

    @property
    def name(self) -> str:
        return f"Bollinger({self.period},{self.num_std}std)"

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
            up = upper.iloc[i]
            if pd.isna(lo):
                position.iloc[i] = 0
                continue
            if not in_trade and p < lo:
                in_trade = True
            elif in_trade and p > up:
                in_trade = False
            position.iloc[i] = 1 if in_trade else 0
        return position

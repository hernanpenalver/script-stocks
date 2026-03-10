import pandas as pd
from .base import BaseStrategy


class SMACrossover(BaseStrategy):
    """
    Long when the fast SMA is above the slow SMA.

    Common pairs:
        fast=20, slow=50  → short-term trend
        fast=50, slow=200 → golden / death cross
    """

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"SMA {self.fast}/{self.slow}"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        sma_fast = prices.rolling(self.fast).mean()
        sma_slow = prices.rolling(self.slow).mean()
        signal = (sma_fast > sma_slow).astype(int)
        return signal.fillna(0)

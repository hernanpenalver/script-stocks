import pandas as pd
from .base import BaseStrategy


class BuyAndHold(BaseStrategy):
    """Always fully invested — classic benchmark."""

    @property
    def name(self) -> str:
        return "Buy & Hold"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        return pd.Series(1, index=prices.index, dtype=int)

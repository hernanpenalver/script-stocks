import pandas as pd
from .rsi import RSIStrategy


class RSIPercentile(RSIStrategy):
    """
    Buy when RSI falls below the historical (expanding) low_pct percentile.
    Sell when RSI rises above the historical (expanding) high_pct percentile.

    Percentiles are computed on an expanding window (all history up to each day),
    so the thresholds adapt dynamically to the asset's RSI distribution over time.
    """

    def __init__(self, period: int = 14, low_pct: float = 0.10, high_pct: float = 0.90):
        super().__init__(period)
        self.low_pct = low_pct
        self.high_pct = high_pct

    @property
    def name(self) -> str:
        return f"RSI-Pct({int(self.low_pct * 100)}/{int(self.high_pct * 100)})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        rsi = self._compute_rsi(prices)

        # Expanding percentiles: thresholds adapt as more RSI history accumulates
        low_threshold = rsi.expanding().quantile(self.low_pct)
        high_threshold = rsi.expanding().quantile(self.high_pct)

        position = pd.Series(0, index=prices.index, dtype=int)
        in_trade = False

        for i in range(len(rsi)):
            val = rsi.iloc[i]
            lo = low_threshold.iloc[i]
            hi = high_threshold.iloc[i]

            if pd.isna(val) or pd.isna(lo) or pd.isna(hi):
                position.iloc[i] = 0
                continue

            if not in_trade and val < lo:
                in_trade = True
            elif in_trade and val > hi:
                in_trade = False

            position.iloc[i] = 1 if in_trade else 0

        return position

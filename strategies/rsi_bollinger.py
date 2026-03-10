import pandas as pd
from .base import BaseStrategy


class RSIBollinger(BaseStrategy):
    """
    Combined RSI + Bollinger Bands mean-reversion strategy.

    Entry (long): price < lower Bollinger band  AND  RSI < oversold
    Exit:         price > upper Bollinger band  OR   RSI > overbought

    Both conditions must be true to enter, but either is enough to exit,
    making entries selective and exits prompt.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return f"RSI+BB({self.rsi_period},{self.bb_period})"

    def _compute_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        alpha = 1.0 / self.rsi_period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # Bollinger Bands
        rolling = prices.rolling(self.bb_period)
        mid = rolling.mean()
        std = rolling.std()
        lower = mid - self.bb_std * std
        upper = mid + self.bb_std * std

        # RSI
        rsi = self._compute_rsi(prices)

        position = pd.Series(0, index=prices.index, dtype=int)
        in_trade = False

        for i in range(len(prices)):
            p = prices.iloc[i]
            lo = lower.iloc[i]
            up = upper.iloc[i]
            r = rsi.iloc[i]

            if pd.isna(lo) or pd.isna(r):
                position.iloc[i] = 0
                continue

            if not in_trade:
                # Enter only when BOTH signals agree
                if p < lo and r < self.oversold:
                    in_trade = True
            else:
                # Exit when EITHER signal triggers
                if p > up or r > self.overbought:
                    in_trade = False

            position.iloc[i] = 1 if in_trade else 0

        return position

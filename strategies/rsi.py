import pandas as pd
from .base import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    Enter long when RSI drops below `oversold`, exit when RSI rises above `overbought`.
    Uses Wilder's smoothing (EWM with adjust=False, alpha=1/period).
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return f"RSI({self.period})"

    def _compute_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        alpha = 1.0 / self.period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        rsi = self._compute_rsi(prices)
        position = pd.Series(0, index=prices.index, dtype=int)
        in_trade = False
        for i in range(len(rsi)):
            val = rsi.iloc[i]
            if pd.isna(val):
                position.iloc[i] = 0
                continue
            if not in_trade and val < self.oversold:
                in_trade = True
            elif in_trade and val > self.overbought:
                in_trade = False
            position.iloc[i] = 1 if in_trade else 0
        return position


class RSIWithStopLoss(RSIStrategy):
    """
    Same entry/exit as RSIStrategy, but adds a stop loss:
    if RSI re-enters the oversold zone after having exited it,
    the position is closed (the bounce failed).
    """

    @property
    def name(self) -> str:
        return f"RSI({self.period})+SL"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        rsi = self._compute_rsi(prices)
        position = pd.Series(0, index=prices.index, dtype=int)
        in_trade = False
        exited_oversold = False  # True once RSI has risen back above oversold after entry
        for i in range(len(rsi)):
            val = rsi.iloc[i]
            if pd.isna(val):
                position.iloc[i] = 0
                continue
            if not in_trade:
                if val < self.oversold:
                    in_trade = True
                    exited_oversold = False
            else:
                if not exited_oversold and val >= self.oversold:
                    exited_oversold = True
                if val > self.overbought:
                    # Normal exit: RSI reached overbought
                    in_trade = False
                    exited_oversold = False
                elif exited_oversold and val < self.oversold:
                    # Stop loss: RSI re-entered oversold after bounce
                    in_trade = False
                    exited_oversold = False
            position.iloc[i] = 1 if in_trade else 0
        return position

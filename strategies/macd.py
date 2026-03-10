import pandas as pd
from .base import BaseStrategy


class MACDStrategy(BaseStrategy):
    """
    Long when the MACD line is above the signal line.

    Default parameters: fast=12, slow=26, signal=9.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        signal = (macd_line > signal_line).astype(int)
        return signal.fillna(0)


class MACDWithStopLoss(MACDStrategy):
    """
    Same entry as MACDStrategy (histogram turns positive), but tolerates small
    negative histogram values. Exits only when the histogram drops below
    `-sl_threshold` as a fraction of price (strong bearish momentum = stop loss).

    Default sl_threshold=0.005 means histogram must reach -0.5% of price to exit.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, sl_threshold: float = 0.005):
        super().__init__(fast, slow, signal)
        self.sl_threshold = sl_threshold

    @property
    def name(self) -> str:
        return f"MACD({self.fast},{self.slow},{self.signal})+SL"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        histogram = macd_line - signal_line
        norm_histogram = histogram / prices  # normalize by price for cross-symbol comparability

        position = pd.Series(0, index=prices.index, dtype=int)
        in_trade = False
        for i in range(len(norm_histogram)):
            h = norm_histogram.iloc[i]
            if pd.isna(h):
                position.iloc[i] = 0
                continue
            if not in_trade:
                if h > 0:
                    in_trade = True
            else:
                if h < -self.sl_threshold:
                    in_trade = False
            position.iloc[i] = 1 if in_trade else 0
        return position

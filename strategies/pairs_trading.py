import pandas as pd
import numpy as np
from .base import BaseStrategy


class PairsTrading(BaseStrategy):
    """
    Pairs Trading (statistical arbitrage).

    Computes the price ratio between the main asset and a paired asset,
    then calculates the z-score of that ratio relative to its rolling mean.

    Enter long when the z-score drops below -entry_z (main asset is
    cheap relative to pair). Exit when z-score reverts above exit_z.

    Parameters
    ----------
    pair_prices : pd.Series
        Price series of the paired asset (must share the same date index).
    pair_name : str
        Ticker of the paired asset (for display).
    period : int
        Lookback window for mean and std of the ratio (default 60).
    entry_z : float
        Z-score threshold to enter long (default 2.0).
    exit_z : float
        Z-score threshold to exit (default 0.0, i.e. mean reversion).
    """

    def __init__(
        self,
        pair_prices: pd.Series,
        pair_name: str,
        period: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
    ):
        self.pair_prices = pair_prices
        self.pair_name = pair_name
        self.period = period
        self.entry_z = entry_z
        self.exit_z = exit_z

    @property
    def name(self) -> str:
        return f"Pairs(vs {self.pair_name})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # Align both series on common dates
        combined = pd.DataFrame({"main": prices, "pair": self.pair_prices}).dropna()

        # Price ratio
        ratio = combined["main"] / combined["pair"]

        # Rolling z-score of the ratio
        rolling_mean = ratio.rolling(self.period).mean()
        rolling_std = ratio.rolling(self.period).std()
        zscore = (ratio - rolling_mean) / rolling_std

        # Reindex to original prices index
        zscore = zscore.reindex(prices.index)

        position = pd.Series(0, index=prices.index, dtype=int)
        in_trade = False
        for i in range(len(prices)):
            z = zscore.iloc[i]
            if pd.isna(z):
                continue
            if not in_trade and z < -self.entry_z:
                # Main asset is cheap relative to pair
                in_trade = True
            elif in_trade and z > -self.exit_z:
                # Ratio reverted to mean
                in_trade = False
            position.iloc[i] = 1 if in_trade else 0
        return position

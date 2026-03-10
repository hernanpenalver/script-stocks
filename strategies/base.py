from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate position signals from a price series.

        Parameters
        ----------
        prices : pd.Series
            Adjusted close prices indexed by date.

        Returns
        -------
        pd.Series
            Integer series of 1 (long) or 0 (out of market).
        """

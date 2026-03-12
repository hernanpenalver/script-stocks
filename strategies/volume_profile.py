import numpy as np
import pandas as pd
from .base import BaseStrategy


class VolumeProfile(BaseStrategy):
    """
    Volume Profile support/resistance strategy.

    Builds a rolling volume profile over the last `lookback` days, binning
    prices into `num_bins` levels weighted by volume.  Identifies the
    Point of Control (POC — price level with highest volume) and the
    Value Area (levels containing `va_pct`% of total volume).

    Trading rules
    -------------
    - Enter long when price drops to or below the Value Area Low (VAL),
      which acts as volume-based support.
    - Exit when price reaches the POC (mean reversion target) or when
      price rises above the Value Area High (VAH).

    Parameters
    ----------
    volume : pd.Series
        Daily volume series (same index as prices).
    lookback : int
        Rolling window to build the volume profile (default 60 days).
    num_bins : int
        Number of price bins for the profile (default 50).
    va_pct : float
        Fraction of total volume that defines the Value Area (default 0.70).
    exit_at : str
        "poc" to exit at Point of Control, "vah" to exit at Value Area High
        (default "poc").
    """

    def __init__(self, volume: pd.Series, lookback: int = 60,
                 num_bins: int = 50, va_pct: float = 0.70,
                 exit_at: str = "poc"):
        self.volume = volume
        self.lookback = lookback
        self.num_bins = num_bins
        self.va_pct = va_pct
        self.exit_at = exit_at

    @property
    def name(self) -> str:
        return f"VolProfile({self.lookback},{self.exit_at})"

    def _build_profile(self, prices_window: np.ndarray,
                       volume_window: np.ndarray):
        """Build volume profile for a window. Returns (poc, val, vah)."""
        price_min = prices_window.min()
        price_max = prices_window.max()

        if price_max == price_min:
            return price_min, price_min, price_max

        bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Assign each day's volume to a price bin
        vol_profile = np.zeros(self.num_bins)
        bin_indices = np.digitize(prices_window, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        for idx, vol in zip(bin_indices, volume_window):
            vol_profile[idx] += vol

        # Point of Control: bin with highest volume
        poc_idx = vol_profile.argmax()
        poc = bin_centers[poc_idx]

        # Value Area: expand from POC until va_pct of total volume
        total_vol = vol_profile.sum()
        if total_vol == 0:
            return poc, price_min, price_max

        va_vol = vol_profile[poc_idx]
        lo_idx = poc_idx
        hi_idx = poc_idx

        while va_vol / total_vol < self.va_pct:
            look_lo = vol_profile[lo_idx - 1] if lo_idx > 0 else 0
            look_hi = vol_profile[hi_idx + 1] if hi_idx < self.num_bins - 1 else 0

            if look_lo == 0 and look_hi == 0:
                break

            if look_lo >= look_hi:
                lo_idx -= 1
                va_vol += vol_profile[lo_idx]
            else:
                hi_idx += 1
                va_vol += vol_profile[hi_idx]

        val = bin_centers[lo_idx]  # Value Area Low
        vah = bin_centers[hi_idx]  # Value Area High

        return poc, val, vah

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # Align volume with prices index
        vol = self.volume.reindex(prices.index).fillna(0).values
        px = prices.values

        position = pd.Series(0, index=prices.index, dtype=int)
        in_trade = False

        for i in range(self.lookback, len(prices)):
            px_window = px[i - self.lookback:i]
            vol_window = vol[i - self.lookback:i]

            poc, val, vah = self._build_profile(px_window, vol_window)

            p = px[i]

            if not in_trade and p <= val:
                in_trade = True
            elif in_trade:
                exit_level = poc if self.exit_at == "poc" else vah
                if p >= exit_level:
                    in_trade = False

            position.iloc[i] = 1 if in_trade else 0

        return position

import numpy as np
import pandas as pd
from .base import BaseStrategy


class TrendExhaustion(BaseStrategy):
    """
    Trend Exhaustion Detector — ported from PineScript v6.

    Detects when a trend is running out of steam by scoring 5 components:
      1. ATR compression (volatility shrinking)
      2. Decreasing volume (linear regression slope < 0)
      3. RSI divergence (bearish in uptrend, bullish in downtrend)
      4. MACD divergence (same logic)
      5. Failed Break of Structure (BOS)

    Score ranges 0–4 (BOS counts double, capped at 4).

    Trading rules
    -------------
    - Long when trend is bullish AND exhaustion score < exit_score (healthy trend)
    - Long when trend is bearish AND exhaustion score >= entry_score (reversal expected)
    - Flat otherwise

    Parameters
    ----------
    ohlcv : pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume.
    ema_fast : int
        Fast EMA period for trend detection (default 21).
    ema_slow : int
        Slow EMA period for trend detection (default 50).
    swing_len : int
        Lookback for swing high/low detection (default 10).
    atr_len : int
        ATR period (default 14).
    atr_ma_len : int
        SMA period for ATR reference (default 30).
    atr_thresh : float
        ATR compression threshold (default 0.7).
    vol_len : int
        Bars for volume slope calculation (default 10).
    rsi_len : int
        RSI period (default 14).
    div_lb : int
        Divergence lookback (default 5).
    macd_fast : int
        MACD fast period (default 12).
    macd_slow : int
        MACD slow period (default 26).
    macd_signal : int
        MACD signal period (default 9).
    bos_len : int
        Swing lookback for BOS (default 10).
    entry_score : int
        Min exhaustion score to enter on reversal (default 3).
    exit_score : int
        Score at which to exit a trend-following position (default 3).
    """

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        ema_fast: int = 21,
        ema_slow: int = 50,
        swing_len: int = 10,
        atr_len: int = 14,
        atr_ma_len: int = 30,
        atr_thresh: float = 0.7,
        vol_len: int = 10,
        rsi_len: int = 14,
        div_lb: int = 5,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bos_len: int = 10,
        entry_score: int = 3,
        exit_score: int = 3,
    ):
        self.ohlcv = ohlcv
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.swing_len = swing_len
        self.atr_len = atr_len
        self.atr_ma_len = atr_ma_len
        self.atr_thresh = atr_thresh
        self.vol_len = vol_len
        self.rsi_len = rsi_len
        self.div_lb = div_lb
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bos_len = bos_len
        self.entry_score = entry_score
        self.exit_score = exit_score

    @property
    def name(self) -> str:
        return f"TrendExhaustion(entry>={self.entry_score},exit>={self.exit_score})"

    # ------------------------------------------------------------------
    # Indicator components
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(close: pd.Series, fast: int, slow: int, signal: int):
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_f - ema_s
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _linreg_slope(series: pd.Series, period: int) -> pd.Series:
        """Rolling linear regression slope (like ta.linreg in Pine)."""
        result = pd.Series(np.nan, index=series.index)
        x = np.arange(period, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        vals = series.values
        for i in range(period - 1, len(vals)):
            y = vals[i - period + 1: i + 1]
            if np.any(np.isnan(y)):
                continue
            y_mean = y.mean()
            result.iloc[i] = ((x * (y - y_mean)).sum()) / x_var
        return result

    @staticmethod
    def _pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
        """Detect pivot highs (like ta.pivothigh). Returns the high value or NaN."""
        result = pd.Series(np.nan, index=high.index)
        vals = high.values
        for i in range(left, len(vals) - right):
            pivot = vals[i]
            if all(pivot >= vals[i - j] for j in range(1, left + 1)) and \
               all(pivot >= vals[i + j] for j in range(1, right + 1)):
                result.iloc[i + right] = pivot  # confirmed after right bars
        return result

    @staticmethod
    def _pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
        """Detect pivot lows (like ta.pivotlow). Returns the low value or NaN."""
        result = pd.Series(np.nan, index=low.index)
        vals = low.values
        for i in range(left, len(vals) - right):
            pivot = vals[i]
            if all(pivot <= vals[i - j] for j in range(1, left + 1)) and \
               all(pivot <= vals[i + j] for j in range(1, right + 1)):
                result.iloc[i + right] = pivot
        return result

    def _compute_trend(self, close: pd.Series, high: pd.Series, low: pd.Series):
        """
        Returns trend_dir series: +1 bullish, -1 bearish, 0 neutral.
        """
        ema_f = self._ema(close, self.ema_fast)
        ema_s = self._ema(close, self.ema_slow)
        ema_bull = (ema_f > ema_s) & (close > ema_s)
        ema_bear = (ema_f < ema_s) & (close < ema_s)

        ph = self._pivot_high(high, self.swing_len, self.swing_len)
        pl = self._pivot_low(low, self.swing_len, self.swing_len)

        # Track last two pivot highs and lows
        trend_dir = pd.Series(0, index=close.index, dtype=int)
        ph1 = ph2 = pl1 = pl2 = np.nan

        for i in range(len(close)):
            if not np.isnan(ph.iloc[i]):
                ph2 = ph1
                ph1 = ph.iloc[i]
            if not np.isnan(pl.iloc[i]):
                pl2 = pl1
                pl1 = pl.iloc[i]

            struct_bull = (not np.isnan(ph1) and not np.isnan(ph2) and
                          not np.isnan(pl1) and not np.isnan(pl2) and
                          ph1 > ph2 and pl1 > pl2)
            struct_bear = (not np.isnan(ph1) and not np.isnan(ph2) and
                          not np.isnan(pl1) and not np.isnan(pl2) and
                          ph1 < ph2 and pl1 < pl2)

            if ema_bull.iloc[i] and struct_bull:
                trend_dir.iloc[i] = 1
            elif ema_bear.iloc[i] and struct_bear:
                trend_dir.iloc[i] = -1

        return trend_dir

    def _compute_scores(self, close: pd.Series, high: pd.Series,
                        low: pd.Series, volume: pd.Series,
                        trend_dir: pd.Series) -> pd.Series:
        """Compute exhaustion score (0–4) for each bar."""
        n = len(close)

        # 1. ATR compression
        atr_val = self._atr(high, low, close, self.atr_len)
        atr_ma = atr_val.rolling(self.atr_ma_len).mean()
        atr_comp = atr_val < (atr_ma * self.atr_thresh)

        # 2. Volume decreasing
        vol_slope = self._linreg_slope(volume, self.vol_len)
        vol_shrink = vol_slope < 0

        # 3. RSI divergence
        rsi = self._rsi(close, self.rsi_len)
        div_rsi = pd.Series(False, index=close.index)
        for i in range(self.div_lb, n):
            lb = self.div_lb
            hi_slice = high.iloc[i - lb:i]
            lo_slice = low.iloc[i - lb:i]
            rsi_slice = rsi.iloc[i - lb:i]

            price_ph = high.iloc[i] >= hi_slice.max()
            price_pl = low.iloc[i] <= lo_slice.min()

            if trend_dir.iloc[i] == 1:
                # Bearish divergence: price makes new high but RSI doesn't
                if price_ph and high.iloc[i] > hi_slice.max() and rsi.iloc[i] < rsi_slice.max():
                    div_rsi.iloc[i] = True
            elif trend_dir.iloc[i] == -1:
                # Bullish divergence: price makes new low but RSI doesn't
                if price_pl and low.iloc[i] < lo_slice.min() and rsi.iloc[i] > rsi_slice.min():
                    div_rsi.iloc[i] = True
            else:
                bear_d = price_ph and high.iloc[i] > hi_slice.max() and rsi.iloc[i] < rsi_slice.max()
                bull_d = price_pl and low.iloc[i] < lo_slice.min() and rsi.iloc[i] > rsi_slice.min()
                div_rsi.iloc[i] = bear_d or bull_d

        # 4. MACD divergence
        macd_line, _ = self._macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        div_macd = pd.Series(False, index=close.index)
        for i in range(self.div_lb, n):
            lb = self.div_lb
            hi_slice = high.iloc[i - lb:i]
            lo_slice = low.iloc[i - lb:i]
            macd_slice = macd_line.iloc[i - lb:i]

            price_ph = high.iloc[i] >= hi_slice.max()
            price_pl = low.iloc[i] <= lo_slice.min()

            if trend_dir.iloc[i] == 1:
                if price_ph and high.iloc[i] > hi_slice.max() and macd_line.iloc[i] < macd_slice.max():
                    div_macd.iloc[i] = True
            elif trend_dir.iloc[i] == -1:
                if price_pl and low.iloc[i] < lo_slice.min() and macd_line.iloc[i] > macd_slice.min():
                    div_macd.iloc[i] = True
            else:
                bear_d = price_ph and high.iloc[i] > hi_slice.max() and macd_line.iloc[i] < macd_slice.max()
                bull_d = price_pl and low.iloc[i] < lo_slice.min() and macd_line.iloc[i] > macd_slice.min()
                div_macd.iloc[i] = bear_d or bull_d

        # 5. Failed BOS
        swing_high = high.rolling(self.bos_len).max()
        swing_low = low.rolling(self.bos_len).min()
        bos_bear_fail = (high > swing_high.shift(1)) & (close < swing_high.shift(1))
        bos_bull_fail = (low < swing_low.shift(1)) & (close > swing_low.shift(1))

        bos_fail = pd.Series(False, index=close.index)
        for i in range(len(close)):
            td = trend_dir.iloc[i]
            if td == 1:
                bos_fail.iloc[i] = bos_bear_fail.iloc[i]
            elif td == -1:
                bos_fail.iloc[i] = bos_bull_fail.iloc[i]
            else:
                bos_fail.iloc[i] = bos_bear_fail.iloc[i] or bos_bull_fail.iloc[i]

        # Score: 0–4 (BOS counts 2)
        score = (
            atr_comp.astype(int) +
            vol_shrink.astype(int) +
            div_rsi.astype(int) +
            div_macd.astype(int) +
            bos_fail.astype(int) * 2
        ).clip(upper=4)

        return score

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        df = self.ohlcv.reindex(prices.index).dropna()
        idx = prices.index
        close = df["Close"].reindex(idx)
        high = df["High"].reindex(idx)
        low = df["Low"].reindex(idx)
        volume = df["Volume"].reindex(idx).fillna(0)

        trend_dir = self._compute_trend(close, high, low)
        score = self._compute_scores(close, high, low, volume, trend_dir)

        position = pd.Series(0, index=idx, dtype=int)
        in_trade = False

        for i in range(len(idx)):
            td = trend_dir.iloc[i]
            sc = score.iloc[i]

            if np.isnan(sc):
                position.iloc[i] = 0
                continue

            if not in_trade:
                # Enter: healthy bull trend OR bearish exhaustion (reversal)
                if td == 1 and sc < self.exit_score:
                    in_trade = True
                elif td == -1 and sc >= self.entry_score:
                    in_trade = True
            else:
                # Exit: bull trend exhausting OR bear trend recovering
                if td == 1 and sc >= self.exit_score:
                    in_trade = False
                elif td == -1 and sc < self.entry_score:
                    in_trade = False
                elif td == 0:
                    # No clear trend — stay but watch for re-entry conditions
                    pass

            position.iloc[i] = 1 if in_trade else 0

        return position

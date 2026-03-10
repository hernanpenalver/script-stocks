import yfinance as yf
import pandas as pd


def download_prices(symbols: list[str], start: str, end: str) -> dict[str, pd.Series]:
    """
    Download adjusted close prices for a list of symbols.

    Returns a dict mapping symbol → pd.Series of daily adjusted close prices.
    Symbols that fail to download (no data) are silently skipped with a warning.
    """
    print(f"Downloading data for {len(symbols)} symbols ({start} to {end}) ...")
    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    prices: dict[str, pd.Series] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        # Multiple tickers → columns are (field, ticker)
        close = raw["Close"]
        for sym in symbols:
            if sym in close.columns:
                series = close[sym].dropna()
                if len(series) > 0:
                    prices[sym] = series
                else:
                    print(f"  [WARN] No data for {sym}, skipping.")
            else:
                print(f"  [WARN] {sym} not found in downloaded data, skipping.")
    else:
        # Single ticker
        sym = symbols[0]
        series = raw["Close"].dropna()
        if len(series) > 0:
            prices[sym] = series
        else:
            print(f"  [WARN] No data for {sym}, skipping.")

    print(f"  Downloaded {len(prices)} symbols successfully.")
    return prices

"""
scanner.py — Daily signal scanner with Telegram notifications.

Runs all strategies on all symbols from 2015 to today, computes Composite
Scores, and sends Telegram alerts whenever a BUY or EXIT signal is detected
for any strategy with Composite Score > SCORE_THRESHOLD.

Trades are executed manually; this module only analyses and notifies.

──────────────────────────────────────────────────────────────────────────────
FIRST-TIME SETUP
──────────────────────────────────────────────────────────────────────────────
1. Create a Telegram bot:
   - Open Telegram, search for @BotFather
   - Send /newbot and follow the prompts → you get a BOT_TOKEN
   - Start a chat with your new bot, then open:
     https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
     (send any message to the bot first so the chat appears)
   - Copy the "id" field from the result → that is your CHAT_ID

2. Set environment variables (local test):
   export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
   export TELEGRAM_CHAT_ID="987654321"
   python scanner.py

3. For GitHub Actions (see .github/workflows/scanner.yml):
   - Go to your repo → Settings → Secrets → Actions
   - Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as repository secrets
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import requests

import config
from backtester import run_backtest
from data_loader import download_prices
from strategies import (
    BollingerBands,
    BuyAndHold,
    MACDOptimized,
    MACDStrategy,
    MACDWithStopLoss,
    RSIPercentile,
    RSIStrategy,
    RSIWithStopLoss,
    SMACrossover,
)

# ── Settings ───────────────────────────────────────────────────────────────────
SCORE_THRESHOLD = 1.2       # only notify strategies with Composite Score above this


def _build_strategies():
    return [
        BuyAndHold(),
        SMACrossover(fast=20, slow=50),
        SMACrossover(fast=50, slow=200),
        RSIStrategy(period=14, oversold=30, overbought=70),
        RSIWithStopLoss(period=14, oversold=30, overbought=70),
        RSIPercentile(period=14, low_pct=0.10, high_pct=0.90),
        MACDStrategy(fast=12, slow=26, signal=9),
        MACDWithStopLoss(fast=12, slow=26, signal=9, sl_threshold=0.005),
        MACDOptimized(train_ratio=0.70),
        BollingerBands(period=20, num_std=2.0),
    ]


# ── Composite score ────────────────────────────────────────────────────────────
def _compute_scores(results_flat: list[dict]) -> dict[tuple, float]:
    """
    Same formula as Strategy Ranking chart:
        score = norm(Sharpe) + norm(CAGR) + norm(−MaxDrawdown)
    Range 0–3. Normalisation is relative to all symbol×strategy pairs.
    """
    df = pd.DataFrame([
        {
            "key":          (r["symbol"], r["strat_name"]),
            "Sharpe":       r["metrics"]["Sharpe"],
            "CAGR":         r["metrics"]["CAGR"],
            "Max Drawdown": r["metrics"]["Max Drawdown"],
        }
        for r in results_flat
    ])
    for col in ("Sharpe", "CAGR", "Max Drawdown"):
        rng = df[col].max() - df[col].min()
        df[f"_n{col}"] = (df[col] - df[col].min()) / rng if rng != 0 else 0.5
    df["_nMax Drawdown"] = 1 - df["_nMax Drawdown"]   # less negative = better
    df["score"] = df["_nSharpe"] + df["_nCAGR"] + df["_nMax Drawdown"]
    return dict(zip(df["key"], df["score"]))


# ── Telegram ───────────────────────────────────────────────────────────────────
def _send_telegram(text: str) -> None:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[WARN] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — printing to stdout.")
        print(text)
        print()
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[ERROR] Telegram send failed: {exc}")


def _format_message(
    symbol: str,
    strat_label: str,
    signal: str,
    score: float,
    metrics: dict,
    run_date: str,
) -> str:
    icon    = "🟢" if signal == "BUY" else "🔴"
    vs_sp   = metrics["CAGR"] - 11.14          # approx S&P 500 CAGR 2015-2024
    vs_str  = f"+{vs_sp:.2f}pp" if vs_sp >= 0 else f"{vs_sp:.2f}pp"
    action  = "Considerar COMPRA" if signal == "BUY" else "Considerar SALIDA"
    return (
        f"{icon} <b>{signal} — {symbol}</b>\n"
        f"📐 Estrategia: <b>{strat_label}</b>\n"
        f"📅 {run_date}\n\n"
        f"<b>Composite Score:</b> {score:.3f} / 3.0\n"
        f"<b>CAGR:</b>    {metrics['CAGR']}%  ({vs_str} vs S&amp;P500)\n"
        f"<b>Sharpe:</b>  {metrics['Sharpe']}\n"
        f"<b>Max DD:</b>  {metrics['Max Drawdown']}%\n"
        f"<b>Win Rate:</b>{metrics['Win Rate']}%\n\n"
        f"⚠️ {action} — trade manual"
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    today    = date.today()
    end_date = today.strftime("%Y-%m-%d")
    run_date = today.strftime("%Y-%m-%d")

    print("=" * 55)
    print(f"  Stock Scanner — {run_date}")
    print("=" * 55)

    # 1. Download prices from 2015 to today
    #    Using the full historical window keeps metrics and MACD-Opt params
    #    consistent with the backtest report.
    all_symbols = config.SYMBOLS_USA + config.SYMBOLS_ARG
    print(f"\nDownloading {config.START_DATE} to {end_date} ...")
    prices_dict = download_prices(all_symbols, config.START_DATE, end_date)
    if not prices_dict:
        print("ERROR: No data downloaded. Check internet connection.")
        sys.exit(1)

    # 2. Run backtests (metrics + signals)
    strategies = _build_strategies()
    total = len(prices_dict) * len(strategies)
    print(f"Running {len(strategies)} strategies × {len(prices_dict)} symbols "
          f"({total} backtests) ...\n")

    results_flat = []
    done = 0
    for symbol, prices in prices_dict.items():
        for strat in strategies:
            result     = run_backtest(prices, strat, config.INITIAL_CAPITAL)
            best_p     = result.get("best_params")
            strat_label = (
                f"MACD-Opt({best_p[0]},{best_p[1]},{best_p[2]})"
                if best_p else strat.name
            )
            results_flat.append({
                "symbol":      symbol,
                "strat_name":  strat.name,
                "strat_label": strat_label,
                "metrics":     result["metrics"],
                "signals":     result["signals"],
            })
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{total} done ...")

    # 3. Composite scores
    scores = _compute_scores(results_flat)
    above  = sum(1 for s in scores.values() if s >= SCORE_THRESHOLD)
    print(f"\nStrategies above threshold (score > {SCORE_THRESHOLD}): {above}/{total}")

    # 4. Detect signal changes (yesterday → today)
    new_signals = []
    for r in results_flat:
        key   = (r["symbol"], r["strat_name"])
        score = scores.get(key, 0.0)
        if score < SCORE_THRESHOLD:
            continue
        sigs = r["signals"]
        if len(sigs) < 2:
            continue
        prev, curr = int(sigs.iloc[-2]), int(sigs.iloc[-1])
        if   curr == 1 and prev == 0:
            signal_type = "BUY"
        elif curr == 0 and prev == 1:
            signal_type = "EXIT"
        else:
            continue
        new_signals.append((
            r["symbol"], r["strat_label"], signal_type, score, r["metrics"]
        ))

    # Sort by score descending
    new_signals.sort(key=lambda x: x[3], reverse=True)

    print(f"New signals detected: {len(new_signals)}\n")

    if not new_signals:
        print("No new buy/exit signals above threshold. Nothing sent.")
        return

    # 5. Send Telegram notifications
    for symbol, strat_label, signal_type, score, metrics in new_signals:
        msg = _format_message(symbol, strat_label, signal_type, score, metrics, run_date)
        _send_telegram(msg)
        print(f"  [{signal_type}] {symbol} — {strat_label}  score={score:.3f}")

    print(f"\nDone. {len(new_signals)} notification(s) sent.")


if __name__ == "__main__":
    main()

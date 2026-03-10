import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Metric display configuration
METRIC_COLS = [
    "Score",
    "Composite Score",
    "Total Return",
    "CAGR",
    "vs S&P500",
    "Sharpe",
    "Sortino",
    "Max Drawdown",
    "Calmar",
    "Win Rate",
    "Num Trades",
]

# Metrics where higher is better (for colour coding)
HIGHER_IS_BETTER = {"Score", "Composite Score", "Total Return", "CAGR", "vs S&P500", "Sharpe", "Sortino", "Calmar", "Win Rate"}
# Metrics where lower (more negative) is worse
LOWER_IS_BETTER_ABS = {"Max Drawdown"}


def _add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two score columns per row:
      Score           — same formula as Strategy Ranking (sum of 3 normalised values, 0–3).
      Score (0–10)    — same, rescaled for easier reading.
    Normalisation is relative to all rows in the table.
    """
    tmp = df.copy()
    for col in ("Sharpe", "CAGR", "Max Drawdown"):
        rng = tmp[col].max() - tmp[col].min()
        tmp[f"_norm_{col}"] = (tmp[col] - tmp[col].min()) / rng if rng != 0 else 0.5
    tmp["_norm_Max Drawdown"] = 1 - tmp["_norm_Max Drawdown"]  # less negative = better
    raw = tmp["_norm_Sharpe"] + tmp["_norm_CAGR"] + tmp["_norm_Max Drawdown"]
    df = df.copy()
    df["Composite Score"] = raw.round(3)          # 0–3, matches Strategy Ranking exactly
    df["Score"] = (raw / 3 * 10).round(2)         # 0–10, easier to read
    return df


def _build_summary_df(results: dict, sp500_cagr: float | None = None) -> pd.DataFrame:
    rows = []
    for symbol, strats in results.items():
        for strat_name, data in strats.items():
            best_p = data.get("best_params")
            label = (f"MACD-Opt({best_p[0]},{best_p[1]},{best_p[2]})"
                     if best_p else strat_name)
            row = {"Symbol": symbol, "Strategy": label, "Strategy Key": strat_name}
            row.update(data["metrics"])
            if sp500_cagr is not None:
                row["vs S&P500"] = round(row["CAGR"] - sp500_cagr, 2)
            rows.append(row)
    return pd.DataFrame(rows)


def _colour_cell(val, col, vmin, vmax):
    """Return an RGBA colour string for a metric cell."""
    if pd.isna(val) or vmin == vmax:
        return "rgba(200,200,200,0.3)"
    norm = (val - vmin) / (vmax - vmin)  # 0..1
    if col in HIGHER_IS_BETTER:
        # green gradient
        r = int(220 - norm * 150)
        g = int(100 + norm * 120)
        b = int(100 - norm * 50)
    elif col in LOWER_IS_BETTER_ABS:
        # Max Drawdown: more negative = worse (red), less negative = better (green)
        # norm=0 means most negative (worst), norm=1 means least negative (best)
        r = int(220 - norm * 150)
        g = int(100 + norm * 120)
        b = int(100 - norm * 50)
    else:
        r, g, b = 200, 200, 200
    return f"rgba({r},{g},{b},0.55)"


def _build_summary_table(df: pd.DataFrame) -> go.Figure:
    available_cols = [c for c in METRIC_COLS if c in df.columns]
    header_values = ["Symbol", "Strategy"] + available_cols
    cell_values = [df["Symbol"].tolist(), df["Strategy"].tolist()]

    fill_colours = [["rgba(240,240,240,0.5)"] * len(df)] * 2  # symbol, strategy cols

    for col in available_cols:
        col_vals = df[col].tolist()
        try:
            vmin, vmax = df[col].min(), df[col].max()
        except Exception:
            vmin, vmax = 0, 1
        colours = [_colour_cell(v, col, vmin, vmax) for v in col_vals]
        fill_colours.append(colours)

        # Format display values
        if col in ("Score", "Composite Score"):
            cell_values.append([f"{v:.3f}" for v in col_vals])
        elif col in ("Total Return", "CAGR", "Win Rate", "Max Drawdown"):
            cell_values.append([f"{v:.2f}%" for v in col_vals])
        elif col == "vs S&P500":
            cell_values.append([f"{'+' if v >= 0 else ''}{v:.2f}pp" for v in col_vals])
        elif col == "Num Trades":
            cell_values.append([str(int(v)) for v in col_vals])
        else:
            cell_values.append([f"{v:.3f}" for v in col_vals])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{h}</b>" for h in header_values],
                    fill_color="rgba(50,100,180,0.85)",
                    font=dict(color="white", size=12),
                    align="center",
                    height=30,
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=fill_colours,
                    font=dict(size=11),
                    align="center",
                    height=25,
                ),
            )
        ]
    )
    fig.update_layout(
        title="<b>Strategy Performance Summary</b> — All Symbols × All Strategies",
        margin=dict(l=10, r=10, t=50, b=10),
        height=max(500, 30 * len(df) + 100),
    )
    return fig


def _build_equity_curves(results: dict, sp500_equity: pd.Series | None = None) -> go.Figure:
    symbols = list(results.keys())
    ncols = min(3, len(symbols))
    nrows = -(-len(symbols) // ncols)  # ceiling division

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=symbols,
        shared_xaxes=False,
    )

    palette = px.colors.qualitative.Plotly
    strategy_names = list(next(iter(results.values())).keys())
    colour_map = {s: palette[i % len(palette)] for i, s in enumerate(strategy_names)}
    shown_legends = set()

    for idx, (symbol, strats) in enumerate(results.items()):
        row = idx // ncols + 1
        col = idx % ncols + 1

        # S&P 500 benchmark line
        if sp500_equity is not None:
            show_sp500 = "S&P 500" not in shown_legends
            shown_legends.add("S&P 500")
            fig.add_trace(
                go.Scatter(
                    x=sp500_equity.index,
                    y=sp500_equity.values,
                    mode="lines",
                    name="S&P 500",
                    line=dict(color="black", width=2, dash="dash"),
                    legendgroup="S&P 500",
                    showlegend=show_sp500,
                    hovertemplate="<b>S&P 500</b><br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        for strat_name, data in strats.items():
            equity = data["equity"]
            show_legend = strat_name not in shown_legends
            shown_legends.add(strat_name)
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    mode="lines",
                    name=strat_name,
                    line=dict(color=colour_map[strat_name], width=1.5),
                    legendgroup=strat_name,
                    showlegend=show_legend,
                    hovertemplate=f"<b>{strat_name}</b><br>%{{x|%Y-%m-%d}}<br>$%{{y:,.0f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title="<b>Equity Curves by Symbol</b> — línea negra punteada = S&P 500 benchmark",
        height=350 * nrows,
        legend=dict(title="Strategy", orientation="v"),
        hovermode="x unified",
    )
    return fig


def _build_drawdown_chart(results: dict, sp500_equity: pd.Series | None = None) -> go.Figure:
    symbols = list(results.keys())
    ncols = min(3, len(symbols))
    nrows = -(-len(symbols) // ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=symbols,
    )

    palette = px.colors.qualitative.Plotly
    strategy_names = list(next(iter(results.values())).keys())
    colour_map = {s: palette[i % len(palette)] for i, s in enumerate(strategy_names)}
    shown_legends = set()

    for idx, (symbol, strats) in enumerate(results.items()):
        row = idx // ncols + 1
        col = idx % ncols + 1

        # S&P 500 drawdown reference
        if sp500_equity is not None:
            sp500_rolling_max = sp500_equity.cummax()
            sp500_dd = (sp500_equity - sp500_rolling_max) / sp500_rolling_max * 100
            show_sp500 = "S&P 500" not in shown_legends
            shown_legends.add("S&P 500")
            fig.add_trace(
                go.Scatter(
                    x=sp500_dd.index,
                    y=sp500_dd.values,
                    mode="lines",
                    name="S&P 500",
                    line=dict(color="black", width=2, dash="dash"),
                    legendgroup="S&P 500",
                    showlegend=show_sp500,
                    hovertemplate="<b>S&P 500</b><br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
                ),
                row=row,
                col=col,
            )

        for strat_name, data in strats.items():
            equity = data["equity"]
            rolling_max = equity.cummax()
            dd = (equity - rolling_max) / rolling_max * 100
            show_legend = strat_name not in shown_legends
            shown_legends.add(strat_name)
            fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd.values,
                    mode="lines",
                    name=strat_name,
                    fill="tozeroy",
                    line=dict(color=colour_map[strat_name], width=1),
                    legendgroup=strat_name,
                    showlegend=show_legend,
                    hovertemplate=f"<b>{strat_name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>",
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title="<b>Drawdown Curves by Symbol</b> — línea negra punteada = S&P 500",
        height=350 * nrows,
        legend=dict(title="Strategy"),
    )
    return fig


def _build_ranking(df: pd.DataFrame) -> go.Figure:
    """Composite score = normalised(Sharpe) + normalised(CAGR) + normalised(-MaxDD)."""
    tmp = df.copy()
    for col in ("Sharpe", "CAGR", "Max Drawdown"):
        rng = tmp[col].max() - tmp[col].min()
        if rng == 0:
            tmp[f"norm_{col}"] = 0.5
        else:
            tmp[f"norm_{col}"] = (tmp[col] - tmp[col].min()) / rng

    # For Max Drawdown: less negative is better → invert normalisation
    tmp["norm_Max Drawdown"] = 1 - tmp["norm_Max Drawdown"]
    tmp["score"] = tmp["norm_Sharpe"] + tmp["norm_CAGR"] + tmp["norm_Max Drawdown"]

    ranking = (
        tmp.groupby("Strategy Key")["score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    ranking.columns = ["Strategy", "Composite Score"]

    fig = px.bar(
        ranking,
        x="Composite Score",
        y="Strategy",
        orientation="h",
        color="Composite Score",
        color_continuous_scale="RdYlGn",
        text=ranking["Composite Score"].map(lambda v: f"{v:.3f}"),
        title="<b>Strategy Ranking</b> — Composite Score (Sharpe + CAGR + DrawdownAdj)",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=400,
    )
    return fig


def _build_yearly_heatmap(results: dict) -> go.Figure:
    """Heatmap of annual returns: one subplot per strategy, rows=symbols, cols=years."""
    strategy_names = list(next(iter(results.values())).keys())
    symbols = list(results.keys())

    ncols = min(2, len(strategy_names))
    nrows = -(-len(strategy_names) // ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=strategy_names,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for idx, strat_name in enumerate(strategy_names):
        row = idx // ncols + 1
        col = idx % ncols + 1

        # Build matrix: rows=symbols, cols=years
        matrix = []
        years = None
        for symbol in symbols:
            yr = results[symbol][strat_name]["yearly"] * 100  # as %
            if years is None:
                years = sorted(yr.index.tolist())
            matrix.append([yr.get(y, float("nan")) for y in years])

        z = matrix
        text = [
            [f"{v:.1f}%" if not pd.isna(v) else "" for v in row_vals]
            for row_vals in z
        ]

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[str(y) for y in years],
                y=symbols,
                text=text,
                texttemplate="%{text}",
                colorscale=[
                    [0.0, "rgb(180,0,0)"],
                    [0.4, "rgb(240,100,100)"],
                    [0.5, "rgb(245,245,245)"],
                    [0.6, "rgb(100,200,100)"],
                    [1.0, "rgb(0,140,0)"],
                ],
                zmid=0,
                showscale=False,
                hovertemplate="<b>%{y}</b> %{x}<br>Return: %{text}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="<b>Annual Returns by Strategy & Symbol</b> — Red=Loss, Green=Gain",
        height=max(400, 60 * len(symbols)) * nrows,
    )
    return fig


def _build_sp500_comparison(df: pd.DataFrame, sp500_metrics: dict) -> go.Figure:
    """Bar chart: CAGR por estrategia (promedio de símbolos) vs línea de referencia S&P 500."""
    sp500_cagr = sp500_metrics["CAGR"]
    avg_cagr = df.groupby("Strategy Key")["CAGR"].mean().reset_index()
    avg_cagr.columns = ["Strategy", "Avg CAGR"]
    avg_cagr = avg_cagr.sort_values("Avg CAGR", ascending=False)
    avg_cagr["Beats S&P500"] = avg_cagr["Avg CAGR"] >= sp500_cagr
    avg_cagr["Color"] = avg_cagr["Beats S&P500"].map({True: "rgba(0,160,80,0.8)", False: "rgba(200,60,60,0.8)"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg_cagr["Strategy"],
        y=avg_cagr["Avg CAGR"],
        marker_color=avg_cagr["Color"].tolist(),
        text=[f"{v:.2f}%" for v in avg_cagr["Avg CAGR"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg CAGR: %{y:.2f}%<extra></extra>",
        name="Avg CAGR",
    ))
    fig.add_hline(
        y=sp500_cagr,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text=f"S&P 500: {sp500_cagr:.2f}%",
        annotation_position="top right",
        annotation_font_color="black",
    )
    fig.update_layout(
        title="<b>CAGR Promedio por Estrategia vs S&P 500</b> — Verde = supera el benchmark",
        xaxis_title="Estrategia",
        yaxis_title="CAGR (%)",
        height=450,
        showlegend=False,
    )
    return fig


def generate_report(results: dict, output_path: str = "reporte.html", sp500_benchmark: dict | None = None) -> None:
    """
    Generate a self-contained interactive HTML report.

    Parameters
    ----------
    results : dict
        Nested dict as returned by backtester.run_all().
    output_path : str
        Path for the output HTML file.
    """
    print("Generating HTML report ...")
    sp500_cagr = sp500_benchmark["metrics"]["CAGR"] if sp500_benchmark else None
    sp500_equity = sp500_benchmark["equity"] if sp500_benchmark else None

    df = _build_summary_df(results, sp500_cagr=sp500_cagr)
    df = _add_composite_score(df)

    fig_table = _build_summary_table(df)
    fig_equity = _build_equity_curves(results, sp500_equity=sp500_equity)
    fig_dd = _build_drawdown_chart(results, sp500_equity=sp500_equity)
    fig_ranking = _build_ranking(df)
    fig_yearly = _build_yearly_heatmap(results)

    sections = [
        ("Strategy Ranking", fig_ranking),
    ]
    if sp500_benchmark is not None:
        sections.append(("vs S&P 500", _build_sp500_comparison(df, sp500_benchmark["metrics"])))
    sections += [
        ("Annual Returns Heatmap", fig_yearly),
        ("Performance Summary Table", fig_table),
        ("Equity Curves", fig_equity),
        ("Drawdown Analysis", fig_dd),
    ]

    sp500_subtitle = (
        f"&nbsp;|&nbsp; S&P 500 CAGR: {sp500_cagr:.2f}%"
        if sp500_cagr is not None else ""
    )

    # Concatenate all figures into one HTML
    html_parts = []

    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stock Backtesting Report</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0; }}
    h1 {{ text-align: center; color: #1a2a4a; padding: 30px 0 10px; margin: 0; }}
    .subtitle {{ text-align: center; color: #555; margin-bottom: 20px; font-size: 14px; }}
    .section {{ background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
               margin: 20px auto; max-width: 98%; padding: 10px; }}
    .section-title {{ color: #1a2a4a; font-size: 16px; font-weight: bold;
                     border-left: 4px solid #3264b4; padding-left: 10px; margin: 10px 0; }}
  </style>
</head>
<body>
  <h1>Stock Backtesting Report</h1>
  <p class="subtitle">Period: 2015-01-01 → 2024-12-31 &nbsp;|&nbsp; 9 Strategies &nbsp;|&nbsp; 10 Symbols (USA + Argentina ADRs){sp500_subtitle}</p>
""")

    for title, fig in sections:
        html_parts.append(f'  <div class="section">')
        html_parts.append(
            fig.to_html(full_html=False, include_plotlyjs="cdn" if title == "Strategy Ranking" else False)
        )
        html_parts.append("  </div>")

    html_parts.append("""
<script>
const COLUMN_TOOLTIPS = {
  "Score":            "Score 0\u201310: el Composite Score reescalado para f\u00e1cil lectura. M\u00e1s alto = mejor.",
  "Composite Score":  "Exactamente el mismo score del gr\u00e1fico 'Strategy Ranking': suma de norm(Sharpe) + norm(CAGR) + norm(\u2212MaxDD). Rango 0\u20133.",
  "Total Return": "Retorno total acumulado en el per\u00edodo completo (2015\u20132024).",
  "CAGR":         "Compound Annual Growth Rate \u2014 tasa de crecimiento anual compuesto.",
  "vs S&P500":    "Diferencia de CAGR contra el S&P 500 en puntos porcentuales (pp). Verde = supera el benchmark.",
  "Sharpe":       "Retorno medio / volatilidad total, anualizado (rf=0). M\u00e1s alto = mejor relaci\u00f3n riesgo-retorno.",
  "Sortino":      "Como Sharpe, pero penaliza solo la volatilidad negativa (downside deviation).",
  "Max Drawdown": "Ca\u00edda m\u00e1xima desde el pico hasta el valle. Refleja el peor escenario de p\u00e9rdida.",
  "Calmar":       "CAGR / |Max Drawdown|. Mide cu\u00e1nto retorno se obtiene por unidad de riesgo de ca\u00edda.",
  "Win Rate":     "Porcentaje de trades individuales cerrados con ganancia.",
  "Num Trades":   "Cantidad total de operaciones (entradas) ejecutadas en el per\u00edodo."
};

window.addEventListener('load', function () {
  setTimeout(function () {
    document.querySelectorAll('.js-plotly-plot text').forEach(function (el) {
      var key = el.textContent.trim();
      if (COLUMN_TOOLTIPS[key]) {
        var svgTitle = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        svgTitle.textContent = COLUMN_TOOLTIPS[key];
        el.appendChild(svgTitle);
        el.style.cursor = 'help';
      }
    });
  }, 1500);
});
</script>
</body></html>""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    abs_path = os.path.abspath(output_path)
    print(f"Report saved: {abs_path}")
    return abs_path

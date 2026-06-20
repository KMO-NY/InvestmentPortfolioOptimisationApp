import streamlit as st
import numpy as np
import pandas as pd

from utils import (
    load_session_state, get_stock_data, normalize_weights,
    brinson_attribution, plot_brinson_waterfall, portfolio_daily_returns,
)

st.set_page_config(page_title="Performance Attribution", page_icon="🔬", layout="wide")
st.title("Performance Attribution — Brinson Model")

st.markdown("""
The **Brinson-Hood-Beebower model** decomposes active return into three effects:

| Effect | Question it answers |
|---|---|
| **Allocation** | Did over/underweighting asset classes add value? |
| **Selection** | Did the chosen stocks outperform their benchmark equivalents? |
| **Interaction** | Combined impact of weighting and stock picking together. |
""")

# ── Load state ─────────────────────────────────────────────────────────────────
tickers, amounts, start_date, end_date = load_session_state()

if not tickers or start_date is None or end_date is None:
    st.warning("Enter your portfolio on the home page first.")
    st.stop()

try:
    stock_data, daily_returns_df = get_stock_data(tickers, start_date, end_date)
except ValueError as e:
    st.error(str(e))
    st.stop()

portfolio_weights = normalize_weights(amounts)

# ── Benchmark weights (user-adjustable) ────────────────────────────────────────
st.write("## Set Benchmark Weights")
st.caption(
    "The benchmark represents the 'passive' allocation you are measuring against. "
    "Default is equal-weight across your holdings."
)

equal_weight = np.ones(len(tickers)) / len(tickers)
benchmark_weights = []
cols = st.columns(len(tickers))
for i, (ticker, col) in enumerate(zip(tickers, cols)):
    bw = col.number_input(
        f"{ticker} benchmark weight",
        min_value=0.0, max_value=1.0,
        value=float(round(equal_weight[i], 4)),
        step=0.01, key=f"bw_{ticker}"
    )
    benchmark_weights.append(bw)

benchmark_weights = np.array(benchmark_weights)
if abs(benchmark_weights.sum() - 1.0) > 0.01:
    st.error(f"Benchmark weights sum to {benchmark_weights.sum():.2%}. They must sum to 100%.")
    st.stop()

# ── Period selection ────────────────────────────────────────────────────────────
st.write("## Attribution Period")
period = st.selectbox("Measure attribution over:", ["Full period", "Last 1 month", "Last 3 months", "Last 6 months"])

period_map = {"Full period": None, "Last 1 month": 21, "Last 3 months": 63, "Last 6 months": 126}
n_days = period_map[period]
if n_days:
    daily_returns_df = daily_returns_df.iloc[-n_days:]

# ── Compute attribution ─────────────────────────────────────────────────────────
portfolio_returns_per_asset = daily_returns_df.mean().values         # mean daily return per asset
benchmark_returns_per_asset = daily_returns_df.mean().values         # same universe; user can extend

# For a richer demo, let the user set hypothetical benchmark asset returns
st.write("### Benchmark Return Assumptions per Asset")
st.caption(
    "By default these match your portfolio's realised returns (active return = selection only). "
    "Adjust to model a different benchmark composition."
)
benchmark_asset_returns = []
cols2 = st.columns(len(tickers))
for i, (ticker, col) in enumerate(zip(tickers, cols2)):
    br = col.number_input(
        f"{ticker} benchmark return (daily)",
        value=float(round(portfolio_returns_per_asset[i], 6)),
        format="%.6f", step=0.0001, key=f"br_{ticker}"
    )
    benchmark_asset_returns.append(br)

benchmark_asset_returns = np.array(benchmark_asset_returns)

if st.button("Run Attribution"):
    attr_df = brinson_attribution(
        portfolio_weights, benchmark_weights,
        portfolio_returns_per_asset, benchmark_asset_returns,
        tickers,
    )

    # Format for display
    pct_cols = ["Portfolio Weight", "Benchmark Weight", "Portfolio Return",
                "Benchmark Return", "Allocation Effect", "Selection Effect",
                "Interaction Effect", "Total Active"]
    display_df = attr_df.copy()
    for c in pct_cols:
        display_df[c] = display_df[c].apply(lambda x: f"{x:.4%}" if isinstance(x, float) else x)

    st.write("## Attribution Results")
    st.dataframe(display_df, use_container_width=True)

    # Summary metrics
    totals = attr_df[attr_df["Asset"] == "TOTAL"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Active Return", f"{totals['Total Active']:.4%}")
    c2.metric("Allocation Effect", f"{totals['Allocation Effect']:.4%}")
    c3.metric("Selection Effect", f"{totals['Selection Effect']:.4%}")
    c4.metric("Interaction Effect", f"{totals['Interaction Effect']:.4%}")

    # Chart
    fig = plot_brinson_waterfall(attr_df)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Interpreting the results"):
        st.markdown("""
- A **positive allocation effect** means you overweighted sectors/assets that outperformed the benchmark.
- A **positive selection effect** means your chosen stocks beat their benchmark counterparts.
- **Interaction** captures whether your tilts and picks reinforced each other.
- A skilled active manager typically shows consistent positive selection effect over time.
        """)
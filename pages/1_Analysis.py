import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_session_state, get_stock_data, get_bond_data,
    normalize_weights, generate_efficient_frontier,
    plot_efficient_frontier, tabulate_portfolio_info,
    suggested_portfolio_split, risk_metrics_summary,
    build_multi_asset_frontier, plot_multi_asset_frontier,
    portfolio_daily_returns, BOND_ETF_OPTIONS,
)

st.set_page_config(page_title="Analysis", page_icon="📈", layout="wide")
st.title("Portfolio Analysis & Optimisation")

# ── Load state ─────────────────────────────────────────────────────────────────
tickers, amounts, start_date, end_date = load_session_state()

if not tickers or start_date is None or end_date is None:
    st.warning("Go back to the home page and enter your portfolio first.")
    st.stop()

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (annual, e.g. 0.083 for 8.3%):", value=0.083, step=0.001, format="%.3f"
)

# ── Fetch data ─────────────────────────────────────────────────────────────────
try:
    stock_data, daily_returns_df = get_stock_data(tickers, start_date, end_date)
except ValueError as e:
    st.error(str(e))
    st.stop()

weights_raw = normalize_weights(amounts)
mean_returns = daily_returns_df.mean()
cov_matrix = daily_returns_df.cov()
trading_days = len(stock_data)

# ── Efficient Frontier ─────────────────────────────────────────────────────────
st.write("## Efficient Frontier")
with st.spinner("Simulating portfolios..."):
    results, weights_record = generate_efficient_frontier(
        mean_returns, cov_matrix, risk_free_rate=risk_free_rate
    )

fig, max_sharpe_idx, min_vol_idx, max_sortino_idx = plot_efficient_frontier(
    results, weights_record, current_weights=weights_raw,
    mean_returns=mean_returns, cov_matrix=cov_matrix,
)
st.plotly_chart(fig, use_container_width=True)

# ── Portfolio comparison table ─────────────────────────────────────────────────
st.write("## Optimised Portfolio Comparison")
portfolio_table = tabulate_portfolio_info(
    mean_returns, cov_matrix, max_sharpe_idx, max_sortino_idx,
    min_vol_idx, weights_record, tickers, risk_free_rate
)
st.dataframe(portfolio_table, use_container_width=True)

# Portfolio selector
st.session_state["suggested_portfolio"] = st.selectbox(
    "View allocation breakdown for:", portfolio_table["Portfolio Type"].tolist()
)
suggested_portfolio_split(portfolio_table, tickers)

# ── Risk Metrics Dashboard ─────────────────────────────────────────────────────
st.write("## Risk Metrics: Your Current Portfolio")
st.caption("Based on your entered weights — not the optimised portfolios above.")

metrics = risk_metrics_summary(weights_raw, daily_returns_df, risk_free_rate)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Annualised Return", f"{metrics['Annualised Return']:.2%}")
col2.metric("Annualised Volatility", f"{metrics['Annualised Volatility']:.2%}")
col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
col4.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")

col5, col6, col7 = st.columns(3)
col5.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
col6.metric("VaR (95%)", f"{metrics['VaR (95%)']:.2%}")
col7.metric("CVaR (95%)", f"{metrics['CVaR (95%)']:.2%}")

with st.expander("What do these metrics mean?"):
    st.markdown("""
| Metric | What it measures |
|---|---|
| **Sharpe Ratio** | Return earned per unit of total risk. Higher is better. |
| **Sortino Ratio** | Like Sharpe, but only penalises *downside* volatility. More relevant for investors. |
| **Max Drawdown** | Largest peak-to-trough loss in the period. Shows worst-case historical experience. |
| **VaR (95%)** | On the worst 5% of days, losses exceeded this threshold. |
| **CVaR (95%)** | The *average* loss on those worst 5% of days. More informative than VaR alone. |
    """)

# ── Multi-asset frontier ───────────────────────────────────────────────────────
bond_etf_ticker = st.session_state.get("bond_etf")
if bond_etf_ticker:
    st.write("## Multi-Asset Frontier (Equity + Fixed Income)")
    st.caption(f"Bond ETF: {bond_etf_ticker}")
    try:
        _, bond_returns = get_bond_data(bond_etf_ticker, start_date, end_date)
        equity_port_returns = portfolio_daily_returns(weights_raw, daily_returns_df)

        # Align on common dates
        common = equity_port_returns.index.intersection(bond_returns.index)
        eq_ret = equity_port_returns.loc[common]
        bo_ret = bond_returns.loc[common]

        # Equity-only frontier as scatter (use results from simulation)
        equity_frontier_df = pd.DataFrame({
            "Volatility": results[0],
            "Return": results[1],
        })
        multi_df = build_multi_asset_frontier(eq_ret, bo_ret)
        fig_ma = plot_multi_asset_frontier(equity_frontier_df, multi_df)
        st.plotly_chart(fig_ma, use_container_width=True)

        # Best multi-asset allocation
        best_idx = multi_df["Sharpe"].idxmax()
        best = multi_df.loc[best_idx]
        st.info(
            f"Highest Sharpe in multi-asset space: **{best['Sharpe']:.2f}** "
            f"at **{best['Equity Weight']:.0%} equity / {best['Bond Weight']:.0%} bonds**. "
            f"Return: {best['Return']:.2%}, Volatility: {best['Volatility']:.2%}."
        )
    except ValueError as e:
        st.error(f"Could not load bond ETF data: {e}")
else:
    st.info("Select a Bond ETF on the home page to enable multi-asset frontier analysis.")
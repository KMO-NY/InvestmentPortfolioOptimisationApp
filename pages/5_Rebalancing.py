import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_session_state, get_stock_data, normalize_weights,
    simulate_rebalancing, plot_rebalancing_comparison,
)

st.set_page_config(page_title="Rebalancing Simulation", page_icon="⚖️", layout="wide")
st.title("Rebalancing Simulation")
st.caption("Models the impact of rebalancing frequency and transaction costs on portfolio value.")

# ── Load state ─────────────────────────────────────────────────────────────────
tickers, amounts, start_date, end_date = load_session_state()

if not tickers or start_date is None or end_date is None:
    st.warning("Enter your portfolio on the home page first.")
    st.stop()

try:
    stock_data, _ = get_stock_data(tickers, start_date, end_date)
except ValueError as e:
    st.error(str(e))
    st.stop()

weights = normalize_weights(amounts)

# ── Parameters ─────────────────────────────────────────────────────────────────
st.write("## Simulation Parameters")

col1, col2, col3 = st.columns(3)
initial_investment = col1.number_input("Initial Investment (R):", value=100_000, step=10_000)
transaction_cost = col2.number_input(
    "Transaction Cost (% per trade):", value=0.5, min_value=0.0, max_value=5.0, step=0.05
) / 100
rebalance_freq = col3.selectbox("Rebalancing Frequency:", ["monthly", "quarterly", "annually"])

# ── Run simulation ─────────────────────────────────────────────────────────────
if st.button("Run Simulation"):
    with st.spinner("Running simulation..."):

        # Comparison chart
        fig = plot_rebalancing_comparison(stock_data, weights, initial_investment)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed output for selected frequency
        port_vals, total_cost, rebalance_dates = simulate_rebalancing(
            stock_data, weights,
            rebalance_frequency=rebalance_freq,
            transaction_cost_pct=transaction_cost,
            initial_investment=initial_investment,
        )

        # Buy-and-hold for comparison
        daily_rets = stock_data.pct_change().dropna()
        port_daily = (daily_rets * weights).sum(axis=1)
        bah_value = initial_investment * (1 + port_daily).cumprod().iloc[-1]

        final_value = port_vals.iloc[-1]
        cost_drag = total_cost / initial_investment

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Portfolio Value", f"R{final_value:,.0f}")
        c2.metric("Buy & Hold Value", f"R{bah_value:,.0f}")
        c3.metric("Total Transaction Costs", f"R{total_cost:,.0f}")
        c4.metric("Cost as % of Initial Capital", f"{cost_drag:.2%}")

        st.write(f"### Rebalance Dates ({rebalance_freq.capitalize()})")
        st.write(f"Portfolio was rebalanced **{len(rebalance_dates)} times** over the period.")

        # Transaction cost sensitivity sweep
        st.write("## Cost Sensitivity Analysis")
        st.caption("How total transaction costs change with different cost rates.")

        cost_rates = np.linspace(0, 0.02, 40)
        cost_totals = []
        for rate in cost_rates:
            _, tc, _ = simulate_rebalancing(
                stock_data, weights,
                rebalance_frequency=rebalance_freq,
                transaction_cost_pct=rate,
                initial_investment=initial_investment,
            )
            cost_totals.append(tc)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=cost_rates * 100, y=cost_totals,
            mode="lines", fill="tozeroy",
            line=dict(color="steelblue", width=2),
            fillcolor="rgba(70, 130, 180, 0.15)",
        ))
        fig2.update_layout(
            title=f"Total Costs vs Transaction Cost Rate ({rebalance_freq.capitalize()} rebalancing)",
            xaxis_title="Transaction Cost Rate (%)",
            yaxis_title="Total Costs (R)",
            yaxis_tickprefix="R",
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Methodology note"):
            st.markdown("""
The simulation:
1. Starts with your specified initial investment split by your portfolio weights.
2. On each rebalance date, calculates the trades needed to restore target weights.
3. Applies transaction costs as a percentage of the absolute value of each trade.
4. Reinvests the remainder at target weights.

Transaction costs here model brokerage and market impact. Typical JSE transaction costs range from 0.2% to 0.6% per trade, plus VAT and securities transfer tax on equity purchases (0.25%).
            """)
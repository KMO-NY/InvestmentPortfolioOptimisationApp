import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_session_state, get_stock_data,
    normalize_weights, portfolio_daily_returns,
)

st.set_page_config(page_title="Monte Carlo Simulation", page_icon="🎲", layout="wide")
st.title("Monte Carlo Simulation & Stress Testing")
st.caption("Projects portfolio outcomes across thousands of simulated paths under different market conditions.")

# ── Load state ─────────────────────────────────────────────────────────────────
ticker, amounts, start_date, end_date = load_session_state()

if not ticker or amounts is None:
    st.error("Add your portfolio on the home page first.")
    st.stop()
if start_date is None or end_date is None:
    st.error("Set a date range on the home page first.")
    st.stop()

try:
    stock_data, daily_returns_df = get_stock_data(ticker, start_date, end_date)
except Exception as e:
    st.error(f"Error fetching stock data: {e}")
    st.stop()

weights = normalize_weights(amounts)
port_daily_rets = portfolio_daily_returns(weights, daily_returns_df)

# ── Simulation parameters ──────────────────────────────────────────────────────
st.write("## Simulation Parameters")

col1, col2, col3, col4 = st.columns(4)
initial_investment = col1.number_input("Initial Investment (R):", value=100_000, step=10_000)
horizon_days = col2.number_input("Horizon (trading days):", value=252, step=21, min_value=21)
n_simulations = col3.number_input("Number of Simulations:", value=1000, step=500, min_value=100, max_value=10000)
target_value = col4.number_input("Target Portfolio Value (R):", value=120_000, step=5_000)

# ── Stress scenario selector ───────────────────────────────────────────────────
st.write("## Stress Scenario (Optional)")
st.caption("Applies a shock to the historical return and volatility before simulating.")

scenario = st.selectbox("Select a scenario:", [
    "None (use historical parameters)",
    "Market Crash (-30% return shock)",
    "High Volatility (2x volatility)",
    "Interest Rate Shock (-15% return, +50% volatility)",
    "Sector Decline (-20% return shock)",
    "Custom",
])

# Base parameters from historical data
base_mean = float(port_daily_rets.mean())
base_std = float(port_daily_rets.std())

if scenario == "None (use historical parameters)":
    sim_mean = base_mean
    sim_std = base_std

elif scenario == "Market Crash (-30% return shock)":
    sim_mean = base_mean - (0.30 / 252)
    sim_std = base_std * 1.5

elif scenario == "High Volatility (2x volatility)":
    sim_mean = base_mean
    sim_std = base_std * 2.0

elif scenario == "Interest Rate Shock (-15% return, +50% volatility)":
    sim_mean = base_mean - (0.15 / 252)
    sim_std = base_std * 1.5

elif scenario == "Sector Decline (-20% return shock)":
    sim_mean = base_mean - (0.20 / 252)
    sim_std = base_std * 1.2

elif scenario == "Custom":
    c1, c2 = st.columns(2)
    return_shock = c1.number_input(
        "Annual return shock (e.g. -0.20 for -20%):",
        value=0.0, step=0.01, format="%.2f"
    )
    vol_multiplier = c2.number_input(
        "Volatility multiplier (e.g. 1.5 for +50% vol):",
        value=1.0, step=0.1, min_value=0.1
    )
    sim_mean = base_mean + (return_shock / 252)
    sim_std = base_std * vol_multiplier

# ── Run simulation ─────────────────────────────────────────────────────────────
if st.button("Run Simulation"):
    with st.spinner(f"Running {n_simulations:,} simulations over {horizon_days} days..."):

        # Simulate paths
        daily_returns_sim = np.random.normal(
            sim_mean, sim_std,
            size=(int(horizon_days), int(n_simulations))
        )
        price_paths = initial_investment * np.cumprod(1 + daily_returns_sim, axis=0)

        # Final values
        final_values = price_paths[-1, :]
        percentiles = {
            "5th": np.percentile(final_values, 5),
            "25th": np.percentile(final_values, 25),
            "50th": np.percentile(final_values, 50),
            "75th": np.percentile(final_values, 75),
            "95th": np.percentile(final_values, 95),
        }

        # ── Summary metrics ────────────────────────────────────────────────────
        prob_profit = float(np.mean(final_values > initial_investment)) * 100
        prob_target = float(np.mean(final_values >= target_value)) * 100
        prob_loss_20 = float(np.mean(final_values < initial_investment * 0.80)) * 100
        expected_final = float(np.mean(final_values))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected Final Value", f"R{expected_final:,.0f}")
        c2.metric("Probability of Profit", f"{prob_profit:.1f}%")
        c3.metric(f"Probability of Reaching R{target_value:,.0f}", f"{prob_target:.1f}%")
        c4.metric("Probability of >20% Loss", f"{prob_loss_20:.1f}%")

        # ── Simulation paths chart ─────────────────────────────────────────────
        st.write("## Simulated Portfolio Paths")

        fig = go.Figure()

        # Plot a sample of paths (plotting all is slow)
        sample_size = min(200, int(n_simulations))
        sample_indices = np.random.choice(int(n_simulations), sample_size, replace=False)

        for i in sample_indices:
            fig.add_trace(go.Scatter(
                y=price_paths[:, i],
                mode="lines",
                line=dict(width=0.5, color="steelblue"),
                opacity=0.15,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Percentile bands
        for pct_label, pct_val, color in [
            ("95th percentile", np.percentile(price_paths, 95, axis=1), "green"),
            ("75th percentile", np.percentile(price_paths, 75, axis=1), "lightgreen"),
            ("Median (50th)", np.percentile(price_paths, 50, axis=1), "white"),
            ("25th percentile", np.percentile(price_paths, 25, axis=1), "orange"),
            ("5th percentile", np.percentile(price_paths, 5, axis=1), "red"),
        ]:
            fig.add_trace(go.Scatter(
                y=pct_val,
                mode="lines",
                line=dict(width=2, color=color),
                name=pct_label,
            ))

        # Initial investment line
        fig.add_hline(
            y=initial_investment,
            line_dash="dash", line_color="gray",
            annotation_text="Initial Investment",
        )

        # Target line
        fig.add_hline(
            y=target_value,
            line_dash="dot", line_color="gold",
            annotation_text=f"Target: R{target_value:,.0f}",
        )

        fig.update_layout(
            xaxis_title="Trading Days",
            yaxis_title="Portfolio Value (R)",
            yaxis_tickprefix="R",
            height=500,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Distribution of final values ───────────────────────────────────────
        st.write("## Distribution of Final Portfolio Values")

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=final_values,
            nbinsx=80,
            marker_color="steelblue",
            opacity=0.75,
            name="Final Values",
        ))

        # Percentile markers
        for label, val, color in [
            ("5th pct", percentiles["5th"], "red"),
            ("Median", percentiles["50th"], "white"),
            ("95th pct", percentiles["95th"], "green"),
            ("Target", target_value, "gold"),
        ]:
            fig2.add_vline(
                x=val, line_dash="dash", line_color=color,
                annotation_text=label, annotation_position="top",
            )

        fig2.update_layout(
            xaxis_title="Final Portfolio Value (R)",
            yaxis_title="Frequency",
            xaxis_tickprefix="R",
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Percentile table ───────────────────────────────────────────────────
        st.write("## Outcome Percentiles")

        pct_df = pd.DataFrame({
            "Percentile": ["5th (worst case)", "25th", "50th (median)", "75th", "95th (best case)"],
            "Final Value": [f"R{v:,.0f}" for v in percentiles.values()],
            "Return": [f"{(v / initial_investment - 1):.2%}" for v in percentiles.values()],
        })
        st.dataframe(pct_df, use_container_width=True, hide_index=True)

        with st.expander("Methodology"):
            st.markdown(f"""
**Parameters used:**
- Daily mean return: `{sim_mean:.6f}` (annualised: `{sim_mean * 252:.2%}`)
- Daily volatility: `{sim_std:.6f}` (annualised: `{sim_std * np.sqrt(252):.2%}`)
- Scenario applied: `{scenario}`

**Method:** Geometric Brownian Motion with normally distributed daily returns drawn from the historical mean and volatility of your weighted portfolio. Each simulation is an independent path.

**Limitation:** GBM assumes normally distributed returns. Equity returns exhibit fat tails and negative skew, so the model underestimates the probability of extreme losses. CVaR on the Analysis page gives a more conservative tail risk estimate based on actual historical returns.
            """)
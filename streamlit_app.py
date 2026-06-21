import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf

from utils import (
    validate_exchange_symbols, initialize_session_state,
    clear_button_clicked, exchanges, zar,
    BOND_ETF_OPTIONS,
)

st.set_page_config(page_title="Portfolio Intelligence", page_icon="📊", layout="wide")

st.title("Investment Portfolio Optimisation App / Intelligence Tool")
st.caption("Built on Modern Portfolio Theory | Brinson Attribution | Institutional Risk Metrics")

st.sidebar.header("Navigation")
st.sidebar.write("Enter your portfolio here, then use the pages to analyse it.")

initialize_session_state()

# ── Clear button ───────────────────────────────────────────────────────────────
_, clear_col = st.columns([4, 1])
if clear_col.button("**:red[Clear Data]**"):
    clear_button_clicked()

# ── Exchange validation (cached) ───────────────────────────────────────────────
country_suffix_map = validate_exchange_symbols(exchanges)
selected_country = st.selectbox("Select Your Country:", options=["South Africa"])
selected_suffix = ".JO"

# ── Date range ─────────────────────────────────────────────────────────────────
date_col1, date_col2 = st.columns(2)
start_date = date_col1.date_input("From:", format="DD/MM/YYYY",
                                   value=st.session_state.get("start_date", None))
end_date = date_col2.date_input("To:", format="DD/MM/YYYY",
                                 value=st.session_state.get("end_date", None))

if start_date:
    st.session_state["start_date"] = start_date
if end_date:
    st.session_state["end_date"] = end_date

# ── Fixed income option ────────────────────────────────────────────────────────
st.write("### Fixed Income Allocation (Optional)")
st.caption("Adding a bond ETF extends the efficient frontier analysis to a multi-asset view.")
selected_bond = st.selectbox("Add a Bond ETF to your analysis:", options=list(BOND_ETF_OPTIONS.keys()))
st.session_state["bond_etf"] = BOND_ETF_OPTIONS[selected_bond]

# ── Stock input ────────────────────────────────────────────────────────────────
st.write("### Add Stocks and Amount Invested")

with st.form("add_stock_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    ticker = col1.text_input("Ticker symbol (without exchange suffix):")
    amount = col2.number_input("Amount Invested (R):", min_value=0.0, step=1.0, format="%.2f")
    if st.form_submit_button("Add Stock"):
        if ticker and amount > 0:
            st.session_state["stock_list"].append({
                "Ticker": ticker.upper(), "Amount Invested": amount
            })
            st.success(f"Added {ticker.upper()}{selected_suffix} — {zar}{amount:,.2f}")
        else:
            st.error("Enter a valid ticker and a positive amount.")

# ── Portfolio table ────────────────────────────────────────────────────────────
if st.session_state["stock_list"]:
    st.write("### Your Current Portfolio")
    st.session_state["portfolio_df"] = pd.DataFrame(st.session_state["stock_list"])
    st.dataframe(st.session_state["portfolio_df"], use_container_width=True)

    if st.button("Load Stock Data"):
        portfolio_df = st.session_state["portfolio_df"]
        fig = px.pie(
            portfolio_df, values="Amount Invested", names="Ticker",
            title="Portfolio Allocation by Amount Invested", hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Head to the Analysis pages in the sidebar to run optimisation and risk analysis.")
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# ─── Constants ────────────────────────────────────────────────────────────────

exchanges = [
    {"country": "South Africa", "symbol": "AGL.JO", "suffix": ".JO"},
]

currency_list = [
    {"country": "South Africa", "currency": "R"}
]

zar = currency_list[0]["currency"]

BENCHMARK_OPTIONS = {
    "S&P 500": "^GSPC",
    "JSE All Share (ALSI)": "^J203.JO",
    "MSCI World": "URTH",
    "Nasdaq 100": "^NDX",
}

BOND_ETF_OPTIONS = {
    "None": None,
    "iShares Global Govt Bond ETF (IGLO)": "IGLO.L",
    "Vanguard Total Bond Market ETF (BND)": "BND",
    "NewFunds GOVI ETF (JSE)": "GOVI.JO",
}

# ─── Session State ─────────────────────────────────────────────────────────────

def initialize_session_state():
    defaults = {
        "stock_list": [],
        "portfolio_df": pd.DataFrame(columns=["Ticker", "Amount Invested"]),
        "start_date": None,
        "end_date": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_session_state():
    ticker, weights, start_date, end_date = [], [], None, None
    if "portfolio_df" in st.session_state and not st.session_state["portfolio_df"].empty:
        portfolio_df = st.session_state["portfolio_df"]
        ticker = portfolio_df["Ticker"].tolist()
        weights = portfolio_df["Amount Invested"]
    if "start_date" in st.session_state:
        start_date = st.session_state["start_date"]
    if "end_date" in st.session_state:
        end_date = st.session_state["end_date"]
    return ticker, weights, start_date, end_date


def clear_button_clicked():
    st.session_state["stock_list"] = []
    st.session_state["portfolio_df"] = pd.DataFrame(columns=["Ticker", "Amount Invested"])
    st.session_state["start_date"] = None
    st.session_state["end_date"] = None

# ─── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Validating exchange symbols...")
def validate_exchange_symbols(exchange_list):
    country_suffix_map = {}
    for exchange in exchange_list:
        country = exchange["country"]
        symbol = exchange["symbol"]
        suffix = exchange["suffix"]
        try:
            ticker = yf.Ticker(symbol)
            if "longName" in ticker.info:
                country_suffix_map[country] = suffix
        except Exception:
            pass
    return country_suffix_map


@st.cache_data(show_spinner="Fetching stock data...")
def get_stock_data(stock_list, start_date, end_date, suffix=".JO"):
    """
    Download adjusted close prices and compute daily returns.
    Appends the exchange suffix to each ticker.
    """
    stocks = [s + suffix for s in stock_list]
    try:
        raw = yf.download(stocks, start=start_date, end=end_date, auto_adjust=True)
        stock_data = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(stock_data, pd.Series):
            stock_data = stock_data.to_frame(name=stocks[0])
        stock_data.columns = stock_list          # strip suffix for display
        daily_returns_df = stock_data.pct_change().dropna()
        return stock_data, daily_returns_df
    except Exception as e:
        raise ValueError(f"Error fetching stock data: {e}")


@st.cache_data(show_spinner="Fetching benchmark data...")
def get_benchmark_data(benchmark_ticker, start_date, end_date):
    """Download benchmark close prices and daily returns."""
    try:
        raw = yf.download(benchmark_ticker, start=start_date, end=end_date, auto_adjust=True)
        prices = raw["Close"]
        returns = prices.pct_change().dropna()
        return prices, returns
    except Exception as e:
        raise ValueError(f"Error fetching benchmark data: {e}")


@st.cache_data(show_spinner="Fetching bond ETF data...")
def get_bond_data(bond_ticker, start_date, end_date):
    """Download bond ETF prices and returns."""
    try:
        raw = yf.download(bond_ticker, start=start_date, end=end_date, auto_adjust=True)
        prices = raw["Close"]
        returns = prices.pct_change().dropna()
        return prices, returns
    except Exception as e:
        raise ValueError(f"Error fetching bond ETF data: {e}")

# ─── Portfolio Calculations ────────────────────────────────────────────────────

def normalize_weights(amounts):
    """Convert rand amounts to portfolio weight fractions."""
    amounts = np.array(amounts, dtype=float)
    return amounts / amounts.sum()


def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0, trading_days=252):
    """
    Return annualised portfolio return, volatility, and Sharpe ratio.
    BUG FIX: original subtracted volatility instead of dividing.
    """
    weights = np.array(weights)
    port_return = np.dot(weights, mean_returns) * trading_days
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0.0
    return port_return, port_volatility, sharpe


def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=0.0, trading_days=252):
    """
    Simulate random portfolios and return a results array plus weights list.
    results rows: [volatility, return, sharpe, sortino]
    """
    num_assets = len(mean_returns)
    results = np.zeros((4, num_portfolios))
    weights_record = []

    daily_rf = risk_free_rate / trading_days

    for i in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= w.sum()
        weights_record.append(w)

        port_return, port_vol, sharpe = calculate_portfolio_performance(
            w, mean_returns, cov_matrix, risk_free_rate, trading_days
        )
        sortino = calculate_sortino_ratio(w, mean_returns, cov_matrix, risk_free_rate, trading_days)

        results[0, i] = port_vol
        results[1, i] = port_return
        results[2, i] = sharpe
        results[3, i] = sortino

    return results, weights_record


def plot_efficient_frontier(results, weights_record, current_weights=None,
                            mean_returns=None, cov_matrix=None, trading_days=252):
    """
    Interactive Plotly efficient frontier.
    Highlights max Sharpe, min volatility, and max Sortino portfolios.
    Optionally overlays the user's current portfolio.
    """
    vols = results[0]
    rets = results[1]
    sharpes = results[2]
    sortinos = results[3]

    max_sharpe_idx = int(np.argmax(sharpes))
    min_vol_idx = int(np.argmin(vols))
    max_sortino_idx = int(np.argmax(sortinos))

    fig = go.Figure()

    # Scatter cloud
    fig.add_trace(go.Scatter(
        x=vols, y=rets,
        mode="markers",
        marker=dict(color=sharpes, colorscale="Viridis", size=4,
                    colorbar=dict(title="Sharpe Ratio")),
        name="Simulated Portfolios",
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    # Key portfolios
    for idx, label, color in [
        (max_sharpe_idx, "Max Sharpe", "red"),
        (min_vol_idx, "Min Volatility", "blue"),
        (max_sortino_idx, "Max Sortino", "green"),
    ]:
        fig.add_trace(go.Scatter(
            x=[vols[idx]], y=[rets[idx]],
            mode="markers",
            marker=dict(symbol="star", size=16, color=color),
            name=label,
        ))

    # Current portfolio overlay
    if current_weights is not None and mean_returns is not None and cov_matrix is not None:
        cp_ret, cp_vol, _ = calculate_portfolio_performance(
            current_weights, mean_returns, cov_matrix, trading_days=trading_days
        )
        fig.add_trace(go.Scatter(
            x=[cp_vol], y=[cp_ret],
            mode="markers",
            marker=dict(symbol="diamond", size=14, color="orange"),
            name="Your Portfolio",
        ))

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualised Volatility (Risk)",
        yaxis_title="Annualised Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
    )
    return fig, max_sharpe_idx, min_vol_idx, max_sortino_idx


def tabulate_portfolio_info(mean_returns, cov_matrix, max_sharpe_idx,
                             max_sortino_idx, min_vol_idx, weights_record, tickers,
                             risk_free_rate=0.0, trading_days=252):
    """Return a DataFrame comparing key optimised portfolios."""
    portfolios = {
        "Max Sharpe Ratio": weights_record[max_sharpe_idx],
        "Max Sortino Ratio": weights_record[max_sortino_idx],
        "Min Volatility": weights_record[min_vol_idx],
    }
    rows = []
    for name, w in portfolios.items():
        ret, vol, sharpe = calculate_portfolio_performance(
            w, mean_returns, cov_matrix, risk_free_rate, trading_days
        )
        sortino = calculate_sortino_ratio(w, mean_returns, cov_matrix, risk_free_rate, trading_days)
        row = {
            "Portfolio Type": name,
            "Return": f"{ret:.2%}",
            "Volatility": f"{vol:.2%}",
            "Sharpe": f"{sharpe:.2f}",
            "Sortino": f"{sortino:.2f}",
        }
        for t, wt in zip(tickers, w):
            row[t] = f"{wt:.2%}"
        rows.append(row)
    return pd.DataFrame(rows)


def suggested_portfolio_split(portfolio_table, tickers):
    """Pie chart for the selected portfolio type."""
    if "suggested_portfolio" not in st.session_state or not st.session_state["suggested_portfolio"]:
        st.error("Please select a portfolio type.")
        return
    selected = st.session_state["suggested_portfolio"]
    row = portfolio_table[portfolio_table["Portfolio Type"] == selected]
    if not row.empty:
        weights = [float(row.iloc[0][t].strip("%")) for t in tickers]
        fig = px.pie(names=tickers, values=weights,
                     title=f"Portfolio Breakdown: {selected}", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"No data for: {selected}")

# ─── Risk Metrics ──────────────────────────────────────────────────────────────

def calculate_sortino_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0,
                             trading_days=252, n_simulations=10000):
    """
    Sortino ratio uses downside deviation instead of total volatility.
    Simulates return distribution from the mean/cov to estimate downside.
    """
    weights = np.array(weights)
    port_return = np.dot(weights, mean_returns) * trading_days

    # Simulate daily portfolio returns
    sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_simulations)
    port_sim = sim_returns @ weights
    downside = port_sim[port_sim < (risk_free_rate / trading_days)]

    if len(downside) == 0:
        return np.inf
    downside_std = np.std(downside) * np.sqrt(trading_days)
    return (port_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0


def calculate_max_drawdown(portfolio_values):
    """
    Maximum drawdown from a time series of portfolio values.
    Returns the drawdown as a negative fraction.
    """
    roll_max = portfolio_values.cummax()
    drawdown = (portfolio_values - roll_max) / roll_max
    return drawdown.min()


def calculate_cvar(daily_returns, confidence_level=0.95):
    """
    Conditional Value at Risk (Expected Shortfall) at given confidence level.
    Returns the mean loss in the worst (1 - confidence_level) of days.
    """
    var_threshold = np.percentile(daily_returns, (1 - confidence_level) * 100)
    cvar = daily_returns[daily_returns <= var_threshold].mean()
    return cvar


def calculate_var(daily_returns, confidence_level=0.95):
    """Value at Risk at the given confidence level."""
    return np.percentile(daily_returns, (1 - confidence_level) * 100)


def portfolio_daily_returns(weights, daily_returns_df):
    """Compute weighted daily portfolio return series."""
    return (daily_returns_df * weights).sum(axis=1)


def risk_metrics_summary(weights, daily_returns_df, risk_free_rate=0.0, trading_days=252):
    """
    Return a dict of institutional-grade risk metrics for a given weight vector.
    """
    port_rets = portfolio_daily_returns(weights, daily_returns_df)
    mean_returns = daily_returns_df.mean()
    cov_matrix = daily_returns_df.cov()

    ann_return, ann_vol, sharpe = calculate_portfolio_performance(
        weights, mean_returns, cov_matrix, risk_free_rate, trading_days
    )
    sortino = calculate_sortino_ratio(weights, mean_returns, cov_matrix, risk_free_rate, trading_days)

    cumulative = (1 + port_rets).cumprod()
    max_dd = calculate_max_drawdown(cumulative)
    cvar_95 = calculate_cvar(port_rets, 0.95)
    var_95 = calculate_var(port_rets, 0.95)

    return {
        "Annualised Return": ann_return,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "CVaR (95%)": cvar_95,
        "VaR (95%)": var_95,
    }

# ─── Multi-Asset (Fixed Income Extension) ─────────────────────────────────────

def build_multi_asset_frontier(equity_returns, bond_returns, num_portfolios=3000,
                                risk_free_rate=0.0, trading_days=252):
    """
    Combine equity portfolio returns and a bond ETF return series.
    Sweeps the equity/bond allocation from 0% to 100% bonds.
    Returns a DataFrame of [equity_weight, bond_weight, return, volatility, sharpe].
    """
    rows = []
    for bond_alloc in np.linspace(0, 1, num_portfolios):
        eq_alloc = 1 - bond_alloc
        blended = eq_alloc * equity_returns + bond_alloc * bond_returns
        ann_ret = blended.mean() * trading_days
        ann_vol = blended.std() * np.sqrt(trading_days)
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        rows.append({
            "Equity Weight": eq_alloc,
            "Bond Weight": bond_alloc,
            "Return": ann_ret,
            "Volatility": ann_vol,
            "Sharpe": sharpe,
        })
    return pd.DataFrame(rows)


def plot_multi_asset_frontier(equity_frontier_df, multi_asset_df):
    """
    Overlay equity-only frontier and multi-asset frontier on one Plotly chart.
    Shows how adding bonds shifts the frontier.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_frontier_df["Volatility"],
        y=equity_frontier_df["Return"],
        mode="markers",
        marker=dict(color="steelblue", size=4, opacity=0.5),
        name="Equity-Only Frontier",
    ))

    fig.add_trace(go.Scatter(
        x=multi_asset_df["Volatility"],
        y=multi_asset_df["Return"],
        mode="lines",
        line=dict(color="darkorange", width=2),
        name="Multi-Asset Frontier (Equity + Bond ETF)",
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title="Efficient Frontier: Equity-Only vs Multi-Asset",
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

# ─── Brinson Performance Attribution ──────────────────────────────────────────

def brinson_attribution(portfolio_weights, benchmark_weights, portfolio_returns,
                         benchmark_returns, tickers):
    """
    Brinson-Hood-Beebower single-period attribution.

    Allocation Effect = (wp - wb) * (rb - R_b)
    Selection Effect  = wb * (rp - rb)
    Interaction       = (wp - wb) * (rp - rb)
    Total Active      = Allocation + Selection + Interaction

    Parameters
    ----------
    portfolio_weights  : array-like, weight per asset in portfolio
    benchmark_weights  : array-like, weight per asset in benchmark (equal-weight default)
    portfolio_returns  : array-like, period return per asset in portfolio
    benchmark_returns  : array-like, period return per asset in benchmark
    tickers            : list of asset names

    Returns
    -------
    DataFrame with per-asset attribution and totals row.
    """
    wp = np.array(portfolio_weights)
    wb = np.array(benchmark_weights)
    rp = np.array(portfolio_returns)
    rb = np.array(benchmark_returns)

    R_b = np.dot(wb, rb)   # benchmark total return

    allocation = (wp - wb) * (rb - R_b)
    selection = wb * (rp - rb)
    interaction = (wp - wb) * (rp - rb)
    total = allocation + selection + interaction

    df = pd.DataFrame({
        "Asset": tickers,
        "Portfolio Weight": wp,
        "Benchmark Weight": wb,
        "Portfolio Return": rp,
        "Benchmark Return": rb,
        "Allocation Effect": allocation,
        "Selection Effect": selection,
        "Interaction Effect": interaction,
        "Total Active": total,
    })

    # Totals row
    totals = pd.DataFrame([{
        "Asset": "TOTAL",
        "Portfolio Weight": wp.sum(),
        "Benchmark Weight": wb.sum(),
        "Portfolio Return": np.dot(wp, rp),
        "Benchmark Return": R_b,
        "Allocation Effect": allocation.sum(),
        "Selection Effect": selection.sum(),
        "Interaction Effect": interaction.sum(),
        "Total Active": total.sum(),
    }])

    return pd.concat([df, totals], ignore_index=True)


def plot_brinson_waterfall(attribution_df):
    """Waterfall bar chart of allocation vs selection vs interaction effects."""
    df = attribution_df[attribution_df["Asset"] != "TOTAL"].copy()

    fig = go.Figure()
    for col, color in [
        ("Allocation Effect", "#2196F3"),
        ("Selection Effect", "#4CAF50"),
        ("Interaction Effect", "#FF9800"),
    ]:
        fig.add_trace(go.Bar(
            name=col,
            x=df["Asset"],
            y=df[col],
            marker_color=color,
        ))

    fig.update_layout(
        barmode="group",
        title="Brinson Attribution: Allocation vs Selection vs Interaction",
        xaxis_title="Asset",
        yaxis_title="Contribution to Active Return",
        yaxis_tickformat=".2%",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

# ─── Benchmark Comparison ──────────────────────────────────────────────────────

def calculate_tracking_error(portfolio_daily_rets, benchmark_daily_rets, trading_days=252):
    """
    Tracking error = annualised std of active return (portfolio - benchmark).
    Aligns series on common dates.
    """
    active = (portfolio_daily_rets - benchmark_daily_rets).dropna()
    return float(active.std()) * np.sqrt(trading_days)


def calculate_information_ratio(portfolio_daily_rets, benchmark_daily_rets, trading_days=252):
    """Information Ratio = Active Return / Tracking Error."""
    active = (portfolio_daily_rets - benchmark_daily_rets).dropna()
    active_return = active.mean() * trading_days
    te = active.std() * np.sqrt(trading_days)
    return active_return / te if te > 0 else 0.0


def benchmark_comparison_table(portfolio_daily_rets, benchmark_daily_rets,
                                risk_free_rate=0.0, trading_days=252):
    """Return a side-by-side comparison dict for portfolio vs benchmark."""
    try:
        def ann_return(r): return float(r.mean()) * trading_days
        def ann_vol(r): return float(r.std()) * np.sqrt(trading_days)
        def sharpe(r): return (ann_return(r) - risk_free_rate) / ann_vol(r) if ann_vol(r) > 0 else 0

        active = (portfolio_daily_rets - benchmark_daily_rets).dropna()
        active_return = ann_return(active)
        te = calculate_tracking_error(portfolio_daily_rets, benchmark_daily_rets, trading_days)
        ir = active_return / te if te > 0 else 0

        return {
            "Metric": ["Annualised Return", "Annualised Volatility", "Sharpe Ratio",
                       "Active Return", "Tracking Error", "Information Ratio"],
            "Portfolio": [
                f"{ann_return(portfolio_daily_rets):.2%}",
                f"{ann_vol(portfolio_daily_rets):.2%}",
                f"{sharpe(portfolio_daily_rets):.2f}",
                f"{active_return:.2%}", f"{te:.2%}", f"{ir:.2f}",
            ],
            "Benchmark": [
                f"{ann_return(benchmark_daily_rets):.2%}",
                f"{ann_vol(benchmark_daily_rets):.2%}",
                f"{sharpe(benchmark_daily_rets):.2f}",
                "—", "—", "—",
            ],
        }
    except Exception as e:
        st.error(f"benchmark_comparison_table failed: {e}")
        return None


def plot_cumulative_returns(portfolio_daily_rets, benchmark_daily_rets, benchmark_name="Benchmark"):
    """Line chart of cumulative growth of R1 invested."""
    port_cum = (1 + portfolio_daily_rets).cumprod()
    bench_cum = (1 + benchmark_daily_rets).cumprod()

    common = port_cum.index.intersection(bench_cum.index)
    port_cum = port_cum.loc[common]
    bench_cum = bench_cum.loc[common]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values,
                             mode="lines", name="Your Portfolio", line=dict(color="steelblue", width=2)))
    fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values,
                             mode="lines", name=benchmark_name, line=dict(color="gray", width=2, dash="dash")))
    fig.update_layout(
        title="Cumulative Return: Portfolio vs Benchmark",
        xaxis_title="Date", yaxis_title="Growth of R1",
        height=400,
    )
    return fig

# ─── Rebalancing Simulation ────────────────────────────────────────────────────

def simulate_rebalancing(stock_data, target_weights, rebalance_frequency="quarterly",
                          transaction_cost_pct=0.005, initial_investment=100_000):
    """
    Simulate portfolio value over time with periodic rebalancing and transaction costs.

    Parameters
    ----------
    stock_data            : DataFrame of adjusted close prices
    target_weights        : array-like of target portfolio weights
    rebalance_frequency   : 'monthly', 'quarterly', or 'annually'
    transaction_cost_pct  : cost per trade as fraction of trade value (e.g. 0.005 = 50bps)
    initial_investment    : starting portfolio value in rands

    Returns
    -------
    portfolio_values : pd.Series of daily portfolio value
    total_costs      : total rand value of transaction costs paid
    rebalance_dates  : list of dates rebalancing occurred
    """
    freq_map = {"monthly": "ME", "quarterly": "QE", "annually": "YE"}
    resample_freq = freq_map.get(rebalance_frequency, "QE")

    target_weights = np.array(target_weights)
    daily_returns = stock_data.pct_change().dropna()

    # Rebalance dates
    rebalance_dates = set(
        daily_returns.resample(resample_freq).last().index.normalize()
    )

    # Initialise holdings in shares
    prices_start = stock_data.iloc[0]
    holdings = (initial_investment * target_weights) / prices_start.values
    portfolio_values = []
    total_costs = 0.0
    rebalance_log = []

    for date, prices in stock_data.iterrows():
        prices = prices.values
        current_value = (holdings * prices).sum()
        portfolio_values.append(current_value)

        if date.normalize() in rebalance_dates:
            target_values = target_weights * current_value
            current_values = holdings * prices
            trades = np.abs(target_values - current_values)
            cost = (trades * transaction_cost_pct).sum()
            total_costs += cost
            current_value_after_cost = current_value - cost
            holdings = (target_weights * current_value_after_cost) / prices
            rebalance_log.append(date)

    port_series = pd.Series(portfolio_values, index=stock_data.index)
    return port_series, total_costs, rebalance_log


def plot_rebalancing_comparison(stock_data, target_weights, initial_investment=100_000):
    """
    Compare buy-and-hold vs three rebalancing frequencies with a cost sensitivity sweep.
    Returns a Plotly figure.
    """
    fig = go.Figure()
    colours = {
        "Buy & Hold": "gray",
        "Monthly": "#E53935",
        "Quarterly": "#1E88E5",
        "Annually": "#43A047",
    }

    # Buy & hold
    daily_returns = stock_data.pct_change().dropna()
    port_daily = (daily_returns * target_weights).sum(axis=1)
    bah = initial_investment * (1 + port_daily).cumprod()
    fig.add_trace(go.Scatter(x=bah.index, y=bah.values, mode="lines",
                              name="Buy & Hold", line=dict(color="gray", dash="dash", width=2)))

    for freq in ["monthly", "quarterly", "annually"]:
        vals, costs, _ = simulate_rebalancing(stock_data, target_weights,
                                               rebalance_frequency=freq,
                                               initial_investment=initial_investment)
        label = freq.capitalize()
        fig.add_trace(go.Scatter(x=vals.index, y=vals.values, mode="lines",
                                  name=f"{label} Rebalancing",
                                  line=dict(color=colours[label], width=2)))

    fig.update_layout(
        title="Portfolio Value: Buy & Hold vs Rebalancing Strategies",
        xaxis_title="Date", yaxis_title="Portfolio Value (R)",
        yaxis_tickprefix="R", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
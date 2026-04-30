"""
DCF Valuation Tutor - Streamlit App

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

GitHub / Streamlit Community Cloud:
    1. Upload app.py and requirements.txt to a GitHub repo.
    2. On Streamlit Community Cloud, select app.py as the main file.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="DCF Valuation Tutor",
    page_icon="📈",
    layout="wide",
)


# -----------------------------
# Formatting helpers
# -----------------------------
def safe_float(value: Any, default: float = np.nan) -> float:
    """Convert value to float without crashing."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def clean_number(value: Any, default: float = 0.0) -> float:
    """Return a finite float for Streamlit number inputs."""
    value = safe_float(value, np.nan)
    if np.isnan(value) or np.isinf(value):
        return float(default)
    return float(value)


def fmt_money(value: Any) -> str:
    value = safe_float(value, np.nan)
    if np.isnan(value):
        return "N/A"

    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 1_000_000_000_000:
        return f"{sign}${value / 1_000_000_000_000:,.2f}T"
    if value >= 1_000_000_000:
        return f"{sign}${value / 1_000_000_000:,.2f}B"
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:,.2f}M"
    return f"{sign}${value:,.0f}"


def fmt_price(value: Any) -> str:
    value = safe_float(value, np.nan)
    if np.isnan(value):
        return "N/A"
    return f"${value:,.2f}"


def fmt_pct(value: Any) -> str:
    value = safe_float(value, np.nan)
    if np.isnan(value):
        return "N/A"
    return f"{value:.1%}"


def first_available(info: Dict[str, Any], keys: Iterable[str], default: Any = np.nan) -> Any:
    for key in keys:
        value = info.get(key)
        if value is not None:
            return value
    return default


def row_value(statement: pd.DataFrame, possible_rows: Iterable[str], default: float = np.nan) -> float:
    """Safely pull the most recent value from a yfinance statement row."""
    if statement is None or statement.empty:
        return default

    for row in possible_rows:
        if row in statement.index:
            values = pd.to_numeric(statement.loc[row], errors="coerce").dropna()
            if not values.empty:
                return safe_float(values.iloc[0], default)

    return default


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_yahoo_data(ticker_symbol: str) -> Dict[str, Any]:
    """Load Yahoo Finance data using yfinance."""
    ticker_symbol = ticker_symbol.strip().upper()
    ticker = yf.Ticker(ticker_symbol)

    # yfinance can occasionally fail on one statement but not others.
    # Each call is isolated so the app still loads where possible.
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    try:
        cashflow = ticker.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    try:
        balance = ticker.balance_sheet
    except Exception:
        balance = pd.DataFrame()

    try:
        income = ticker.financials
    except Exception:
        income = pd.DataFrame()

    try:
        hist = ticker.history(period="5d", auto_adjust=False)
    except Exception:
        hist = pd.DataFrame()

    return {
        "info": info,
        "cashflow": cashflow if cashflow is not None else pd.DataFrame(),
        "balance": balance if balance is not None else pd.DataFrame(),
        "income": income if income is not None else pd.DataFrame(),
        "hist": hist if hist is not None else pd.DataFrame(),
    }


def estimate_yahoo_inputs(data: Dict[str, Any]) -> Dict[str, float]:
    info = data["info"]
    cashflow = data["cashflow"]
    balance = data["balance"]
    income = data["income"]
    hist = data["hist"]

    price = safe_float(first_available(info, ["currentPrice", "regularMarketPrice", "previousClose"]))
    if (np.isnan(price) or price <= 0) and not hist.empty and "Close" in hist:
        close_series = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if not close_series.empty:
            price = safe_float(close_series.iloc[-1])

    market_cap = safe_float(first_available(info, ["marketCap"]))
    shares = safe_float(first_available(info, ["sharesOutstanding", "impliedSharesOutstanding"]))

    cash = safe_float(first_available(info, ["totalCash", "cash"]))
    if np.isnan(cash):
        cash = row_value(
            balance,
            [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash",
            ],
        )

    debt = safe_float(first_available(info, ["totalDebt"]))
    if np.isnan(debt):
        current_debt = row_value(balance, ["Current Debt", "Short Long Term Debt"], 0.0)
        long_debt = row_value(balance, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0.0)
        debt = clean_number(current_debt) + clean_number(long_debt)

    operating_cf = row_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = row_value(cashflow, ["Capital Expenditure", "Capital Expenditures"])

    # In yfinance, capital expenditure is usually negative.
    if not np.isnan(operating_cf) and not np.isnan(capex):
        base_fcf = operating_cf + capex
    else:
        base_fcf = safe_float(first_available(info, ["freeCashflow"]))

    revenue = row_value(income, ["Total Revenue"])
    ebit = row_value(income, ["EBIT", "Operating Income"])

    tax_provision = row_value(income, ["Tax Provision", "Income Tax Expense"])
    pretax_income = row_value(income, ["Pretax Income", "Income Before Tax"])
    if not np.isnan(tax_provision) and not np.isnan(pretax_income) and pretax_income > 0:
        tax_rate = min(max(tax_provision / pretax_income, 0.0), 0.40)
    else:
        tax_rate = 0.21

    return {
        "price": price,
        "market_cap": market_cap,
        "shares": shares,
        "cash": cash,
        "debt": debt,
        "net_debt": clean_number(debt) - clean_number(cash),
        "operating_cf": operating_cf,
        "capex": capex,
        "base_fcf": base_fcf,
        "revenue": revenue,
        "ebit": ebit,
        "tax_rate": tax_rate,
        "beta": safe_float(first_available(info, ["beta"])),
        "yahoo_enterprise_value": safe_float(first_available(info, ["enterpriseValue"])),
    }


# -----------------------------
# DCF calculations
# -----------------------------
def project_fcfs(base_fcf: float, growth_rates: np.ndarray) -> np.ndarray:
    values = []
    current = base_fcf
    for growth in growth_rates:
        current = current * (1 + growth)
        values.append(current)
    return np.array(values, dtype=float)


def terminal_value_gordon(final_fcf: float, discount_rate: float, terminal_growth: float) -> float:
    if discount_rate <= terminal_growth:
        return np.nan
    return final_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)


def calculate_dcf(
    base_fcf: float,
    growth_rates: np.ndarray,
    discount_rate: float,
    terminal_growth: float,
    net_debt: float,
    shares: float,
    current_price: float,
) -> Dict[str, Any]:
    fcfs = project_fcfs(base_fcf, growth_rates)
    years = np.arange(1, len(fcfs) + 1)
    discount_factors = 1 / ((1 + discount_rate) ** years)
    pv_fcfs = fcfs * discount_factors

    terminal_value = terminal_value_gordon(fcfs[-1], discount_rate, terminal_growth)
    pv_terminal_value = terminal_value / ((1 + discount_rate) ** len(fcfs))

    enterprise_value = pv_fcfs.sum() + pv_terminal_value
    equity_value = enterprise_value - net_debt
    implied_share_price = equity_value / shares if shares > 0 else np.nan
    upside_downside = implied_share_price / current_price - 1 if current_price > 0 else np.nan

    if np.isnan(upside_downside):
        signal = "No Signal"
    elif upside_downside > 0.10:
        signal = "Undervalued"
    elif upside_downside < -0.10:
        signal = "Overvalued"
    else:
        signal = "Fairly Valued"

    return {
        "fcfs": fcfs,
        "discount_factors": discount_factors,
        "pv_fcfs": pv_fcfs,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "implied_share_price": implied_share_price,
        "upside_downside": upside_downside,
        "signal": signal,
    }


# -----------------------------
# App header
# -----------------------------
st.title("📈 DCF Valuation Tutor")
st.write(
    "Enter a ticker, let Yahoo Finance pre-fill the model, then adjust the assumptions or override the source data."
)

with st.expander("How this app works", expanded=False):
    st.markdown(
        """
        This app uses a free-cash-flow DCF:

        1. Pulls company data from Yahoo Finance through `yfinance`.
        2. Estimates base FCF as **Operating Cash Flow + Capital Expenditures**.
        3. Forecasts FCF over your chosen forecast period.
        4. Discounts forecast FCF using WACC.
        5. Calculates terminal value using the Gordon Growth method.
        6. Calculates enterprise value, equity value, implied share price, upside/downside, and a valuation signal.

        The model is educational. Real valuation work usually needs cleaned financials, normalized FCF,
        diluted share adjustments, lease/debt adjustments, stock-based compensation review, and scenario analysis.
        """
    )


# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Company")
ticker_symbol = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()

if not ticker_symbol:
    st.warning("Please enter a ticker symbol.")
    st.stop()

with st.spinner("Loading Yahoo Finance data..."):
    data = load_yahoo_data(ticker_symbol)

info = data["info"]
auto = estimate_yahoo_inputs(data)

company_name = info.get("longName") or info.get("shortName") or ticker_symbol
currency = info.get("financialCurrency") or info.get("currency") or "USD"

st.sidebar.header("Model Assumptions")
projection_years = st.sidebar.slider("Projection years", min_value=3, max_value=10, value=5)
year_1_growth = st.sidebar.number_input("Year 1 FCF growth", value=0.08, step=0.01, format="%.2f")
final_year_growth = st.sidebar.number_input("Final year FCF growth", value=0.03, step=0.01, format="%.2f")
discount_rate = st.sidebar.number_input("Discount rate / WACC", min_value=0.01, max_value=0.50, value=0.09, step=0.005, format="%.3f")
terminal_growth = st.sidebar.number_input("Terminal growth", min_value=-0.05, max_value=0.10, value=0.025, step=0.005, format="%.3f")

st.sidebar.header("Yahoo Data + Overrides")
use_yahoo = st.sidebar.checkbox("Use Yahoo Finance as starting point", value=True)

base_fcf_default = clean_number(auto["base_fcf"]) if use_yahoo else 0.0
cash_default = clean_number(auto["cash"]) if use_yahoo else 0.0
debt_default = clean_number(auto["debt"]) if use_yahoo else 0.0
shares_default = clean_number(auto["shares"]) if use_yahoo else 0.0
price_default = clean_number(auto["price"]) if use_yahoo else 0.0

base_fcf = st.sidebar.number_input("Base free cash flow ($)", value=base_fcf_default, step=100_000_000.0)
cash = st.sidebar.number_input("Cash & equivalents ($)", value=cash_default, step=100_000_000.0)
debt = st.sidebar.number_input("Total debt ($)", value=debt_default, step=100_000_000.0)
shares = st.sidebar.number_input("Diluted shares outstanding", value=shares_default, step=10_000_000.0)
current_price = st.sidebar.number_input("Current share price ($)", value=price_default, step=1.0)

tax_rate = st.sidebar.number_input(
    "Tax rate reference",
    min_value=0.00,
    max_value=0.50,
    value=clean_number(auto["tax_rate"], 0.21),
    step=0.01,
    format="%.2f",
)


# -----------------------------
# Validation
# -----------------------------
warnings = []

if not info:
    warnings.append("Yahoo Finance returned limited company information. Manual overrides may be needed.")

if base_fcf <= 0:
    warnings.append("Base free cash flow must be positive for this simple DCF. Enter a normalized FCF estimate.")

if shares <= 0:
    warnings.append("Shares outstanding must be positive. Enter diluted shares manually.")

if current_price <= 0:
    warnings.append("Current share price must be positive. Enter the market price manually.")

if discount_rate <= terminal_growth:
    warnings.append("Discount rate / WACC must be greater than terminal growth.")

if warnings:
    for warning in warnings:
        st.warning(warning)
    st.stop()


# -----------------------------
# Run model
# -----------------------------
net_debt = debt - cash
growth_rates = np.linspace(year_1_growth, final_year_growth, projection_years)

dcf = calculate_dcf(
    base_fcf=base_fcf,
    growth_rates=growth_rates,
    discount_rate=discount_rate,
    terminal_growth=terminal_growth,
    net_debt=net_debt,
    shares=shares,
    current_price=current_price,
)


# -----------------------------
# Output
# -----------------------------
st.subheader(f"{company_name} ({ticker_symbol})")
st.caption(f"Currency: {currency} | Last app refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

metric_cols = st.columns(5)
metric_cols[0].metric("Market Price", fmt_price(current_price))
metric_cols[1].metric("Implied Share Price", fmt_price(dcf["implied_share_price"]))
metric_cols[2].metric("Upside / Downside", fmt_pct(dcf["upside_downside"]))
metric_cols[3].metric("Enterprise Value", fmt_money(dcf["enterprise_value"]))
metric_cols[4].metric("Signal", dcf["signal"])

tabs = st.tabs(
    [
        "Valuation",
        "Yahoo vs Inputs",
        "Forecast",
        "Sensitivity",
        "Teaching",
        "Raw Data",
    ]
)


with tabs[0]:
    st.header("Valuation Summary")

    summary = pd.DataFrame(
        {
            "Metric": [
                "PV of projected FCF",
                "Terminal value",
                "PV of terminal value",
                "Enterprise value",
                "Net debt",
                "Equity value",
                "Diluted shares",
                "Implied share price",
                "Current share price",
                "Upside / downside",
                "Signal",
            ],
            "Calculation": [
                "Sum of discounted forecast FCF",
                "Final FCF × (1 + g) / (WACC − g)",
                "Terminal value discounted to today",
                "PV of FCF + PV of terminal value",
                "Debt − cash",
                "Enterprise value − net debt",
                "User input or Yahoo Finance",
                "Equity value / shares",
                "User input or Yahoo Finance",
                "Implied price / market price − 1",
                "+10% undervalued, -10% overvalued",
            ],
            "Value": [
                fmt_money(dcf["pv_fcfs"].sum()),
                fmt_money(dcf["terminal_value"]),
                fmt_money(dcf["pv_terminal_value"]),
                fmt_money(dcf["enterprise_value"]),
                fmt_money(net_debt),
                fmt_money(dcf["equity_value"]),
                f"{shares:,.0f}",
                fmt_price(dcf["implied_share_price"]),
                fmt_price(current_price),
                fmt_pct(dcf["upside_downside"]),
                dcf["signal"],
            ],
        }
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"Year {i}" for i in range(1, projection_years + 1)] + ["Terminal Value PV"],
            y=list(dcf["pv_fcfs"]) + [dcf["pv_terminal_value"]],
            name="Present Value",
        )
    )
    fig.update_layout(
        title="Present Value Contribution",
        xaxis_title="DCF Component",
        yaxis_title="Present Value",
    )
    st.plotly_chart(fig, use_container_width=True)


with tabs[1]:
    st.header("Yahoo Finance Data vs Values Used")

    comparison = pd.DataFrame(
        {
            "Input": [
                "Current share price",
                "Base free cash flow",
                "Cash",
                "Total debt",
                "Net debt",
                "Shares outstanding",
                "Yahoo enterprise value",
                "Operating cash flow",
                "Capital expenditures",
                "Revenue",
                "EBIT",
                "Beta",
                "Tax rate reference",
            ],
            "Yahoo / Estimated": [
                auto["price"],
                auto["base_fcf"],
                auto["cash"],
                auto["debt"],
                auto["net_debt"],
                auto["shares"],
                auto["yahoo_enterprise_value"],
                auto["operating_cf"],
                auto["capex"],
                auto["revenue"],
                auto["ebit"],
                auto["beta"],
                auto["tax_rate"],
            ],
            "Used in Model": [
                current_price,
                base_fcf,
                cash,
                debt,
                net_debt,
                shares,
                dcf["enterprise_value"],
                auto["operating_cf"],
                auto["capex"],
                auto["revenue"],
                auto["ebit"],
                auto["beta"],
                tax_rate,
            ],
        }
    )

    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.info("Use the sidebar override fields when Yahoo Finance is missing data or when you want a custom scenario.")


with tabs[2]:
    st.header("Forecast Table")

    forecast = pd.DataFrame(
        {
            "Year": [f"Year {i}" for i in range(1, projection_years + 1)],
            "FCF Growth": growth_rates,
            "Projected FCF": dcf["fcfs"],
            "Discount Factor": dcf["discount_factors"],
            "PV of FCF": dcf["pv_fcfs"],
        }
    )

    display = forecast.copy()
    display["FCF Growth"] = display["FCF Growth"].map(fmt_pct)
    display["Projected FCF"] = display["Projected FCF"].map(fmt_money)
    display["Discount Factor"] = display["Discount Factor"].map(lambda x: f"{x:.3f}")
    display["PV of FCF"] = display["PV of FCF"].map(fmt_money)

    st.dataframe(display, use_container_width=True, hide_index=True)


with tabs[3]:
    st.header("Sensitivity Analysis")
    st.caption("Rows are discount rate / WACC. Columns are terminal growth. Cells are implied share price.")

    wacc_values = np.round(np.arange(max(0.01, discount_rate - 0.02), discount_rate + 0.025, 0.005), 4)
    terminal_growth_values = np.round(
        np.arange(max(-0.02, terminal_growth - 0.01), terminal_growth + 0.0125, 0.005),
        4,
    )

    sensitivity = pd.DataFrame(
        index=[fmt_pct(x) for x in wacc_values],
        columns=[fmt_pct(x) for x in terminal_growth_values],
        dtype=float,
    )

    for w in wacc_values:
        for g in terminal_growth_values:
            if w <= g:
                sensitivity.loc[fmt_pct(w), fmt_pct(g)] = np.nan
            else:
                result = calculate_dcf(
                    base_fcf=base_fcf,
                    growth_rates=growth_rates,
                    discount_rate=w,
                    terminal_growth=g,
                    net_debt=net_debt,
                    shares=shares,
                    current_price=current_price,
                )
                sensitivity.loc[fmt_pct(w), fmt_pct(g)] = result["implied_share_price"]

    st.dataframe(sensitivity.style.format("${:,.2f}"), use_container_width=True)


with tabs[4]:
    st.header("Teaching Mode")

    st.markdown(
        """
        ### 1. Free Cash Flow
        The app estimates base free cash flow from Yahoo Finance:

        **FCF = Operating Cash Flow + Capital Expenditures**

        Capex is commonly negative in financial statements, so adding capex usually reduces operating cash flow.

        ### 2. Forecast FCF
        The app grows base FCF using your growth assumptions. Growth fades from Year 1 growth to final-year growth.

        ### 3. Discount Cash Flows
        Future cash flows are discounted by WACC:

        **PV = Future Cash Flow / (1 + WACC)^Year**

        ### 4. Terminal Value
        The app uses the Gordon Growth formula:

        **Terminal Value = Final FCF × (1 + Terminal Growth) / (WACC − Terminal Growth)**

        ### 5. Enterprise Value
        **Enterprise Value = PV of Forecast FCF + PV of Terminal Value**

        ### 6. Equity Value
        **Equity Value = Enterprise Value − Net Debt**

        where:

        **Net Debt = Total Debt − Cash**

        ### 7. Implied Share Price
        **Implied Share Price = Equity Value / Diluted Shares**

        ### 8. Upside / Downside
        **Upside / Downside = Implied Share Price / Current Market Price − 1**

        Signal rule used in this app:

        - Above +10%: **Undervalued**
        - Below -10%: **Overvalued**
        - Between -10% and +10%: **Fairly Valued**
        """
    )


with tabs[5]:
    st.header("Raw Yahoo Finance Data")

    with st.expander("Cash Flow Statement"):
        st.dataframe(data["cashflow"], use_container_width=True)

    with st.expander("Balance Sheet"):
        st.dataframe(data["balance"], use_container_width=True)

    with st.expander("Income Statement"):
        st.dataframe(data["income"], use_container_width=True)

    with st.expander("Company Info"):
        if info:
            info_df = pd.DataFrame(list(info.items()), columns=["Field", "Value"])
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        else:
            st.write("No company info returned by Yahoo Finance.")


st.divider()
st.caption(
    "Data note: yfinance is an unofficial Yahoo Finance wrapper. Data may be delayed, missing, or categorized differently by company. "
    "Always review and override model inputs when needed."
)

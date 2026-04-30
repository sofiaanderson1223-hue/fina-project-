# dcf_valuation_app.py
# Streamlit DCF valuation app with Yahoo Finance / yfinance data and user overrides
#
# How to run in Spyder:
# 1) Save this file as dcf_valuation_app.py
# 2) In Spyder's console, install packages if needed:
#       pip install streamlit yfinance pandas numpy plotly
# 3) Run:
#       streamlit run dcf_valuation_app.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import datetime


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="DCF Valuation Tutor", page_icon="📈", layout="wide")

st.title("📈 DCF Valuation Tutor")
st.caption(
    "Pulls Yahoo Finance data through yfinance, lets you override inputs, and teaches each valuation step. "
    "For education only, not investment advice."
)


# -----------------------------
# Utility functions
# -----------------------------
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def fmt_money(x):
    if pd.isna(x):
        return "N/A"
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e12:
        return f"{sign}${x/1e12:,.2f}T"
    if x >= 1e9:
        return f"{sign}${x/1e9:,.2f}B"
    if x >= 1e6:
        return f"{sign}${x/1e6:,.2f}M"
    return f"{sign}${x:,.0f}"


def fmt_num(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.0f}"


def fmt_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.1%}"


def first_available(info, keys, default=np.nan):
    for key in keys:
        value = info.get(key)
        if value is not None:
            return value
    return default


def row_value(df, possible_rows, default=np.nan):
    if df is None or df.empty:
        return default
    for row in possible_rows:
        if row in df.index:
            s = df.loc[row].dropna()
            if len(s) > 0:
                return safe_float(s.iloc[0])
    return default


def get_latest_price(info, hist):
    price = safe_float(first_available(info, ["currentPrice", "regularMarketPrice", "previousClose"]))
    if (pd.isna(price) or price <= 0) and hist is not None and not hist.empty:
        price = safe_float(hist["Close"].dropna().iloc[-1])
    return price


@st.cache_data(ttl=900)
def load_yahoo_data(ticker_symbol):
    t = yf.Ticker(ticker_symbol)
    info = t.info
    cashflow = t.cashflow
    balance = t.balance_sheet
    income = t.financials
    hist = t.history(period="5d")
    return info, cashflow, balance, income, hist


def estimate_from_yahoo(info, cashflow, balance, income, hist):
    current_price = get_latest_price(info, hist)
    market_cap = safe_float(first_available(info, ["marketCap"]))
    shares = safe_float(first_available(info, ["sharesOutstanding", "impliedSharesOutstanding"]))

    cash = safe_float(first_available(info, ["totalCash", "cash"]))
    if pd.isna(cash):
        cash = row_value(balance, [
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash",
        ])

    debt = safe_float(first_available(info, ["totalDebt"]))
    if pd.isna(debt):
        current_debt = row_value(balance, ["Current Debt", "Short Long Term Debt"], 0)
        long_debt = row_value(balance, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
        debt = safe_float(current_debt, 0) + safe_float(long_debt, 0)

    operating_cf = row_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = row_value(cashflow, ["Capital Expenditure", "Capital Expenditures"])

    # Yahoo/yfinance often reports capex as negative, so FCF = CFO + CapEx.
    if not pd.isna(operating_cf) and not pd.isna(capex):
        fcf = operating_cf + capex
    else:
        fcf = safe_float(first_available(info, ["freeCashflow"]))

    revenue = row_value(income, ["Total Revenue"])
    ebit = row_value(income, ["EBIT", "Operating Income"])
    tax_provision = row_value(income, ["Tax Provision", "Income Tax Expense"])
    pretax_income = row_value(income, ["Pretax Income", "Income Before Tax"])

    if not pd.isna(tax_provision) and not pd.isna(pretax_income) and pretax_income > 0:
        tax_rate = max(0, min(0.40, tax_provision / pretax_income))
    else:
        tax_rate = 0.21

    beta = safe_float(first_available(info, ["beta"]))
    yahoo_ev = safe_float(first_available(info, ["enterpriseValue"]))

    return {
        "current_price": current_price,
        "market_cap": market_cap,
        "shares": shares,
        "cash": cash,
        "debt": debt,
        "net_debt": safe_float(debt, 0) - safe_float(cash, 0),
        "operating_cf": operating_cf,
        "capex": capex,
        "base_fcf": fcf,
        "revenue": revenue,
        "ebit": ebit,
        "tax_rate": tax_rate,
        "beta": beta,
        "yahoo_enterprise_value": yahoo_ev,
    }


def project_fcfs(base_fcf, growth_rates):
    fcfs = []
    current = base_fcf
    for g in growth_rates:
        current *= 1 + g
        fcfs.append(current)
    return np.array(fcfs)


def terminal_value_gordon(final_fcf, wacc, terminal_growth):
    if wacc <= terminal_growth:
        return np.nan
    return final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)


def calculate_dcf(base_fcf, growth_rates, wacc, terminal_growth, net_debt, shares, current_price):
    fcfs = project_fcfs(base_fcf, growth_rates)
    years = np.arange(1, len(fcfs) + 1)
    discount_factors = 1 / ((1 + wacc) ** years)
    pv_fcfs = fcfs * discount_factors

    terminal_value = terminal_value_gordon(fcfs[-1], wacc, terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** len(fcfs))

    enterprise_value = pv_fcfs.sum() + pv_terminal_value
    equity_value = enterprise_value - net_debt
    implied_share_price = equity_value / shares
    upside_downside = (implied_share_price / current_price) - 1 if current_price > 0 else np.nan

    if pd.isna(upside_downside):
        signal = "NO SIGNAL"
    elif upside_downside > 0.10:
        signal = "UNDERVALUED"
    elif upside_downside < -0.10:
        signal = "OVERVALUED"
    else:
        signal = "FAIRLY VALUED"

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
# Sidebar: Yahoo data and assumptions
# -----------------------------
st.sidebar.header("Company")
ticker_symbol = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()

if not ticker_symbol:
    st.warning("Enter a ticker symbol.")
    st.stop()

try:
    info, cashflow, balance, income, hist = load_yahoo_data(ticker_symbol)
except Exception as e:
    st.error(f"Could not pull Yahoo Finance data for {ticker_symbol}: {e}")
    st.stop()

auto = estimate_from_yahoo(info, cashflow, balance, income, hist)

company_name = info.get("longName", ticker_symbol)
currency = info.get("financialCurrency", info.get("currency", "USD"))

st.sidebar.header("Data Source")
use_yahoo_defaults = st.sidebar.checkbox(
    "Start with Yahoo Finance values",
    value=True,
    help="When checked, the model pre-fills operating cash flow, capex, cash, debt, shares, and market price from Yahoo Finance."
)

st.sidebar.header("DCF Assumptions")
projection_years = st.sidebar.slider("Projection years", 3, 10, 5)

year_1_growth = st.sidebar.number_input("Year 1 FCF growth", value=0.08, step=0.01, format="%.2f")
final_year_growth = st.sidebar.number_input("Final forecast year FCF growth", value=0.03, step=0.01, format="%.2f")
wacc = st.sidebar.number_input("Discount rate / WACC", min_value=0.01, max_value=0.50, value=0.09, step=0.005, format="%.3f")
terminal_growth = st.sidebar.number_input("Terminal growth", min_value=-0.05, max_value=0.10, value=0.025, step=0.005, format="%.3f")

st.sidebar.header("User Inputs / Overrides")
st.sidebar.caption("Use these if Yahoo data is missing, unusual, or you want a custom scenario.")

default_fcf = auto["base_fcf"] if use_yahoo_defaults and not pd.isna(auto["base_fcf"]) else 0.0
base_fcf = st.sidebar.number_input("Base free cash flow, $", value=float(default_fcf), step=100_000_000.0)

default_cash = auto["cash"] if use_yahoo_defaults and not pd.isna(auto["cash"]) else 0.0
cash = st.sidebar.number_input("Cash & equivalents, $", value=float(default_cash), step=100_000_000.0)

default_debt = auto["debt"] if use_yahoo_defaults and not pd.isna(auto["debt"]) else 0.0
debt = st.sidebar.number_input("Total debt, $", value=float(default_debt), step=100_000_000.0)

default_shares = auto["shares"] if use_yahoo_defaults and not pd.isna(auto["shares"]) else 0.0
shares = st.sidebar.number_input("Diluted shares outstanding", value=float(default_shares), step=10_000_000.0)

default_price = auto["current_price"] if use_yahoo_defaults and not pd.isna(auto["current_price"]) else 0.0
current_price = st.sidebar.number_input("Current share price, $", value=float(default_price), step=1.0)

tax_rate = st.sidebar.number_input(
    "Tax rate, for teaching/reference",
    min_value=0.0,
    max_value=0.50,
    value=float(auto["tax_rate"]),
    step=0.01,
    format="%.2f",
    help="Included to teach DCF inputs. This FCF-based model does not directly use tax rate unless you build FCFF from EBIT."
)


# -----------------------------
# Validation and model
# -----------------------------
if base_fcf <= 0:
    st.error("Base free cash flow must be positive. Enter a normalized FCF estimate or use a different company.")
    st.stop()

if shares <= 0:
    st.error("Shares outstanding must be positive. Enter diluted shares manually if Yahoo Finance did not provide it.")
    st.stop()

if current_price <= 0:
    st.error("Current share price must be positive. Enter it manually if Yahoo Finance did not provide it.")
    st.stop()

if wacc <= terminal_growth:
    st.error("WACC must be greater than terminal growth.")
    st.stop()

net_debt = debt - cash
growth_rates = np.linspace(year_1_growth, final_year_growth, projection_years)
dcf = calculate_dcf(base_fcf, growth_rates, wacc, terminal_growth, net_debt, shares, current_price)


# -----------------------------
# Main UI
# -----------------------------
st.subheader(f"{company_name} ({ticker_symbol})")
st.write(f"Currency: **{currency}** | Refreshed: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")

cols = st.columns(5)
cols[0].metric("Current Price", f"${current_price:,.2f}")
cols[1].metric("Implied Price", f"${dcf['implied_share_price']:,.2f}")
cols[2].metric("Upside / Downside", fmt_pct(dcf["upside_downside"]))
cols[3].metric("Enterprise Value", fmt_money(dcf["enterprise_value"]))
cols[4].metric("Signal", dcf["signal"])

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Valuation Output",
    "Yahoo Inputs vs User Inputs",
    "Teaching Mode",
    "Forecast & Sensitivity",
    "Raw Yahoo Data"
])

with tab1:
    st.header("DCF Valuation Output")

    summary_df = pd.DataFrame({
        "Line Item": [
            "Present value of projected free cash flows",
            "Terminal value",
            "Present value of terminal value",
            "Enterprise value",
            "Less: net debt",
            "Equity value",
            "Diluted shares",
            "Implied share price",
            "Current share price",
            "Upside / downside",
            "Signal"
        ],
        "Formula / Meaning": [
            "Sum of discounted annual FCF",
            "Value after forecast period",
            "Terminal value discounted to today",
            "PV of projected FCF + PV of terminal value",
            "Debt minus cash",
            "Enterprise value minus net debt",
            "Shares used to calculate per-share value",
            "Equity value / shares",
            "Market price from Yahoo or override",
            "Implied price / current price - 1",
            "Rule-based undervalued/overvalued call"
        ],
        "Value": [
            fmt_money(dcf["pv_fcfs"].sum()),
            fmt_money(dcf["terminal_value"]),
            fmt_money(dcf["pv_terminal_value"]),
            fmt_money(dcf["enterprise_value"]),
            fmt_money(net_debt),
            fmt_money(dcf["equity_value"]),
            fmt_num(shares),
            f"${dcf['implied_share_price']:,.2f}",
            f"${current_price:,.2f}",
            fmt_pct(dcf["upside_downside"]),
            dcf["signal"]
        ]
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Year {i+1}" for i in range(projection_years)] + ["Terminal PV"],
        y=list(dcf["pv_fcfs"]) + [dcf["pv_terminal_value"]],
        name="PV Contribution"
    ))
    fig.update_layout(
        title="What Drives Enterprise Value?",
        xaxis_title="DCF Component",
        yaxis_title="Present Value"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Yahoo Finance Inputs vs User Inputs")

    input_compare = pd.DataFrame({
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
            "Tax rate"
        ],
        "Yahoo Finance / Estimated": [
            auto["current_price"],
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
            auto["tax_rate"]
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
            tax_rate
        ]
    })

    st.dataframe(input_compare, use_container_width=True, hide_index=True)

    st.info(
        "The model starts with Yahoo Finance numbers, but every critical input can be changed in the sidebar. "
        "This is important because normalized FCF, diluted shares, excess cash, leases, and one-time items often need analyst judgment."
    )

with tab3:
    st.header("Teaching Mode")

    st.markdown(
        """
        ### Step 1: Start with free cash flow
        This model starts with Yahoo Finance cash flow data:

        **Free Cash Flow = Operating Cash Flow + Capital Expenditures**

        Capital expenditures are usually negative in Yahoo/yfinance, so adding capex subtracts investment spending.

        ### Step 2: Forecast future FCF
        The app grows FCF from your Year 1 growth assumption to your final-year growth assumption.

        ### Step 3: Discount cash flows
        Future cash is worth less than cash today:

        **Present Value = Future Cash Flow / (1 + WACC) ^ Year**

        ### Step 4: Estimate terminal value
        The terminal value estimates all cash flows after the explicit forecast:

        **Terminal Value = Final FCF × (1 + Terminal Growth) / (WACC − Terminal Growth)**

        ### Step 5: Calculate enterprise value
        **Enterprise Value = PV of Forecast FCF + PV of Terminal Value**

        ### Step 6: Convert enterprise value to equity value
        **Equity Value = Enterprise Value − Net Debt**

        where:

        **Net Debt = Total Debt − Cash**

        ### Step 7: Calculate implied share price
        **Implied Share Price = Equity Value / Diluted Shares**

        ### Step 8: Compare to market price
        **Upside / Downside = Implied Share Price / Current Share Price − 1**

        Signal rule:
        - Above +10%: **Undervalued**
        - Below -10%: **Overvalued**
        - Between -10% and +10%: **Fairly valued**
        """
    )

with tab4:
    st.header("Forecast & Sensitivity")

    forecast_df = pd.DataFrame({
        "Year": [f"Year {i+1}" for i in range(projection_years)],
        "FCF Growth": growth_rates,
        "Projected FCF": dcf["fcfs"],
        "Discount Factor": dcf["discount_factors"],
        "PV of FCF": dcf["pv_fcfs"]
    })

    display_forecast = forecast_df.copy()
    display_forecast["FCF Growth"] = display_forecast["FCF Growth"].map(lambda x: f"{x:.1%}")
    display_forecast["Projected FCF"] = display_forecast["Projected FCF"].map(fmt_money)
    display_forecast["PV of FCF"] = display_forecast["PV of FCF"].map(fmt_money)
    display_forecast["Discount Factor"] = display_forecast["Discount Factor"].map(lambda x: f"{x:.3f}")

    st.subheader("Forecast Table")
    st.dataframe(display_forecast, use_container_width=True, hide_index=True)

    st.subheader("Sensitivity: Implied Share Price")
    wacc_values = np.arange(max(0.01, wacc - 0.02), wacc + 0.025, 0.005)
    terminal_values = np.arange(max(-0.02, terminal_growth - 0.01), terminal_growth + 0.0125, 0.005)

    sensitivity = pd.DataFrame(
        index=[f"{x:.1%}" for x in wacc_values],
        columns=[f"{x:.1%}" for x in terminal_values],
        dtype=float
    )

    for w in wacc_values:
        for tg in terminal_values:
            if w <= tg:
                sensitivity.loc[f"{w:.1%}", f"{tg:.1%}"] = np.nan
            else:
                temp = calculate_dcf(base_fcf, growth_rates, w, tg, net_debt, shares, current_price)
                sensitivity.loc[f"{w:.1%}", f"{tg:.1%}"] = temp["implied_share_price"]

    st.dataframe(sensitivity.style.format("${:,.2f}"), use_container_width=True)
    st.caption("Rows are WACC. Columns are terminal growth. Cells are implied share price.")

with tab5:
    st.header("Raw Yahoo Finance Data")

    with st.expander("Cash Flow Statement"):
        st.dataframe(cashflow, use_container_width=True)

    with st.expander("Balance Sheet"):
        st.dataframe(balance, use_container_width=True)

    with st.expander("Income Statement"):
        st.dataframe(income, use_container_width=True)

    with st.expander("Company Info"):
        info_df = pd.DataFrame(list(info.items()), columns=["Field", "Value"])
        st.dataframe(info_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Important: Yahoo/yfinance data can be missing, delayed, restated, or categorized differently across companies. "
    "Use the override fields when building a real valuation case."
)

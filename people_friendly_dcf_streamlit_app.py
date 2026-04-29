"""
People-Friendly DCF Valuation App
=================================


"""

import csv
import io
from dataclasses import dataclass

import streamlit as st


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="People-Friendly DCF App",
    page_icon="💰",
    layout="wide",
)


# -----------------------------
# Styling
# -----------------------------

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #061826 0%, #0f2742 45%, #24153d 100%);
            color: #f8fafc;
        }

        .main-title {
            padding: 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #2563eb, #9333ea);
            box-shadow: 0 20px 60px rgba(0,0,0,0.35);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.2);
        }

        .main-title h1 {
            color: white;
            font-size: 2.8rem;
            margin-bottom: 0.4rem;
        }

        .main-title p {
            color: #dbeafe;
            font-size: 1.05rem;
            line-height: 1.6;
        }

        .card {
            background: rgba(15, 23, 42, 0.82);
            padding: 1.2rem;
            border-radius: 22px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 16px 40px rgba(0,0,0,0.25);
            min-height: 140px;
        }

        .card-label {
            color: #94a3b8;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .card-value {
            color: white;
            font-size: 1.7rem;
            font-weight: 900;
            margin-bottom: 0.35rem;
        }

        .card-note {
            color: #cbd5e1;
            font-size: 0.82rem;
            line-height: 1.45;
        }

        .explain-box {
            background: rgba(30, 41, 59, 0.78);
            padding: 1rem 1.2rem;
            border-radius: 18px;
            border-left: 5px solid #38bdf8;
            margin-bottom: 1rem;
        }

        .custom-table {
            width: 100%;
            border-collapse: collapse;
            border-radius: 18px;
            overflow: hidden;
            font-size: 0.92rem;
            margin-top: 0.5rem;
        }

        .custom-table th {
            background: linear-gradient(135deg, #1d4ed8, #7e22ce);
            color: white;
            padding: 0.75rem;
            text-align: right;
            border: 1px solid rgba(255,255,255,0.12);
        }

        .custom-table th:first-child {
            text-align: left;
        }

        .custom-table td {
            padding: 0.7rem;
            text-align: right;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.76);
            color: #e5e7eb;
        }

        .custom-table td:first-child {
            text-align: left;
            font-weight: 700;
            color: #cbd5e1;
        }

        .custom-table tr:nth-child(even) td {
            background: rgba(30, 41, 59, 0.76);
        }

        .bar-shell {
            height: 14px;
            background: rgba(148, 163, 184, 0.25);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 0.45rem;
        }

        .bar-fill {
            height: 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #38bdf8);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617, #0f172a);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data model
# -----------------------------

@dataclass
class Inputs:
    company_name: str
    currency: str
    current_revenue: float
    revenue_growth: float
    years: int
    ebitda_margin: float
    depreciation_percent: float
    capex_percent: float
    nwc_percent: float
    tax_rate: float
    discount_rate: float
    terminal_growth: float
    net_debt: float
    cash_and_investments: float
    shares: float
    current_price: float


# -----------------------------
# Helper functions
# -----------------------------

def pct(x):
    return x / 100


def money(x, symbol="$"):
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}{symbol}{x / 1_000_000:,.2f}M"
    if x >= 1_000:
        return f"{sign}{symbol}{x / 1_000:,.2f}K"
    return f"{sign}{symbol}{x:,.2f}"


def percent(x):
    return f"{x:.1%}"


def table(headers, rows):
    html = '<table class="custom-table"><thead><tr>'
    for header in headers:
        html += f"<th>{header}</th>"
    html += "</tr></thead><tbody>"
    for row in rows:
        html += "<tr>"
        for cell in row:
            html += f"<td>{cell}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


def card(label, value, note):
    return f"""
    <div class="card">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        <div class="card-note">{note}</div>
    </div>
    """


def simple_bar(title, value, max_value, display):
    width = 0 if max_value == 0 else max(0, min(100, value / max_value * 100))
    return f"""
    <div class="card">
        <div class="card-label">{title}</div>
        <div class="card-value">{display}</div>
        <div class="bar-shell"><div class="bar-fill" style="width:{width:.1f}%;"></div></div>
    </div>
    """


def calculate_dcf(i: Inputs):
    rows = []
    revenue = i.current_revenue

    for year in range(1, i.years + 1):
        old_revenue = revenue
        revenue = revenue * (1 + i.revenue_growth)
        revenue_change = revenue - old_revenue

        ebitda = revenue * i.ebitda_margin
        depreciation = revenue * i.depreciation_percent
        ebit = ebitda - depreciation
        taxes = max(0, ebit * i.tax_rate)
        nopat = ebit - taxes
        capex = revenue * i.capex_percent
        change_nwc = revenue_change * i.nwc_percent
        fcf = nopat + depreciation - capex - change_nwc
        discount_factor = 1 / ((1 + i.discount_rate) ** year)
        pv_fcf = fcf * discount_factor

        rows.append({
            "year": year,
            "revenue": revenue,
            "revenue_growth": i.revenue_growth,
            "ebitda": ebitda,
            "depreciation": depreciation,
            "ebit": ebit,
            "taxes": taxes,
            "nopat": nopat,
            "capex": capex,
            "change_nwc": change_nwc,
            "fcf": fcf,
            "fcf_margin": fcf / revenue if revenue else 0,
            "discount_factor": discount_factor,
            "pv_fcf": pv_fcf,
        })

    final_fcf = rows[-1]["fcf"]
    terminal_value = final_fcf * (1 + i.terminal_growth) / (i.discount_rate - i.terminal_growth)
    pv_terminal_value = terminal_value / ((1 + i.discount_rate) ** i.years)
    pv_fcf = sum(r["pv_fcf"] for r in rows)
    enterprise_value = pv_fcf + pv_terminal_value
    equity_value = enterprise_value - i.net_debt + i.cash_and_investments
    fair_price = equity_value / i.shares if i.shares else 0
    upside = fair_price / i.current_price - 1 if i.current_price else 0

    summary = {
        "pv_fcf": pv_fcf,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_price": fair_price,
        "upside": upside,
        "terminal_value_weight": pv_terminal_value / enterprise_value if enterprise_value else 0,
    }

    return rows, summary


def scenario_analysis(base_inputs):
    cases = {
        "Bear Case": {
            "revenue_growth": base_inputs.revenue_growth - 0.03,
            "ebitda_margin": base_inputs.ebitda_margin - 0.03,
            "discount_rate": base_inputs.discount_rate + 0.01,
            "terminal_growth": max(0, base_inputs.terminal_growth - 0.005),
        },
        "Base Case": {},
        "Bull Case": {
            "revenue_growth": base_inputs.revenue_growth + 0.03,
            "ebitda_margin": base_inputs.ebitda_margin + 0.03,
            "discount_rate": max(0.01, base_inputs.discount_rate - 0.01),
            "terminal_growth": base_inputs.terminal_growth + 0.005,
        },
    }

    output = []
    for name, changes in cases.items():
        scenario = Inputs(**{**base_inputs.__dict__, **changes})
        if scenario.discount_rate <= scenario.terminal_growth:
            continue
        _, result = calculate_dcf(scenario)
        output.append({
            "case": name,
            "enterprise_value": result["enterprise_value"],
            "equity_value": result["equity_value"],
            "fair_price": result["fair_price"],
            "upside": result["upside"],
        })
    return output


def sensitivity_analysis(base_inputs):
    discount_rates = [base_inputs.discount_rate - 0.02, base_inputs.discount_rate - 0.01, base_inputs.discount_rate, base_inputs.discount_rate + 0.01, base_inputs.discount_rate + 0.02]
    terminal_growth_rates = [max(0, base_inputs.terminal_growth - 0.01), max(0, base_inputs.terminal_growth - 0.005), base_inputs.terminal_growth, base_inputs.terminal_growth + 0.005, base_inputs.terminal_growth + 0.01]

    results = []
    for tg in terminal_growth_rates:
        row = {"terminal_growth": tg}
        for dr in discount_rates:
            if dr <= tg:
                row[dr] = None
            else:
                scenario = Inputs(**{**base_inputs.__dict__, "discount_rate": dr, "terminal_growth": tg})
                _, result = calculate_dcf(scenario)
                row[dr] = result["fair_price"]
        results.append(row)

    return discount_rates, terminal_growth_rates, results


def csv_download(rows):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Year", "Revenue", "EBITDA", "Depreciation", "EBIT", "Taxes", "NOPAT",
        "Capex", "Change in NWC", "Free Cash Flow", "PV of FCF"
    ])
    for r in rows:
        writer.writerow([
            r["year"], r["revenue"], r["ebitda"], r["depreciation"], r["ebit"],
            r["taxes"], r["nopat"], r["capex"], r["change_nwc"], r["fcf"], r["pv_fcf"]
        ])
    return output.getvalue()


# -----------------------------
# Sidebar controls
# -----------------------------

with st.sidebar:
    st.title("⚙️ Assumptions")
    st.caption("Use these controls to build your DCF valuation.")

    company_name = st.text_input("Company name", "Sample Company")
    currency = st.selectbox("Currency", ["$", "€", "£", "¥", "₹"], index=0)

    st.subheader("Company size")
    current_revenue = st.number_input("Current revenue", min_value=0.0, value=1000.0, step=50.0)
    shares = st.number_input("Shares outstanding", min_value=0.01, value=100.0, step=5.0)
    current_price = st.number_input("Current share price", min_value=0.01, value=10.0, step=0.25)

    st.subheader("Forecast")
    years = st.slider("Forecast years", 3, 10, 5)
    revenue_growth = pct(st.slider("Annual revenue growth", -20.0, 50.0, 8.0, 0.5))
    ebitda_margin = pct(st.slider("EBITDA margin", -20.0, 70.0, 25.0, 0.5))

    st.subheader("Cash flow")
    depreciation_percent = pct(st.slider("Depreciation as % of revenue", 0.0, 25.0, 4.0, 0.5))
    capex_percent = pct(st.slider("Capex as % of revenue", 0.0, 40.0, 6.0, 0.5))
    nwc_percent = pct(st.slider("NWC as % of revenue growth", 0.0, 50.0, 8.0, 0.5))
    tax_rate = pct(st.slider("Tax rate", 0.0, 50.0, 21.0, 0.5))

    st.subheader("Valuation")
    discount_rate = pct(st.slider("Discount rate / WACC", 1.0, 30.0, 10.0, 0.25))
    terminal_growth = pct(st.slider("Terminal growth", 0.0, 8.0, 3.0, 0.25))
    net_debt = st.number_input("Net debt", value=250.0, step=25.0)
    cash_and_investments = st.number_input("Cash and investments", value=50.0, step=10.0)


inputs = Inputs(
    company_name=company_name,
    currency=currency,
    current_revenue=current_revenue,
    revenue_growth=revenue_growth,
    years=years,
    ebitda_margin=ebitda_margin,
    depreciation_percent=depreciation_percent,
    capex_percent=capex_percent,
    nwc_percent=nwc_percent,
    tax_rate=tax_rate,
    discount_rate=discount_rate,
    terminal_growth=terminal_growth,
    net_debt=net_debt,
    cash_and_investments=cash_and_investments,
    shares=shares,
    current_price=current_price,
)

if inputs.discount_rate <= inputs.terminal_growth:
    st.error("Discount rate must be greater than terminal growth rate.")
    st.stop()

forecast_rows, valuation = calculate_dcf(inputs)
scenario_rows = scenario_analysis(inputs)
discount_rates, terminal_growth_rates, sensitivity_rows = sensitivity_analysis(inputs)


# -----------------------------
# App header
# -----------------------------

st.markdown(
    f"""
    <div class="main-title">
        <h1>💰 People-Friendly DCF Valuation App</h1>
        <p>
            Build a discounted cash flow valuation for <b>{inputs.company_name}</b>. This app explains the key numbers,
            calculates enterprise value, converts it to equity value, estimates fair value per share, and shows scenarios.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

signal = "Undervalued" if valuation["upside"] > 0 else "Overvalued"
signal_icon = "🟢" if valuation["upside"] > 0 else "🔴"

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(card("Enterprise Value", money(valuation["enterprise_value"], inputs.currency), "Value of the whole business before debt/cash adjustments."), unsafe_allow_html=True)
with k2:
    st.markdown(card("Equity Value", money(valuation["equity_value"], inputs.currency), "Value available to common shareholders."), unsafe_allow_html=True)
with k3:
    st.markdown(card("Fair Price / Share", money(valuation["fair_price"], inputs.currency), "DCF estimate divided by shares outstanding."), unsafe_allow_html=True)
with k4:
    st.markdown(card("Upside / Downside", percent(valuation["upside"]), "Compared with the current share price."), unsafe_allow_html=True)
with k5:
    st.markdown(card("Model Signal", f"{signal_icon} {signal}", "Based only on the assumptions entered."), unsafe_allow_html=True)

st.write("")


# -----------------------------
# Tabs
# -----------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Explain It",
    "🧮 DCF Model",
    "🌉 Value Bridge",
    "🔥 Sensitivity",
    "📤 Export",
])

with tab1:
    st.markdown("### What this app is doing")
    st.markdown(
        """
        <div class="explain-box">
        A DCF valuation estimates what a business is worth today by forecasting future free cash flow,
        discounting those cash flows back to today, adding a terminal value, and then adjusting for debt and cash.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    max_value = max(abs(forecast_rows[-1]["revenue"]), abs(forecast_rows[-1]["ebitda"]), abs(forecast_rows[-1]["fcf"]), 1)
    with c1:
        st.markdown(simple_bar("Final-Year Revenue", abs(forecast_rows[-1]["revenue"]), max_value, money(forecast_rows[-1]["revenue"], inputs.currency)), unsafe_allow_html=True)
    with c2:
        st.markdown(simple_bar("Final-Year EBITDA", abs(forecast_rows[-1]["ebitda"]), max_value, money(forecast_rows[-1]["ebitda"], inputs.currency)), unsafe_allow_html=True)
    with c3:
        st.markdown(simple_bar("Final-Year Free Cash Flow", abs(forecast_rows[-1]["fcf"]), max_value, money(forecast_rows[-1]["fcf"], inputs.currency)), unsafe_allow_html=True)

    st.markdown("### Key interpretation")
    if valuation["terminal_value_weight"] > 0.80:
        st.warning("Terminal value is more than 80% of enterprise value. The valuation depends heavily on long-term assumptions.")
    else:
        st.success("Terminal value is below 80% of enterprise value, which is generally easier to defend than an extremely terminal-value-heavy model.")

with tab2:
    st.markdown("### Forecasted DCF Model")
    headers = ["Line Item"] + [f"Year {r['year']}" for r in forecast_rows]
    rows = [
        ["Revenue"] + [money(r["revenue"], inputs.currency) for r in forecast_rows],
        ["Revenue Growth"] + [percent(r["revenue_growth"]) for r in forecast_rows],
        ["EBITDA"] + [money(r["ebitda"], inputs.currency) for r in forecast_rows],
        ["Depreciation"] + [money(r["depreciation"], inputs.currency) for r in forecast_rows],
        ["EBIT"] + [money(r["ebit"], inputs.currency) for r in forecast_rows],
        ["Taxes"] + [money(r["taxes"], inputs.currency) for r in forecast_rows],
        ["NOPAT"] + [money(r["nopat"], inputs.currency) for r in forecast_rows],
        ["Capex"] + [money(r["capex"], inputs.currency) for r in forecast_rows],
        ["Change in NWC"] + [money(r["change_nwc"], inputs.currency) for r in forecast_rows],
        ["Free Cash Flow"] + [money(r["fcf"], inputs.currency) for r in forecast_rows],
        ["FCF Margin"] + [percent(r["fcf_margin"]) for r in forecast_rows],
        ["Discount Factor"] + [f"{r['discount_factor']:.3f}" for r in forecast_rows],
        ["PV of FCF"] + [money(r["pv_fcf"], inputs.currency) for r in forecast_rows],
    ]
    st.markdown(table(headers, rows), unsafe_allow_html=True)

with tab3:
    st.markdown("### Valuation Bridge")
    bridge_rows = [
        ["PV of Explicit Free Cash Flow", money(valuation["pv_fcf"], inputs.currency)],
        ["Terminal Value", money(valuation["terminal_value"], inputs.currency)],
        ["PV of Terminal Value", money(valuation["pv_terminal_value"], inputs.currency)],
        ["Enterprise Value", money(valuation["enterprise_value"], inputs.currency)],
        ["Less: Net Debt", money(-inputs.net_debt, inputs.currency)],
        ["Add: Cash and Investments", money(inputs.cash_and_investments, inputs.currency)],
        ["Equity Value", money(valuation["equity_value"], inputs.currency)],
        ["Shares Outstanding", f"{inputs.shares:,.2f}"],
        ["Fair Price Per Share", money(valuation["fair_price"], inputs.currency)],
        ["Current Share Price", money(inputs.current_price, inputs.currency)],
        ["Upside / Downside", percent(valuation["upside"])],
    ]
    st.markdown(table(["Item", "Value"], bridge_rows), unsafe_allow_html=True)

    st.markdown("### Scenario Analysis")
    scenario_table = []
    for r in scenario_rows:
        scenario_table.append([
            r["case"],
            money(r["enterprise_value"], inputs.currency),
            money(r["equity_value"], inputs.currency),
            money(r["fair_price"], inputs.currency),
            percent(r["upside"]),
        ])
    st.markdown(table(["Case", "Enterprise Value", "Equity Value", "Fair Price", "Upside / Downside"], scenario_table), unsafe_allow_html=True)

with tab4:
    st.markdown("### Sensitivity Table")
    st.caption("Rows change terminal growth. Columns change discount rate. Cells show fair price per share.")

    sensitivity_headers = ["Terminal Growth"] + [percent(dr) for dr in discount_rates]
    sensitivity_table = []
    for row in sensitivity_rows:
        new_row = [percent(row["terminal_growth"])]
        for dr in discount_rates:
            value = row[dr]
            new_row.append("N/A" if value is None else money(value, inputs.currency))
        sensitivity_table.append(new_row)

    st.markdown(table(sensitivity_headers, sensitivity_table), unsafe_allow_html=True)

with tab5:
    st.markdown("### Download Forecast")
    st.caption("This creates a simple CSV file. No Excel package is needed.")

    st.download_button(
        label="⬇️ Download DCF Forecast CSV",
        data=csv_download(forecast_rows),
        file_name=f"{inputs.company_name.lower().replace(' ', '_')}_dcf_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### How to run from Spyder")
    st.code(
        """# Install Streamlit once if needed:
pip install streamlit

# Run app from terminal:
streamlit run people_friendly_dcf_streamlit_app.py""",
        language="bash",
    )

st.caption("Educational model only. Not investment advice.")




"""
Financial Analyzer — Streamlit Dashboard
Run with: python -m streamlit run dashboard.py
"""

import os
import sys
import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetcher import fetch_yfinance


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock Analyzer",
    page_icon="📊",
    layout="wide",
)


# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────
def show_statement(df: pd.DataFrame, title: str):
    st.subheader(title)
    if df.empty or len(df) < 2:
        st.warning("Not enough data available for this statement.")
        return
    display_df = df.copy()
    for col in display_df.columns[1:]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


_ABSOLUTE_CR = {
    "revenue", "pat", "ebitda", "cfo", "cfi", "cff", "fcf",
    "long-term debt", "short-term debt",
}

def _format_ratio(key: str, value) -> str:
    """Return a human-readable value string with the correct unit."""
    if not isinstance(value, (int, float)):
        return str(value)
    k = key.lower()

    # Percentages first — prevents "EBITDA Margin %" being caught by "ebitda" substring
    if "%" in key:
        return f"{value:,.2f}%"

    # Days
    if "days" in k or "cycle" in k:
        return f"{value:,.1f} days"

    # Revenue per Employee — must come before the ₹ Cr check (also contains "revenue")
    if "₹ l" in k:
        return f"₹ {value:,.2f} L"

    # Absolute ₹ Crore values — exact key match to avoid substring false positives
    if k in _ABSOLUTE_CR:
        return f"₹ {value:,.2f} Cr"

    # Everything else is a multiplier (ratios, coverage, turnover, D/E, CFO/PAT …)
    return f"{value:,.2f}x"


def show_ratios(ratios: dict, label: str):
    st.subheader(label)
    if not ratios:
        st.info("No ratios computed — data may be unavailable.")
        return
    items = [(k, v) for k, v in ratios.items() if v is not None]
    cols = st.columns(4)
    for i, (k, v) in enumerate(items):
        with cols[i % 4]:
            st.metric(k, _format_ratio(k, v))


def waterfall_chart(result: dict):
    """P&L waterfall: Revenue → deductions → PAT."""
    pl       = result["pl"]
    curr_col = result["curr_year"]

    def _v(label_substr):
        for _, row in pl.iterrows():
            if label_substr.lower() in str(row.get("Particulars", "")).lower():
                try:
                    v = float(row.get(curr_col))
                    return v if not pd.isna(v) else None
                except (TypeError, ValueError):
                    return None
        return None

    revenue   = _v("revenue from operations")
    gross     = _v("gross profit")
    ebitda    = _v("ebitda")
    ebit      = _v("operating profit")
    pbt       = _v("profit before tax")
    pat       = _v("profit for the year")

    # Build stages: pick whatever chain we have
    stages = []
    if revenue:
        stages.append(("Revenue", revenue, "absolute"))
    if gross and revenue:
        stages.append(("Cost of Revenue", gross - revenue, "relative"))   # negative
        stages.append(("Gross Profit",    0,               "total"))
    if ebitda and (gross or revenue):
        base  = gross if gross else revenue
        stages.append(("Operating Costs", ebitda - base, "relative"))
        stages.append(("EBITDA",          0,             "total"))
    if ebit and ebitda:
        stages.append(("D&A",  ebit - ebitda, "relative"))
        stages.append(("EBIT", 0,             "total"))
    if pbt and (ebit or ebitda or revenue):
        base  = ebit if ebit else (ebitda if ebitda else revenue)
        stages.append(("Finance Costs & Other", pbt - base, "relative"))
        stages.append(("PBT", 0, "total"))
    if pat and pbt:
        stages.append(("Tax", pat - pbt, "relative"))
        stages.append(("PAT", 0, "total"))

    if len(stages) < 2:
        st.info("Not enough P&L data to draw waterfall chart.")
        return

    labels   = [s[0] for s in stages]
    vals     = [s[1] for s in stages]
    measures = [s[2] for s in stages]
    colors   = ["#26A65B" if m == "total" or v >= 0 else "#E74C3C"
                for v, m in zip(vals, measures)]

    fig = go.Figure(go.Waterfall(
        name="P&L", orientation="v",
        measure=measures,
        x=labels, y=vals,
        connector={"line": {"color": "rgb(63,63,63)"}},
        increasing={"marker": {"color": "#26A65B"}},
        decreasing={"marker": {"color": "#E74C3C"}},
        totals={"marker":    {"color": "#3498DB"}},
    ))
    fig.update_layout(
        title=f"P&L Waterfall — {result['curr_year']} (₹ Crores)",
        height=420, showlegend=False,
        margin=dict(l=20, r=20, t=45, b=60),
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig, use_container_width=True)


def debt_maturity_chart(result: dict):
    """Long-term vs short-term debt split."""
    ratios = result.get("ratios_sa", {})
    lt      = ratios.get("Long-term Debt")
    st_debt = ratios.get("Short-term Debt")

    if lt is None and st_debt is None:
        return

    # Metrics row
    c1, c2, c3 = st.columns(3)
    total_debt = (lt or 0) + (st_debt or 0)
    c1.metric("Total Debt (₹ Cr)",       f"{total_debt:,.1f}")
    c2.metric("Long-term Debt (₹ Cr)",   f"{lt:,.1f}"     if lt      is not None else "—")
    c3.metric("Short-term Debt (₹ Cr)",  f"{st_debt:,.1f}" if st_debt is not None else "—")

    if lt is not None and st_debt is not None and total_debt > 0:
        fig = go.Figure(go.Pie(
            labels=["Long-term Debt", "Short-term Debt"],
            values=[lt, st_debt],
            hole=0.45,
            marker_colors=["#636EFA", "#EF553B"],
            textinfo="label+percent",
        ))
        fig.update_layout(
            title="Debt Maturity Profile",
            height=320,
            margin=dict(l=20, r=20, t=45, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def bar_chart(df: pd.DataFrame, title: str, top_n: int = 10):
    if df.empty or len(df.columns) < 3:
        return
    num_df = df.copy()
    col_curr = num_df.columns[1]
    col_prev = num_df.columns[2]
    num_df[col_curr] = pd.to_numeric(num_df[col_curr], errors="coerce")
    num_df[col_prev] = pd.to_numeric(num_df[col_prev], errors="coerce")
    num_df = num_df.dropna(subset=[col_curr]).head(top_n)

    fig = go.Figure()
    fig.add_bar(name=col_prev, x=num_df["Particulars"], y=num_df[col_prev], marker_color="#636EFA")
    fig.add_bar(name=col_curr, x=num_df["Particulars"], y=num_df[col_curr], marker_color="#EF553B")
    fig.update_layout(
        title=title, barmode="group",
        xaxis_tickangle=-35, height=400,
        margin=dict(l=20, r=20, t=40, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)


TREND_GROUPS = [
    ("Revenue, PAT & EBITDA  (₹ Cr)", ["Revenue", "PAT", "EBITDA"]),
    ("Margins (%)",                    ["Net Profit Margin %", "EBITDA Margin %",
                                        "Return on Equity %", "Return on Assets %"]),
    ("Leverage",                       ["Debt-to-Equity", "Interest Coverage"]),
    ("Growth (%)",                     ["Revenue Growth YoY %", "PAT Growth YoY %"]),
    ("Cash Quality",                   ["CFO / PAT", "Current Ratio"]),
]

COMPARE_KEYS = [
    "Revenue", "Net Profit Margin %", "EBITDA Margin %",
    "Return on Equity %", "Return on Assets %",
    "Revenue Growth YoY %", "PAT Growth YoY %",
    "Debt-to-Equity", "Interest Coverage",
    "Current Ratio", "CFO / PAT", "Receivables Days",
    "Cash Conversion Cycle",
]

BAR_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]


def trend_charts(result: dict):
    history = result.get("ratios_history", {})
    if len(history) < 2:
        st.info("Need at least 2 years of data for trend charts.")
        return

    years = sorted(history.keys())   # oldest → newest on x-axis

    for group_name, keys in TREND_GROUPS:
        series = {}
        for k in keys:
            vals = [history[y].get(k) for y in years]
            if any(v is not None for v in vals):
                series[k] = vals
        if not series:
            continue

        fig = go.Figure()
        for k, vals in series.items():
            fig.add_trace(go.Scatter(
                x=years, y=vals, name=k,
                mode="lines+markers+text",
                text=[f"{v:.1f}" if v is not None else "" for v in vals],
                textposition="top center",
                connectgaps=True,
            ))
        fig.update_layout(
            title=group_name, height=320,
            margin=dict(l=20, r=20, t=45, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)


def compare_section(main_result: dict, peers: list):
    all_cos = {main_result["company"]: main_result["ratios_sa"]}
    for p in peers:
        all_cos[p["company"]] = p["ratios"]
    companies = list(all_cos.keys())

    # ── Comparison table ──────────────────────────────────────────────────
    rows = []
    for k in COMPARE_KEYS:
        row = {"Metric": k}
        for c in companies:
            v = all_cos[c].get(k)
            row[c] = _format_ratio(k, v) if v is not None else "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Bar charts for key metrics ────────────────────────────────────────
    st.divider()
    bar_metrics = [
        "Net Profit Margin %", "Return on Equity %",
        "Revenue Growth YoY %", "Debt-to-Equity", "Current Ratio",
    ]
    chart_cols = st.columns(2)
    ci = 0
    for k in bar_metrics:
        pairs = [(c, all_cos[c].get(k)) for c in companies
                 if all_cos[c].get(k) is not None]
        if len(pairs) < 2:
            continue
        fig = go.Figure(go.Bar(
            x=[p[0] for p in pairs],
            y=[p[1] for p in pairs],
            marker_color=BAR_COLORS[:len(pairs)],
            text=[_format_ratio(k, p[1]) for p in pairs],
            textposition="auto",
        ))
        fig.update_layout(
            title=k, height=300,
            margin=dict(l=20, r=20, t=45, b=30),
            showlegend=False,
        )
        with chart_cols[ci % 2]:
            st.plotly_chart(fig, use_container_width=True)
        ci += 1


def _sector_filter(ratios: dict, sector: str) -> tuple:
    """
    Returns (filtered_ratios, notes, hidden_categories).
    filtered_ratios  — ratios dict with sector-irrelevant items removed.
    notes            — list of st.info() messages to show in the Ratios tab.
    hidden_hc        — set of Health Check category names to suppress.
    """
    s = (sector or "").lower()
    is_banking = any(x in s for x in ["financial", "bank", "insurance", "nbfc"])
    is_it      = any(x in s for x in ["technology", "software", "it service"])
    is_fmcg    = any(x in s for x in ["consumer", "fmcg", "food", "beverage", "retail"])

    hide       = set()
    notes      = []
    hidden_hc  = set()   # Health Check category names to suppress

    if is_banking:
        hide = {
            "Current Ratio", "Quick Ratio", "Cash Ratio",
            "Inventory Days", "Payables Days", "Cash Conversion Cycle",
            "Debt-to-Equity", "Debt-to-Assets", "Long-term Debt", "Short-term Debt",
        }
        hidden_hc = {"Liquidity"}
        notes.append(
            "**Banking sector detected.** Standard liquidity and leverage ratios (Current Ratio, D/E) "
            "are hidden — they are structurally misleading for banks.  \n"
            "Key banking metrics — **NIM, CAR, GNPA %, CASA ratio** — require RBI/exchange filing data "
            "not available via Yahoo Finance and are not shown here."
        )

    elif is_it:
        hide = {"Inventory Days", "Payables Days", "Cash Conversion Cycle"}
        hidden_hc = {"Efficiency"}   # CCC/Inventory Days not meaningful for IT
        rpe = ratios.get("Revenue per Employee (₹ L)")
        if rpe:
            notes.append(
                f"**IT sector:** Revenue per employee ₹ {rpe:,.2f} L/year.  \n"
                "Indian IT majors (TCS, Infosys, Wipro) typically range ₹ 35–55 L per employee."
            )

    elif is_fmcg:
        cr = ratios.get("Current Ratio")
        if cr is not None and cr < 1:
            notes.append(
                "**FMCG sector:** Current Ratio below 1 is intentional and healthy here — "
                "large FMCG companies collect from consumers immediately but pay suppliers on "
                "60–90 day credit terms, effectively running negative working capital."
            )
        ccc = ratios.get("Cash Conversion Cycle")
        if ccc is not None and ccc <= 0:
            notes.append(
                f"**FMCG sector:** Negative Cash Conversion Cycle ({ccc:.0f} days) is a strength — "
                "suppliers fund the business. This is a hallmark of dominant FMCG brands."
            )

    filtered = {k: v for k, v in ratios.items() if k not in hide}
    return filtered, notes, hidden_hc


def render_results(result: dict):
    company = result["company"]
    info    = result.get("info", {})

    # ── Market data strip
    mc = info.get("marketCap")
    pe = info.get("trailingPE")
    hi = info.get("fiftyTwoWeekHigh", "—")
    lo = info.get("fiftyTwoWeekLow",  "—")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Cap",      f"₹{mc/1e7:,.0f} Cr" if mc else "—")
    c2.metric("Sector",          info.get("sector") or "—")
    c3.metric("P/E (TTM)",       f"{pe:.1f}x" if pe else "—")
    c4.metric("52W High / Low",  f"₹{hi} / ₹{lo}")
    st.divider()

    # ── Warnings
    for w in result.get("warnings", []):
        st.info(w)

    # ── Sector filter (computed once, used in Ratios + Health Check tabs)
    sector = info.get("sector", "")
    filtered_ratios, sector_notes, hidden_hc = _sector_filter(result["ratios_sa"], sector)

    # ── Peer session state — reset when main company changes ──────────────
    if st.session_state.get("_peer_anchor") != result["ticker"]:
        st.session_state["peers"]        = []
        st.session_state["_peer_anchor"] = result["ticker"]
    if "peers" not in st.session_state:
        st.session_state["peers"] = []

    # ── Tabs
    tabs = st.tabs(["Financials", "Ratios", "Charts", "Trends", "Health Check", "Compare"])

    # Tab 0 — Financials
    with tabs[0]:
        show_statement(result["bs"], f"Balance Sheet — {company}  (₹ Crores)")
        st.divider()
        show_statement(result["pl"], f"Profit & Loss — {company}  (₹ Crores)")
        st.divider()
        show_statement(result["cf"], f"Cash Flow — {company}  (₹ Crores)")

    # Tab 1 — Ratios
    with tabs[1]:
        for note in sector_notes:
            st.info(note)
        show_ratios(filtered_ratios, "Key Financial Ratios")
        if result["interp"]:
            st.divider()
            st.subheader("Interpretation")
            for line in result["interp"]:
                st.write(f"- {line}")

    # Tab 2 — Charts
    with tabs[2]:
        waterfall_chart(result)
        st.divider()
        debt_maturity_chart(result)
        st.divider()
        bar_chart(result["pl"], "P&L — Year-on-Year Comparison", top_n=10)
        st.divider()
        bar_chart(result["bs"], "Balance Sheet — Year-on-Year Comparison", top_n=10)

    # Tab 3 — Trends
    with tabs[3]:
        st.subheader(f"4-Year Trend — {company}")
        trend_charts(result)

    # Tab 4 — Health Check
    with tabs[4]:
        findings = result.get("findings", [])
        if not findings:
            st.info("Not enough ratio data to run benchmarking.")
        else:
            # ── Summary counts ────────────────────────────────────────────
            counts = {v: sum(1 for f in findings if f.get("verdict") == v)
                      for v in ("strong", "good", "watch", "concern")}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🟢 Strong",  counts["strong"])
            c2.metric("🟢 Good",    counts["good"])
            c3.metric("🟡 Watch",   counts["watch"])
            c4.metric("🔴 Concern", counts["concern"])
            st.divider()

            # ── Red Flags ─────────────────────────────────────────────────
            red_flags    = [f for f in findings if f.get("verdict") == "concern"]

            # Compound red flag checks
            ratios = result.get("ratios_sa", {})
            compound = []
            if ratios.get("Debt-to-Equity", 0) > 1 and (ratios.get("Revenue Growth YoY %") or 0) < 0:
                compound.append("High debt combined with falling revenue — debt repayment capacity at risk.")
            if (ratios.get("Current Ratio") or 2) < 1 and (ratios.get("CFO / PAT") or 1) < 0.5:
                compound.append("Weak liquidity AND poor cash conversion — potential near-term cash crunch.")
            if (ratios.get("Interest Coverage") or 5) < 1.5 and (ratios.get("Debt-to-Equity") or 0) > 1.5:
                compound.append("High debt load with insufficient interest coverage — elevated default risk.")
            if (ratios.get("PAT Growth YoY %") or 0) < 0 and (ratios.get("Net Profit Margin %") or 10) < 5:
                compound.append("Profits declining AND margins already thin — profitability under serious pressure.")

            if red_flags or compound:
                st.subheader("🚨 Red Flags")
                for f in red_flags:
                    st.error(f"**{f['ratio']}** — {f['message']}  \n`Benchmark: {f['benchmark']}`")
                for c in compound:
                    st.error(f"**Combined Risk:** {c}")
                st.divider()

            # ── Categorised findings ──────────────────────────────────────
            ALL_CATEGORIES = {
                "Profitability": ["Net Profit Margin %", "EBITDA Margin %",
                                  "Return on Equity %", "Return on Assets %"],
                "Liquidity":     ["Current Ratio", "Quick Ratio"],
                "Leverage":      ["Debt-to-Equity", "Interest Coverage"],
                "Growth":        ["Revenue Growth YoY %", "PAT Growth YoY %"],
                "Cash Quality":  ["CFO / PAT"],
                "Efficiency":    ["Receivables Days", "Cash Conversion Cycle"],
            }
            CATEGORIES = {k: v for k, v in ALL_CATEGORIES.items() if k not in hidden_hc}
            findings_map = {f["ratio"]: f for f in findings}

            def _show(f):
                v       = f.get("verdict", "watch")
                label   = {"strong": "🟢 Strong", "good": "🟢 Good",
                           "watch": "🟡 Watch", "concern": "🔴 Concern"}.get(v, "🟡")
                text    = f"**{f['ratio']}** &nbsp;·&nbsp; `{f['benchmark']}`  \n{label} — {f['message']}"
                if v in ("strong", "good"):
                    st.success(text)
                elif v == "watch":
                    st.warning(text)
                else:
                    st.error(text)

            for category, ratio_names in CATEGORIES.items():
                cat_findings = [findings_map[r] for r in ratio_names if r in findings_map]
                if not cat_findings:
                    continue
                st.subheader(category)
                for f in cat_findings:
                    _show(f)
                st.divider()

    # Tab 5 — Compare
    with tabs[5]:
        st.subheader("Peer Comparison")
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            peer_input = st.text_input(
                "Add a peer (ticker or company name)", key="peer_input",
                placeholder="e.g. WIPRO, TCS, Tata Consultancy",
            )
        with c2:
            add_btn = st.button("Add Peer", use_container_width=True)
        with c3:
            clear_btn = st.button("Clear Peers", use_container_width=True)

        if clear_btn:
            st.session_state["peers"] = []

        if add_btn and peer_input.strip():
            already = [p["ticker"] for p in st.session_state["peers"]]
            with st.spinner(f"Fetching {peer_input.strip()}…"):
                try:
                    pr = fetch_yfinance(peer_input.strip())
                    if pr["ticker"] == result["ticker"]:
                        st.warning("That's the same company you're already analysing.")
                    elif pr["ticker"] in already:
                        st.warning(f"{pr['company']} is already in the comparison.")
                    else:
                        st.session_state["peers"].append({
                            "company": pr["company"],
                            "ticker":  pr["ticker"],
                            "ratios":  pr["ratios_sa"],
                            "sector":  pr["info"].get("sector", ""),
                        })
                        st.success(f"Added {pr['company']} ({pr['ticker']})")
                except Exception as e:
                    st.error(f"Could not fetch '{peer_input.strip()}': {e}")

        peers = st.session_state.get("peers", [])
        if not peers:
            st.info("Add 1–3 peer companies above to compare side by side.")
        else:
            st.caption(f"Comparing: **{company}** (main)  +  " +
                       "  |  ".join(f"**{p['company']}**" for p in peers))
            st.divider()
            compare_section(result, peers)

    # ── Download
    st.divider()
    json_out = json.dumps({
        "company": company,
        "balance_sheet":   result["bs"].to_dict(orient="records"),
        "profit_and_loss": result["pl"].to_dict(orient="records"),
        "cash_flow":       result["cf"].to_dict(orient="records"),
        "ratios":          result["ratios_sa"],
        "cross_reference": result["findings"],
    }, indent=2, default=str)

    st.download_button(
        label="Download analysis as JSON",
        data=json_out,
        file_name=f"analysis_{company[:30].replace(' ', '_')}.json",
        mime="application/json",
    )


# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.title("📊 Indian Stock Financial Analyzer")
st.caption("Enter any NSE / BSE listed company ticker to instantly fetch and analyze its financials.")

col_input, col_btn = st.columns([4, 1])
with col_input:
    ticker_input = st.text_input(
        "Stock ticker",
        placeholder="Ticker: INFY · SHAKTIPUMP · HDFCBANK   or   Name: Shakti Pumps · Infosys",
        label_visibility="collapsed",
    )
with col_btn:
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

st.caption("Enter a ticker (e.g. `SHAKTIPUMP`) or a company name (e.g. `Shakti Pumps`). NSE suffix added automatically.")

# ── On Analyze click: fetch and store in session_state ───────────────────────
if analyze_btn:
    if not ticker_input.strip():
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching financials for **{ticker_input.strip().upper()}**…"):
            try:
                result = fetch_yfinance(ticker_input.strip())
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                st.stop()

        if result["pl"].empty and result["bs"].empty:
            st.error(
                f"No financial data found for **{ticker_input.strip().upper()}**. "
                "Double-check the ticker symbol and try again."
            )
        else:
            # Store in session_state so it survives button reruns
            st.session_state["fin_result"] = result
            st.session_state["fin_ticker"] = ticker_input.strip()

# ── Always render if we have data in session_state ───────────────────────────
if "fin_result" in st.session_state:
    result      = st.session_state["fin_result"]
    ticker_used = st.session_state["fin_ticker"]
    company     = result["company"]

    st.success(
        f"**{company}** ({result['ticker']})  ·  "
        f"{result['curr_year']} & {result['prev_year']}  ·  Source: Yahoo Finance"
    )
    render_results(result)

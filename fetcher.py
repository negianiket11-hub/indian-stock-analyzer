"""
yfinance data fetcher and ratio engine for Indian listed companies.
Fetches annual financials, reshapes them, and computes financial ratios.

Usage:
    from fetcher import fetch_yfinance
    result = fetch_yfinance("INFY")         # NSE suffix added automatically
    result = fetch_yfinance("INFY.NS")      # explicit NSE
    result = fetch_yfinance("500209.BO")    # BSE code
"""

import numpy as np
import re

import pandas as pd
import yfinance as yf

# ── Unit conversion ───────────────────────────────────────────────────────────
# yfinance returns absolute INR values. Divide by 1 crore (10^7) for readability.
CRORE = 1_00_00_000  # 10,000,000

# ── Field maps ────────────────────────────────────────────────────────────────
# (yfinance_key, label_in_dataframe)
# Labels are chosen to match keyword searches inside compute_ratios().

INCOME_MAP = [
    ("Total Revenue",                  "Revenue from operations"),
    ("Gross Profit",                   "Gross profit"),
    ("EBITDA",                         "EBITDA"),
    ("Operating Income",               "Operating profit (EBIT)"),
    ("Pretax Income",                  "Profit before tax"),
    ("Tax Provision",                  "Tax expense"),
    ("Net Income",                     "Profit for the year"),
    ("Interest Expense",               "Finance costs"),
    ("Reconciled Depreciation",        "Depreciation and amortization"),
    ("Total Expenses",                 "Total expenses"),
]

BALANCE_MAP = [
    ("Total Assets",                                    "Total assets"),
    ("Current Assets",                                  "Total current assets"),
    ("Cash And Cash Equivalents",                       "Cash and cash equivalents"),
    ("Cash Cash Equivalents And Short Term Investments","Cash and cash equivalents"),
    ("Accounts Receivable",                             "Trade receivables"),
    ("Inventory",                                       "Inventories"),
    ("Net PPE",                                         "Property, plant and equipment"),
    ("Goodwill And Other Intangible Assets",            "Intangible assets"),
    ("Total Non Current Assets",                        "Total non-current assets"),
    ("Total Liabilities Net Minority Interest",         "Total liabilities"),
    ("Current Liabilities",                             "Total current liabilities"),
    ("Accounts Payable",                                "Trade payables"),
    ("Total Debt",                                      "Total borrowings"),
    ("Long Term Debt",                                  "Long-term borrowings"),
    ("Stockholders Equity",                             "Total equity"),
    ("Retained Earnings",                               "Retained earnings"),
]

CASHFLOW_MAP = [
    ("Operating Cash Flow",            "Net cash from operating"),
    ("Investing Cash Flow",            "Net cash from investing"),
    ("Financing Cash Flow",            "Net cash from financing"),
    ("Free Cash Flow",                 "Free cash flow"),
    ("Capital Expenditure",            "Capital expenditure"),
    ("Depreciation And Amortization",  "Depreciation and amortization"),
    ("Change In Working Capital",      "Changes in working capital"),
    ("Dividends Paid",                 "Dividends paid"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _fy_label(date) -> str:
    """Convert a pandas Timestamp to a fiscal year label like 'FY2025'."""
    try:
        return f"FY{pd.Timestamp(date).year}"
    except Exception:
        return str(date)[:6]


def _to_df(
    raw: pd.DataFrame,
    field_map: list,
    curr_col: str,
    prev_col: str,
) -> pd.DataFrame:
    """
    Reshape a yfinance statement DataFrame into:
        Particulars | <curr_col> | <prev_col>
    Values are converted from absolute INR to ₹ Crores.
    """
    empty = pd.DataFrame(columns=["Particulars", curr_col, prev_col])
    if raw is None or raw.empty:
        return empty

    cols_sorted = sorted(raw.columns, reverse=True)
    date_curr = cols_sorted[0] if len(cols_sorted) >= 1 else None
    date_prev = cols_sorted[1] if len(cols_sorted) >= 2 else None

    records = []
    seen_labels: set[str] = set()

    for yf_key, label in field_map:
        if yf_key not in raw.index or label in seen_labels:
            continue
        seen_labels.add(label)
        row = raw.loc[yf_key]

        curr_val = _safe_float(row[date_curr]) if date_curr is not None else None
        prev_val = _safe_float(row[date_prev]) if date_prev is not None else None

        if curr_val is not None:
            curr_val = round(curr_val / CRORE, 2)
        if prev_val is not None:
            prev_val = round(prev_val / CRORE, 2)

        if curr_val is not None or prev_val is not None:
            records.append({"Particulars": label, curr_col: curr_val, prev_col: prev_val})

    if not records:
        return empty
    return pd.DataFrame(records).reset_index(drop=True)


def _to_df_multi(
    raw: pd.DataFrame,
    field_map: list,
    year_labels: list,          # e.g. ["FY2025", "FY2024", "FY2023", "FY2022"]
) -> pd.DataFrame:
    """Like _to_df but retains all available year columns for trend analysis."""
    cols = ["Particulars"] + year_labels
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)

    date_cols = sorted(raw.columns, reverse=True)[:len(year_labels)]
    records   = []
    seen: set = set()

    for yf_key, label in field_map:
        if yf_key not in raw.index or label in seen:
            continue
        seen.add(label)
        row   = raw.loc[yf_key]
        entry = {"Particulars": label}
        has_data = False
        for date_col, fy_label in zip(date_cols, year_labels):
            val = _safe_float(row.get(date_col))
            if val is not None:
                val      = round(val / CRORE, 2)
                has_data = True
            entry[fy_label] = val
        if has_data:
            records.append(entry)

    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records).reset_index(drop=True)


# ── Ratio engine (moved from main.py) ────────────────────────────────────────

def get_val(df: pd.DataFrame, keywords: list[str], year: str = "FY2025") -> float | None:
    for kw in keywords:
        matches = df[df["Particulars"].str.lower().str.contains(kw.lower(), na=False)]
        if not matches.empty:
            val = matches.iloc[-1][year]
            try:
                v = float(val)
                if not np.isnan(v):
                    return v
            except (TypeError, ValueError):
                pass
    return None


def compute_ratios(bs: pd.DataFrame, pl: pd.DataFrame, cf: pd.DataFrame,
                   curr_year: str = "FY2025", prev_year: str = "FY2024") -> dict:
    def g(df, keys, year=None):
        return get_val(df, keys, year if year is not None else curr_year)

    total_assets    = g(bs, ["total assets", "sum of assets"])
    total_equity    = g(bs, ["total equity", "net worth", "stockholders equity", "shareholders"])
    total_debt      = g(bs, ["total borrowings", "borrowings"])
    long_term_debt  = g(bs, ["long-term borrowings"])
    current_assets  = g(bs, ["total current assets", "current assets"])
    current_liab    = g(bs, ["total current liabilities", "current liabilities"])
    inventory       = g(bs, ["inventories"])
    receivables     = g(bs, ["trade receivables"])
    payables        = g(bs, ["trade payables"])
    cash            = g(bs, ["cash and cash equivalents"])

    revenue      = g(pl, ["revenue from operations", "net revenue",
                           "income from operations", "revenue from contracts"])
    pbt          = g(pl, ["profit before tax"])
    pat          = g(pl, ["profit for the year", "profit after tax",
                           "net income", "profit attributable"])
    depreciation = g(pl, ["depreciation and amortization", "depreciation & amortisation",
                           "depreciation"])
    interest     = g(pl, ["finance costs"])
    gross_profit = g(pl, ["gross profit"])
    ebitda       = g(pl, ["ebitda"])

    revenue_24      = g(pl, ["revenue from operations", "net revenue",
                              "income from operations", "revenue from contracts"], prev_year)
    pat_24          = g(pl, ["profit for the year", "profit after tax",
                              "net income", "profit attributable"], prev_year)
    total_assets_24 = g(bs, ["total assets", "sum of assets"], prev_year)

    cfo = g(cf, ["net cash from operating", "net cash generated from operating",
                  "net cash generated from operations", "net cash from operations",
                  "cash generated from operations",
                  "net cash provided by operating"])
    cfi = g(cf, ["net cash from investing", "net cash used in investing"])
    cff = g(cf, ["net cash from financing", "net cash used in financing"])

    if ebitda is None and pbt is not None and depreciation is not None and interest is not None:
        ebitda = pbt + depreciation + interest

    def sd(a, b):
        return round(a / b, 4) if (a is not None and b and b != 0) else None

    def pct(a, b):
        v = sd(a, b)
        return round(v * 100, 2) if v is not None else None

    def yoy(curr, prev):
        if curr is None or not prev or prev == 0: return None
        return round((curr - prev) / abs(prev) * 100, 2)

    ratios = {
        "Revenue":               revenue,
        "PAT":                   pat,
        "EBITDA":                ebitda,
        "Gross Profit Margin %": pct(gross_profit, revenue),
        "EBITDA Margin %":       pct(ebitda, revenue),
        "Net Profit Margin %":   pct(pat, revenue),
        "Return on Assets %":    pct(pat, total_assets),
        "Return on Equity %":    pct(pat, total_equity),
        "Current Ratio":         sd(current_assets, current_liab),
        "Quick Ratio":           sd((current_assets - inventory)
                                    if current_assets and inventory else None, current_liab),
        "Cash Ratio":            sd(cash, current_liab),
        "Debt-to-Equity":        sd(total_debt, total_equity),
        "Debt-to-Assets":        sd(total_debt, total_assets),
        "Interest Coverage":     sd(ebitda, interest),
        "Asset Turnover":        sd(revenue, total_assets),
        "Receivables Days":      round(sd(receivables, revenue) * 365, 1)
                                   if sd(receivables, revenue) else None,
        "Inventory Days":        round(sd(inventory, revenue) * 365, 1)
                                   if sd(inventory, revenue) else None,
        "Payables Days":         round(sd(payables, revenue) * 365, 1)
                                   if sd(payables, revenue) else None,
        "Cash Conversion Cycle": round(
                                   (sd(inventory,   revenue) or 0) * 365 +
                                   (sd(receivables, revenue) or 0) * 365 -
                                   (sd(payables,    revenue) or 0) * 365, 1)
                                   if (inventory is not None and receivables is not None
                                       and payables is not None and revenue) else None,
        "Long-term Debt":        long_term_debt,
        "Short-term Debt":       round(total_debt - long_term_debt, 2)
                                   if (total_debt is not None and long_term_debt is not None) else None,
        "CFO":                   cfo,
        "CFI":                   cfi,
        "CFF":                   cff,
        "FCF":                   round(cfo + cfi, 2) if cfo and cfi else None,
        "CFO / PAT":             sd(cfo, pat),
        "Revenue Growth YoY %":  yoy(revenue, revenue_24),
        "PAT Growth YoY %":      yoy(pat, pat_24),
        "Asset Growth YoY %":    yoy(total_assets, total_assets_24),
    }
    return {k: v for k, v in ratios.items() if v is not None}


def interpret_ratios(ratios: dict, sector: str = None) -> list[str]:
    notes = []
    s = (sector or "").lower()

    is_it      = any(x in s for x in ["technology", "software", "it service"])
    is_banking = any(x in s for x in ["financial", "bank", "insurance", "nbfc"])
    is_fmcg    = any(x in s for x in ["consumer", "fmcg", "food", "beverage", "retail"])
    is_pharma  = any(x in s for x in ["healthcare", "pharma", "biotech", "drug"])
    is_infra   = any(x in s for x in ["real estate", "construction", "infrastructure",
                                       "utility", "utilities", "energy", "power"])
    is_mfg     = any(x in s for x in ["industrial", "manufactur", "material",
                                       "chemical", "metal", "auto", "engineering"])

    npm = ratios.get("Net Profit Margin %")
    if npm is not None:
        if is_it:
            if npm >= 20:
                notes.append(f"Net Margin {npm:.1f}% — excellent for an IT company. Indian IT majors (TCS, Infosys) typically operate at 18–25%. This indicates strong pricing power and cost discipline.")
            elif npm >= 15:
                notes.append(f"Net Margin {npm:.1f}% — healthy for IT, though below the 20%+ seen in tier-1 players. Watch for rising employee costs or client pricing pressure.")
            elif npm >= 10:
                notes.append(f"Net Margin {npm:.1f}% — below the IT sector norm of 15%+. Could indicate wage inflation, project mix shift to lower-margin work, or ramp-up costs.")
            else:
                notes.append(f"Net Margin {npm:.1f}% — weak for IT where asset-light models should produce 15%+ margins. Investigate attrition costs, offshore-onshore mix, or pricing erosion.")
        elif is_banking:
            notes.append(f"Net Margin {npm:.1f}% — for banks, Net Interest Margin (NIM) and Return on Assets are more meaningful than net margin. This ratio should be interpreted with caution.")
        elif is_fmcg:
            if npm >= 15:
                notes.append(f"Net Margin {npm:.1f}% — excellent for FMCG. Companies like HUL and Nestle operate at 12–18%. Strong brand premiums are likely driving this.")
            elif npm >= 8:
                notes.append(f"Net Margin {npm:.1f}% — healthy FMCG margin. Volumes and distribution efficiency are holding profitability at a reasonable level.")
            elif npm >= 4:
                notes.append(f"Net Margin {npm:.1f}% — thin for FMCG. Rising raw material or advertising costs may be squeezing margins. Watch gross margin trends.")
            else:
                notes.append(f"Net Margin {npm:.1f}% — weak for FMCG. FMCG businesses should consistently earn 8%+ margins. Check for a price war or cost spike.")
        elif is_infra:
            if npm >= 10:
                notes.append(f"Net Margin {npm:.1f}% — strong for an infra/real estate company where 5–8% is typical. Execution quality or asset monetisation may be driving this.")
            elif npm >= 5:
                notes.append(f"Net Margin {npm:.1f}% — acceptable for a capital-heavy infra business. Ensure debt servicing costs are not about to compress this further.")
            else:
                notes.append(f"Net Margin {npm:.1f}% — thin for infra. With significant debt typical in this sector, thin margins leave little cushion for cost overruns or rate hikes.")
        elif is_mfg:
            if npm >= 12:
                notes.append(f"Net Margin {npm:.1f}% — strong for manufacturing. Suggests value-added products, pricing power, or superior cost management vs peers.")
            elif npm >= 6:
                notes.append(f"Net Margin {npm:.1f}% — reasonable for manufacturing. Keep an eye on raw material cost cycles which can swing margins quickly.")
            else:
                notes.append(f"Net Margin {npm:.1f}% — thin for manufacturing. Commodity pricing or wage inflation may be weighing on margins.")
        elif is_pharma:
            if npm >= 18:
                notes.append(f"Net Margin {npm:.1f}% — excellent for pharma. Branded generics or patented products typically drive margins above 18%.")
            elif npm >= 10:
                notes.append(f"Net Margin {npm:.1f}% — healthy for pharma. Generic businesses typically earn 10–18%; US market access boosts this range.")
            else:
                notes.append(f"Net Margin {npm:.1f}% — below pharma norms. API or contract manufacturing margins are lower, but branded generics should do better.")
        else:
            if npm >= 15: notes.append(f"Net Margin {npm:.1f}% — excellent profitability.")
            elif npm >= 8: notes.append(f"Net Margin {npm:.1f}% — healthy profitability.")
            else:          notes.append(f"Net Margin {npm:.1f}% — thin margins; limited buffer against cost shocks.")

    cr = ratios.get("Current Ratio")
    if cr is not None:
        if is_banking:
            notes.append(f"Current Ratio {cr:.2f}x — this metric is not meaningful for banks, which operate on fractional reserves and have structural current liability mismatches by design.")
        elif is_fmcg:
            if cr < 1:
                notes.append(f"Current Ratio {cr:.2f}x — below 1, which is actually common and healthy in FMCG. Large FMCG companies often run negative working capital: they collect from consumers immediately but pay suppliers on credit terms of 60–90 days.")
            elif cr >= 1.5:
                notes.append(f"Current Ratio {cr:.2f}x — strong liquidity. Higher than typical FMCG, suggesting conservative cash management or low payable leverage with suppliers.")
            else:
                notes.append(f"Current Ratio {cr:.2f}x — adequate for FMCG where near-1 ratios are normal.")
        elif is_infra:
            if cr >= 1.5:
                notes.append(f"Current Ratio {cr:.2f}x — good for an infra/real estate company where project-based billing can cause lumpy receivables.")
            elif cr >= 1:
                notes.append(f"Current Ratio {cr:.2f}x — adequate but tight for infra. Watch for delayed project payments stressing working capital.")
            else:
                notes.append(f"Current Ratio {cr:.2f}x — below 1. For infra companies dependent on milestone billing, this signals near-term liquidity risk.")
        else:
            if cr >= 2:   notes.append(f"Current Ratio {cr:.2f}x — strong short-term liquidity. Company can comfortably meet near-term obligations.")
            elif cr >= 1: notes.append(f"Current Ratio {cr:.2f}x — adequate liquidity, though leaving limited buffer.")
            else:         notes.append(f"Current Ratio {cr:.2f}x — current liabilities exceed current assets. Monitor cash flow closely.")

    de = ratios.get("Debt-to-Equity")
    if de is not None:
        if is_banking:
            notes.append(f"D/E {de:.2f}x — high leverage is structurally normal for banks. Capital Adequacy Ratio (CAR) and GNPA % are the relevant risk metrics, not D/E.")
        elif is_infra:
            if de < 1:
                notes.append(f"D/E {de:.2f}x — very conservatively leveraged for infra/real estate, where 2–3x is the sector norm. Could indicate under-investment or asset-light model.")
            elif de < 2.5:
                notes.append(f"D/E {de:.2f}x — normal for infra. Capital-intensive projects are typically funded with significant long-term debt. Key is whether operating cash flows comfortably service it.")
            else:
                notes.append(f"D/E {de:.2f}x — elevated even for infra. Refinancing risk rises sharply when D/E exceeds 3x in capital-intensive sectors.")
        elif is_it:
            if de < 0.3:
                notes.append(f"D/E {de:.2f}x — effectively debt-free, typical of strong Indian IT companies. Asset-light model means debt is rarely needed.")
            elif de < 1:
                notes.append(f"D/E {de:.2f}x — low leverage. For IT, any meaningful debt warrants scrutiny — check if it's funding acquisitions or working capital.")
            else:
                notes.append(f"D/E {de:.2f}x — unusual for IT where most companies are cash-rich. Investigate the source of debt — acquisition financing or operational stress.")
        elif is_mfg:
            if de < 0.5:
                notes.append(f"D/E {de:.2f}x — low leverage for manufacturing. Strong balance sheet with headroom for capex-led growth.")
            elif de < 1.5:
                notes.append(f"D/E {de:.2f}x — manageable for manufacturing. Typical for companies funding capacity expansion.")
            else:
                notes.append(f"D/E {de:.2f}x — elevated for manufacturing. Ensure EBITDA growth is keeping pace with debt levels.")
        else:
            if de < 0.5:  notes.append(f"D/E {de:.2f}x — conservatively leveraged. Strong balance sheet.")
            elif de < 1:  notes.append(f"D/E {de:.2f}x — moderate, manageable leverage.")
            else:         notes.append(f"D/E {de:.2f}x — high leverage. Monitor debt repayment capacity carefully.")

    icr = ratios.get("Interest Coverage")
    if icr is not None:
        if is_it:
            if icr >= 10:
                notes.append(f"Interest Coverage {icr:.1f}x — very comfortable, consistent with an IT company that should barely need debt at all.")
            else:
                notes.append(f"Interest Coverage {icr:.1f}x — IT companies should have very high coverage given their cash-generative nature. This warrants checking the debt composition.")
        elif is_infra:
            if icr >= 3:
                notes.append(f"Interest Coverage {icr:.1f}x — solid for infra. Even with high debt, earnings cover interest well.")
            elif icr >= 1.5:
                notes.append(f"Interest Coverage {icr:.1f}x — tight but common during project execution phases in infra. Will improve as projects stabilise.")
            else:
                notes.append(f"Interest Coverage {icr:.1f}x — dangerously low for infra. Project delays or cost overruns could push this below 1x, triggering covenant breaches.")
        else:
            if icr >= 5:  notes.append(f"Interest Coverage {icr:.1f}x — very comfortable. Earnings cover interest payments many times over.")
            elif icr >= 2: notes.append(f"Interest Coverage {icr:.1f}x — adequate but not generous. A significant earnings dip would tighten this.")
            else:          notes.append(f"Interest Coverage {icr:.1f}x — tight coverage. At risk if earnings deteriorate even moderately.")

    cfo_pat = ratios.get("CFO / PAT")
    if cfo_pat is not None:
        if is_infra:
            notes.append(f"CFO/PAT {cfo_pat:.2f}x — for infra/real estate, this ratio can be lumpy due to project-linked cash flows. A single year below 1x is not necessarily alarming; look at the 3-year average.")
        elif cfo_pat >= 1:
            notes.append(f"CFO/PAT {cfo_pat:.2f}x — cash flows exceed reported profits. Very high earnings quality — the company is not booking paper profits.")
        elif cfo_pat >= 0.7:
            notes.append(f"CFO/PAT {cfo_pat:.2f}x — reasonable cash conversion. Most profits are translating into actual cash.")
        else:
            notes.append(f"CFO/PAT {cfo_pat:.2f}x — profits are not fully converting to cash. Check for ballooning receivables or aggressive revenue recognition.")

    rg = ratios.get("Revenue Growth YoY %")
    pg = ratios.get("PAT Growth YoY %")
    if rg is not None and pg is not None:
        if is_it:
            if rg >= 15:
                notes.append(f"Revenue grew {rg:.1f}% YoY — strong for IT. Industry tailwinds (cloud, AI adoption) may be driving this; check if deal wins support sustaining it.")
            elif rg >= 8:
                notes.append(f"Revenue grew {rg:.1f}% YoY — moderate for IT. Large IT companies typically target 8–15% constant-currency growth.")
            else:
                notes.append(f"Revenue grew only {rg:.1f}% YoY — sluggish for IT. Could reflect macro headwinds, client budget freezes, or deal ramp delays.")
        elif is_fmcg:
            if rg >= 12:
                notes.append(f"Revenue grew {rg:.1f}% YoY — strong for FMCG, suggesting volume growth and/or pricing power beyond inflation.")
            else:
                notes.append(f"Revenue grew {rg:.1f}% YoY — moderate for FMCG. Decompose into volume vs price to understand quality of growth.")
        else:
            if pg > rg:
                notes.append(f"PAT grew faster than revenue ({pg:.1f}% vs {rg:.1f}%) — operating leverage at work. Fixed costs are being spread over a larger revenue base.")
            elif rg > 0:
                notes.append(f"Revenue grew {rg:.1f}% YoY, PAT grew {pg:.1f}% YoY. {'Margins expanding.' if pg >= rg else 'Some margin compression — costs are growing faster than revenue.'}")
            else:
                notes.append(f"Revenue declined {abs(rg):.1f}% YoY. {'PAT also declined ' + str(abs(pg)) + '% — across-the-board contraction.' if pg < 0 else 'PAT held up despite revenue decline — cost cuts are working.'}")

    ccc = ratios.get("Cash Conversion Cycle")
    if ccc is not None:
        if is_fmcg:
            if ccc <= 0:
                notes.append(f"Cash Conversion Cycle {ccc:.0f} days — negative CCC is a hallmark of great FMCG businesses (think DMart, HUL). Suppliers effectively fund operations.")
            else:
                notes.append(f"Cash Conversion Cycle {ccc:.0f} days — positive CCC is unusual for FMCG where supplier credit typically exceeds inventory + receivables. Check if payable days are shrinking.")
        elif is_it:
            notes.append(f"Cash Conversion Cycle {ccc:.0f} days — for IT, CCC is dominated by receivables (60–90 days on large enterprise contracts). Inventory component is negligible.")
        elif is_mfg:
            if ccc <= 60:
                notes.append(f"Cash Conversion Cycle {ccc:.0f} days — efficient for manufacturing. Working capital is recovered in under 2 months.")
            elif ccc <= 120:
                notes.append(f"Cash Conversion Cycle {ccc:.0f} days — moderate for manufacturing. The business ties up working capital for ~{ccc//30} months per cycle.")
            else:
                notes.append(f"Cash Conversion Cycle {ccc:.0f} days — high. Over {ccc//30} months of working capital locked in the cycle. This requires significant funding and constrains free cash flow.")
        else:
            if ccc <= 30:
                notes.append(f"Cash Conversion Cycle {ccc:.0f} days — very efficient working capital management.")
            elif ccc <= 90:
                notes.append(f"Cash Conversion Cycle {ccc:.0f} days — moderate. The business takes ~{ccc} days to convert inventory/receivables back to cash.")
            else:
                notes.append(f"Cash Conversion Cycle {ccc:.0f} days — high. Significant cash is tied up in operations. Watch free cash flow.")

    return notes


def benchmark_ratios(ratios: dict, sector: str = None) -> list[dict]:
    findings = []

    def add(name, benchmark, verdict, message):
        val = ratios.get(name)
        if val is None:
            return
        findings.append({
            "ratio":     name,
            "value":     val,
            "benchmark": benchmark,
            "verdict":   verdict,
            "message":   message,
        })

    s = (sector or "").lower()
    is_it        = any(x in s for x in ["technology", "software", "it service"])
    is_banking   = any(x in s for x in ["financial", "bank", "insurance", "nbfc"])
    is_fmcg      = any(x in s for x in ["consumer", "fmcg", "food", "beverage", "retail"])
    is_pharma    = any(x in s for x in ["healthcare", "pharma", "biotech", "drug"])
    is_infra     = any(x in s for x in ["real estate", "construction", "infrastructure", "utility", "utilities", "energy", "power"])
    is_mfg       = any(x in s for x in ["industrial", "manufactur", "material", "chemical", "metal", "auto", "engineering"])

    sector_label = (
        "IT/Tech" if is_it else
        "Banking/Financial" if is_banking else
        "FMCG/Consumer" if is_fmcg else
        "Pharma/Healthcare" if is_pharma else
        "Infra/Real Estate/Energy" if is_infra else
        "Manufacturing" if is_mfg else
        "General"
    )
    sector_note = f" [{sector_label} benchmark]" if sector else ""

    npm = ratios.get("Net Profit Margin %")
    if npm is not None:
        if is_it:
            if   npm >= 20: add("Net Profit Margin %", f"≥ 20% excellent{sector_note}", "strong",  f"{npm:.1f}% — excellent, typical of high-quality IT businesses.")
            elif npm >= 15: add("Net Profit Margin %", f"15–20% healthy{sector_note}",  "good",    f"{npm:.1f}% — healthy for an IT company.")
            elif npm >= 10: add("Net Profit Margin %", f"10–15% moderate{sector_note}", "watch",   f"{npm:.1f}% — below the IT sector norm of 15%+.")
            else:           add("Net Profit Margin %", f"< 10% weak{sector_note}",      "concern", f"{npm:.1f}% — weak for IT. Margin pressure needs investigation.")
        elif is_fmcg:
            if   npm >= 15: add("Net Profit Margin %", f"≥ 15% excellent{sector_note}", "strong",  f"{npm:.1f}% — excellent for an FMCG company.")
            elif npm >=  8: add("Net Profit Margin %", f"8–15% healthy{sector_note}",   "good",    f"{npm:.1f}% — healthy FMCG margin.")
            elif npm >=  4: add("Net Profit Margin %", f"4–8% thin{sector_note}",       "watch",   f"{npm:.1f}% — thin for FMCG. Rising input costs may squeeze further.")
            else:           add("Net Profit Margin %", f"< 4% weak{sector_note}",       "concern", f"{npm:.1f}% — weak. FMCG companies should maintain 8%+ margins.")
        elif is_infra:
            if   npm >= 12: add("Net Profit Margin %", f"≥ 12% strong{sector_note}",   "strong",  f"{npm:.1f}% — strong for a capital-heavy sector.")
            elif npm >=  6: add("Net Profit Margin %", f"6–12% acceptable{sector_note}","good",    f"{npm:.1f}% — acceptable for infra/real estate.")
            elif npm >=  3: add("Net Profit Margin %", f"3–6% tight{sector_note}",      "watch",   f"{npm:.1f}% — tight margins for this sector.")
            else:           add("Net Profit Margin %", f"< 3% weak{sector_note}",       "concern", f"{npm:.1f}% — very weak. Debt servicing may erode remaining profits.")
        elif is_mfg:
            if   npm >= 15: add("Net Profit Margin %", f"≥ 15% excellent{sector_note}", "strong",  f"{npm:.1f}% — excellent for a manufacturing company.")
            elif npm >=  8: add("Net Profit Margin %", f"8–15% good{sector_note}",      "good",    f"{npm:.1f}% — good manufacturing margin.")
            elif npm >=  4: add("Net Profit Margin %", f"4–8% moderate{sector_note}",   "watch",   f"{npm:.1f}% — moderate. Watch raw material costs.")
            else:           add("Net Profit Margin %", f"< 4% weak{sector_note}",       "concern", f"{npm:.1f}% — weak for manufacturing. Cost control needed.")
        else:
            if   npm >= 20: add("Net Profit Margin %", "≥ 20% excellent",  "strong",  f"{npm:.1f}% — excellent. Company keeps a large share of every rupee earned.")
            elif npm >= 10: add("Net Profit Margin %", "10–20% healthy",   "good",    f"{npm:.1f}% — healthy and above average.")
            elif npm >=  5: add("Net Profit Margin %", "5–10% thin",       "watch",   f"{npm:.1f}% — thin margin, leaves little room for cost surprises.")
            else:           add("Net Profit Margin %", "< 5% weak",        "concern", f"{npm:.1f}% — very weak. Profitability needs urgent attention.")

    em = ratios.get("EBITDA Margin %")
    if em is not None:
        if is_it:
            if   em >= 30: add("EBITDA Margin %", f"≥ 30% excellent{sector_note}", "strong",  f"{em:.1f}% — excellent operating leverage for an IT company.")
            elif em >= 20: add("EBITDA Margin %", f"20–30% healthy{sector_note}",  "good",    f"{em:.1f}% — healthy IT operating margin.")
            elif em >= 15: add("EBITDA Margin %", f"15–20% moderate{sector_note}", "watch",   f"{em:.1f}% — below typical IT EBITDA of 20%+.")
            else:          add("EBITDA Margin %", f"< 15% weak{sector_note}",      "concern", f"{em:.1f}% — weak for IT. Check if margin erosion is structural.")
        elif is_mfg or is_infra:
            if   em >= 20: add("EBITDA Margin %", f"≥ 20% strong{sector_note}",   "strong",  f"{em:.1f}% — strong operating efficiency.")
            elif em >= 12: add("EBITDA Margin %", f"12–20% healthy{sector_note}",  "good",    f"{em:.1f}% — healthy for a capital-intensive sector.")
            elif em >=  8: add("EBITDA Margin %", f"8–12% moderate{sector_note}",  "watch",   f"{em:.1f}% — moderate. Rising costs or debt could pressure PAT.")
            else:          add("EBITDA Margin %", f"< 8% weak{sector_note}",       "concern", f"{em:.1f}% — weak. Debt servicing on thin EBITDA is high risk.")
        else:
            if   em >= 25: add("EBITDA Margin %", "≥ 25% strong",    "strong",  f"{em:.1f}% — strong operating efficiency.")
            elif em >= 15: add("EBITDA Margin %", "15–25% healthy",  "good",    f"{em:.1f}% — healthy operating margin.")
            elif em >=  8: add("EBITDA Margin %", "8–15% moderate",  "watch",   f"{em:.1f}% — moderate. Watch for rising costs.")
            else:          add("EBITDA Margin %", "< 8% weak",       "concern", f"{em:.1f}% — weak operating efficiency.")

    roe = ratios.get("Return on Equity %")
    if roe is not None:
        if   roe >= 20: add("Return on Equity %", "≥ 20% excellent",  "strong",  f"{roe:.1f}% — management is delivering excellent returns on shareholder money.")
        elif roe >= 15: add("Return on Equity %", "15–20% good",      "good",    f"{roe:.1f}% — good capital efficiency.")
        elif roe >= 10: add("Return on Equity %", "10–15% acceptable","watch",   f"{roe:.1f}% — acceptable but below high-quality benchmark of 15%+.")
        else:           add("Return on Equity %", "< 10% poor",       "concern", f"{roe:.1f}% — poor returns on equity. Capital is not being deployed well.")

    roa = ratios.get("Return on Assets %")
    if roa is not None:
        if is_infra or is_mfg:
            if   roa >= 6: add("Return on Assets %", f"≥ 6% good{sector_note}",     "good",    f"{roa:.1f}% — good asset utilisation for a capital-intensive business.")
            elif roa >= 3: add("Return on Assets %", f"3–6% moderate{sector_note}",  "watch",   f"{roa:.1f}% — moderate. Heavy assets weigh on this ratio.")
            else:          add("Return on Assets %", f"< 3% weak{sector_note}",      "concern", f"{roa:.1f}% — weak. Assets not generating adequate returns.")
        else:
            if   roa >= 10: add("Return on Assets %", "≥ 10% excellent", "strong",  f"{roa:.1f}% — assets generating strong returns.")
            elif roa >=  5: add("Return on Assets %", "5–10% good",      "good",    f"{roa:.1f}% — good asset utilisation.")
            elif roa >=  2: add("Return on Assets %", "2–5% moderate",   "watch",   f"{roa:.1f}% — moderate. Assets could be sweated harder.")
            else:           add("Return on Assets %", "< 2% weak",       "concern", f"{roa:.1f}% — weak. Assets are not generating adequate returns.")

    cr = ratios.get("Current Ratio")
    if cr is not None:
        if is_fmcg:
            if   cr >= 1.5: add("Current Ratio", f"≥ 1.5 strong{sector_note}",   "strong",  f"{cr:.2f}x — strong liquidity.")
            elif cr >= 0.8: add("Current Ratio", f"0.8–1.5 normal{sector_note}", "good",    f"{cr:.2f}x — normal for FMCG. Negative working capital is often intentional.")
            else:           add("Current Ratio", f"< 0.8 watch{sector_note}",    "watch",   f"{cr:.2f}x — watch closely even for FMCG.")
        else:
            if   cr >= 2:   add("Current Ratio", "≥ 2 strong",    "strong",  f"{cr:.2f}x — strong. Company can comfortably pay short-term bills.")
            elif cr >= 1.5: add("Current Ratio", "1.5–2 healthy", "good",    f"{cr:.2f}x — healthy liquidity.")
            elif cr >= 1:   add("Current Ratio", "1–1.5 tight",   "watch",   f"{cr:.2f}x — just adequate. Working capital needs monitoring.")
            else:           add("Current Ratio", "< 1 risky",     "concern", f"{cr:.2f}x — current liabilities exceed current assets. Liquidity risk.")

    qr = ratios.get("Quick Ratio")
    if qr is not None:
        if   qr >= 1:   add("Quick Ratio", "≥ 1 good",       "good",    f"{qr:.2f}x — can meet obligations without touching inventory.")
        elif qr >= 0.5: add("Quick Ratio", "0.5–1 watch",    "watch",   f"{qr:.2f}x — moderate. Some reliance on inventory for liquidity.")
        else:           add("Quick Ratio", "< 0.5 concern",  "concern", f"{qr:.2f}x — low quick ratio. High dependence on inventory to pay bills.")

    de = ratios.get("Debt-to-Equity")
    if de is not None:
        if is_infra:
            if   de < 1:   add("Debt-to-Equity", f"< 1 low{sector_note}",        "strong",  f"{de:.2f}x — very conservative leverage for infra/real estate.")
            elif de < 2:   add("Debt-to-Equity", f"1–2 normal{sector_note}",     "good",    f"{de:.2f}x — normal leverage for a capital-intensive sector.")
            elif de < 3:   add("Debt-to-Equity", f"2–3 elevated{sector_note}",   "watch",   f"{de:.2f}x — elevated. Ensure cash flows cover debt service comfortably.")
            else:          add("Debt-to-Equity", f"> 3 high{sector_note}",        "concern", f"{de:.2f}x — high even for infra. Refinancing risk is elevated.")
        elif is_mfg:
            if   de < 0.5: add("Debt-to-Equity", f"< 0.5 low{sector_note}",      "strong",  f"{de:.2f}x — very low debt for manufacturing.")
            elif de < 1.5: add("Debt-to-Equity", f"0.5–1.5 acceptable{sector_note}", "good", f"{de:.2f}x — manageable leverage for manufacturing.")
            elif de < 2.5: add("Debt-to-Equity", f"1.5–2.5 elevated{sector_note}","watch",   f"{de:.2f}x — elevated for manufacturing. Monitor capex funding.")
            else:          add("Debt-to-Equity", f"> 2.5 high{sector_note}",      "concern", f"{de:.2f}x — high debt load for manufacturing. Stress risk.")
        else:
            if   de < 0.5: add("Debt-to-Equity", "< 0.5 conservative", "strong",  f"{de:.2f}x — very low debt. Financially resilient.")
            elif de < 1:   add("Debt-to-Equity", "0.5–1 moderate",     "good",    f"{de:.2f}x — moderate and manageable leverage.")
            elif de < 2:   add("Debt-to-Equity", "1–2 elevated",       "watch",   f"{de:.2f}x — elevated. Monitor debt repayment capacity.")
            else:          add("Debt-to-Equity", "> 2 high risk",      "concern", f"{de:.2f}x — high debt load. Risk of financial stress if earnings fall.")

    icr = ratios.get("Interest Coverage")
    if icr is not None:
        if   icr >= 5:   add("Interest Coverage", "≥ 5 very safe",  "strong",  f"{icr:.1f}x — earnings cover interest payments very comfortably.")
        elif icr >= 3:   add("Interest Coverage", "3–5 adequate",   "good",    f"{icr:.1f}x — adequate buffer over interest payments.")
        elif icr >= 1.5: add("Interest Coverage", "1.5–3 tight",    "watch",   f"{icr:.1f}x — tight. A dip in earnings could strain debt servicing.")
        else:            add("Interest Coverage", "< 1.5 danger",   "concern", f"{icr:.1f}x — dangerously low. At risk of defaulting on interest payments.")

    rg = ratios.get("Revenue Growth YoY %")
    if rg is not None:
        if   rg >= 20: add("Revenue Growth YoY %", "≥ 20% strong",    "strong",  f"{rg:.1f}% — strong top-line growth momentum.")
        elif rg >= 10: add("Revenue Growth YoY %", "10–20% healthy",  "good",    f"{rg:.1f}% — healthy and above inflation.")
        elif rg >=  0: add("Revenue Growth YoY %", "0–10% slow",      "watch",   f"{rg:.1f}% — slow growth. Check if industry-wide or company-specific.")
        else:          add("Revenue Growth YoY %", "< 0% declining",  "concern", f"{abs(rg):.1f}% decline — investigate the cause immediately.")

    pg = ratios.get("PAT Growth YoY %")
    if pg is not None:
        if   pg >= 20: add("PAT Growth YoY %", "≥ 20% strong",   "strong",  f"{pg:.1f}% — strong earnings growth.")
        elif pg >= 10: add("PAT Growth YoY %", "10–20% healthy", "good",    f"{pg:.1f}% — healthy profit growth.")
        elif pg >=  0: add("PAT Growth YoY %", "0–10% slow",     "watch",   f"{pg:.1f}% — slow profit growth.")
        else:          add("PAT Growth YoY %", "< 0% declining", "concern", f"{abs(pg):.1f}% decline in PAT — profitability is deteriorating.")

    cfo_pat = ratios.get("CFO / PAT")
    if cfo_pat is not None:
        if   cfo_pat >= 1:   add("CFO / PAT", "≥ 1 excellent", "strong",  f"{cfo_pat:.2f}x — cash flows exceed reported profits. Very high earnings quality.")
        elif cfo_pat >= 0.7: add("CFO / PAT", "0.7–1 good",    "good",    f"{cfo_pat:.2f}x — good cash conversion from profits.")
        elif cfo_pat >= 0.5: add("CFO / PAT", "0.5–0.7 watch", "watch",   f"{cfo_pat:.2f}x — profits partially not converting to cash. Watch receivables.")
        else:                add("CFO / PAT", "< 0.5 concern", "concern", f"{cfo_pat:.2f}x — earnings quality is poor. Cash flows lag far behind profits.")

    rd = ratios.get("Receivables Days")
    if rd is not None:
        if is_it:
            if   rd <= 60:  add("Receivables Days", f"≤ 60 days good{sector_note}",     "good",    f"{rd:.0f} days — healthy for an IT company with enterprise contracts.")
            elif rd <= 90:  add("Receivables Days", f"60–90 days watch{sector_note}",   "watch",   f"{rd:.0f} days — on the higher end for IT. Monitor large customer payments.")
            else:           add("Receivables Days", f"> 90 days concern{sector_note}",  "concern", f"{rd:.0f} days — very slow collections for IT. Bad debt risk rising.")
        else:
            if   rd <= 30: add("Receivables Days", "≤ 30 days excellent", "strong",  f"{rd:.0f} days — very fast collections. Excellent working capital management.")
            elif rd <= 60: add("Receivables Days", "30–60 days good",     "good",    f"{rd:.0f} days — healthy collection cycle.")
            elif rd <= 90: add("Receivables Days", "60–90 days watch",    "watch",   f"{rd:.0f} days — slow collections. Monitor for bad debts.")
            else:          add("Receivables Days", "> 90 days concern",   "concern", f"{rd:.0f} days — very slow collections. High bad debt risk.")

    ccc = ratios.get("Cash Conversion Cycle")
    if ccc is not None:
        if is_fmcg:
            if   ccc <= 0:  add("Cash Conversion Cycle", f"≤ 0 days excellent{sector_note}", "strong",  f"{ccc:.0f} days — negative CCC. Suppliers finance operations. Classic FMCG strength.")
            elif ccc <= 30: add("Cash Conversion Cycle", f"0–30 days good{sector_note}",     "good",    f"{ccc:.0f} days — good working capital cycle.")
            else:           add("Cash Conversion Cycle", f"> 30 days watch{sector_note}",    "watch",   f"{ccc:.0f} days — positive CCC is unusual for FMCG. Check payable terms.")
        elif is_mfg:
            if   ccc <= 60:  add("Cash Conversion Cycle", f"≤ 60 days good{sector_note}",    "good",    f"{ccc:.0f} days — efficient cycle for manufacturing.")
            elif ccc <= 120: add("Cash Conversion Cycle", f"60–120 days watch{sector_note}", "watch",   f"{ccc:.0f} days — moderate. Working capital is tied up for ~4 months.")
            else:            add("Cash Conversion Cycle", f"> 120 days concern{sector_note}","concern", f"{ccc:.0f} days — high. Large working capital requirement strains cash flow.")
        else:
            if   ccc <= 30:  add("Cash Conversion Cycle", "≤ 30 days excellent", "strong",  f"{ccc:.0f} days — very efficient working capital cycle.")
            elif ccc <= 60:  add("Cash Conversion Cycle", "30–60 days good",     "good",    f"{ccc:.0f} days — good working capital management.")
            elif ccc <= 90:  add("Cash Conversion Cycle", "60–90 days watch",    "watch",   f"{ccc:.0f} days — moderate. Cash is tied up for 3 months.")
            else:            add("Cash Conversion Cycle", "> 90 days concern",   "concern", f"{ccc:.0f} days — high. Significant working capital strain.")

    return findings


# ── Main fetch function ───────────────────────────────────────────────────────

def _indian_search(query: str) -> str | None:
    """
    Search Yahoo Finance for an Indian stock (NSE preferred, BSE fallback).
    Returns the best matching symbol or None.
    """
    try:
        results = yf.Search(query, max_results=20).quotes
        for suffix in (".NS", ".BO"):
            for r in results:
                sym = r.get("symbol", "")
                if sym.endswith(suffix):
                    return sym
    except Exception:
        pass
    return None


def _build_search_candidates(raw: str) -> list[str]:
    """
    Generate multiple search strings from user input so we cast a wide net.
    e.g. "shakti pumps india ltd" →
         ["shakti pumps india ltd", "shakti pumps", "shakti", "SHAKTIPUMPSINDIALTD"]
    """
    raw = raw.strip()
    candidates = [raw]

    # Strip common legal / noise words
    cleaned = re.sub(
        r'\b(limited|ltd\.?|pvt\.?|private|india|industries|industry|'
        r'enterprises|enterprise|corporation|corp|group|holdings|'
        r'technologies|technology|solutions|services|infra|infrastructure)\b',
        '', raw, flags=re.I
    ).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned and cleaned.lower() != raw.lower():
        candidates.append(cleaned)

    # First two words of cleaned name
    words = cleaned.split()
    if len(words) >= 2:
        candidates.append(" ".join(words[:2]))
    if len(words) >= 1:
        candidates.append(words[0])

    # Joined (no spaces) version → NSE ticker guess
    joined = raw.replace(" ", "").upper()
    if joined:
        candidates.append(joined)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for c in candidates:
        key = c.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _resolve_symbol(ticker_input: str) -> str:
    """
    Turn any user input into a yfinance symbol using multiple strategies:

    1. Already has a dot  →  use as-is             (INFY.NS, 500209.BO)
    2. No spaces, no dot  →  try as NSE ticker      (SHAKTIPUMP → SHAKTIPUMP.NS)
    3. Has spaces         →  search Yahoo Finance   (Shakti Pumps → SHAKTIPUMP.NS)

    For cases 2 & 3, if the direct ticker returns no data, falls back to
    Yahoo Finance search using multiple cleaned variations of the input.
    """
    raw = ticker_input.strip()

    # Already fully qualified
    if "." in raw:
        return raw.upper()

    # Looks like a ticker — return directly; fetch_yfinance will try search if empty
    if " " not in raw:
        return raw.upper() + ".NS"

    # Looks like a company name — search immediately
    for candidate in _build_search_candidates(raw):
        sym = _indian_search(candidate)
        if sym:
            return sym

    # Last resort: smash words together as NSE ticker
    return raw.replace(" ", "").upper() + ".NS"


def _fetch_statements(symbol: str) -> tuple:
    """Fetch income, balance sheet, cashflow for a symbol. Returns (income, bs, cf)."""
    t = yf.Ticker(symbol)
    try:
        income = t.financials
    except Exception:
        income = pd.DataFrame()
    try:
        bs = t.balance_sheet
    except Exception:
        bs = pd.DataFrame()
    try:
        cf = t.cashflow
    except Exception:
        cf = pd.DataFrame()
    return t, income, bs, cf


def fetch_yfinance(ticker_input: str) -> dict:
    """
    Fetch annual financial data for any Indian listed company.

    Accepts any format:
      - Ticker:       INFY, SHAKTIPUMP, HDFCBANK
      - With suffix:  INFY.NS, 500209.BO
      - Company name: Shakti Pumps, Infosys, HDFC Bank, hdfc bank limited
      - Partial name: shakti, hdfc, wipro tech
    """
    symbol = _resolve_symbol(ticker_input)
    t, income, bs_raw, cf_raw = _fetch_statements(symbol)

    # If direct symbol returned no data and input was ticker-like, try search as fallback
    if income.empty and bs_raw.empty and " " not in ticker_input.strip():
        for candidate in _build_search_candidates(ticker_input.strip()):
            found = _indian_search(candidate)
            if found and found != symbol:
                t2, inc2, bs2, cf2 = _fetch_statements(found)
                if not inc2.empty or not bs2.empty:
                    symbol, t, income, bs_raw, cf_raw = found, t2, inc2, bs2, cf2
                    break

    # ── Determine year column labels from actual data dates ───────────────
    ref = next((df for df in [income, bs_raw, cf_raw] if not df.empty), None)
    if ref is not None:
        dates     = sorted(ref.columns, reverse=True)
        curr_col  = _fy_label(dates[0])
        prev_col  = _fy_label(dates[1]) if len(dates) >= 2 else f"FY{int(curr_col[2:]) - 1}"
    else:
        curr_col, prev_col = "FY2025", "FY2024"

    # ── Reshape to Particulars DataFrames ─────────────────────────────────
    pl = _to_df(income,  INCOME_MAP,   curr_col, prev_col)
    bs = _to_df(bs_raw,  BALANCE_MAP,  curr_col, prev_col)
    cf = _to_df(cf_raw,  CASHFLOW_MAP, curr_col, prev_col)

    # ── Company info (market data from Yahoo Finance) ─────────────────────
    try:
        info = t.info or {}
    except Exception:
        info = {}

    company = (
        info.get("longName")
        or info.get("shortName")
        or symbol.replace(".NS", "").replace(".BO", "")
    )

    # ── Warnings ──────────────────────────────────────────────────────────
    warnings = ["Values are in ₹ Crores  |  Source: Yahoo Finance (consolidated only)"]
    if pl.empty:
        warnings.append("Income statement unavailable — ticker may be wrong or delisted")
    if bs.empty:
        warnings.append("Balance sheet unavailable")
    if cf.empty:
        warnings.append("Cash flow statement unavailable")

    # ── Ratios & analysis ─────────────────────────────────────────────────
    sector   = info.get("sector")
    ratios   = compute_ratios(bs, pl, cf, curr_year=curr_col, prev_year=prev_col)

    # ── Sector-specific extras ─────────────────────────────────────────────
    employees  = info.get("fullTimeEmployees")
    revenue_cr = ratios.get("Revenue")
    if employees and employees > 0 and revenue_cr:
        # revenue_cr is in ₹ Crores; result in ₹ Lakhs per employee
        ratios["Revenue per Employee (₹ L)"] = round(revenue_cr * 100 / employees, 2)

    interp   = interpret_ratios(ratios, sector=sector)
    findings = benchmark_ratios(ratios, sector=sector)

    # ── Multi-year ratio history (up to 4 years) ──────────────────────────
    ref_raw = next((df for df in [income, bs_raw, cf_raw] if not df.empty), None)
    if ref_raw is not None and len(ref_raw.columns) >= 2:
        all_dates     = sorted(ref_raw.columns, reverse=True)[:4]
        all_fy_labels = [_fy_label(d) for d in all_dates]
        pl_full = _to_df_multi(income, INCOME_MAP,   all_fy_labels)
        bs_full = _to_df_multi(bs_raw, BALANCE_MAP,  all_fy_labels)
        cf_full = _to_df_multi(cf_raw, CASHFLOW_MAP, all_fy_labels)
        ratios_history = {}
        for i, fy in enumerate(all_fy_labels):
            if i + 1 < len(all_fy_labels):
                prev = all_fy_labels[i + 1]
            else:
                prev = None   # oldest year — no prior data
            r = compute_ratios(bs_full, pl_full, cf_full,
                               curr_year=fy, prev_year=prev or fy)
            if prev is None:
                # Strip YoY growth keys that are meaningless without a prior year
                for k in ["Revenue Growth YoY %", "PAT Growth YoY %", "Asset Growth YoY %"]:
                    r.pop(k, None)
            ratios_history[fy] = r
    else:
        ratios_history = {}

    return {
        "company":         company,
        "ticker":          symbol,
        "curr_year":       curr_col,
        "prev_year":       prev_col,
        "total_pages":     None,
        "page_map":        {},
        "warnings":        warnings,
        # yfinance only provides consolidated; expose as both sa and co
        "bs":   bs,   "pl":   pl,   "cf":   cf,
        "bs_c": bs,   "pl_c": pl,   "cf_c": cf,
        "ratios_sa":       ratios,
        "ratios_co":       ratios,
        "key_statements":  {},
        "findings":        findings,
        "interp":          interp,
        "mda_text":        "",
        "info":            info,
        "ratios_history":  ratios_history,
    }

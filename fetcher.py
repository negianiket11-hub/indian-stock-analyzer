"""
yfinance data fetcher for Indian listed companies.
Fetches annual financials and reshapes them to match the format
expected by main.py's analysis functions (compute_ratios, etc.).

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

from main import compute_ratios, interpret_ratios, benchmark_ratios

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

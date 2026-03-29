"""
Annual Report Financial Analyzer — Auto-detect edition
Change only PDF_PATH below. Everything else is automatic.
"""

import re
import pdfplumber
import pandas as pd
import numpy as np
import json
import warnings
import sys
import io
warnings.filterwarnings("ignore")

if __name__ == "__main__" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ════════════════════════════════════════════════════════
#  ONLY CHANGE THIS LINE
# ════════════════════════════════════════════════════════
PDF_PATH = r"C:\Users\Aniket Negi\Coding\Financial analysis\infosys-ar-25.pdf"
# ════════════════════════════════════════════════════════


# ══════════════════════════════════════════════
# 1. AUTO-DETECTION: COMPANY NAME
# ══════════════════════════════════════════════

def detect_company_name(pdf) -> str:
    """
    Extract company name using two strategies:
    1. Auditor's Report 'To the Members of X' — reliable universal pattern
    2. Short standalone lines ending in 'Limited' as fallback
    """
    # Strategy 1: "To the Members of Infosys Limited" / "TO THE MEMBERS OF ..."
    members_pat = re.compile(
        r'to the members of\s+([A-Z][A-Za-z ()&,]+?(?:Limited|Ltd\.?))\b',
        re.IGNORECASE
    )
    for i in range(min(len(pdf.pages), 300)):
        text = pdf.pages[i].extract_text() or ""
        m = members_pat.search(text)
        if m:
            name = m.group(1).strip().rstrip(",")
            if 5 < len(name) < 80:
                return name

    # Strategy 2: short standalone line ending in Limited (cover / title page)
    name_pat = re.compile(
        r'^([A-Z][A-Za-z&()]+(?:\s[A-Za-z&()]+){1,5})\s+(?:Limited|Ltd\.?)$',
        re.MULTILINE | re.IGNORECASE
    )
    for i in range(min(50, len(pdf.pages))):
        text = pdf.pages[i].extract_text() or ""
        for m in name_pat.finditer(text):
            name = m.group(0).strip()
            if 5 < len(name) < 80:
                return name

    return "Company"


# ══════════════════════════════════════════════
# 2. AUTO-DETECTION: PAGE NUMBERS
# ══════════════════════════════════════════════

def auto_detect_pages(pdf) -> dict:
    """
    Scan all pages and return page ranges for each financial section.
    Looks for standard section headers used in Indian annual reports.
    """
    # Full-prefix names (with STANDALONE/CONSOLIDATED) use startswith to allow
    # slight variations. Short names (Balance Sheet, etc.) use exact match only
    # to avoid false positives in Auditor's Report text.
    SECTION_MARKERS = {
        "standalone_bs":   [
            ("prefix", "STANDALONE BALANCE SHEET"),
            ("exact",  "BALANCE SHEET"),           # Infosys / non-prefixed style
            ("exact",  "STAND ALONE BALANCE SHEET"),
            ("exact",  "BALANCE SHEET AS AT"),
        ],
        "standalone_pl":   [
            ("prefix", "STANDALONE STATEMENT OF PROFIT AND LOSS"),
            ("prefix", "STANDALONE STATEMENT OF PROFIT & LOSS"),
            ("prefix", "STANDALONE PROFIT AND LOSS ACCOUNT"),
            ("exact",  "STATEMENT OF PROFIT AND LOSS"),
            ("exact",  "PROFIT AND LOSS ACCOUNT"),
            ("exact",  "PROFIT AND LOSS STATEMENT"),
            ("exact",  "STATEMENT OF PROFIT & LOSS"),
        ],
        "standalone_cf":   [
            ("prefix", "STANDALONE STATEMENT OF CASH FLOWS"),
            ("prefix", "STANDALONE CASH FLOW STATEMENT"),
            ("exact",  "STATEMENT OF CASH FLOWS"),
            ("exact",  "CASH FLOW STATEMENT"),
            ("exact",  "CASH FLOW STATEMENT"),
            ("exact",  "STATEMENT OF CASH FLOW"),
        ],
        "consolidated_bs": [
            ("prefix", "CONSOLIDATED BALANCE SHEET"),
        ],
        "consolidated_pl": [
            ("prefix", "CONSOLIDATED STATEMENT OF PROFIT AND LOSS"),
            ("prefix", "CONSOLIDATED STATEMENT OF PROFIT & LOSS"),
            ("prefix", "CONSOLIDATED PROFIT AND LOSS ACCOUNT"),
            ("exact",  "CONSOLIDATED PROFIT AND LOSS"),
            ("exact",  "CONSOLIDATED STATEMENT OF PROFIT & LOSS"),
        ],
        "consolidated_cf": [
            ("prefix", "CONSOLIDATED STATEMENT OF CASH FLOWS"),
            ("prefix", "CONSOLIDATED CASH FLOW STATEMENT"),
            ("exact",  "CONSOLIDATED CASH FLOW STATEMENT"),
        ],
        "mda": [
            ("prefix", "MANAGEMENT DISCUSSION AND ANALYSIS"),
            ("prefix", "MANAGEMENT'S DISCUSSION AND ANALYSIS"),
            ("prefix", "MANAGEMENT DISCUSSION & ANALYSIS"),
        ],
    }

    # Markers that signal the END of a financial statement section
    NOTES_MARKERS = [
        "NOTES TO THE",
        "NOTES TO STANDALONE",
        "NOTES TO CONSOLIDATED",
    ]

    def line_matches(line: str, match_type: str, marker: str) -> bool:
        if match_type == "exact":
            return line == marker
        else:  # prefix
            return line == marker or line.startswith(marker + " ")

    starts = {}
    print("  Scanning pages for financial sections...")
    for i, page in enumerate(pdf.pages):
        text  = page.extract_text() or ""
        lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
        for key, marker_list in SECTION_MARKERS.items():
            if key not in starts:
                for (match_type, m) in marker_list:
                    if any(line_matches(line, match_type, m) for line in lines[:15]):
                        starts[key] = i + 1
                        print(f"    Found [{key}] on page {i + 1}")
                        break

    # Build a set of pages where "NOTES TO THE ..." appears, to use as end markers
    notes_pages = set()
    for i, page in enumerate(pdf.pages):
        text  = page.extract_text() or ""
        lines = [l.strip().upper() for l in text.split("\n") if l.strip()]
        for line in lines[:15]:
            if any(line.startswith(nm) for nm in NOTES_MARKERS):
                notes_pages.add(i + 1)  # 1-based
                break

    # Build page ranges for financial statements
    stmt_keys = ["standalone_bs", "standalone_pl", "standalone_cf",
                 "consolidated_bs", "consolidated_pl", "consolidated_cf"]
    stmt_starts = sorted([(starts[k], k) for k in stmt_keys if k in starts])

    page_map = {}
    for i, (start, key) in enumerate(stmt_starts):
        # Determine the upper bound from the next statement start
        if i + 1 < len(stmt_starts):
            next_start = stmt_starts[i + 1][0]
            upper_bound = next_start - 1
        else:
            upper_bound = start + 4

        # Scan forward from start to find the first "NOTES TO THE ..." page
        notes_end = None
        for pg in range(start + 1, upper_bound + 1):
            if pg in notes_pages:
                notes_end = pg - 1  # end just before the notes page
                break

        if notes_end is not None:
            end = notes_end
        else:
            # Fall back to capping at start + 4 (as before)
            end = min(upper_bound, start + 4)

        page_map[key] = list(range(start, end + 1))

    # MD&A: allow up to 35 pages
    if "mda" in starts:
        mda_s = starts["mda"]
        page_map["mda"] = list(range(mda_s, min(mda_s + 35, len(pdf.pages) + 1)))

    # Warn about missing sections (improvement #6)
    for key in stmt_keys:
        if key not in page_map:
            print(f"    WARNING: Could not find [{key}] — this statement will be empty")
            page_map[key] = []

    return page_map


# ══════════════════════════════════════════════
# 3. COORDINATE-BASED TABLE PARSER
# ══════════════════════════════════════════════

def group_words_by_row(words: list[dict], y_tol: float = 4.0) -> list[list[dict]]:
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (round(w["top"] / y_tol), w["x0"]))
    rows, current_row, current_y = [], [], None
    for w in sorted_words:
        y = w["top"]
        if current_y is None or abs(y - current_y) <= y_tol:
            current_row.append(w)
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            rows.append(current_row)
            current_row = [w]
            current_y = y
    if current_row:
        rows.append(current_row)
    return rows


def _extract_year_int(text: str) -> int | None:
    """
    Given a year token (e.g. "FY2025", "FY25", "Mar-25", "2024-25", "2025"),
    return the most recent 4-digit year as an integer, or None if not parseable.
    """
    t = text.strip().upper()

    # "2024-25" or "2023-24" fiscal year range — take the later year
    m = re.match(r'^(20\d{2})-(\d{2})$', t)
    if m:
        base = int(m.group(1))
        return base + 1  # e.g. 2024-25 → 2025

    # Bare 4-digit year "2025"
    m = re.match(r'^(20\d{2})[,)]?$', t)
    if m:
        return int(m.group(1))

    # "FY2025" or "FY 2025"
    m = re.match(r'^FY\s*(20\d{2})$', t)
    if m:
        return int(m.group(1))

    # "FY25" or "FY 25"
    m = re.match(r'^FY\s*(\d{2})$', t)
    if m:
        yy = int(m.group(1))
        return 2000 + yy

    # "Mar-25", "Mar'25", "Mar-2025"
    m = re.match(r'^[A-Z]{3}[-\'](\d{2,4})$', t)
    if m:
        yy = int(m.group(1))
        if yy < 100:
            return 2000 + yy
        return yy

    return None


def detect_columns(words: list[dict], next_page_words: list[dict] | None = None) -> dict:
    """
    Detect column x-positions by finding year numbers in the page header.
    Handles bare years (2025), FY2025, FY25, FY 2025, FY 25, Mar-25, Mar'25,
    2024-25 fiscal year ranges, etc.

    If no year words are found on the current page and next_page_words is
    provided, falls back to searching the next page's header (for "(contd.)"
    pages that lack a year row).

    If still no year words are found at all, clusters numeric words in the
    right half of the page and takes the 2 rightmost clusters as column
    positions (fallback).

    Returns {"label_max", "fy25_x", "fy24_x", "curr_year", "prev_year"}
    """
    # Broad pattern: anything that could be a year token
    year_token_pat = re.compile(
        r'^(?:'
        r'20\d{2}[,)]?'           # 2025, 2024,
        r'|FY\s*20\d{2}'          # FY2025, FY 2025
        r'|FY\s*\d{2}'            # FY25, FY 25
        r'|[A-Z][a-z]{2}[-\']\d{2,4}'  # Mar-25, Mar'25, Mar-2025
        r'|20\d{2}-\d{2}'         # 2024-25
        r')$',
        re.IGNORECASE
    )

    def _find_year_avg(word_list):
        """Return {year_int: avg_x} dict from a word list."""
        header_words = [w for w in word_list if w["top"] <= 250]
        matched = [w for w in header_words if year_token_pat.match(w["text"])]
        # Only right side of page (x > 350) to avoid title/date text
        matched = [w for w in matched if w["x0"] > 350]

        year_x: dict[int, list[float]] = {}
        for yw in matched:
            yr_int = _extract_year_int(yw["text"])
            if yr_int is None:
                continue
            x = (yw["x0"] + yw["x1"]) / 2
            year_x.setdefault(yr_int, []).append(x)

        return {yr: sum(xs) / len(xs) for yr, xs in year_x.items()}

    year_avg = _find_year_avg(words)

    # Improvement #8: if no years found, try the next page's words
    if len(year_avg) < 2 and next_page_words:
        year_avg_next = _find_year_avg(next_page_words)
        if len(year_avg_next) >= len(year_avg):
            year_avg = year_avg_next

    if len(year_avg) >= 2:
        # Take the 2 most recent years (by numeric sort)
        sorted_yrs = sorted(year_avg.keys(), reverse=True)
        curr_yr_int, prev_yr_int = sorted_yrs[0], sorted_yrs[1]
        fy25_x = year_avg[curr_yr_int]
        fy24_x = year_avg[prev_yr_int]
        curr_yr = str(curr_yr_int)
        prev_yr = str(prev_yr_int)
    elif len(year_avg) == 1:
        curr_yr_int = list(year_avg.keys())[0]
        prev_yr_int = curr_yr_int - 1
        curr_yr = str(curr_yr_int)
        prev_yr = str(prev_yr_int)
        fy25_x  = list(year_avg.values())[0]
        fy24_x  = fy25_x + 75
    else:
        # Fallback: cluster numeric words in the right half of the page
        # and take the 2 rightmost cluster centres as column positions
        num_pat = re.compile(r'^[\d,.()\-]+$')
        page_width_est = max((w["x1"] for w in words), default=600)
        right_numeric = [
            w for w in words
            if w["x0"] > page_width_est * 0.45 and num_pat.match(w["text"])
        ]
        if len(right_numeric) >= 2:
            # Cluster by x using simple 30pt buckets
            buckets: dict[int, list[float]] = {}
            for w in right_numeric:
                cx = (w["x0"] + w["x1"]) / 2
                bucket_key = int(cx // 30)
                buckets.setdefault(bucket_key, []).append(cx)
            cluster_centres = sorted(
                [sum(xs) / len(xs) for xs in buckets.values()], reverse=True
            )
            fy25_x = cluster_centres[0]
            fy24_x = cluster_centres[1] if len(cluster_centres) > 1 else fy25_x + 75
        else:
            fy25_x, fy24_x = 450.0, 525.0
        curr_yr, prev_yr = "2025", "2024"

    label_max = min(min(fy25_x, fy24_x) - 80, 380)

    return {
        "label_max": label_max,
        "fy25_x":    fy25_x,
        "fy24_x":    fy24_x,
        "curr_year": curr_yr,
        "prev_year": prev_yr,
    }


def words_near_x(row_words: list[dict], target_x: float, tolerance: float = 55.0) -> str:
    near = [w["text"] for w in row_words
            if abs((w["x0"] + w["x1"]) / 2 - target_x) <= tolerance]
    return " ".join(near).strip()


def clean_value(val: str) -> float | None:
    if not val or val.strip() in ("", "-", "–", "—", "Nil"):
        return None
    v = val.strip()

    # Improvement #4: filter note references like "2.1", "2.12", "3(a)", "1.1,"
    # A note reference is short (< 6 chars), has no comma (except trailing),
    # and matches patterns like X.XX or X(X)
    v_stripped = v.rstrip(",")
    if len(v_stripped) < 6 and "," not in v_stripped:
        if re.match(r'^\d+\.\d{1,2}$', v_stripped):
            return None  # e.g. "2.1", "2.12"
        if re.match(r'^\d+\([a-zA-Z]\)$', v_stripped):
            return None  # e.g. "3(a)"

    v = v.replace(",", "").replace(" ", "")
    if v.startswith("(") and v.endswith(")"):
        v = "-" + v[1:-1]
    try:
        return float(v)
    except ValueError:
        return None


GENERIC_SKIP = {
    "particulars", "note no.", "note", "", "sr.", "s.no",
    "corporate", "overview", "statutory", "reports", "financial",
    "statements", "standalone", "consolidated",
}
GENERIC_SKIP_CONTAINS = [
    "particulars", "corporate", "statutory", "overview",
    "all amounts", "unless otherwise", "notes to", "annexure",
    "for the year ended", "as at march", "as at 31",
    "annual report", "integrated report", "financial statements",
]


def extract_statement_from_pages(pdf, page_numbers: list[int]) -> pd.DataFrame:
    """Extract BS or P&L from given pages using word-position detection."""
    if not page_numbers:
        return pd.DataFrame(columns=["Particulars", "FY2025", "FY2024"])

    records = []
    col_detected = False
    label_max = fy25_x = fy24_x = None

    for pg_idx, pg_num in enumerate(page_numbers):
        page  = pdf.pages[pg_num - 1]
        words = page.extract_words(x_tolerance=4, y_tolerance=3)

        if not col_detected:
            # Improvement #8: pass next page's words for contd. pages
            next_words = None
            if pg_idx + 1 < len(page_numbers):
                next_pg = page_numbers[pg_idx + 1]
                next_words = pdf.pages[next_pg - 1].extract_words(x_tolerance=4, y_tolerance=3)

            cols      = detect_columns(words, next_page_words=next_words)
            fy25_x    = cols["fy25_x"]
            fy24_x    = cols["fy24_x"]

            # Improvement #3: adaptive label_max
            # Find the gap between label text and the first numeric column
            # by looking at all words in the data area (top > 150)
            data_words = [w for w in words if w["top"] > 150]
            left_col_x = min(fy25_x, fy24_x)
            # Collect x-centres of words that are clearly to the left of the columns
            label_xs = [
                (w["x0"] + w["x1"]) / 2
                for w in data_words
                if (w["x0"] + w["x1"]) / 2 < left_col_x - 40
            ]
            if label_xs:
                # The gap position is the max x of label words
                gap_pos = max(label_xs)
                label_max = min(gap_pos * 0.80, 380)
            else:
                label_max = min(left_col_x - 80, 380)

            col_detected = True

        page_width = float(page.width)
        words = [w for w in words if w["x0"] < page_width - 30]
        rows  = group_words_by_row(words, y_tol=4.0)

        for row in rows:
            label_words = [w for w in row if w["x0"] < label_max and w["x1"] < label_max + 30]
            label = " ".join(w["text"] for w in sorted(label_words, key=lambda x: x["x0"])).strip()

            if not label or label.lower() in GENERIC_SKIP:
                continue
            if any(s in label.lower() for s in GENERIC_SKIP_CONTAINS):
                continue

            fy25 = clean_value(words_near_x(row, fy25_x, tolerance=50))
            fy24 = clean_value(words_near_x(row, fy24_x, tolerance=50))

            if fy25 is not None or fy24 is not None:
                records.append({"Particulars": label, "FY2025": fy25, "FY2024": fy24})

    if not records:
        return pd.DataFrame(columns=["Particulars", "FY2025", "FY2024"])

    df = pd.DataFrame(records)
    df = df[df["Particulars"].str.len() > 2]
    df = df.drop_duplicates(subset=["Particulars", "FY2025"])
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════
# 4. CASH FLOW SPECIAL PARSER
# ══════════════════════════════════════════════

def extract_cashflow_from_pages(pdf, page_numbers: list[int]) -> pd.DataFrame:
    """
    Cash flow statements have an indented two-column value structure.
    Sub-items are in one x-zone, section totals in another.
    """
    if not page_numbers:
        return pd.DataFrame(columns=["Particulars", "FY2025", "FY2024"])

    SKIP_CF = GENERIC_SKIP | {"financial statements", "notes to"}
    records = []

    # Detect column layout from first page (improvement #8: try next page too)
    first_page_words = pdf.pages[page_numbers[0] - 1].extract_words(x_tolerance=4, y_tolerance=3)
    next_words_cf = None
    if len(page_numbers) > 1:
        next_words_cf = pdf.pages[page_numbers[1] - 1].extract_words(x_tolerance=4, y_tolerance=3)
    cols   = detect_columns(first_page_words, next_page_words=next_words_cf)
    fy25_x = cols["fy25_x"]
    fy24_x = cols["fy24_x"]

    # CF-specific zone boundaries derived from detected column positions
    # Use midpoint between the two year columns as the hard separator —
    # this prevents the FY2025 zone from bleeding into FY2024 values
    midpoint    = (fy25_x + fy24_x) / 2
    col_gap     = fy24_x - fy25_x          # distance between columns
    label_end   = min(fy25_x - 120, 295)
    sub_x_start = label_end + 15
    sub_x_end   = fy25_x - 20
    tot_x_start = fy25_x - col_gap * 0.6   # current year zone
    tot_x_end   = midpoint                  # hard stop at midpoint
    py_x_start  = midpoint                  # prior year starts at midpoint
    py_x_end    = fy24_x + col_gap * 0.6

    for pg_num in page_numbers:
        page  = pdf.pages[pg_num - 1]
        words = page.extract_words(x_tolerance=4, y_tolerance=3)
        words = [w for w in words if w["x0"] < float(page.width) - 30]
        rows  = group_words_by_row(words, y_tol=4.0)

        for row in rows:
            label_words = [w for w in row if w["x1"] < label_end + 10]
            label = " ".join(w["text"] for w in sorted(label_words, key=lambda x: x["x0"])).strip()

            if not label or label.lower() in SKIP_CF:
                continue
            if any(s in label.lower() for s in GENERIC_SKIP_CONTAINS):
                continue

            sub_str = " ".join(w["text"] for w in row if sub_x_start <= w["x0"] <= sub_x_end).strip()
            tot_str = " ".join(w["text"] for w in row if tot_x_start <= w["x0"] <= tot_x_end).strip()
            py_str  = " ".join(w["text"] for w in row if py_x_start  <= w["x0"] <= py_x_end).strip()

            fy25 = clean_value(tot_str) if clean_value(tot_str) is not None else clean_value(sub_str)
            fy24 = clean_value(py_str)

            if fy25 is not None or fy24 is not None:
                records.append({"Particulars": label, "FY2025": fy25, "FY2024": fy24})

    if not records:
        return pd.DataFrame(columns=["Particulars", "FY2025", "FY2024"])

    df = pd.DataFrame(records)
    df = df[df["Particulars"].str.len() > 2]
    df = df.drop_duplicates(subset=["Particulars", "FY2025"])
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════
# 5. NARRATIVE / MD&A EXTRACTION
# ══════════════════════════════════════════════

def extract_narrative(pdf, page_numbers: list[int]) -> str:
    return "\n\n".join(
        pdf.pages[pg - 1].extract_text().strip()
        for pg in page_numbers
        if pdf.pages[pg - 1].extract_text()
    )


def extract_key_statements(narrative: str) -> dict:
    themes = {
        "revenue_growth":   ["revenue", "sales", "turnover", "grew", "growth"],
        "profitability":    ["profit", "margin", "ebitda", "pat", "earnings"],
        "debt":             ["debt", "borrowing", "loan", "leverage", "repay"],
        "capex":            ["capex", "capital expenditure", "investment", "expansion"],
        "order_book":       ["order book", "backlog", "order inflow", "pipeline"],
        "guidance":         ["guidance", "target", "outlook", "expect", "anticipate"],
        "risks":            ["risk", "challenge", "concern", "headwind", "uncertainty"],
    }
    sentences = [s.strip() for s in narrative.replace("\n", " ").split(".") if len(s.strip()) > 30]
    results   = {theme: [] for theme in themes}
    for sentence in sentences:
        sl = sentence.lower()
        for theme, keywords in themes.items():
            if any(kw in sl for kw in keywords):
                results[theme].append(sentence)
    return {k: v[:3] for k, v in results.items() if v}


# ══════════════════════════════════════════════
# 6. FINANCIAL RATIO ENGINE
# ══════════════════════════════════════════════

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

    # Balance Sheet — improvement #5: more keyword variants
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

    # P&L — improvement #5: more keyword variants
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

    # Prior year
    revenue_24      = g(pl, ["revenue from operations", "net revenue",
                              "income from operations", "revenue from contracts"], prev_year)
    pat_24          = g(pl, ["profit for the year", "profit after tax",
                              "net income", "profit attributable"], prev_year)
    total_assets_24 = g(bs, ["total assets", "sum of assets"], prev_year)

    # Cash Flow — improvement #5: more keyword variants
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


# ══════════════════════════════════════════════
# 7. CROSS-REFERENCE & INTERPRETATION
# ══════════════════════════════════════════════

def cross_reference(ratios: dict, key_statements: dict) -> list[dict]:
    findings = []

    def finding(claim, metric, verdict):
        findings.append({"Claim": claim[:120], "Metric": metric, "Verdict": verdict})

    rev_growth    = ratios.get("Revenue Growth YoY %")
    ebitda_margin = ratios.get("EBITDA Margin %")
    de_ratio      = ratios.get("Debt-to-Equity")
    cfo_pat       = ratios.get("CFO / PAT")

    if rev_growth is not None:
        for stmt in key_statements.get("revenue_growth", [])[:1]:
            finding(stmt, f"Revenue Growth: {rev_growth}%",
                    "Confirmed" if rev_growth > 0 else "Contradicts narrative")

    if de_ratio is not None:
        for stmt in key_statements.get("debt", [])[:1]:
            finding(stmt, f"Debt/Equity: {de_ratio}x",
                    "Low leverage (D/E < 1)" if de_ratio < 1 else "Check debt levels (D/E > 1)")

    if ebitda_margin is not None:
        for stmt in key_statements.get("profitability", [])[:1]:
            finding(stmt, f"EBITDA Margin: {ebitda_margin}%",
                    "Healthy margins" if ebitda_margin > 12 else "Margin pressure")

    if cfo_pat is not None:
        finding("Cash conversion from reported profits", f"CFO/PAT: {cfo_pat}x",
                "Strong earnings quality" if cfo_pat > 0.8 else "Earnings quality concern -- CFO lags PAT")

    return findings


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

    # ── Net Profit Margin ────────────────────────────────────────────────
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

    # ── Current Ratio ────────────────────────────────────────────────────
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

    # ── Debt-to-Equity ───────────────────────────────────────────────────
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

    # ── Interest Coverage ────────────────────────────────────────────────
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

    # ── CFO / PAT ────────────────────────────────────────────────────────
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

    # ── Revenue & PAT Growth ─────────────────────────────────────────────
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

    # ── Cash Conversion Cycle ────────────────────────────────────────────
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


# ══════════════════════════════════════════════
# 8. RATIO BENCHMARKING
# ══════════════════════════════════════════════

def benchmark_ratios(ratios: dict, sector: str = None) -> list[dict]:
    """
    Compare computed ratios against sector-adjusted financial benchmarks.
    Returns a list of findings, each with:
      ratio, value, benchmark, verdict (strong/good/watch/concern), message
    """
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

    # ── Sector classification ─────────────────────────────────────────────
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

    # ── Profitability ─────────────────────────────────────────────────────
    npm = ratios.get("Net Profit Margin %")
    if npm is not None:
        if is_it:
            # IT: 15%+ is normal, 20%+ is excellent
            if   npm >= 20: add("Net Profit Margin %", f"≥ 20% excellent{sector_note}", "strong",  f"{npm:.1f}% — excellent, typical of high-quality IT businesses.")
            elif npm >= 15: add("Net Profit Margin %", f"15–20% healthy{sector_note}",  "good",    f"{npm:.1f}% — healthy for an IT company.")
            elif npm >= 10: add("Net Profit Margin %", f"10–15% moderate{sector_note}", "watch",   f"{npm:.1f}% — below the IT sector norm of 15%+.")
            else:           add("Net Profit Margin %", f"< 10% weak{sector_note}",      "concern", f"{npm:.1f}% — weak for IT. Margin pressure needs investigation.")
        elif is_fmcg:
            # FMCG: 8%+ is good, volumes drive business
            if   npm >= 15: add("Net Profit Margin %", f"≥ 15% excellent{sector_note}", "strong",  f"{npm:.1f}% — excellent for an FMCG company.")
            elif npm >=  8: add("Net Profit Margin %", f"8–15% healthy{sector_note}",   "good",    f"{npm:.1f}% — healthy FMCG margin.")
            elif npm >=  4: add("Net Profit Margin %", f"4–8% thin{sector_note}",       "watch",   f"{npm:.1f}% — thin for FMCG. Rising input costs may squeeze further.")
            else:           add("Net Profit Margin %", f"< 4% weak{sector_note}",       "concern", f"{npm:.1f}% — weak. FMCG companies should maintain 8%+ margins.")
        elif is_infra:
            # Infra/Real Estate: 5%+ acceptable due to capital-heavy nature
            if   npm >= 12: add("Net Profit Margin %", f"≥ 12% strong{sector_note}",   "strong",  f"{npm:.1f}% — strong for a capital-heavy sector.")
            elif npm >=  6: add("Net Profit Margin %", f"6–12% acceptable{sector_note}","good",    f"{npm:.1f}% — acceptable for infra/real estate.")
            elif npm >=  3: add("Net Profit Margin %", f"3–6% tight{sector_note}",      "watch",   f"{npm:.1f}% — tight margins for this sector.")
            else:           add("Net Profit Margin %", f"< 3% weak{sector_note}",       "concern", f"{npm:.1f}% — very weak. Debt servicing may erode remaining profits.")
        elif is_mfg:
            # Manufacturing: 6%+ good
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

    # ── Liquidity ─────────────────────────────────────────────────────────
    cr = ratios.get("Current Ratio")
    if cr is not None:
        if is_fmcg:
            # FMCG often runs negative working capital (suppliers fund inventory) — < 1 can be fine
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

    # ── Leverage ──────────────────────────────────────────────────────────
    de = ratios.get("Debt-to-Equity")
    if de is not None:
        if is_infra:
            # Infra/real estate: 2-3x D/E is structurally normal
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

    # ── Growth ────────────────────────────────────────────────────────────
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

    # ── Cash Quality ─────────────────────────────────────────────────────
    cfo_pat = ratios.get("CFO / PAT")
    if cfo_pat is not None:
        if   cfo_pat >= 1:   add("CFO / PAT", "≥ 1 excellent", "strong",  f"{cfo_pat:.2f}x — cash flows exceed reported profits. Very high earnings quality.")
        elif cfo_pat >= 0.7: add("CFO / PAT", "0.7–1 good",    "good",    f"{cfo_pat:.2f}x — good cash conversion from profits.")
        elif cfo_pat >= 0.5: add("CFO / PAT", "0.5–0.7 watch", "watch",   f"{cfo_pat:.2f}x — profits partially not converting to cash. Watch receivables.")
        else:                add("CFO / PAT", "< 0.5 concern", "concern", f"{cfo_pat:.2f}x — earnings quality is poor. Cash flows lag far behind profits.")

    # ── Efficiency ───────────────────────────────────────────────────────
    rd = ratios.get("Receivables Days")
    if rd is not None:
        if is_it:
            # IT: large enterprise contracts — 60-90 days is normal
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


# ══════════════════════════════════════════════
# 9. OPTIONAL OLLAMA ANALYSIS
# ══════════════════════════════════════════════

def ollama_analyze(company: str, ratios: dict, mda_text: str, model: str = "llama3.2") -> str:
    try:
        import ollama
        ratio_summary = "\n".join(f"  {k}: {v}" for k, v in ratios.items())
        prompt = f"""You are a senior equity analyst. Analyze {company}.

FINANCIAL RATIOS:
{ratio_summary}

MD&A EXCERPT:
{mda_text[:3000]}

Provide structured analysis:
1. Business performance summary
2. Key strengths from the numbers
3. Red flags or concerns
4. Alignment between narrative and financials
5. Overall investment outlook (1-2 sentences)"""
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except ImportError:
        return "Ollama not installed. Run: pip install ollama  then  ollama pull llama3.2"
    except Exception as e:
        return f"Ollama unavailable: {e}\nInstall from https://ollama.com then run: ollama pull llama3.2"


# ══════════════════════════════════════════════
# 9. REPORT PRINTER
# ══════════════════════════════════════════════

DIV = "=" * 70

def print_section(title: str, df: pd.DataFrame = None, data: dict = None, items: list = None):
    print(f"\n{DIV}")
    print(f"  {title}")
    print(DIV)
    if df is not None:
        if df.empty:
            print("  (no data extracted)")
        else:
            pd.set_option("display.max_colwidth", 55)
            pd.set_option("display.width", 120)
            print(df.to_string(index=False))
    if data:
        for k, v in data.items():
            print(f"  {k:<38} {v}")
    if items:
        for item in items:
            if isinstance(item, dict):
                print(f"\n  Claim  : {item.get('Claim', '')}")
                print(f"  Metric : {item.get('Metric', '')}")
                print(f"  Verdict: {item.get('Verdict', '')}")
            else:
                print(f"  - {item}")


# ══════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════

def main():
    print(f"\n{DIV}")
    print("  ANNUAL REPORT FINANCIAL ANALYZER")
    print(f"  PDF: {PDF_PATH}")
    print(DIV)

    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
        print(f"\n  PDF loaded: {total_pages} pages")

        # Auto-detect everything
        company  = detect_company_name(pdf)
        page_map = auto_detect_pages(pdf)

        print(f"\n  Company detected: {company}")
        print(f"  Page map: { {k: v for k, v in page_map.items() if k != 'mda'} }")

        print("\n  Extracting standalone statements...")
        bs  = extract_statement_from_pages(pdf, page_map.get("standalone_bs",  []))
        pl  = extract_statement_from_pages(pdf, page_map.get("standalone_pl",  []))
        cf  = extract_cashflow_from_pages( pdf, page_map.get("standalone_cf",  []))

        # Improvement #7: warn if extracted DataFrames are suspiciously thin
        for stmt_name, df_stmt in [
            ("Standalone Balance Sheet", bs),
            ("Standalone P&L", pl),
            ("Standalone Cash Flow", cf),
        ]:
            if len(df_stmt) < 3:
                print(f"  WARNING: {stmt_name} has only {len(df_stmt)} rows — layout may differ from standard")

        print("  Extracting consolidated statements...")
        bs_c = extract_statement_from_pages(pdf, page_map.get("consolidated_bs", []))
        pl_c = extract_statement_from_pages(pdf, page_map.get("consolidated_pl", []))
        cf_c = extract_cashflow_from_pages( pdf, page_map.get("consolidated_cf", []))

        # Improvement #7: warn for consolidated statements too
        for stmt_name, df_stmt in [
            ("Consolidated Balance Sheet", bs_c),
            ("Consolidated P&L", pl_c),
            ("Consolidated Cash Flow", cf_c),
        ]:
            if len(df_stmt) < 3:
                print(f"  WARNING: {stmt_name} has only {len(df_stmt)} rows — layout may differ from standard")

        print("  Extracting MD&A narrative...")
        mda_text       = extract_narrative(pdf, page_map.get("mda", []))
        key_statements = extract_key_statements(mda_text)

    print("  Computing financial ratios...")
    ratios_sa = compute_ratios(bs,   pl,   cf)
    ratios_co = compute_ratios(bs_c, pl_c, cf_c)
    findings  = cross_reference(ratios_sa, key_statements)

    # Print report
    print_section(f"STANDALONE BALANCE SHEET -- {company}", df=bs)
    print_section(f"STANDALONE P&L STATEMENT -- {company}", df=pl)
    print_section(f"STANDALONE CASH FLOW -- {company}", df=cf)

    print_section(f"CONSOLIDATED BALANCE SHEET -- {company}", df=bs_c)
    print_section(f"CONSOLIDATED P&L STATEMENT -- {company}", df=pl_c)
    print_section(f"CONSOLIDATED CASH FLOW -- {company}", df=cf_c)

    print_section("STANDALONE FINANCIAL RATIOS", data=ratios_sa)
    print_section("CONSOLIDATED FINANCIAL RATIOS", data=ratios_co)

    if key_statements:
        print_section("KEY MANAGEMENT STATEMENTS (MD&A)",
                      items=[s for stmts in key_statements.values() for s in stmts[:1]])

    if findings:
        print_section("NARRATIVE vs NUMBERS -- CROSS REFERENCE", items=findings)

    interp = interpret_ratios(ratios_sa)
    if interp:
        print_section("QUICK INTERPRETATION", items=interp)

    ai_narrative = ollama_analyze(company, ratios_sa, mda_text)
    if not ai_narrative.startswith("Ollama"):
        print(f"\n{DIV}")
        print("  AI NARRATIVE ANALYSIS (Ollama)")
        print(DIV)
        print(ai_narrative)

    # Save output JSON next to the PDF
    import os
    out_path = os.path.join(os.path.dirname(PDF_PATH),
                            f"analysis_{company[:30].replace(' ', '_')}.json")
    output = {
        "company": company,
        "pdf":     PDF_PATH,
        "standalone":   {
            "balance_sheet":   bs.to_dict(orient="records"),
            "profit_and_loss": pl.to_dict(orient="records"),
            "cash_flow":       cf.to_dict(orient="records"),
            "ratios":          ratios_sa,
        },
        "consolidated": {
            "balance_sheet":   bs_c.to_dict(orient="records"),
            "profit_and_loss": pl_c.to_dict(orient="records"),
            "cash_flow":       cf_c.to_dict(orient="records"),
            "ratios":          ratios_co,
        },
        "key_management_statements": key_statements,
        "cross_reference_findings":  findings,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Full data saved to: {out_path}")
    print(f"{DIV}\n")


if __name__ == "__main__":
    main()

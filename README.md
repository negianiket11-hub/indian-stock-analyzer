# Indian Stock Financial Analyzer

A Streamlit dashboard for instantly fetching and analyzing financials of any NSE/BSE listed company using live Yahoo Finance data.

## Features

- **Live data** — fetch balance sheet, P&L, and cash flow for any Indian stock by ticker or company name (e.g. `INFY`, `Shakti Pumps`, `HDFC Bank`)
- **PDF parser** — extract and analyze financials directly from a company's annual report PDF
- **Key ratios** — profitability, liquidity, leverage, growth, and efficiency ratios computed automatically
- **Health check** — traffic-light benchmarking (Strong / Good / Watch / Concern) with compound risk flags
- **Sector awareness** — hides irrelevant ratios for banking, IT, and FMCG sectors and adds sector-specific context
- **4-year trend charts** — revenue, margins, leverage, growth, and cash quality over time
- **Peer comparison** — add up to 3 peers and compare side-by-side with a table and bar charts
- **P&L waterfall** — visual breakdown from Revenue to PAT
- **Debt maturity profile** — long-term vs short-term debt split
- **JSON export** — download the full analysis as a structured JSON file

## Project Structure

```
.
├── dashboard.py   # Streamlit UI — run this to launch the app
├── fetcher.py     # yfinance data fetcher and reshaper for Indian stocks
└── main.py        # PDF annual report parser + ratio engine (compute_ratios, interpret_ratios, benchmark_ratios)
```

## Setup

**Requirements:** Python 3.10+

```bash
pip install streamlit yfinance pandas numpy plotly pdfplumber
```

## Usage

### Streamlit Dashboard (live data from Yahoo Finance)

```bash
python -m streamlit run dashboard.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

Enter any of the following in the search box:
- **Ticker:** `INFY`, `SHAKTIPUMP`, `HDFCBANK`
- **Company name:** `Shakti Pumps`, `Infosys`, `HDFC Bank Limited`
- **BSE code:** `500209.BO`

### PDF Annual Report Analyzer

Edit the `PDF_PATH` variable at the top of `main.py` to point to your annual report PDF, then run:

```bash
python main.py
```

The script auto-detects the company name, locates all financial statement sections (standalone and consolidated), extracts the tables, computes ratios, and prints a full analysis.

## How It Works

| Component | Description |
|-----------|-------------|
| `_resolve_symbol()` | Converts any ticker/name input to a valid yfinance symbol using search and fallback strategies |
| `compute_ratios()` | Calculates ~30 financial ratios from the three statements |
| `interpret_ratios()` | Generates plain-English commentary on each ratio |
| `benchmark_ratios()` | Scores each ratio against sector-appropriate benchmarks |
| `_sector_filter()` | Suppresses misleading ratios for banking, IT, and FMCG companies |

## Data Source

Live financial data is fetched from **Yahoo Finance** via the `yfinance` library. All values are in **₹ Crores**. Only consolidated financials are available through this source.

For standalone vs consolidated comparison, use the PDF parser (`main.py`) with the company's official annual report.

## Notes

- Market cap, P/E, and 52-week high/low are pulled from Yahoo Finance's `info` endpoint and reflect real-time market data.
- Banking-sector metrics (NIM, CAR, GNPA %) require RBI/exchange filing data and are not available via Yahoo Finance.

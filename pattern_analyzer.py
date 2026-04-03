"""
Stock Pattern Analyzer — Technical chart pattern detection and visualization.
Detects 22 patterns across 10 families, scores by confidence, and builds
annotated candlestick charts.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Optional

# ── Timeframes ────────────────────────────────────────────────────────────────
# period: yfinance period string
# interval: candle size (daily for ≤1Y, weekly for longer)
# default_order: sensible pivot sensitivity for this timeframe
TIMEFRAME_MAP: dict[str, dict] = {
    "1 Month":  {"period": "1mo",  "interval": "1d",  "default_order": 3},
    "3 Months": {"period": "3mo",  "interval": "1d",  "default_order": 4},
    "6 Months": {"period": "6mo",  "interval": "1d",  "default_order": 5},
    "1 Year":   {"period": "1y",   "interval": "1d",  "default_order": 7},
    "2 Years":  {"period": "2y",   "interval": "1wk", "default_order": 5},
    "5 Years":  {"period": "5y",   "interval": "1wk", "default_order": 6},
}

# ── Timeframe validity per pattern ────────────────────────────────────────────
# Each pattern only makes technical sense on certain timeframes.
# Flags need short bursts; Cup & Handle needs months of base-building.
PATTERN_TIMEFRAMES: dict[str, set[str]] = {
    # ── Short-term (days to weeks) ────────────────────────────────────────────
    "Bull Flag":                {"1 Month", "3 Months", "6 Months"},
    "Bear Flag":                {"1 Month", "3 Months", "6 Months"},
    "Bull Pennant":             {"1 Month", "3 Months", "6 Months"},
    "Bear Pennant":             {"1 Month", "3 Months", "6 Months"},
    # ── Short to medium-term (weeks to months) ────────────────────────────────
    "Ascending Triangle":       {"1 Month", "3 Months", "6 Months", "1 Year"},
    "Descending Triangle":      {"1 Month", "3 Months", "6 Months", "1 Year"},
    "Symmetrical Triangle":     {"1 Month", "3 Months", "6 Months", "1 Year"},
    "Rising Wedge":             {"1 Month", "3 Months", "6 Months", "1 Year"},
    "Falling Wedge":            {"1 Month", "3 Months", "6 Months", "1 Year"},
    "Rectangle":                {"1 Month", "3 Months", "6 Months", "1 Year", "2 Years"},
    # ── Medium-term (months) ─────────────────────────────────────────────────
    "Double Top":               {"3 Months", "6 Months", "1 Year", "2 Years"},
    "Double Bottom":            {"3 Months", "6 Months", "1 Year", "2 Years"},
    "Broadening Formation":     {"3 Months", "6 Months", "1 Year", "2 Years"},
    "Three Drives Up":          {"3 Months", "6 Months", "1 Year"},
    "Three Drives Down":        {"3 Months", "6 Months", "1 Year"},
    # ── Medium to long-term (months to years) ────────────────────────────────
    "Head & Shoulders":         {"6 Months", "1 Year", "2 Years", "5 Years"},
    "Inverse Head & Shoulders": {"6 Months", "1 Year", "2 Years", "5 Years"},
    "Triple Top":               {"6 Months", "1 Year", "2 Years", "5 Years"},
    "Triple Bottom":            {"6 Months", "1 Year", "2 Years", "5 Years"},
    "Cup & Handle":             {"6 Months", "1 Year", "2 Years", "5 Years"},
    # ── Long-term (years) ────────────────────────────────────────────────────
    "Rounding Bottom":          {"1 Year", "2 Years", "5 Years"},
    "Rounding Top":             {"1 Year", "2 Years", "5 Years"},
}

# ── Colors per pattern signal ─────────────────────────────────────────────────
_BULL_COLOR = "#26A65B"
_BEAR_COLOR = "#E74C3C"
_NEUT_COLOR = "#F39C12"


# ── PatternResult ─────────────────────────────────────────────────────────────
@dataclass
class PatternResult:
    name: str
    signal: str           # "Bullish" | "Bearish" | "Neutral"
    confidence: float     # 0–100 (Python float, NOT numpy)
    status: str           # "Forming" | "Breakout" | "Breakdown"
    description: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    breakout_level: Optional[float]
    lines: list = field(default_factory=list)        # plotly add_shape kwargs
    annotations: list = field(default_factory=list)  # plotly add_annotation kwargs


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _f(x) -> Optional[float]:
    """Safe conversion to Python float (no numpy scalars)."""
    if x is None:
        return None
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _pct(a, b) -> float:
    """Percentage difference between two values."""
    avg = (abs(a) + abs(b)) / 2
    return abs(a - b) / avg * 100 if avg > 0 else 0.0


def _iso(ts) -> str:
    """pandas Timestamp → ISO-8601 string for plotly x-axis."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance multi-level columns and keep only OHLCV."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    wanted = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[wanted].copy()


def find_pivots(
    high: pd.Series,
    low: pd.Series,
    order: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find local pivot highs and lows using a rolling window of size 2*order+1.
    Returns integer index positions (into the DataFrame) as numpy arrays.
    """
    h = high.values
    l = low.values
    n = len(h)
    highs, lows = [], []
    for i in range(order, n - order):
        window_h = h[i - order: i + order + 1]
        window_l = l[i - order: i + order + 1]
        if h[i] == window_h.max():
            highs.append(i)
        if l[i] == window_l.min():
            lows.append(i)
    return np.array(highs, dtype=int), np.array(lows, dtype=int)


# ── Shape / annotation factories ──────────────────────────────────────────────

def _hline(x0: str, x1: str, y: float, color: str, dash: str = "dash",
           width: float = 1.5, opacity: float = 1.0) -> dict:
    return {
        "type": "line",
        "x0": x0, "x1": x1, "y0": y, "y1": y,
        "xref": "x", "yref": "y",
        "opacity": opacity,
        "line": {"color": color, "dash": dash, "width": width},
    }


def _tline(x0: str, y0: float, x1: str, y1: float, color: str,
           dash: str = "dash", opacity: float = 1.0) -> dict:
    return {
        "type": "line",
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "xref": "x", "yref": "y",
        "opacity": opacity,
        "line": {"color": color, "dash": dash, "width": 1.5},
    }


def _ann(x: str, y: float, text: str, color: str, size: int = 11) -> dict:
    return {
        "x": x, "y": y, "text": text,
        "showarrow": False,
        "xref": "x", "yref": "y",
        "xanchor": "left",
        "font": {"color": color, "size": size},
    }


def _future_date(df: pd.DataFrame, n_bars: int) -> str:
    """Compute an ISO date string n_bars ahead of the last candle."""
    if len(df) < 2:
        return _iso(df.index[-1])
    avg_delta = (df.index[-1] - df.index[0]) / (len(df) - 1)
    return _iso(df.index[-1] + avg_delta * n_bars)


# ── Projection colors (visually distinct from historical pattern colors) ──────
_PROJ_COLOR   = "#00E5FF"   # Bright cyan  — all projected trendlines
_TARGET_COLOR = "#69FF47"   # Lime green   — projected target price
_STOP_COLOR   = "#FF6B6B"   # Coral red    — projected stop loss


def _project_from(shape: dict, x_future: str, opacity: float = 0.85,
                  width: float = 1.5) -> dict:
    """
    Build a NEW projection segment starting from where shape ends (x1, y1)
    and extending to x_future in the universal projection color.
    Horizontal lines stay flat; diagonal lines continue their slope.
    """
    x1, y1 = shape["x1"], float(shape["y1"])
    x0, y0 = shape["x0"], float(shape["y0"])
    x1_dt  = pd.Timestamp(x1)
    xf_dt  = pd.Timestamp(x_future)
    x0_dt  = pd.Timestamp(x0)
    days_hist = (x1_dt - x0_dt).days
    days_proj = (xf_dt - x1_dt).days

    if days_hist > 0 and abs(y1 - y0) > 1e-6:   # diagonal — project slope
        y_future = float(y1 + (y1 - y0) / days_hist * days_proj)
    else:                                          # horizontal — flat
        y_future = y1

    return {
        "type": "line",
        "x0": x1, "y0": y1,
        "x1": x_future, "y1": y_future,
        "xref": "x", "yref": "y",
        "opacity": opacity,
        "line": {"color": _PROJ_COLOR, "dash": "dash", "width": width},
    }


# ═══════════════════════════════════════════════════════════════
#  PATTERN DETECTORS
# ═══════════════════════════════════════════════════════════════

def detect_head_shoulders(
    df: pd.DataFrame,
    ph: np.ndarray,
    pl: np.ndarray,
) -> list[PatternResult]:
    results = []
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    # ── Bearish Head & Shoulders ──────────────────────────────────────────────
    for i in range(len(ph) - 2):
        ls_i, hd_i, rs_i = ph[i], ph[i + 1], ph[i + 2]
        ls, hd, rs = h[ls_i], h[hd_i], h[rs_i]

        if not (hd > ls and hd > rs):
            continue
        if _pct(ls, rs) > 8:
            continue
        if (hd - max(ls, rs)) / hd * 100 < 3:
            continue

        lt_arr = pl[(pl > ls_i) & (pl < hd_i)]
        rt_arr = pl[(pl > hd_i) & (pl < rs_i)]
        if len(lt_arr) == 0 or len(rt_arr) == 0:
            continue

        lt_i, rt_i = int(lt_arr[-1]), int(rt_arr[0])
        nl1, nl2   = l[lt_i], l[rt_i]
        neckline   = float((nl1 + nl2) / 2)
        target     = float(neckline - (hd - neckline))
        curr       = float(c[-1])

        conf = 60.0
        if _pct(ls, rs)   < 3: conf += 15
        if _pct(nl1, nl2) < 3: conf += 10
        if curr < neckline:    conf += 15
        conf = min(conf, 95.0)

        status = "Breakdown" if curr < neckline else "Forming"
        color  = _BEAR_COLOR
        x_end  = _iso(df.index[-1])

        lines = [
            _hline(_iso(df.index[lt_i]), x_end, neckline, color),
            _hline(_iso(df.index[rs_i]), x_end, target, color, dash="dot", width=1),
        ]
        anns = [
            _ann(x_end, neckline, f" Neckline ₹{neckline:.0f}", color),
            _ann(x_end, target,   f" Target ₹{target:.0f}",     color, size=10),
        ]

        results.append(PatternResult(
            name="Head & Shoulders", signal="Bearish",
            confidence=conf, status=status,
            description=(
                f"Three peaks — left shoulder ₹{ls:.0f}, head ₹{hd:.0f}, right shoulder ₹{rs:.0f}. "
                f"Neckline at ₹{neckline:.0f}. "
                + ("Confirmed — price broke below neckline." if status == "Breakdown"
                   else f"Awaiting breakdown below ₹{neckline:.0f}.")
            ),
            target_price=target, stop_loss=_f(hd * 1.02), breakout_level=neckline,
            lines=lines, annotations=anns,
        ))

    # ── Bullish Inverse Head & Shoulders ─────────────────────────────────────
    for i in range(len(pl) - 2):
        ls_i, hd_i, rs_i = pl[i], pl[i + 1], pl[i + 2]
        ls, hd, rs = l[ls_i], l[hd_i], l[rs_i]

        if not (hd < ls and hd < rs):
            continue
        if _pct(ls, rs) > 8:
            continue
        if (min(ls, rs) - hd) / min(ls, rs) * 100 < 3:
            continue

        lp_arr = ph[(ph > ls_i) & (ph < hd_i)]
        rp_arr = ph[(ph > hd_i) & (ph < rs_i)]
        if len(lp_arr) == 0 or len(rp_arr) == 0:
            continue

        lp_i, rp_i = int(lp_arr[-1]), int(rp_arr[0])
        nl1, nl2   = h[lp_i], h[rp_i]
        neckline   = float((nl1 + nl2) / 2)
        target     = float(neckline + (neckline - hd))
        curr       = float(c[-1])

        conf = 60.0
        if _pct(ls, rs)   < 3: conf += 15
        if _pct(nl1, nl2) < 3: conf += 10
        if curr > neckline:    conf += 15
        conf = min(conf, 95.0)

        status = "Breakout" if curr > neckline else "Forming"
        color  = _BULL_COLOR
        x_end  = _iso(df.index[-1])

        results.append(PatternResult(
            name="Inverse Head & Shoulders", signal="Bullish",
            confidence=conf, status=status,
            description=(
                f"Three troughs — left shoulder ₹{ls:.0f}, head ₹{hd:.0f}, right shoulder ₹{rs:.0f}. "
                f"Neckline at ₹{neckline:.0f}. "
                + ("Confirmed — price broke above neckline." if status == "Breakout"
                   else f"Awaiting breakout above ₹{neckline:.0f}.")
            ),
            target_price=target, stop_loss=_f(hd * 0.98), breakout_level=neckline,
            lines=[
                _hline(_iso(df.index[lp_i]), x_end, neckline, color),
                _hline(_iso(df.index[rs_i]), x_end, target, color, dash="dot", width=1),
            ],
            annotations=[
                _ann(x_end, neckline, f" Neckline ₹{neckline:.0f}", color),
                _ann(x_end, target,   f" Target ₹{target:.0f}",     color, size=10),
            ],
        ))

    return results


def detect_double_top_bottom(
    df: pd.DataFrame,
    ph: np.ndarray,
    pl: np.ndarray,
) -> list[PatternResult]:
    results = []
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    x_end = _iso(df.index[-1])

    # ── Double Top ────────────────────────────────────────────────────────────
    for i in range(len(ph) - 1):
        p1_i, p2_i = int(ph[i]), int(ph[i + 1])
        p1, p2 = h[p1_i], h[p2_i]
        if _pct(p1, p2) > 3:
            continue
        troughs = pl[(pl > p1_i) & (pl < p2_i)]
        if len(troughs) == 0:
            continue
        t_i = int(troughs[0])
        neck = float(l[t_i])
        avg_top = float((p1 + p2) / 2)
        target  = float(neck - (avg_top - neck))
        curr    = float(c[-1])

        conf = 65.0
        if _pct(p1, p2) < 1.5: conf += 15
        if curr < neck:         conf += 15
        conf = min(conf, 95.0)
        status = "Breakdown" if curr < neck else "Forming"
        color  = _BEAR_COLOR

        results.append(PatternResult(
            name="Double Top", signal="Bearish",
            confidence=conf, status=status,
            description=(
                f"Two peaks at ₹{p1:.0f} and ₹{p2:.0f} with a valley at ₹{neck:.0f}. "
                + ("Neckline broken — bearish confirmed." if status == "Breakdown"
                   else f"Break below ₹{neck:.0f} confirms the pattern.")
            ),
            target_price=target, stop_loss=_f(avg_top * 1.02), breakout_level=neck,
            lines=[
                _hline(_iso(df.index[t_i]), x_end, neck,   color),
                _hline(_iso(df.index[p2_i]), x_end, target, color, dash="dot", width=1),
            ],
            annotations=[
                _ann(x_end, neck,   f" Neckline ₹{neck:.0f}",   color),
                _ann(x_end, target, f" Target ₹{target:.0f}",   color, size=10),
            ],
        ))

    # ── Double Bottom ─────────────────────────────────────────────────────────
    for i in range(len(pl) - 1):
        p1_i, p2_i = int(pl[i]), int(pl[i + 1])
        p1, p2 = l[p1_i], l[p2_i]
        if _pct(p1, p2) > 3:
            continue
        peaks = ph[(ph > p1_i) & (ph < p2_i)]
        if len(peaks) == 0:
            continue
        pk_i = int(peaks[0])
        neck    = float(h[pk_i])
        avg_bot = float((p1 + p2) / 2)
        target  = float(neck + (neck - avg_bot))
        curr    = float(c[-1])

        conf = 65.0
        if _pct(p1, p2) < 1.5: conf += 15
        if curr > neck:         conf += 15
        conf = min(conf, 95.0)
        status = "Breakout" if curr > neck else "Forming"
        color  = _BULL_COLOR

        results.append(PatternResult(
            name="Double Bottom", signal="Bullish",
            confidence=conf, status=status,
            description=(
                f"Two troughs at ₹{p1:.0f} and ₹{p2:.0f} with a peak at ₹{neck:.0f}. "
                + ("Neckline broken — bullish confirmed." if status == "Breakout"
                   else f"Break above ₹{neck:.0f} confirms the pattern.")
            ),
            target_price=target, stop_loss=_f(avg_bot * 0.98), breakout_level=neck,
            lines=[
                _hline(_iso(df.index[pk_i]), x_end, neck,   color),
                _hline(_iso(df.index[p2_i]), x_end, target, color, dash="dot", width=1),
            ],
            annotations=[
                _ann(x_end, neck,   f" Neckline ₹{neck:.0f}",   color),
                _ann(x_end, target, f" Target ₹{target:.0f}",   color, size=10),
            ],
        ))

    return results


def detect_triple_top_bottom(
    df: pd.DataFrame,
    ph: np.ndarray,
    pl: np.ndarray,
) -> list[PatternResult]:
    results = []
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    x_end = _iso(df.index[-1])

    # ── Triple Top ────────────────────────────────────────────────────────────
    if len(ph) >= 3:
        for i in range(len(ph) - 2):
            i1, i2, i3 = int(ph[i]), int(ph[i + 1]), int(ph[i + 2])
            v1, v2, v3 = h[i1], h[i2], h[i3]
            if _pct(v1, v2) > 3 or _pct(v2, v3) > 3:
                continue
            # Need two valleys between the three peaks
            t1_arr = pl[(pl > i1) & (pl < i2)]
            t2_arr = pl[(pl > i2) & (pl < i3)]
            if len(t1_arr) == 0 or len(t2_arr) == 0:
                continue
            neck = float(min(l[t1_arr[0]], l[t2_arr[0]]))
            avg_top = float((v1 + v2 + v3) / 3)
            target  = float(neck - (avg_top - neck))
            curr    = float(c[-1])

            conf = 75.0
            if _pct(v1, v3) < 1.5: conf += 10
            if curr < neck:         conf += 10
            conf = min(conf, 95.0)
            status = "Breakdown" if curr < neck else "Forming"

            results.append(PatternResult(
                name="Triple Top", signal="Bearish",
                confidence=conf, status=status,
                description=(
                    f"Three peaks near ₹{avg_top:.0f} with neckline at ₹{neck:.0f}. "
                    "Very reliable reversal pattern. "
                    + ("Confirmed breakdown." if status == "Breakdown"
                       else f"Breakdown below ₹{neck:.0f} targets ₹{target:.0f}.")
                ),
                target_price=target, stop_loss=_f(avg_top * 1.02), breakout_level=neck,
                lines=[_hline(_iso(df.index[i1]), x_end, neck, _BEAR_COLOR)],
                annotations=[_ann(x_end, neck, f" Neckline ₹{neck:.0f}", _BEAR_COLOR)],
            ))

    # ── Triple Bottom ─────────────────────────────────────────────────────────
    if len(pl) >= 3:
        for i in range(len(pl) - 2):
            i1, i2, i3 = int(pl[i]), int(pl[i + 1]), int(pl[i + 2])
            v1, v2, v3 = l[i1], l[i2], l[i3]
            if _pct(v1, v2) > 3 or _pct(v2, v3) > 3:
                continue
            p1_arr = ph[(ph > i1) & (ph < i2)]
            p2_arr = ph[(ph > i2) & (ph < i3)]
            if len(p1_arr) == 0 or len(p2_arr) == 0:
                continue
            neck    = float(max(h[p1_arr[0]], h[p2_arr[0]]))
            avg_bot = float((v1 + v2 + v3) / 3)
            target  = float(neck + (neck - avg_bot))
            curr    = float(c[-1])

            conf = 75.0
            if _pct(v1, v3) < 1.5: conf += 10
            if curr > neck:         conf += 10
            conf = min(conf, 95.0)
            status = "Breakout" if curr > neck else "Forming"

            results.append(PatternResult(
                name="Triple Bottom", signal="Bullish",
                confidence=conf, status=status,
                description=(
                    f"Three troughs near ₹{avg_bot:.0f} with neckline at ₹{neck:.0f}. "
                    "Very reliable reversal pattern. "
                    + ("Confirmed breakout." if status == "Breakout"
                       else f"Breakout above ₹{neck:.0f} targets ₹{target:.0f}.")
                ),
                target_price=target, stop_loss=_f(avg_bot * 0.98), breakout_level=neck,
                lines=[_hline(_iso(df.index[i1]), x_end, neck, _BULL_COLOR)],
                annotations=[_ann(x_end, neck, f" Neckline ₹{neck:.0f}", _BULL_COLOR)],
            ))

    return results


def detect_triangles(
    df: pd.DataFrame,
    ph: np.ndarray,
    pl: np.ndarray,
) -> list[PatternResult]:
    results = []
    if len(ph) < 3 or len(pl) < 3:
        return results

    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)
    x_end = _iso(df.index[-1])

    # Use last 5 pivots for trendline fitting
    ph_u = ph[-5:]
    pl_u = pl[-5:]
    h_pts = h[ph_u].astype(float)
    l_pts = l[pl_u].astype(float)

    m_h, b_h = np.polyfit(ph_u.astype(float), h_pts, 1)
    m_l, b_l = np.polyfit(pl_u.astype(float), l_pts, 1)

    # Projected values at current bar
    res_now = float(m_h * (n - 1) + b_h)
    sup_now = float(m_l * (n - 1) + b_l)
    curr    = float(c[-1])
    price_range = float(h_pts.mean())
    thresh  = price_range * 0.0008

    def trendline_pts(idx_arr, coeff):
        m, b = coeff
        x0_i, x1_i = int(idx_arr[0]), n - 1
        return _iso(df.index[x0_i]), float(m * x0_i + b), _iso(df.index[x1_i]), float(m * x1_i + b)

    # Ascending Triangle: flat resistance (slope ≈ 0), rising support
    h_flat   = abs(m_h) < thresh
    l_rising = m_l > thresh
    if h_flat and l_rising:
        resistance = float(h_pts.mean())
        target     = float(resistance + (resistance - l_pts.mean()))
        conf       = 65.0
        if len(ph_u) >= 4: conf += 10
        if curr > resistance: conf += 20
        conf   = min(conf, 92.0)
        status = "Breakout" if curr > resistance else "Forming"
        color  = _BULL_COLOR
        x0t, y0t, x1t, y1t = trendline_pts(pl_u, np.polyfit(pl_u.astype(float), l_pts, 1))

        results.append(PatternResult(
            name="Ascending Triangle", signal="Bullish",
            confidence=conf, status=status,
            description=(
                f"Flat resistance near ₹{resistance:.0f} with higher lows (rising support). "
                + ("Bullish breakout confirmed." if status == "Breakout"
                   else f"Awaiting breakout above ₹{resistance:.0f}.")
            ),
            target_price=_f(target), stop_loss=_f(sup_now * 0.99), breakout_level=_f(resistance),
            lines=[
                _hline(_iso(df.index[int(ph_u[0])]), x_end, resistance, color),
                _tline(x0t, y0t, x1t, y1t, color),
            ],
            annotations=[_ann(x_end, resistance, f" Resistance ₹{resistance:.0f}", color)],
        ))

    # Descending Triangle: flat support (slope ≈ 0), falling resistance
    l_flat    = abs(m_l) < thresh
    h_falling = m_h < -thresh
    if l_flat and h_falling:
        support = float(l_pts.mean())
        target  = float(support - (h_pts.mean() - support))
        conf    = 65.0
        if len(pl_u) >= 4: conf += 10
        if curr < support: conf += 20
        conf   = min(conf, 92.0)
        status = "Breakdown" if curr < support else "Forming"
        color  = _BEAR_COLOR
        x0t, y0t, x1t, y1t = trendline_pts(ph_u, np.polyfit(ph_u.astype(float), h_pts, 1))

        results.append(PatternResult(
            name="Descending Triangle", signal="Bearish",
            confidence=conf, status=status,
            description=(
                f"Flat support near ₹{support:.0f} with lower highs (falling resistance). "
                + ("Bearish breakdown confirmed." if status == "Breakdown"
                   else f"Awaiting breakdown below ₹{support:.0f}.")
            ),
            target_price=_f(target), stop_loss=_f(res_now * 1.01), breakout_level=_f(support),
            lines=[
                _hline(_iso(df.index[int(pl_u[0])]), x_end, support, color),
                _tline(x0t, y0t, x1t, y1t, color),
            ],
            annotations=[_ann(x_end, support, f" Support ₹{support:.0f}", color)],
        ))

    # Symmetrical Triangle: falling highs + rising lows, converging
    h_falling2 = m_h < -thresh
    l_rising2  = m_l > thresh
    if h_falling2 and l_rising2:
        mid   = float((res_now + sup_now) / 2)
        conf  = 60.0
        if len(ph_u) >= 4 and len(pl_u) >= 4: conf += 15
        conf  = min(conf, 82.0)
        x0h, y0h, x1h, y1h = trendline_pts(ph_u, np.polyfit(ph_u.astype(float), h_pts, 1))
        x0l, y0l, x1l, y1l = trendline_pts(pl_u, np.polyfit(pl_u.astype(float), l_pts, 1))

        results.append(PatternResult(
            name="Symmetrical Triangle", signal="Neutral",
            confidence=conf, status="Forming",
            description=(
                "Converging highs and lows forming a symmetrical triangle. "
                "Breakout direction typically follows the prior trend. "
                f"Apex near ₹{mid:.0f}."
            ),
            target_price=None, stop_loss=None, breakout_level=_f(mid),
            lines=[
                _tline(x0h, y0h, x1h, y1h, _NEUT_COLOR),
                _tline(x0l, y0l, x1l, y1l, _NEUT_COLOR),
            ],
            annotations=[_ann(x_end, mid, f" Apex ₹{mid:.0f}", _NEUT_COLOR)],
        ))

    return results


def detect_wedges(
    df: pd.DataFrame,
    ph: np.ndarray,
    pl: np.ndarray,
) -> list[PatternResult]:
    results = []
    if len(ph) < 3 or len(pl) < 3:
        return results

    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)
    x_end = _iso(df.index[-1])

    ph_u = ph[-4:]
    pl_u = pl[-4:]
    h_pts = h[ph_u].astype(float)
    l_pts = l[pl_u].astype(float)

    m_h, b_h = np.polyfit(ph_u.astype(float), h_pts, 1)
    m_l, b_l = np.polyfit(pl_u.astype(float), l_pts, 1)

    sup_now = float(m_l * (n - 1) + b_l)
    res_now = float(m_h * (n - 1) + b_h)
    curr    = float(c[-1])
    thresh  = float(h_pts.mean()) * 0.0005

    def tl(idx_arr, m, b):
        x0_i = int(idx_arr[0])
        return _iso(df.index[x0_i]), float(m * x0_i + b), x_end, float(m * (n - 1) + b)

    # Rising Wedge: both slopes positive, lower slope > upper slope (converging upward) — bearish
    if m_h > thresh and m_l > thresh and m_l > m_h * 1.05:
        conf   = 60.0
        if curr < sup_now: conf += 20
        conf   = min(conf, 88.0)
        status = "Breakdown" if curr < sup_now else "Forming"

        results.append(PatternResult(
            name="Rising Wedge", signal="Bearish",
            confidence=conf, status=status,
            description=(
                "Both support and resistance rising but converging — rising wedge. "
                + ("Price broke below rising support — bearish signal." if status == "Breakdown"
                   else "Watch for break below rising support line.")
            ),
            target_price=_f(l_pts[0] * 0.97), stop_loss=_f(h_pts[-1] * 1.01),
            breakout_level=_f(sup_now),
            lines=[_tline(*tl(ph_u, m_h, b_h), _BEAR_COLOR),
                   _tline(*tl(pl_u, m_l, b_l), _BEAR_COLOR)],
            annotations=[_ann(x_end, sup_now, f" Support ₹{sup_now:.0f}", _BEAR_COLOR)],
        ))

    # Falling Wedge: both slopes negative, upper slope < lower slope (converging downward) — bullish
    if m_h < -thresh and m_l < -thresh and m_h < m_l * 1.05:
        conf   = 60.0
        if curr > res_now: conf += 20
        conf   = min(conf, 88.0)
        status = "Breakout" if curr > res_now else "Forming"

        results.append(PatternResult(
            name="Falling Wedge", signal="Bullish",
            confidence=conf, status=status,
            description=(
                "Both support and resistance falling but converging — falling wedge. "
                + ("Price broke above falling resistance — bullish signal." if status == "Breakout"
                   else "Watch for break above falling resistance line.")
            ),
            target_price=_f(h_pts[0] * 1.03), stop_loss=_f(l_pts[-1] * 0.99),
            breakout_level=_f(res_now),
            lines=[_tline(*tl(ph_u, m_h, b_h), _BULL_COLOR),
                   _tline(*tl(pl_u, m_l, b_l), _BULL_COLOR)],
            annotations=[_ann(x_end, res_now, f" Resistance ₹{res_now:.0f}", _BULL_COLOR)],
        ))

    return results


def detect_flags(
    df: pd.DataFrame,
) -> list[PatternResult]:
    results = []
    c = df["Close"].values
    n = len(c)
    if n < 25:
        return results

    x_end = _iso(df.index[-1])
    window   = min(60, n)
    half     = window // 3
    pole_seg = c[n - window: n - window + half]
    flag_seg = c[n - window + half:]

    if len(pole_seg) < 5 or len(flag_seg) < 5:
        return results

    pole_ret   = float((pole_seg[-1] - pole_seg[0]) / pole_seg[0] * 100)
    flag_range = float((max(flag_seg) - min(flag_seg)) / abs(pole_seg[-1]) * 100)
    flag_slope = float(np.polyfit(range(len(flag_seg)), flag_seg, 1)[0])

    # Bull Flag: sharp rally + slight pullback consolidation
    if pole_ret > 8 and flag_range < abs(pole_ret) * 0.6 and flag_slope <= 0:
        pole_h    = float(pole_seg[-1] - pole_seg[0])
        target    = float(c[-1] + pole_h)
        stop      = float(min(flag_seg) * 0.99)
        bl        = float(max(flag_seg))

        conf = 65.0
        if flag_range < 4:          conf += 15
        if pole_ret > 15:           conf += 10
        if c[-1] > max(flag_seg):   conf += 10
        conf   = min(conf, 92.0)
        status = "Breakout" if float(c[-1]) > bl else "Forming"

        results.append(PatternResult(
            name="Bull Flag", signal="Bullish",
            confidence=conf, status=status,
            description=(
                f"Sharp rally of {pole_ret:.1f}% (flagpole) followed by tight consolidation. "
                + ("Breakout confirmed — continuation expected." if status == "Breakout"
                   else f"Break above ₹{bl:.0f} signals continuation of the rally.")
            ),
            target_price=_f(target), stop_loss=_f(stop), breakout_level=_f(bl),
            lines=[_hline(_iso(df.index[n - window + half]), x_end, bl, _BULL_COLOR, dash="dot")],
            annotations=[_ann(x_end, bl, f" Flag High ₹{bl:.0f}", _BULL_COLOR)],
        ))

    # Bear Flag: sharp decline + slight bounce consolidation
    if pole_ret < -8 and flag_range < abs(pole_ret) * 0.6 and flag_slope >= 0:
        pole_h  = float(abs(pole_seg[0] - pole_seg[-1]))
        target  = float(c[-1] - pole_h)
        stop    = float(max(flag_seg) * 1.01)
        bl      = float(min(flag_seg))

        conf = 65.0
        if flag_range < 4:          conf += 15
        if abs(pole_ret) > 15:      conf += 10
        if float(c[-1]) < bl:       conf += 10
        conf   = min(conf, 92.0)
        status = "Breakdown" if float(c[-1]) < bl else "Forming"

        results.append(PatternResult(
            name="Bear Flag", signal="Bearish",
            confidence=conf, status=status,
            description=(
                f"Sharp decline of {abs(pole_ret):.1f}% (flagpole) followed by tight consolidation. "
                + ("Breakdown confirmed — continuation expected." if status == "Breakdown"
                   else f"Break below ₹{bl:.0f} signals continuation of the decline.")
            ),
            target_price=_f(target), stop_loss=_f(stop), breakout_level=_f(bl),
            lines=[_hline(_iso(df.index[n - window + half]), x_end, bl, _BEAR_COLOR, dash="dot")],
            annotations=[_ann(x_end, bl, f" Flag Low ₹{bl:.0f}", _BEAR_COLOR)],
        ))

    return results


def detect_cup_handle(df: pd.DataFrame) -> Optional[PatternResult]:
    """Cup & Handle — bullish continuation pattern."""
    c = df["Close"].values
    h = df["High"].values
    n = len(c)
    if n < 45:
        return None

    x_end  = _iso(df.index[-1])
    window = min(90, n)
    seg    = c[n - window:]
    m      = len(seg)

    thirds  = m // 3
    left_s  = seg[:thirds]
    mid_s   = seg[thirds: 2 * thirds]
    right_s = seg[2 * thirds:]

    left_hi  = float(max(left_s))
    right_hi = float(max(right_s))
    cup_low  = float(min(mid_s))

    if _pct(left_hi, right_hi) > 10:
        return None

    cup_depth = float((left_hi - cup_low) / left_hi * 100)
    if not (10 <= cup_depth <= 50):
        return None

    handle_hi  = float(max(right_s[-len(right_s) // 3:]))
    handle_lo  = float(min(right_s[-len(right_s) // 3:]))
    handle_dep = float((handle_hi - handle_lo) / handle_hi * 100)
    if handle_dep > 15 or handle_dep < 2:
        return None

    rim    = float((left_hi + right_hi) / 2)
    target = float(rim + (rim - cup_low))
    curr   = float(c[-1])

    conf = 65.0
    if _pct(left_hi, right_hi) < 5: conf += 15
    if curr > rim:                   conf += 15
    conf   = min(conf, 92.0)
    status = "Breakout" if curr > rim else "Forming"

    return PatternResult(
        name="Cup & Handle", signal="Bullish",
        confidence=conf, status=status,
        description=(
            f"U-shaped base ({cup_depth:.0f}% deep) with rim at ₹{rim:.0f} followed by a handle. "
            + ("Breakout above cup rim confirmed." if status == "Breakout"
               else f"Watch for breakout above ₹{rim:.0f}.")
        ),
        target_price=_f(target), stop_loss=_f(handle_lo * 0.99), breakout_level=_f(rim),
        lines=[_hline(_iso(df.index[n - window]), x_end, rim, _BULL_COLOR)],
        annotations=[_ann(x_end, rim, f" Cup Rim ₹{rim:.0f}", _BULL_COLOR)],
    )


def detect_pennants(df: pd.DataFrame) -> list[PatternResult]:
    """
    Bull/Bear Pennant — like a flag but the consolidation is a small
    symmetrical triangle (converging trendlines) instead of a rectangle.
    """
    results = []
    c = df["Close"].values
    h = df["High"].values
    l = df["Low"].values
    n = len(c)
    if n < 20:
        return results

    x_end   = _iso(df.index[-1])
    window  = min(50, n)
    half    = window // 3
    pole    = c[n - window: n - window + half]
    consol  = c[n - window + half:]
    ch      = h[n - window + half:]
    cl      = l[n - window + half:]

    if len(pole) < 4 or len(consol) < 6:
        return results

    pole_ret = float((pole[-1] - pole[0]) / pole[0] * 100)
    xs       = np.arange(len(consol), dtype=float)
    mh, _bh  = np.polyfit(xs, ch.astype(float), 1)
    ml, _bl  = np.polyfit(xs, cl.astype(float), 1)

    # Pennant: slopes converge (opposite signs or same sign but converging)
    converging = (mh < 0 and ml > 0)

    # Bull Pennant: sharp up-pole + converging consolidation
    if pole_ret > 8 and converging:
        bl     = float(ch.max())
        target = float(c[-1] + (pole[-1] - pole[0]))
        stop   = float(cl.min() * 0.99)
        conf   = 65.0
        if pole_ret > 15:           conf += 10
        if float(c[-1]) > bl:       conf += 15
        conf   = min(conf, 90.0)
        status = "Breakout" if float(c[-1]) > bl else "Forming"
        idx0   = n - window + half

        results.append(PatternResult(
            name="Bull Pennant", signal="Bullish",
            confidence=conf, status=status,
            description=(
                f"Sharp rally of {pole_ret:.1f}% followed by a converging triangular consolidation. "
                + ("Breakout above pennant — continuation expected." if status == "Breakout"
                   else f"Awaiting breakout above ₹{bl:.0f} to target ₹{target:.0f}.")
            ),
            target_price=_f(target), stop_loss=_f(stop), breakout_level=_f(bl),
            lines=[
                _hline(_iso(df.index[idx0]), x_end, bl, _BULL_COLOR, dash="dot"),
                _tline(_iso(df.index[idx0]), float(ch[0]), x_end, float(mh * (len(consol) - 1) + _bh), _BULL_COLOR),
                _tline(_iso(df.index[idx0]), float(cl[0]), x_end, float(ml * (len(consol) - 1) + _bl), _BULL_COLOR),
            ],
            annotations=[_ann(x_end, bl, f" Pennant High ₹{bl:.0f}", _BULL_COLOR)],
        ))

    # Bear Pennant: sharp down-pole + converging consolidation
    if pole_ret < -8 and converging:
        bl     = float(cl.min())
        target = float(c[-1] - abs(pole[-1] - pole[0]))
        stop   = float(ch.max() * 1.01)
        conf   = 65.0
        if abs(pole_ret) > 15:      conf += 10
        if float(c[-1]) < bl:       conf += 15
        conf   = min(conf, 90.0)
        status = "Breakdown" if float(c[-1]) < bl else "Forming"
        idx0   = n - window + half

        results.append(PatternResult(
            name="Bear Pennant", signal="Bearish",
            confidence=conf, status=status,
            description=(
                f"Sharp decline of {abs(pole_ret):.1f}% followed by a converging triangular consolidation. "
                + ("Breakdown below pennant — continuation expected." if status == "Breakdown"
                   else f"Awaiting breakdown below ₹{bl:.0f} to target ₹{target:.0f}.")
            ),
            target_price=_f(target), stop_loss=_f(stop), breakout_level=_f(bl),
            lines=[
                _hline(_iso(df.index[idx0]), x_end, bl, _BEAR_COLOR, dash="dot"),
                _tline(_iso(df.index[idx0]), float(ch[0]), x_end, float(mh * (len(consol) - 1) + _bh), _BEAR_COLOR),
                _tline(_iso(df.index[idx0]), float(cl[0]), x_end, float(ml * (len(consol) - 1) + _bl), _BEAR_COLOR),
            ],
            annotations=[_ann(x_end, bl, f" Pennant Low ₹{bl:.0f}", _BEAR_COLOR)],
        ))

    return results


def detect_rectangle(
    df: pd.DataFrame,
    ph: np.ndarray,
    pl: np.ndarray,
) -> Optional[PatternResult]:
    """
    Rectangle (Trading Range) — price oscillates between flat support and flat
    resistance. Neutral until breakout direction is known.
    """
    if len(ph) < 2 or len(pl) < 2:
        return None

    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    x_end = _iso(df.index[-1])

    resistance = float(np.mean(h[ph[-4:]]))  # average of last 4 pivot highs
    support    = float(np.mean(l[pl[-4:]]))  # average of last 4 pivot lows
    height     = resistance - support
    if height <= 0:
        return None

    # Require oscillation: range must be meaningful (>3% of price)
    if height / resistance * 100 < 3:
        return None

    # Check that recent highs are within 3% of resistance and recent lows within 3% of support
    recent_highs = h[ph[-3:]]
    recent_lows  = l[pl[-3:]]
    if (np.any(np.abs(recent_highs - resistance) / resistance > 0.04) or
            np.any(np.abs(recent_lows - support) / support > 0.04)):
        return None

    curr   = float(c[-1])
    conf   = 62.0
    if len(ph) >= 4 and len(pl) >= 4: conf += 12
    conf   = min(conf, 82.0)

    if curr > resistance:
        signal = "Bullish"; status = "Breakout"
        target = _f(resistance + height)
        stop   = _f(support)
        color  = _BULL_COLOR
        label  = f" Breakout ₹{resistance:.0f}"
    elif curr < support:
        signal = "Bearish"; status = "Breakdown"
        target = _f(support - height)
        stop   = _f(resistance)
        color  = _BEAR_COLOR
        label  = f" Breakdown ₹{support:.0f}"
    else:
        signal = "Neutral"; status = "Forming"
        target = None; stop = None; color = _NEUT_COLOR
        label  = f" Range ₹{support:.0f}–₹{resistance:.0f}"

    x0 = _iso(df.index[int(ph[-4]) if len(ph) >= 4 else 0])
    return PatternResult(
        name="Rectangle", signal=signal,
        confidence=conf, status=status,
        description=(
            f"Price consolidating between support ₹{support:.0f} and resistance ₹{resistance:.0f} "
            f"({height / resistance * 100:.1f}% range). "
            + (f"Bullish breakout — target ₹{resistance + height:.0f}." if status == "Breakout"
               else f"Bearish breakdown — target ₹{support - height:.0f}." if status == "Breakdown"
               else f"Watch for breakout above ₹{resistance:.0f} or breakdown below ₹{support:.0f}.")
        ),
        target_price=target, stop_loss=stop, breakout_level=_f(resistance if signal != "Bearish" else support),
        lines=[
            _hline(x0, x_end, resistance, color),
            _hline(x0, x_end, support,    color),
        ],
        annotations=[
            _ann(x_end, resistance, f" Resistance ₹{resistance:.0f}", color),
            _ann(x_end, support,    f" Support ₹{support:.0f}",       color),
        ],
    )


def detect_broadening(
    df: pd.DataFrame,
    ph: np.ndarray,
    pl: np.ndarray,
) -> Optional[PatternResult]:
    """
    Broadening Formation (Megaphone) — expanding price swings with diverging
    trendlines. Often signals distribution / increased volatility.
    """
    if len(ph) < 3 or len(pl) < 3:
        return None

    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)
    x_end = _iso(df.index[-1])

    ph_u = ph[-4:]
    pl_u = pl[-4:]
    h_pts = h[ph_u].astype(float)
    l_pts = l[pl_u].astype(float)

    m_h, b_h = np.polyfit(ph_u.astype(float), h_pts, 1)
    m_l, b_l = np.polyfit(pl_u.astype(float), l_pts, 1)

    thresh = float(h_pts.mean()) * 0.0005

    # Broadening: highs rising AND lows falling (diverging trendlines)
    if not (m_h > thresh and m_l < -thresh):
        return None

    res_now = float(m_h * (n - 1) + b_h)
    sup_now = float(m_l * (n - 1) + b_l)
    curr    = float(c[-1])

    conf = 58.0
    if len(ph_u) >= 4 and len(pl_u) >= 4: conf += 12
    conf = min(conf, 78.0)

    x0h = _iso(df.index[int(ph_u[0])])
    x0l = _iso(df.index[int(pl_u[0])])

    return PatternResult(
        name="Broadening Formation", signal="Bearish",
        confidence=conf, status="Forming",
        description=(
            "Expanding price swings with rising highs and falling lows — megaphone pattern. "
            "Signals market indecision and increasing volatility. Typically bearish at tops. "
            f"Current upper boundary ₹{res_now:.0f}, lower ₹{sup_now:.0f}."
        ),
        target_price=_f(sup_now * 0.97), stop_loss=_f(res_now * 1.02), breakout_level=_f(sup_now),
        lines=[
            _tline(x0h, float(m_h * ph_u[0] + b_h), x_end, res_now, _BEAR_COLOR),
            _tline(x0l, float(m_l * pl_u[0] + b_l), x_end, sup_now, _BEAR_COLOR),
        ],
        annotations=[
            _ann(x_end, res_now, f" Upper ₹{res_now:.0f}", _BEAR_COLOR),
            _ann(x_end, sup_now, f" Lower ₹{sup_now:.0f}", _BEAR_COLOR),
        ],
    )


def detect_three_drives(
    df: pd.DataFrame,
    ph: np.ndarray,
    pl: np.ndarray,
) -> list[PatternResult]:
    """
    Three Drives — three equal, symmetrical pushes in the same direction,
    each separated by a corrective pullback. Signals trend exhaustion.
    """
    results = []
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    x_end = _iso(df.index[-1])

    # Three Drives Up (bearish reversal) — three pivot highs ascending
    if len(ph) >= 3:
        for i in range(len(ph) - 2):
            i1, i2, i3 = int(ph[i]), int(ph[i + 1]), int(ph[i + 2])
            v1, v2, v3 = h[i1], h[i2], h[i3]
            if not (v3 > v2 > v1):
                continue
            # Each drive should be roughly equal in size
            d1 = v2 - v1
            d2 = v3 - v2
            if d1 <= 0 or d2 <= 0:
                continue
            if _pct(d1, d2) > 30:   # drives within 30% of each other
                continue
            # Need corrections between drives
            corr1 = pl[(pl > i1) & (pl < i2)]
            corr2 = pl[(pl > i2) & (pl < i3)]
            if len(corr1) == 0 or len(corr2) == 0:
                continue

            avg_drive = float((d1 + d2) / 2)
            curr      = float(c[-1])
            stop      = float(v3 * 1.01)
            target    = float(l[corr1[0]] if len(corr1) else v1)

            conf = 60.0
            if _pct(d1, d2) < 15: conf += 15
            if curr < v3:          conf += 10
            conf   = min(conf, 88.0)
            status = "Breakdown" if curr < float(l[corr2[0]]) else "Forming"

            results.append(PatternResult(
                name="Three Drives Up", signal="Bearish",
                confidence=conf, status=status,
                description=(
                    f"Three ascending drives to ₹{v1:.0f}, ₹{v2:.0f}, ₹{v3:.0f} — "
                    f"each drive ~₹{avg_drive:.0f}. "
                    "Equal measured moves signal buyer exhaustion. "
                    + ("Reversal likely underway." if status == "Breakdown"
                       else f"Reversal below ₹{float(l[corr2[0]]):.0f} confirms.")
                ),
                target_price=_f(target), stop_loss=_f(stop), breakout_level=_f(float(l[corr2[-1]])),
                lines=[_hline(_iso(df.index[i1]), x_end, float(v3), _BEAR_COLOR, dash="dot")],
                annotations=[_ann(x_end, float(v3), f" Drive 3 ₹{v3:.0f}", _BEAR_COLOR)],
            ))

    # Three Drives Down (bullish reversal) — three pivot lows descending
    if len(pl) >= 3:
        for i in range(len(pl) - 2):
            i1, i2, i3 = int(pl[i]), int(pl[i + 1]), int(pl[i + 2])
            v1, v2, v3 = l[i1], l[i2], l[i3]
            if not (v3 < v2 < v1):
                continue
            d1 = v1 - v2
            d2 = v2 - v3
            if d1 <= 0 or d2 <= 0:
                continue
            if _pct(d1, d2) > 30:
                continue
            peaks1 = ph[(ph > i1) & (ph < i2)]
            peaks2 = ph[(ph > i2) & (ph < i3)]
            if len(peaks1) == 0 or len(peaks2) == 0:
                continue

            avg_drive = float((d1 + d2) / 2)
            curr      = float(c[-1])
            stop      = float(v3 * 0.99)
            target    = float(h[peaks1[0]] if len(peaks1) else v1)

            conf = 60.0
            if _pct(d1, d2) < 15: conf += 15
            if curr > v3:          conf += 10
            conf   = min(conf, 88.0)
            status = "Breakout" if curr > float(h[peaks2[0]]) else "Forming"

            results.append(PatternResult(
                name="Three Drives Down", signal="Bullish",
                confidence=conf, status=status,
                description=(
                    f"Three descending drives to ₹{v1:.0f}, ₹{v2:.0f}, ₹{v3:.0f} — "
                    f"each drive ~₹{avg_drive:.0f}. "
                    "Equal measured moves signal seller exhaustion. "
                    + ("Reversal likely underway." if status == "Breakout"
                       else f"Reversal above ₹{float(h[peaks2[0]]):.0f} confirms.")
                ),
                target_price=_f(target), stop_loss=_f(stop), breakout_level=_f(float(h[peaks2[-1]])),
                lines=[_hline(_iso(df.index[i1]), x_end, float(v3), _BULL_COLOR, dash="dot")],
                annotations=[_ann(x_end, float(v3), f" Drive 3 ₹{v3:.0f}", _BULL_COLOR)],
            ))

    return results


def detect_rounding(df: pd.DataFrame) -> list[PatternResult]:
    """
    Rounding Bottom (Saucer) and Rounding Top — slow, gradual curve reversal.
    Works on longer timeframes where the pattern takes months to form.
    """
    results = []
    c = df["Close"].values
    n = len(c)
    if n < 60:
        return results

    x_end = _iso(df.index[-1])

    # Use last 80% of data to find the curve
    seg_len = int(n * 0.8)
    seg     = c[n - seg_len:]
    xs      = np.linspace(0, 1, len(seg))

    # Fit a quadratic (parabola): positive a = U-shape (bottom), negative a = ∩-shape (top)
    try:
        a, b, cc_coeff = np.polyfit(xs, seg, 2)
    except (np.linalg.LinAlgError, ValueError):
        return results

    a = float(a)
    if abs(a) < 0.001 * float(seg.mean()):   # curve not pronounced enough
        return results

    fitted  = np.polyval([a, b, cc_coeff], xs)
    resid   = float(np.std(seg - fitted))
    price_scale = float(seg.mean())
    fit_quality = 1.0 - (resid / price_scale)   # 1 = perfect fit

    if fit_quality < 0.90:   # require tight fit to parabola
        return results

    curr = float(c[-1])
    rim  = float(max(seg[0], seg[-1]))   # the two ends of the curve

    # ── Rounding Bottom ───────────────────────────────────────────────────────
    if a > 0:
        bottom  = float(min(seg))
        depth   = float((rim - bottom) / rim * 100)
        if depth < 5:
            return results
        target  = float(rim + (rim - bottom))
        conf    = float(min(50 + fit_quality * 40, 88))
        status  = "Breakout" if curr > rim else "Forming"

        results.append(PatternResult(
            name="Rounding Bottom", signal="Bullish",
            confidence=conf, status=status,
            description=(
                f"Gradual U-shaped base over the past {seg_len} candles — "
                f"depth {depth:.0f}% from rim ₹{rim:.0f}. "
                "Slow accumulation pattern indicating a major trend reversal. "
                + ("Breakout above rim confirmed." if status == "Breakout"
                   else f"Watch for close above rim ₹{rim:.0f}.")
            ),
            target_price=_f(target), stop_loss=_f(bottom * 0.98), breakout_level=_f(rim),
            lines=[_hline(_iso(df.index[n - seg_len]), x_end, rim, _BULL_COLOR)],
            annotations=[_ann(x_end, rim, f" Rim ₹{rim:.0f}", _BULL_COLOR)],
        ))

    # ── Rounding Top ──────────────────────────────────────────────────────────
    elif a < 0:
        peak    = float(max(seg))
        height  = float((peak - rim) / peak * 100)
        if height < 5:
            return results
        target  = float(rim - (peak - rim))
        conf    = float(min(50 + fit_quality * 40, 85))
        status  = "Breakdown" if curr < rim else "Forming"

        results.append(PatternResult(
            name="Rounding Top", signal="Bearish",
            confidence=conf, status=status,
            description=(
                f"Gradual ∩-shaped top over the past {seg_len} candles — "
                f"height {height:.0f}% above rim ₹{rim:.0f}. "
                "Slow distribution pattern indicating a major trend reversal. "
                + ("Breakdown below rim confirmed." if status == "Breakdown"
                   else f"Watch for close below rim ₹{rim:.0f}.")
            ),
            target_price=_f(target), stop_loss=_f(peak * 1.02), breakout_level=_f(rim),
            lines=[_hline(_iso(df.index[n - seg_len]), x_end, rim, _BEAR_COLOR)],
            annotations=[_ann(x_end, rim, f" Rim ₹{rim:.0f}", _BEAR_COLOR)],
        ))

    return results


# ═══════════════════════════════════════════════════════════════
#  MASTER DETECTOR
# ═══════════════════════════════════════════════════════════════

def detect_all_patterns(
    df: pd.DataFrame,
    order: int = 5,
    timeframe: str = "1 Year",
) -> list[PatternResult]:
    """
    Run all pattern detectors valid for the given timeframe.
    Patterns outside their natural timeframe are silently skipped.
    Returns list sorted by confidence (highest first).
    """
    df = flatten_ohlcv(df)
    min_needed = max(30, order * 6)
    if len(df) < min_needed:
        return []

    valid = PATTERN_TIMEFRAMES   # shorthand
    ph, pl = find_pivots(df["High"], df["Low"], order)
    all_results: list[PatternResult] = []

    if len(ph) >= 3:
        all_results.extend(detect_head_shoulders(df, ph, pl))
    if len(ph) >= 2 and len(pl) >= 2:
        all_results.extend(detect_double_top_bottom(df, ph, pl))
    if len(ph) >= 3 and len(pl) >= 3:
        all_results.extend(detect_triple_top_bottom(df, ph, pl))
    if len(ph) >= 3 and len(pl) >= 3:
        all_results.extend(detect_triangles(df, ph, pl))
        all_results.extend(detect_wedges(df, ph, pl))
        all_results.extend(detect_three_drives(df, ph, pl))
        rect = detect_rectangle(df, ph, pl)
        if rect:
            all_results.append(rect)
        broad = detect_broadening(df, ph, pl)
        if broad:
            all_results.append(broad)
    all_results.extend(detect_flags(df))
    all_results.extend(detect_pennants(df))
    all_results.extend(detect_rounding(df))
    cup = detect_cup_handle(df)
    if cup:
        all_results.append(cup)

    # Filter: keep only patterns valid for this timeframe
    all_results = [p for p in all_results if timeframe in valid.get(p.name, set())]

    # Deduplicate: keep highest-confidence per pattern name
    best: dict[str, PatternResult] = {}
    for p in all_results:
        if p.name not in best or p.confidence > best[p.name].confidence:
            best[p.name] = p

    return sorted(best.values(), key=lambda x: x.confidence, reverse=True)


# ═══════════════════════════════════════════════════════════════
#  CHART BUILDER
# ═══════════════════════════════════════════════════════════════

def build_pattern_chart(
    df: pd.DataFrame,
    patterns: list[PatternResult],
    ticker: str = "",
    timeframe: str = "",
) -> go.Figure:
    """
    Candlestick + volume + EMA chart.

    Color scheme
    ────────────
    Historical pattern lines  → solid, per-rank palette (gold / blue / purple …)
    Future projection lines   → dashed CYAN (#00E5FF) — universal projection color
    Target price projection   → dotted LIME GREEN (#69FF47)
    Stop loss projection      → dash-dot CORAL RED (#FF6B6B)
    """
    df    = flatten_ohlcv(df)
    close = df["Close"]
    n     = len(df)

    # ── Geometry ──────────────────────────────────────────────────────────────
    n_future = min(60, max(10, n // 5))
    x_last   = _iso(df.index[-1])
    x_future = _future_date(df, n_future)

    # ── Per-pattern historical line colors (rank 0 = best) ────────────────────
    PAT_COLORS  = ["#FFD700", "#3498DB", "#9B59B6", "#1ABC9C", "#E67E22", "#E91E63"]
    PAT_OPACITY = [1.0,        0.80,      0.65,      0.55,      0.50,      0.45]
    PAT_WIDTH   = [2.5,        1.8,       1.5,       1.4,       1.2,       1.2]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )

    # ── Candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing_line_color=_BULL_COLOR,
        decreasing_line_color=_BEAR_COLOR,
        whiskerwidth=0.5,
    ), row=1, col=1)

    # ── EMA 20 / 50 / 200 ────────────────────────────────────────────────────
    for span, color, name in [(20, "#F39C12", "EMA 20"),
                               (50, "#3498DB", "EMA 50"),
                               (200, "#E91E63", "EMA 200")]:
        if n >= span:
            ema = close.ewm(span=span, adjust=False).mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=ema, name=name,
                line=dict(color=color, width=1.1),
                opacity=0.75, showlegend=True,
            ), row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_colors = [
        _BULL_COLOR if float(c) >= float(o) else _BEAR_COLOR
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vol_colors, opacity=0.55,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    # ── Shaded future zone ────────────────────────────────────────────────────
    fig.add_vrect(
        x0=x_last, x1=x_future,
        fillcolor="rgba(0,229,255,0.04)", line_width=0,
        annotation_text="◀ HISTORY  |  PROJECTION ▶",
        annotation_position="top left",
        annotation_font=dict(color="rgba(0,229,255,0.40)", size=9),
        row=1, col=1,
    )
    fig.add_vline(
        x=x_last, line_dash="dot",
        line_color="rgba(0,229,255,0.35)", line_width=1.2,
        row=1, col=1,
    )

    # ── Current price reference ───────────────────────────────────────────────
    curr = float(close.iloc[-1])
    fig.add_shape(
        type="line", xref="x", yref="y",
        x0=x_last, x1=x_future, y0=curr, y1=curr,
        line=dict(color="rgba(255,255,255,0.25)", dash="dot", width=1),
    )
    fig.add_annotation(
        x=x_future, y=curr,
        text=f" CMP ₹{curr:,.0f}",
        showarrow=False, xref="x", yref="y",
        xanchor="left", font=dict(color="rgba(255,255,255,0.45)", size=10),
    )

    # ── Entry zone shading (best pattern breakout level) ──────────────────────
    if patterns and patterns[0].breakout_level:
        bl  = patterns[0].breakout_level
        buf = bl * 0.005
        fc  = ("rgba(38,166,91,0.13)"  if patterns[0].signal == "Bullish"
               else "rgba(231,76,60,0.13)")
        fig.add_hrect(
            y0=bl - buf, y1=bl + buf,
            fillcolor=fc, line_width=0,
            annotation_text="Entry Zone",
            annotation_position="right",
            annotation_font=dict(size=9, color="rgba(255,255,255,0.45)"),
            row=1, col=1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DRAW PATTERNS
    # Historical lines  → solid,  per-rank color
    # Projection lines  → dashed, cyan  (_PROJ_COLOR)
    # Target projection → dotted, lime green  (_TARGET_COLOR)
    # Stop projection   → dash-dot, coral red  (_STOP_COLOR)
    # ─────────────────────────────────────────────────────────────────────────
    drawn: set[float] = set()

    def _is_new(level: float) -> bool:
        for e in drawn:
            if abs(e - level) / max(abs(level), 1) < 0.003:
                return False
        drawn.add(level)
        return True

    for rank, pat in enumerate(patterns[:6]):
        h_color  = PAT_COLORS[rank % len(PAT_COLORS)]
        opacity  = PAT_OPACITY[rank]
        lw       = PAT_WIDTH[rank]

        # ── 1. Historical pattern lines (solid, rank color) ───────────────────
        for shape in pat.lines:
            hist = dict(shape)
            hist["line"] = {**shape["line"], "color": h_color,
                            "dash": "solid", "width": lw}
            hist["opacity"] = opacity
            fig.add_shape(**hist)

            # ── 2. Projection of that same line (dashed cyan) ─────────────────
            proj = _project_from(shape, x_future, opacity=opacity * 0.85, width=lw * 0.8)
            fig.add_shape(**proj)

        # ── 3. Annotations — best pattern only ───────────────────────────────
        if rank == 0:
            for ann in pat.annotations:
                a = dict(ann)
                a["font"] = {**ann.get("font", {}), "color": h_color}
                fig.add_annotation(**a)
            if pat.breakout_level:
                fig.add_annotation(
                    x=x_last, y=pat.breakout_level,
                    text=f"◀ {pat.name}",
                    showarrow=False, xref="x", yref="y",
                    xanchor="right",
                    font=dict(color=h_color, size=11, family="monospace"),
                )

        # ── 4. Target projection (lime green, dotted) ─────────────────────────
        if pat.target_price and _is_new(pat.target_price):
            fig.add_shape(
                type="line", xref="x", yref="y",
                x0=x_last, x1=x_future,
                y0=pat.target_price, y1=pat.target_price,
                opacity=opacity,
                line=dict(color=_TARGET_COLOR, dash="dot",
                          width=2.0 if rank == 0 else 1.2),
            )
            fig.add_annotation(
                x=x_future, y=pat.target_price,
                text=f" 🎯 ₹{pat.target_price:,.0f}",
                showarrow=False, xref="x", yref="y", xanchor="left",
                font=dict(color=_TARGET_COLOR, size=10 if rank == 0 else 9),
            )

        # ── 5. Stop loss projection (coral red, dash-dot) ─────────────────────
        if pat.stop_loss and _is_new(pat.stop_loss):
            fig.add_shape(
                type="line", xref="x", yref="y",
                x0=x_last, x1=x_future,
                y0=pat.stop_loss, y1=pat.stop_loss,
                opacity=opacity * 0.85,
                line=dict(color=_STOP_COLOR, dash="dashdot",
                          width=1.8 if rank == 0 else 1.0),
            )
            fig.add_annotation(
                x=x_future, y=pat.stop_loss,
                text=f" 🛑 ₹{pat.stop_loss:,.0f}",
                showarrow=False, xref="x", yref="y", xanchor="left",
                font=dict(color=_STOP_COLOR, size=9),
            )

    # ── Legend: pattern entries + color key ───────────────────────────────────
    for rank, pat in enumerate(patterns[:4]):
        sig = {"Bullish": "▲", "Bearish": "▼", "Neutral": "◆"}.get(pat.signal, "◆")
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"{sig} {pat.name} ({pat.confidence:.0f}%)",
            line=dict(color=PAT_COLORS[rank % len(PAT_COLORS)], width=2),
            showlegend=True,
        ), row=1, col=1)

    # Color-key entries for the two projection types
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        name="── Projection (cyan)",
        line=dict(color=_PROJ_COLOR, width=1.5, dash="dash"),
        showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        name="🎯 Target  🛑 Stop",
        line=dict(color=_TARGET_COLOR, width=1.5, dash="dot"),
        showlegend=True,
    ), row=1, col=1)

    title = f"Pattern Analysis — {ticker}  ({timeframe})" if ticker else "Pattern Analysis"
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=680,
        margin=dict(l=20, r=170, t=55, b=20),
        legend=dict(
            orientation="v", x=1.01, y=1,
            xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0.45)",
            font=dict(size=10),
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1,
                     showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title_text="Volume",    row=2, col=1, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

    return fig

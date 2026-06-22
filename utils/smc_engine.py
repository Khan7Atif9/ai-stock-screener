# ── SMC Engine ────────────────────────────────────────────────
# Smart Money Concepts detection
# BOS, CHoCH, Order Blocks, FVG, Inducement, Liquidity Sweeps

import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from config import SWING_LENGTH, FVG_THRESHOLD


# ═══════════════════════════════════════════════════════════════
# SWING DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_swings(df: pd.DataFrame, n: int = SWING_LENGTH):
    """
    Detects swing highs and swing lows.
    A swing high = highest point with n bars on each side.
    A swing low  = lowest  point with n bars on each side.
    """
    highs, lows = [], []

    for i in range(n, len(df) - n):
        window_h = df["high"].iloc[i - n : i + n + 1]
        window_l = df["low"].iloc[i  - n : i + n + 1]

        if df["high"].iloc[i] == window_h.max():
            highs.append({
                "index": i,
                "price": float(df["high"].iloc[i]),
                "date":  df.index[i],
            })

        if df["low"].iloc[i] == window_l.min():
            lows.append({
                "index": i,
                "price": float(df["low"].iloc[i]),
                "date":  df.index[i],
            })

    return highs, lows


# ═══════════════════════════════════════════════════════════════
# BOS — BREAK OF STRUCTURE
# ═══════════════════════════════════════════════════════════════

def detect_bos(swing_highs: list, swing_lows: list) -> dict:
    """
    BOS = continuation signal.
    Bullish BOS  → new Higher High (HH)
    Bearish BOS  → new Lower Low  (LL)
    """
    bull_bos = []
    bear_bos = []

    # Check swing highs for HH (bullish BOS)
    for i in range(1, len(swing_highs)):
        curr = swing_highs[i]
        prev = swing_highs[i - 1]
        if curr["price"] > prev["price"]:
            bull_bos.append({
                "index":  curr["index"],
                "date":   curr["date"],
                "price":  curr["price"],
                "type":   "Bullish BOS — Higher High",
            })

    # Check swing lows for LL (bearish BOS)
    for i in range(1, len(swing_lows)):
        curr = swing_lows[i]
        prev = swing_lows[i - 1]
        if curr["price"] < prev["price"]:
            bear_bos.append({
                "index":  curr["index"],
                "date":   curr["date"],
                "price":  curr["price"],
                "type":   "Bearish BOS — Lower Low",
            })

    return {"bullish": bull_bos, "bearish": bear_bos}


# ═══════════════════════════════════════════════════════════════
# CHoCH — CHANGE OF CHARACTER
# ═══════════════════════════════════════════════════════════════

def detect_choch(swing_highs: list, swing_lows: list) -> dict:
    """
    CHoCH = reversal signal.
    Bullish CHoCH → swing low makes Higher Low (structure flip up)
    Bearish CHoCH → swing high makes Lower High (structure flip down)
    """
    bull_choch = []
    bear_choch = []

    # Bullish CHoCH: swing low is higher than previous swing low
    for i in range(1, len(swing_lows)):
        curr = swing_lows[i]
        prev = swing_lows[i - 1]
        if curr["price"] > prev["price"]:
            bull_choch.append({
                "index": curr["index"],
                "date":  curr["date"],
                "price": curr["price"],
                "type":  "Bullish CHoCH — Structure Flip Up",
            })

    # Bearish CHoCH: swing high is lower than previous swing high
    for i in range(1, len(swing_highs)):
        curr = swing_highs[i]
        prev = swing_highs[i - 1]
        if curr["price"] < prev["price"]:
            bear_choch.append({
                "index": curr["index"],
                "date":  curr["date"],
                "price": curr["price"],
                "type":  "Bearish CHoCH — Structure Flip Down",
            })

    return {"bullish": bull_choch, "bearish": bear_choch}


# ═══════════════════════════════════════════════════════════════
# ORDER BLOCKS
# ═══════════════════════════════════════════════════════════════

def detect_order_blocks(
    df: pd.DataFrame,
    swing_highs: list,
    swing_lows: list,
    lookback: int = 20,
) -> list:
    """
    Order Block = last opposing candle before a strong move.
    Bullish OB = last bearish candle before a strong rally.
    Bearish OB = last bullish candle before a strong drop.
    """
    obs = []

    # Bullish OBs — find last red candle before each swing high
    for sh in swing_highs[-8:]:
        idx   = sh["index"]
        start = max(0, idx - lookback)

        for j in range(idx - 1, start, -1):
            # Red candle = bearish
            if df["close"].iloc[j] < df["open"].iloc[j]:
                mitigated = _check_mitigated(
                    df, j, idx, "bullish"
                )
                obs.append({
                    "type":      "Bullish OB",
                    "index":     j,
                    "date":      df.index[j],
                    "high":      float(df["high"].iloc[j]),
                    "low":       float(df["low"].iloc[j]),
                    "mitigated": mitigated,
                    "active":    not mitigated,
                })
                break

    # Bearish OBs — find last green candle before each swing low
    for sl in swing_lows[-8:]:
        idx   = sl["index"]
        start = max(0, idx - lookback)

        for j in range(idx - 1, start, -1):
            # Green candle = bullish
            if df["close"].iloc[j] > df["open"].iloc[j]:
                mitigated = _check_mitigated(
                    df, j, idx, "bearish"
                )
                obs.append({
                    "type":      "Bearish OB",
                    "index":     j,
                    "date":      df.index[j],
                    "high":      float(df["high"].iloc[j]),
                    "low":       float(df["low"].iloc[j]),
                    "mitigated": mitigated,
                    "active":    not mitigated,
                })
                break

    return obs


def _check_mitigated(df, ob_idx, swing_idx, ob_type):
    """Check if price came back and mitigated the order block."""
    ob_high = df["high"].iloc[ob_idx]
    ob_low  = df["low"].iloc[ob_idx]
    end     = min(swing_idx + 30, len(df))

    for i in range(ob_idx + 1, end):
        if ob_type == "bullish" and df["low"].iloc[i] <= ob_high:
            return True
        if ob_type == "bearish" and df["high"].iloc[i] >= ob_low:
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# FAIR VALUE GAPS (FVG / IMBALANCE)
# ═══════════════════════════════════════════════════════════════

def detect_fvg(df: pd.DataFrame) -> list:
    """
    FVG = 3-candle imbalance where candle 1 and 3 don't overlap.
    Bullish FVG → candle 3 low > candle 1 high (gap up)
    Bearish FVG → candle 3 high < candle 1 low (gap down)
    Price usually returns to fill these gaps.
    """
    fvgs = []

    for i in range(1, len(df) - 1):
        c1_high = float(df["high"].iloc[i - 1])
        c1_low  = float(df["low"].iloc[i - 1])
        c3_high = float(df["high"].iloc[i + 1])
        c3_low  = float(df["low"].iloc[i + 1])
        mid_close = float(df["close"].iloc[i])

        # Bullish FVG
        gap_up = c3_low - c1_high
        if gap_up > 0 and gap_up / mid_close > FVG_THRESHOLD:
            filled = _check_fvg_filled(df, i, c1_high, c3_low, "bullish")
            fvgs.append({
                "type":   "Bullish FVG",
                "index":  i,
                "date":   df.index[i],
                "high":   c3_low,
                "low":    c1_high,
                "gap":    round(gap_up, 2),
                "filled": filled,
                "active": not filled,
            })

        # Bearish FVG
        gap_down = c1_low - c3_high
        if gap_down > 0 and gap_down / mid_close > FVG_THRESHOLD:
            filled = _check_fvg_filled(df, i, c3_high, c1_low, "bearish")
            fvgs.append({
                "type":   "Bearish FVG",
                "index":  i,
                "date":   df.index[i],
                "high":   c1_low,
                "low":    c3_high,
                "gap":    round(gap_down, 2),
                "filled": filled,
                "active": not filled,
            })

    return fvgs[-30:]   # Return last 30


def _check_fvg_filled(df, start, low, high, fvg_type):
    """Check if price came back to fill the gap."""
    for i in range(start + 2, min(start + 40, len(df))):
        if fvg_type == "bullish" and df["low"].iloc[i] <= low:
            return True
        if fvg_type == "bearish" and df["high"].iloc[i] >= high:
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# LIQUIDITY SWEEPS
# ═══════════════════════════════════════════════════════════════

def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swing_highs: list,
    swing_lows: list,
) -> list:
    """
    Liquidity Sweep = price wicks above/below a swing point
    then immediately reverses back.
    Smart money hunts stop losses before the real move.
    """
    sweeps = []

    # Sweep above swing highs (bearish sweep)
    for sh in swing_highs:
        idx = sh["index"]
        for j in range(idx + 1, min(idx + 10, len(df))):
            if (df["high"].iloc[j] > sh["price"] and
                    df["close"].iloc[j] < sh["price"]):
                sweeps.append({
                    "type":  "Bearish Sweep",
                    "index": j,
                    "date":  df.index[j],
                    "level": sh["price"],
                    "high":  float(df["high"].iloc[j]),
                })
                break

    # Sweep below swing lows (bullish sweep)
    for sl in swing_lows:
        idx = sl["index"]
        for j in range(idx + 1, min(idx + 10, len(df))):
            if (df["low"].iloc[j] < sl["price"] and
                    df["close"].iloc[j] > sl["price"]):
                sweeps.append({
                    "type":  "Bullish Sweep",
                    "index": j,
                    "date":  df.index[j],
                    "level": sl["price"],
                    "low":   float(df["low"].iloc[j]),
                })
                break

    return sweeps


# ═══════════════════════════════════════════════════════════════
# FULL SMC ANALYSIS — single function call
# ═══════════════════════════════════════════════════════════════

def full_smc_analysis(df: pd.DataFrame) -> dict:
    """
    Runs complete SMC analysis and returns clean summary.
    Used by the Streamlit app.
    """
    swing_highs, swing_lows = detect_swings(df)
    bos     = detect_bos(swing_highs, swing_lows)
    choch   = detect_choch(swing_highs, swing_lows)
    obs     = detect_order_blocks(df, swing_highs, swing_lows)
    fvgs    = detect_fvg(df)
    sweeps  = detect_liquidity_sweeps(df, swing_highs, swing_lows)

    # Recent signals (last 15 candles)
    last    = len(df) - 1
    recent  = 15

    recent_bull_bos   = [b for b in bos["bullish"]
                         if b["index"] >= last - recent]
    recent_bear_bos   = [b for b in bos["bearish"]
                         if b["index"] >= last - recent]
    recent_bull_choch = [b for b in choch["bullish"]
                         if b["index"] >= last - recent]
    recent_bear_choch = [b for b in choch["bearish"]
                         if b["index"] >= last - recent]

    active_bull_obs   = [o for o in obs if o["type"] == "Bullish OB"
                         and o["active"]]
    active_bear_obs   = [o for o in obs if o["type"] == "Bearish OB"
                         and o["active"]]
    active_fvgs       = [f for f in fvgs if f["active"]]
    recent_sweeps     = [s for s in sweeps
                         if s["index"] >= last - recent]

    # Current price
    current_price = float(df["close"].iloc[-1])

    # Is price inside an active bullish OB?
    in_bull_ob = any(
        ob["low"] <= current_price <= ob["high"]
        for ob in active_bull_obs
    )

    # SMC Bias Score (0–10)
    bull_score = (
        len(recent_bull_bos)   * 2 +
        len(recent_bull_choch) * 3 +
        int(in_bull_ob)        * 2 +
        len(active_bull_obs)   * 1
    )
    bear_score = (
        len(recent_bear_bos)   * 2 +
        len(recent_bear_choch) * 3 +
        len(active_bear_obs)   * 1
    )

    # Overall SMC bias
    if bull_score > bear_score + 2:
        bias = "🟢 BULLISH"
    elif bear_score > bull_score + 2:
        bias = "🔴 BEARISH"
    else:
        bias = "🟡 NEUTRAL"

    return {
        # Raw signals
        "swing_highs":       swing_highs[-5:],
        "swing_lows":        swing_lows[-5:],
        "bos":               bos,
        "choch":             choch,
        "order_blocks":      obs,
        "fvgs":              fvgs[-10:],
        "sweeps":            sweeps[-5:],

        # Summary for app display
        "summary": {
            "bias":              bias,
            "bull_score":        bull_score,
            "bear_score":        bear_score,
            "recent_bull_bos":   len(recent_bull_bos),
            "recent_bear_bos":   len(recent_bear_bos),
            "recent_bull_choch": len(recent_bull_choch),
            "recent_bear_choch": len(recent_bear_choch),
            "active_bull_obs":   len(active_bull_obs),
            "active_bear_obs":   len(active_bear_obs),
            "active_fvgs":       len(active_fvgs),
            "recent_sweeps":     len(recent_sweeps),
            "price_in_bull_ob":  in_bull_ob,
            "current_price":     current_price,
        }
    }
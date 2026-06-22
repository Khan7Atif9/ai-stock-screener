# ── Candlestick Pattern Detection ────────────────────────────
# Detects all patterns from your Module 4.3 notes
# Hammer, Engulfing, Morning Star, Doji, Shooting Star etc.

import pandas as pd
import numpy as np


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all pattern detectors on OHLCV df.
    Adds a new column for each pattern (1 = pattern found, 0 = not).
    """
    df = df.copy()
    df = _hammer(df)
    df = _inverted_hammer(df)
    df = _bullish_engulfing(df)
    df = _bearish_engulfing(df)
    df = _morning_star(df)
    df = _evening_star(df)
    df = _shooting_star(df)
    df = _doji(df)
    df = _dragonfly_doji(df)
    df = _three_white_soldiers(df)
    df = _three_black_crows(df)
    df = _bullish_abandoned_baby(df)
    df = _bearish_abandoned_baby(df)
    df = _pattern_summary(df)
    return df


# ── Helper: body and shadow sizes ────────────────────────────
def _parts(df):
    body         = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    return body, candle_range, upper_shadow, lower_shadow


# ── Hammer ────────────────────────────────────────────────────
# Small body at top, long lower shadow — buyers dominating
# Appears at bottom of downtrend = BULLISH REVERSAL
def _hammer(df):
    body, rng, upper, lower = _parts(df)
    df["hammer"] = (
        (lower >= 2 * body) &          # Lower shadow 2x body
        (upper <= 0.1 * rng) &         # Very small upper shadow
        (body > 0) &                   # Has a body
        (rng > 0)
    ).astype(int)
    return df


# ── Inverted Hammer ───────────────────────────────────────────
# Small body at bottom, long upper shadow
# Appears at bottom of downtrend = BULLISH REVERSAL
def _inverted_hammer(df):
    body, rng, upper, lower = _parts(df)
    df["inverted_hammer"] = (
        (upper >= 2 * body) &
        (lower <= 0.1 * rng) &
        (body > 0) &
        (rng > 0)
    ).astype(int)
    return df


# ── Bullish Engulfing ─────────────────────────────────────────
# Small red candle followed by large green that fully covers it
# Strong buying signal — G >= 2*R rule from your notes
def _bullish_engulfing(df):
    prev_bearish = df["close"].shift(1) < df["open"].shift(1)
    curr_bullish = df["close"] > df["open"]

    prev_body = (df["open"].shift(1) - df["close"].shift(1)).abs()
    curr_body = (df["close"] - df["open"]).abs()

    df["bullish_engulfing"] = (
        prev_bearish &
        curr_bullish &
        (df["open"]  < df["close"].shift(1)) &   # Opens below prev close
        (df["close"] > df["open"].shift(1)) &    # Closes above prev open
        (curr_body   >= 1.5 * prev_body)         # Green body >= 1.5x red
    ).astype(int)
    return df


# ── Bearish Engulfing ─────────────────────────────────────────
# Small green candle followed by large red that fully covers it
# Strong selling signal
def _bearish_engulfing(df):
    prev_bullish = df["close"].shift(1) > df["open"].shift(1)
    curr_bearish = df["close"] < df["open"]

    prev_body = (df["close"].shift(1) - df["open"].shift(1)).abs()
    curr_body = (df["open"] - df["close"]).abs()

    df["bearish_engulfing"] = (
        prev_bullish &
        curr_bearish &
        (df["open"]  > df["close"].shift(1)) &
        (df["close"] < df["open"].shift(1)) &
        (curr_body   >= 1.5 * prev_body)
    ).astype(int)
    return df


# ── Morning Star ──────────────────────────────────────────────
# 3-candle pattern: Long red → Small body → Long green
# Sellers losing control, buyers taking over = BULLISH
def _morning_star(df):
    # Candle 1: large bearish
    c1_bearish   = df["close"].shift(2) < df["open"].shift(2)
    c1_body      = (df["open"].shift(2) - df["close"].shift(2)).abs()

    # Candle 2: small body (star) — can be either color
    c2_body      = (df["close"].shift(1) - df["open"].shift(1)).abs()
    c2_range     = df["high"].shift(1) - df["low"].shift(1)
    c2_small     = c2_body <= 0.3 * c2_range

    # Candle 3: large bullish
    c3_bullish   = df["close"] > df["open"]
    c3_body      = (df["close"] - df["open"]).abs()

    df["morning_star"] = (
        c1_bearish &
        c2_small &
        c3_bullish &
        (c3_body >= 0.5 * c1_body) &   # Green recovers at least half of red
        (df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2)
    ).astype(int)
    return df


# ── Evening Star ──────────────────────────────────────────────
# 3-candle pattern: Long green → Small body → Long red
# Uptrend losing momentum = BEARISH REVERSAL
def _evening_star(df):
    c1_bullish = df["close"].shift(2) > df["open"].shift(2)
    c1_body    = (df["close"].shift(2) - df["open"].shift(2)).abs()

    c2_body    = (df["close"].shift(1) - df["open"].shift(1)).abs()
    c2_range   = df["high"].shift(1) - df["low"].shift(1)
    c2_small   = c2_body <= 0.3 * c2_range

    c3_bearish = df["close"] < df["open"]
    c3_body    = (df["open"] - df["close"]).abs()

    df["evening_star"] = (
        c1_bullish &
        c2_small &
        c3_bearish &
        (c3_body >= 0.5 * c1_body)
    ).astype(int)
    return df


# ── Shooting Star ─────────────────────────────────────────────
# Small body, long upper shadow, at top of uptrend
# Buyers tried to push up but sellers took over = BEARISH
def _shooting_star(df):
    body, rng, upper, lower = _parts(df)
    df["shooting_star"] = (
        (upper >= 2 * body) &
        (lower <= 0.1 * rng) &
        (body > 0) &
        (rng > 0)
    ).astype(int)
    return df


# ── Doji ──────────────────────────────────────────────────────
# Open ≈ Close — indecision in market
def _doji(df):
    body, rng, _, _ = _parts(df)
    df["doji"] = (
        (body <= 0.1 * rng) &
        (rng > 0)
    ).astype(int)
    return df


# ── Dragonfly Doji ────────────────────────────────────────────
# Almost no body, very long lower shadow
# Like hammer but open=high=close — BULLISH
def _dragonfly_doji(df):
    body, rng, upper, lower = _parts(df)
    df["dragonfly_doji"] = (
        (body <= 0.05 * rng) &      # Near-zero body
        (lower >= 0.6 * rng) &      # Long lower shadow
        (upper <= 0.1 * rng) &
        (rng > 0)
    ).astype(int)
    return df


# ── Three White Soldiers ──────────────────────────────────────
# 3 consecutive long bullish candles
# Strong trend reversal from bear to bull
def _three_white_soldiers(df):
    c1 = df["close"].shift(2) > df["open"].shift(2)
    c2 = df["close"].shift(1) > df["open"].shift(1)
    c3 = df["close"] > df["open"]

    # Each opens within previous body and closes higher
    c2_opens_in_c1 = df["open"].shift(1) > df["open"].shift(2)
    c3_opens_in_c2 = df["open"] > df["open"].shift(1)

    c2_higher = df["close"].shift(1) > df["close"].shift(2)
    c3_higher = df["close"] > df["close"].shift(1)

    df["three_white_soldiers"] = (
        c1 & c2 & c3 &
        c2_opens_in_c1 & c3_opens_in_c2 &
        c2_higher & c3_higher
    ).astype(int)
    return df


# ── Three Black Crows ─────────────────────────────────────────
# 3 consecutive long bearish candles
# Strong downtrend continuation
def _three_black_crows(df):
    c1 = df["close"].shift(2) < df["open"].shift(2)
    c2 = df["close"].shift(1) < df["open"].shift(1)
    c3 = df["close"] < df["open"]

    c2_lower = df["close"].shift(1) < df["close"].shift(2)
    c3_lower = df["close"] < df["close"].shift(1)

    df["three_black_crows"] = (
        c1 & c2 & c3 &
        c2_lower & c3_lower
    ).astype(int)
    return df


# ── Bullish Abandoned Baby ────────────────────────────────────
# Long red → Doji (gaps down) → Long green (gaps up)
# Very strong bullish reversal
def _bullish_abandoned_baby(df):
    body1, rng1, _, _ = _parts(df.shift(2))

    c1_bearish = df["close"].shift(2) < df["open"].shift(2)

    # Doji in middle
    body2 = (df["close"].shift(1) - df["open"].shift(1)).abs()
    rng2  = df["high"].shift(1) - df["low"].shift(1)
    c2_doji = (body2 <= 0.1 * rng2) & (rng2 > 0)

    c3_bullish = df["close"] > df["open"]

    df["bullish_abandoned_baby"] = (
        c1_bearish &
        c2_doji &
        c3_bullish
    ).astype(int)
    return df


# ── Bearish Abandoned Baby ────────────────────────────────────
# Long green → Doji (gaps up) → Long red (gaps down)
# Very strong bearish reversal
def _bearish_abandoned_baby(df):
    c1_bullish = df["close"].shift(2) > df["open"].shift(2)

    body2 = (df["close"].shift(1) - df["open"].shift(1)).abs()
    rng2  = df["high"].shift(1) - df["low"].shift(1)
    c2_doji = (body2 <= 0.1 * rng2) & (rng2 > 0)

    c3_bearish = df["close"] < df["open"]

    df["bearish_abandoned_baby"] = (
        c1_bullish &
        c2_doji &
        c3_bearish
    ).astype(int)
    return df


# ── Pattern Summary ───────────────────────────────────────────
# Single score combining all patterns for ML input
def _pattern_summary(df):
    bullish_patterns = [
        "hammer", "inverted_hammer", "bullish_engulfing",
        "morning_star", "dragonfly_doji", "three_white_soldiers",
        "bullish_abandoned_baby"
    ]
    bearish_patterns = [
        "bearish_engulfing", "evening_star", "shooting_star",
        "three_black_crows", "bearish_abandoned_baby"
    ]

    df["bullish_pattern_score"] = df[bullish_patterns].sum(axis=1)
    df["bearish_pattern_score"] = df[bearish_patterns].sum(axis=1)
    df["any_bullish_pattern"]   = (df["bullish_pattern_score"] > 0).astype(int)
    df["any_bearish_pattern"]   = (df["bearish_pattern_score"] > 0).astype(int)

    return df


# ── Get latest pattern for a stock ───────────────────────────
def get_latest_patterns(df: pd.DataFrame) -> dict:
    """
    Returns which patterns fired on the last candle.
    Used in the analysis page of the app.
    """
    df = detect_all(df)
    last = df.iloc[-1]

    all_patterns = [
        "hammer","inverted_hammer","bullish_engulfing","bearish_engulfing",
        "morning_star","evening_star","shooting_star","doji",
        "dragonfly_doji","three_white_soldiers","three_black_crows",
        "bullish_abandoned_baby","bearish_abandoned_baby"
    ]

    found = {p: int(last[p]) for p in all_patterns if p in last.index}
    found["bullish_score"] = int(last.get("bullish_pattern_score", 0))
    found["bearish_score"] = int(last.get("bearish_pattern_score", 0))

    return found
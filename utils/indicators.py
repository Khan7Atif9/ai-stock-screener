# ── Technical Indicators ─────────────────────────────────────
# 30+ indicators used for ML features + chart signals

import pandas as pd
import numpy as np


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function — runs all indicators on OHLCV df.
    Returns same df with new indicator columns added.
    """
    df = df.copy()
    df = _moving_averages(df)
    df = _rsi(df)
    df = _macd(df)
    df = _bollinger_bands(df)
    df = _atr(df)
    df = _volume_indicators(df)
    df = _momentum(df)
    df = _price_action(df)
    return df.dropna()


# ── Moving Averages ───────────────────────────────────────────
def _moving_averages(df):
    # Simple Moving Averages
    for p in [20, 50, 200]:
        df[f"sma_{p}"] = df["close"].rolling(p).mean()

    # Exponential Moving Averages
    for p in [9, 21, 50, 200]:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

    # Signals
    df["golden_cross"]       = (df["sma_50"] > df["sma_200"]).astype(int)
    df["above_200sma"]       = (df["close"]  > df["sma_200"]).astype(int)
    df["above_50sma"]        = (df["close"]  > df["sma_50"]).astype(int)

    # All EMAs aligned upward = strong uptrend
    df["ema_aligned"] = (
        (df["ema_9"]  > df["ema_21"]) &
        (df["ema_21"] > df["ema_50"]) &
        (df["ema_50"] > df["ema_200"])
    ).astype(int)

    return df


# ── RSI ───────────────────────────────────────────────────────
def _rsi(df, period=14):
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"]        = 100 - (100 / (1 + rs))
    df["rsi_ob"]     = (df["rsi"] > 70).astype(int)   # Overbought
    df["rsi_os"]     = (df["rsi"] < 30).astype(int)   # Oversold
    df["rsi_rising"] = (df["rsi"] > df["rsi"].shift(5)).astype(int)
    return df


# ── MACD ──────────────────────────────────────────────────────
def _macd(df):
    ema_fast = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"]         = ema_fast - ema_slow
    df["macd_signal"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]    = df["macd"] - df["macd_signal"]
    df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)

    # Crossover: macd just crossed above signal
    df["macd_cross_up"] = (
        (df["macd"] > df["macd_signal"]) &
        (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)

    return df


# ── Bollinger Bands ───────────────────────────────────────────
def _bollinger_bands(df, period=20, std=2):
    sma = df["close"].rolling(period).mean()
    sd  = df["close"].rolling(period).std()

    df["bb_upper"]    = sma + (std * sd)
    df["bb_middle"]   = sma
    df["bb_lower"]    = sma - (std * sd)
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # Where is price inside the bands? 0=lower band, 1=upper band
    df["bb_position"] = (
        (df["close"] - df["bb_lower"]) /
        (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    )

    # Squeeze = bands narrow = breakout coming
    df["bb_squeeze"]  = (
        df["bb_width"] < df["bb_width"].rolling(20).mean()
    ).astype(int)

    return df


# ── ATR (Average True Range) ──────────────────────────────────
def _atr(df, period=14):
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    df["atr"]     = tr.ewm(span=period, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"] * 100   # ATR as % of price
    return df


# ── Volume Indicators ─────────────────────────────────────────
def _volume_indicators(df):
    df["vol_sma20"]  = df["volume"].rolling(20).mean()
    df["vol_ratio"]  = df["volume"] / df["vol_sma20"]
    df["high_vol"]   = (df["vol_ratio"] > 2.0).astype(int)

    # OBV — On Balance Volume
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"]        = (df["volume"] * direction).cumsum()
    df["obv_rising"] = (df["obv"] > df["obv"].shift(10)).astype(int)

    return df


# ── Momentum ──────────────────────────────────────────────────
def _momentum(df):
    # Rate of Change
    for p in [5, 10, 21]:
        df[f"roc_{p}"] = df["close"].pct_change(p) * 100

    # Stochastic Oscillator
    low14  = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    denom  = (high14 - low14).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low14) / denom
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Williams %R
    df["williams_r"] = -100 * (high14 - df["close"]) / denom

    return df


# ── Price Action Features ─────────────────────────────────────
def _price_action(df):
    df["body"]         = (df["close"] - df["open"]).abs()
    df["candle_range"] = df["high"] - df["low"]
    df["body_pct"]     = df["body"] / df["candle_range"].replace(0, np.nan)

    df["upper_shadow"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_shadow"] = df[["open","close"]].min(axis=1) - df["low"]
    df["bullish"]      = (df["close"] > df["open"]).astype(int)

    # Returns
    df["ret_1d"]  = df["close"].pct_change(1)
    df["ret_5d"]  = df["close"].pct_change(5)
    df["ret_21d"] = df["close"].pct_change(21)

    # Volatility (annualised)
    df["volatility"] = df["ret_1d"].rolling(20).std() * np.sqrt(252) * 100

    # Gap up / gap down
    df["gap_up"]   = (df["open"] > df["close"].shift(1) * 1.01).astype(int)
    df["gap_down"] = (df["open"] < df["close"].shift(1) * 0.99).astype(int)

    return df


# ── Feature list used by ML models ────────────────────────────
ML_FEATURES = [
    "rsi", "rsi_ob", "rsi_os", "rsi_rising",
    "macd", "macd_hist", "macd_bullish", "macd_cross_up",
    "bb_position", "bb_width", "bb_squeeze",
    "atr_pct", "vol_ratio", "obv_rising", "high_vol",
    "roc_5", "roc_10", "roc_21",
    "stoch_k", "stoch_d", "williams_r",
    "golden_cross", "above_200sma", "above_50sma", "ema_aligned",
    "body_pct", "upper_shadow", "lower_shadow", "bullish",
    "volatility", "gap_up", "gap_down",
]
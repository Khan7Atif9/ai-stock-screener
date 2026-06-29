"""
data/fetcher.py — Robust yfinance data fetcher with retry + caching.

Fixes:
  1. Upgrade to yfinance ≥ 0.2.54 (handles NSE cookie/crumb refresh automatically).
  2. Exponential back-off retry (3 attempts, 2-4-8 s delays) so a transient
     429 / empty response doesn't crash the whole page.
  3. Streamlit disk-cache (TTL = 15 min) so repeated page loads don't
     hammer Yahoo Finance and hit rate limits.
  4. Graceful empty-DataFrame return with a warning — callers already handle it.
"""

import time
import logging
import requests
import pandas as pd
import yfinance as yf
import streamlit as st

logger = logging.getLogger(__name__)

# ── Retry helper ──────────────────────────────────────────────

def _retry(fn, retries: int = 3, base_delay: float = 2.0):
    """Call *fn* up to *retries* times with exponential back-off."""
    for attempt in range(retries):
        try:
            result = fn()
            return result
        except Exception as exc:
            wait = base_delay * (2 ** attempt)
            logger.warning(
                "Attempt %d/%d failed: %s — retrying in %.0fs",
                attempt + 1, retries, exc, wait,
            )
            if attempt < retries - 1:
                time.sleep(wait)
    return None


# ── OHLCV ────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)   # 15-minute cache
def get_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Returns an empty DataFrame on failure so callers can show a
    friendly warning rather than crashing.
    """
    def _download():
        tk = yf.Ticker(ticker)
        df = tk.history(
            period=period,
            interval="1d",
            auto_adjust=True,
            repair=True,         # yfinance ≥ 0.2.54: fixes split/dividend glitches
        )
        if df.empty:
            raise ValueError(f"{ticker}: history() returned empty DataFrame")
        return df

    df = _retry(_download, retries=3, base_delay=3.0)
    if df is None or df.empty:
        logger.error("Could not fetch OHLCV for %s after retries", ticker)
        return pd.DataFrame()

    # Normalise column names to lower-case
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df


# ── Fundamentals ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)  # 1-hour cache (fundamentals change slowly)
def get_fundamentals(ticker: str) -> dict:
    """
    Fetch key fundamental fields via yfinance .info dict.

    Returns an empty dict on failure — callers display "N/A" gracefully.
    """
    def _fetch():
        tk = yf.Ticker(ticker)
        info = tk.info
        # yfinance sometimes returns a stub dict with only 'trailingPegRatio'
        if not info or len(info) < 5:
            raise ValueError(f"{ticker}: info dict too small — likely rate-limited")
        return info

    info = _retry(_fetch, retries=3, base_delay=4.0)
    if not info:
        logger.error("Could not fetch fundamentals for %s after retries", ticker)
        return {}

    # ── Map yfinance keys → our app keys ──────────────────
    def _safe(key, scale=1):
        val = info.get(key)
        if val is None or val == "Infinity":
            return None
        try:
            return round(float(val) * scale, 2)
        except (TypeError, ValueError):
            return None

    result = {
        "current_price":      _safe("currentPrice") or _safe("regularMarketPrice"),
        "pe_ratio":           _safe("trailingPE"),
        "forward_pe":         _safe("forwardPE"),
        "peg_ratio":          _safe("trailingPegRatio"),
        "pb_ratio":           _safe("priceToBook"),
        "roe":                _safe("returnOnEquity", scale=100),
        "roa":                _safe("returnOnAssets", scale=100),
        "roce":               _safe("returnOnCapitalEmployed", scale=100),
        "debt_to_equity":     _safe("debtToEquity"),
        "current_ratio":      _safe("currentRatio"),
        "operating_margins":  _safe("operatingMargins", scale=100),
        "profit_margins":     _safe("profitMargins", scale=100),
        "revenue_growth":     _safe("revenueGrowth", scale=100),
        "earnings_growth":    _safe("earningsGrowth", scale=100),
        "interest_coverage":  _safe("interestCoverage"),
        "market_cap":         _safe("marketCap"),
        "52w_high":           _safe("fiftyTwoWeekHigh"),
        "52w_low":            _safe("fiftyTwoWeekLow"),
        "dividend_yield":     _safe("dividendYield", scale=100),
        "beta":               _safe("beta"),
        "sector":             info.get("sector", ""),
        "industry":           info.get("industry", ""),
        "company_name":       info.get("longName", ticker),
    }
    return result

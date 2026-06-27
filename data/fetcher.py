# ── Data Fetcher — Multi-source with fallbacks ────────────────
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
import requests
from datetime import datetime, timedelta

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── NSE direct API headers ────────────────────────────────────
HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;"
                       "q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}


# ═══════════════════════════════════════════════════════════════
# OHLCV — tries 3 methods before giving up
# ═══════════════════════════════════════════════════════════════

def get_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetches OHLCV data using 3 fallback methods:
    1. yfinance with custom session
    2. yfinance download (different endpoint)
    3. Stooq (free alternative data source)
    """
    cache_file = os.path.join(
        CACHE_DIR,
        f"{ticker.replace('.', '_')}_{period}.parquet"
    )

    # Return cache if less than 6 hours old
    if os.path.exists(cache_file):
        age = datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(cache_file)
        )
        if age < timedelta(hours=6):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and len(df) > 10:
                    return df
            except Exception:
                pass

    # Method 1 — yfinance Ticker with session
    df = _fetch_yfinance_ticker(ticker, period)
    if not df.empty:
        _save_cache(df, cache_file)
        return df

    # Method 2 — yfinance download
    df = _fetch_yfinance_download(ticker, period)
    if not df.empty:
        _save_cache(df, cache_file)
        return df

    # Method 3 — Stooq fallback
    df = _fetch_stooq(ticker, period)
    if not df.empty:
        _save_cache(df, cache_file)
        return df

    return pd.DataFrame()


# ── Method 1: yfinance Ticker ─────────────────────────────────
def _fetch_yfinance_ticker(ticker: str, period: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            session = requests.Session()
            session.headers.update(HEADERS)
            stock = yf.Ticker(ticker, session=session)
            df    = stock.history(
                period  = period,
                timeout = 20,
            )
            if not df.empty and len(df) > 10:
                return _clean_ohlcv(df)
        except Exception as e:
            print(f"[M1 Attempt {attempt+1}] {ticker}: {e}")
        time.sleep(2)
    return pd.DataFrame()


# ── Method 2: yfinance download ───────────────────────────────
def _fetch_yfinance_download(ticker: str, period: str) -> pd.DataFrame:
    try:
        # Convert period to start date
        days_map = {
            "1mo": 30,  "3mo": 90,  "6mo": 180,
            "1y":  365, "2y":  730, "3y": 1095,
            "5y": 1825,
        }
        days  = days_map.get(period, 365)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        df = yf.download(
            ticker,
            start    = start,
            progress = False,
            timeout  = 20,
        )
        if not df.empty and len(df) > 10:
            return _clean_ohlcv(df)
    except Exception as e:
        print(f"[M2] {ticker}: {e}")
    return pd.DataFrame()


# ── Method 3: Stooq (free data, no API key needed) ───────────
def _fetch_stooq(ticker: str, period: str) -> pd.DataFrame:
    """
    Stooq is a free data source that works on cloud servers.
    Converts NSE ticker format: TCS.NS → TCS.IN
    """
    try:
        # Convert ticker to stooq format
        stooq_ticker = _to_stooq_ticker(ticker)
        if not stooq_ticker:
            return pd.DataFrame()

        days_map = {
            "1mo": 30,  "3mo": 90,  "6mo": 180,
            "1y":  365, "2y":  730, "3y": 1095,
            "5y": 1825,
        }
        days  = days_map.get(period, 365)
        end   = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        url = (
            f"https://stooq.com/q/d/l/"
            f"?s={stooq_ticker}&d1={start}&d2={end}&i=d"
        )

        session = requests.Session()
        session.headers.update(HEADERS)
        resp = session.get(url, timeout=20)

        if resp.status_code != 200:
            return pd.DataFrame()

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))

        if df.empty or "Date" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df.columns = [c.lower() for c in df.columns]
        df = df[["open","high","low","close","volume"]].dropna()

        if len(df) > 10:
            print(f"[Stooq] ✅ {ticker} — {len(df)} rows")
            return df

    except Exception as e:
        print(f"[M3 Stooq] {ticker}: {e}")

    return pd.DataFrame()


def _to_stooq_ticker(ticker: str) -> str:
    """Convert yfinance ticker to stooq format."""
    # TCS.NS → tcs.in
    # RELIANCE.NS → reliance.in
    if ".NS" in ticker:
        symbol = ticker.replace(".NS", "").lower()
        return f"{symbol}.in"
    if ".BO" in ticker:
        symbol = ticker.replace(".BO", "").lower()
        return f"{symbol}.in"
    return ticker.lower()


# ── Clean OHLCV ───────────────────────────────────────────────
def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and clean data."""
    df = df.copy()

    # Handle MultiIndex columns from yf.download
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rename columns
    col_map = {
        "Open": "open", "High": "high",
        "Low":  "low",  "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=col_map)

    # Keep only needed columns
    needed = [c for c in ["open","high","low","close","volume"]
              if c in df.columns]
    df = df[needed].dropna()
    df.index = pd.to_datetime(df.index)

    # Remove timezone info
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def _save_cache(df: pd.DataFrame, path: str):
    try:
        df.to_parquet(path)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# FUNDAMENTALS — yfinance with fallback mock
# ═══════════════════════════════════════════════════════════════

def get_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamentals with retry.
    Returns best available data.
    """
    cache_file = os.path.join(
        CACHE_DIR,
        f"{ticker.replace('.', '_')}_fundamentals.json"
    )

    # Return cache if fresh
    if os.path.exists(cache_file):
        age = datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(cache_file)
        )
        if age < timedelta(hours=6):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    if data.get("current_price", 0) > 0:
                        return data
            except Exception:
                pass

    for attempt in range(3):
        try:
            session = requests.Session()
            session.headers.update(HEADERS)
            stock = yf.Ticker(ticker, session=session)
            info  = stock.info

            if not info or len(info) < 5:
                time.sleep(3)
                continue

            price = (info.get("currentPrice") or
                     info.get("regularMarketPrice") or
                     info.get("previousClose") or 0)

            if price == 0:
                time.sleep(3)
                continue

            data = {
                "ticker":            ticker,
                "name":              info.get("longName", ticker),
                "sector":            info.get("sector", "Unknown"),
                "industry":          info.get("industry", "Unknown"),
                "market_cap":        info.get("marketCap", 0),
                "current_price":     price,
                "pe_ratio":          info.get("trailingPE"),
                "forward_pe":        info.get("forwardPE"),
                "pb_ratio":          info.get("priceToBook"),
                "roe":               _pct(info.get("returnOnEquity")),
                "roa":               _pct(info.get("returnOnAssets")),
                "debt_to_equity":    info.get("debtToEquity"),
                "current_ratio":     info.get("currentRatio"),
                "revenue_growth":    _pct(info.get("revenueGrowth")),
                "earnings_growth":   _pct(info.get("earningsGrowth")),
                "operating_margins": _pct(info.get("operatingMargins")),
                "profit_margins":    _pct(info.get("profitMargins")),
                "gross_margins":     _pct(info.get("grossMargins")),
                "eps":               info.get("trailingEps"),
                "book_value":        info.get("bookValue"),
                "dividend_yield":    _pct(info.get("dividendYield")),
                "beta":              info.get("beta"),
                "week_52_high":      info.get("fiftyTwoWeekHigh"),
                "week_52_low":       info.get("fiftyTwoWeekLow"),
                "avg_volume":        info.get("averageVolume"),
                "free_cashflow":     info.get("freeCashflow"),
                "ebitda":            info.get("ebitda"),
                "revenue":           info.get("totalRevenue"),
            }

            try:
                with open(cache_file, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass

            return data

        except Exception as e:
            print(f"[Attempt {attempt+1}] Fundamentals {ticker}: {e}")
            time.sleep(3 * (attempt + 1))

    # Last resort — return price from OHLCV
    return _fundamentals_from_ohlcv(ticker)


def _fundamentals_from_ohlcv(ticker: str) -> dict:
    """Build minimal fundamentals dict from OHLCV if API fails."""
    df = get_ohlcv(ticker, "1mo")
    price = float(df["close"].iloc[-1]) if not df.empty else 0

    return {
        "ticker":        ticker,
        "name":          ticker.replace(".NS","").replace(".BO",""),
        "sector":        "Unknown",
        "industry":      "Unknown",
        "market_cap":    0,
        "current_price": price,
        "pe_ratio":      None,
        "pb_ratio":      None,
        "roe":           None,
        "roa":           None,
        "debt_to_equity":None,
        "current_ratio": None,
        "revenue_growth":None,
        "earnings_growth":None,
        "operating_margins":None,
        "profit_margins":None,
        "eps":           None,
        "book_value":    None,
        "ebitda":        None,
        "revenue":       None,
        "error":         "Fundamental data unavailable — showing price only",
    }


def _pct(val):
    if val is None:
        return None
    return round(val * 100, 2) if abs(val) < 10 else round(val, 2)


def get_multiple(tickers: list, period: str = "1y") -> dict:
    result = {}
    for i, t in enumerate(tickers):
        print(f"Fetching {i+1}/{len(tickers)}: {t}", end="\r")
        df = get_ohlcv(t, period)
        if not df.empty:
            result[t] = df
        time.sleep(0.5)
    print(f"\nDone — {len(result)}/{len(tickers)} fetched")
    return result
# ── Data Fetcher — GitHub JSON + Stooq OHLCV ─────────────────
# Fundamentals: stored in repo as JSON (no API calls on cloud)
# OHLCV: Stooq direct CSV (works on cloud)

import pandas as pd
import numpy as np
import os, json, time, requests
from datetime import datetime, timedelta
from io import StringIO

CACHE_DIR    = "data/cache"
FUND_FILE    = "data/fundamentals.json"
os.makedirs(CACHE_DIR, exist_ok=True)

# GitHub raw URL for fundamentals backup
GITHUB_FUND_URL = (
    "https://raw.githubusercontent.com/"
    "Khan7Atif9/ai-stock-screener/main/data/fundamentals.json"
)

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ═══════════════════════════════════════════════════════════════
# FUNDAMENTALS — from local JSON or GitHub raw
# ═══════════════════════════════════════════════════════════════

# Cache in memory so we don't re-read file every call
_fund_cache = {}


def _load_fundamentals_db() -> dict:
    """Load fundamentals from local file or GitHub raw."""
    global _fund_cache
    if _fund_cache:
        return _fund_cache

    # Try local file first
    if os.path.exists(FUND_FILE):
        try:
            with open(FUND_FILE) as f:
                _fund_cache = json.load(f)
            print(f"✅ Loaded {len(_fund_cache)} tickers from local fundamentals.json")
            return _fund_cache
        except Exception as e:
            print(f"[Local JSON] {e}")

    # Fallback — fetch from GitHub raw
    try:
        print("Fetching fundamentals from GitHub...")
        session = requests.Session()
        session.headers.update(HEADERS)
        resp = session.get(GITHUB_FUND_URL, timeout=20)
        if resp.status_code == 200:
            _fund_cache = resp.json()
            # Save locally for next time
            os.makedirs("data", exist_ok=True)
            with open(FUND_FILE, "w") as f:
                json.dump(_fund_cache, f)
            print(f"✅ Loaded {len(_fund_cache)} tickers from GitHub")
            return _fund_cache
    except Exception as e:
        print(f"[GitHub JSON] {e}")

    return {}


def get_fundamentals(ticker: str) -> dict:
    """Get fundamentals from pre-built JSON database."""
    db   = _load_fundamentals_db()
    data = db.get(ticker)

    if data and data.get("current_price", 0) > 0:
        return data

    # Not in DB — build minimal from OHLCV
    return _fund_from_ohlcv(ticker)


# ═══════════════════════════════════════════════════════════════
# OHLCV — Stooq as primary source
# ═══════════════════════════════════════════════════════════════

def get_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    cache_file = os.path.join(
        CACHE_DIR,
        f"{ticker.replace('.','_')}_{period}.parquet"
    )

    # Return cache if under 6 hours old
    if os.path.exists(cache_file):
        age = datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(cache_file)
        )
        if age < timedelta(hours=6):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and len(df) > 20:
                    return df
            except Exception:
                pass

    start, end = _period_to_dates(period)

    # Try sources in order
    for name, fn in [
        ("Stooq",   _from_stooq),
        ("yfinance",_from_yfinance),
    ]:
        try:
            print(f"[{name}] Fetching {ticker}...")
            df = fn(ticker, start, end)
            if not df.empty and len(df) > 20:
                df = _clean(df)
                try:
                    df.to_parquet(cache_file)
                except Exception:
                    pass
                print(f"✅ {ticker} — {len(df)} rows via {name}")
                return df
        except Exception as e:
            print(f"[{name}] {e}")
        time.sleep(1)

    return pd.DataFrame()


def _period_to_dates(period: str):
    days = {
        "1mo": 30,  "3mo": 90,  "6mo": 180,
        "1y":  365, "2y":  730, "3y": 1095, "5y": 1825
    }.get(period, 365)
    end   = datetime.now()
    start = end - timedelta(days=days)
    return start, end


def _from_stooq(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Stooq direct CSV download."""
    stooq  = ticker.replace(".NS",".in").replace(".BO",".in").lower()
    s      = start.strftime("%Y%m%d")
    e      = end.strftime("%Y%m%d")
    url    = f"https://stooq.com/q/d/l/?s={stooq}&d1={s}&d2={e}&i=d"

    session = requests.Session()
    session.headers.update(HEADERS)
    resp    = session.get(url, timeout=20)

    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}")

    text = resp.text
    if "No data" in text or len(text) < 50:
        raise Exception("No data returned")

    df = pd.read_csv(StringIO(text))

    if "Date" not in df.columns or len(df) < 5:
        raise Exception("Invalid CSV")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def _from_yfinance(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """yfinance fallback."""
    import yfinance as yf
    df = yf.download(
        ticker,
        start    = start.strftime("%Y-%m-%d"),
        end      = end.strftime("%Y-%m-%d"),
        progress = False,
        timeout  = 20,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    keep = [c for c in ["open","high","low","close","volume"]
            if c in df.columns]
    df   = df[keep].dropna()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index,"tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def _fund_from_ohlcv(ticker: str) -> dict:
    """Minimal fundamentals from price only."""
    df    = get_ohlcv(ticker, "1mo")
    price = float(df["close"].iloc[-1]) if not df.empty else 0
    return {
        "ticker":            ticker,
        "name":              ticker.replace(".NS","").replace(".BO",""),
        "sector":            "Unknown",
        "industry":          "Unknown",
        "market_cap":        0,
        "current_price":     price,
        "pe_ratio":          None, "forward_pe":        None,
        "pb_ratio":          None, "roe":               None,
        "roa":               None, "debt_to_equity":    None,
        "current_ratio":     None, "revenue_growth":    None,
        "earnings_growth":   None, "operating_margins": None,
        "profit_margins":    None, "gross_margins":     None,
        "eps":               None, "book_value":        None,
        "dividend_yield":    None, "beta":              None,
        "week_52_high":      None, "week_52_low":       None,
        "free_cashflow":     None, "ebitda":            None,
        "revenue":           None,
        "note": "Limited data — add to fundamentals.json",
    }


def get_multiple(tickers: list, period: str = "1y") -> dict:
    result = {}
    for i, t in enumerate(tickers):
        print(f"Fetching {i+1}/{len(tickers)}: {t}", end="\r")
        df = get_ohlcv(t, period)
        if not df.empty:
            result[t] = df
        time.sleep(0.5)
    return result
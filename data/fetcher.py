# ── Data Fetcher — Cloud Compatible ──────────────────────────
import yfinance as yf
import pandas as pd
import os
import json
import time
import requests
from datetime import datetime, timedelta

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Fix yfinance headers for cloud ────────────────────────────
yf.utils.get_json = yf.utils.get_json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV with retry logic + cloud-safe headers.
    """
    cache_file = os.path.join(
        CACHE_DIR,
        f"{ticker.replace('.', '_')}_{period}.parquet"
    )

    # Return cache if fresh
    if os.path.exists(cache_file):
        age = datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(cache_file)
        )
        if age < timedelta(hours=6):
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

    # Try fetching with retries
    for attempt in range(3):
        try:
            session = requests.Session()
            session.headers.update(HEADERS)

            stock = yf.Ticker(ticker, session=session)
            df    = stock.history(
                period   = period,
                interval = "1d",
                timeout  = 30,
            )

            if df.empty:
                time.sleep(2)
                continue

            df = df[["Open","High","Low","Close","Volume"]].copy()
            df.columns = ["open","high","low","close","volume"]
            df.index   = pd.to_datetime(df.index)
            df         = df.dropna()

            if len(df) < 10:
                time.sleep(2)
                continue

            # Save cache
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass

            return df

        except Exception as e:
            print(f"[Attempt {attempt+1}] {ticker}: {e}")
            time.sleep(3 * (attempt + 1))

    print(f"[FAIL] Could not fetch {ticker}")
    return pd.DataFrame()


def get_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamentals with retry + cloud-safe headers.
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
                    return json.load(f)
            except Exception:
                pass

    for attempt in range(3):
        try:
            session = requests.Session()
            session.headers.update(HEADERS)

            stock = yf.Ticker(ticker, session=session)
            info  = stock.info

            # Check we got real data
            if not info or len(info) < 5:
                time.sleep(2)
                continue

            data = {
                "ticker":            ticker,
                "name":              info.get("longName", ticker),
                "sector":            info.get("sector", "Unknown"),
                "industry":          info.get("industry", "Unknown"),
                "market_cap":        info.get("marketCap", 0),
                "current_price":     info.get(
                                        "currentPrice",
                                        info.get("regularMarketPrice", 0)
                                     ),
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

    return {"ticker": ticker, "name": ticker, "error": "Fetch failed"}


def get_multiple(tickers: list, period: str = "1y") -> dict:
    """Fetch OHLCV for multiple tickers with delay between calls."""
    result = {}
    for i, t in enumerate(tickers):
        print(f"Fetching {i+1}/{len(tickers)}: {t}", end="\r")
        df = get_ohlcv(t, period)
        if not df.empty:
            result[t] = df
        time.sleep(0.5)   # Avoid rate limiting
    print(f"\nDone — {len(result)}/{len(tickers)} fetched")
    return result


def _pct(val):
    if val is None:
        return None
    return round(val * 100, 2) if abs(val) < 10 else round(val, 2)
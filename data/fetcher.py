# ── Data Fetcher ─────────────────────────────────────────────
# Downloads stock data from yfinance with local caching

import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timedelta

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV price data.
    Caches locally so we don't re-download every run.
    """
    cache_file = os.path.join(
        CACHE_DIR, f"{ticker.replace('.','_')}_{period}.parquet"
    )

    # Return cache if fresh (less than 1 day old)
    if os.path.exists(cache_file):
        age = datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(cache_file)
        )
        if age < timedelta(days=1):
            return pd.read_parquet(cache_file)

    # Download fresh data
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            print(f"[WARN] No data for {ticker}")
            return pd.DataFrame()

        # Clean column names
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df = df.dropna()

        # Save to cache
        df.to_parquet(cache_file)
        return df

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return pd.DataFrame()


def get_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental data: P/E, ROE, D/E, margins etc.
    """
    cache_file = os.path.join(
        CACHE_DIR, f"{ticker.replace('.','_')}_fundamentals.json"
    )

    # Return cache if fresh
    if os.path.exists(cache_file):
        age = datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(cache_file)
        )
        if age < timedelta(days=1):
            with open(cache_file) as f:
                return json.load(f)

    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        data = {
            "ticker":           ticker,
            "name":             info.get("longName", ticker),
            "sector":           info.get("sector", "Unknown"),
            "industry":         info.get("industry", "Unknown"),
            "market_cap":       info.get("marketCap", 0),
            "current_price":    info.get("currentPrice",
                                info.get("regularMarketPrice", 0)),
            "pe_ratio":         info.get("trailingPE", None),
            "forward_pe":       info.get("forwardPE", None),
            "pb_ratio":         info.get("priceToBook", None),
            "roe":              _pct(info.get("returnOnEquity")),
            "roa":              _pct(info.get("returnOnAssets")),
            "debt_to_equity":   info.get("debtToEquity", None),
            "current_ratio":    info.get("currentRatio", None),
            "revenue_growth":   _pct(info.get("revenueGrowth")),
            "earnings_growth":  _pct(info.get("earningsGrowth")),
            "operating_margins":_pct(info.get("operatingMargins")),
            "profit_margins":   _pct(info.get("profitMargins")),
            "gross_margins":    _pct(info.get("grossMargins")),
            "eps":              info.get("trailingEps", None),
            "book_value":       info.get("bookValue", None),
            "dividend_yield":   _pct(info.get("dividendYield")),
            "beta":             info.get("beta", None),
            "week_52_high":     info.get("fiftyTwoWeekHigh", None),
            "week_52_low":      info.get("fiftyTwoWeekLow", None),
            "avg_volume":       info.get("averageVolume", None),
            "free_cashflow":    info.get("freeCashflow", None),
            "ebitda":           info.get("ebitda", None),
            "revenue":          info.get("totalRevenue", None),
        }

        with open(cache_file, "w") as f:
            json.dump(data, f)

        return data

    except Exception as e:
        print(f"[ERROR] Fundamentals {ticker}: {e}")
        return {"ticker": ticker, "name": ticker, "error": str(e)}


def get_multiple(tickers: list, period: str = "1y") -> dict:
    """Fetch OHLCV for a list of tickers. Returns {ticker: df}"""
    result = {}
    for i, t in enumerate(tickers):
        print(f"Fetching {i+1}/{len(tickers)}: {t}", end="\r")
        df = get_ohlcv(t, period)
        if not df.empty:
            result[t] = df
    print(f"\nDone — {len(result)}/{len(tickers)} fetched")
    return result


def _pct(val):
    """Convert 0.25 → 25.0 if the value looks like a decimal ratio."""
    if val is None:
        return None
    return round(val * 100, 2) if abs(val) < 10 else round(val, 2)
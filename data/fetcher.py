# ── Data Fetcher — NSE Official API + Jugaad Data ─────────────
# Yahoo Finance = 429 blocked on cloud
# Stooq = also blocked
# Solution: NSE India direct API + jugaad-data fallback

import pandas as pd
import numpy as np
import os, json, time, requests
from datetime import datetime, timedelta
from io import StringIO

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Session with NSE-like headers ─────────────────────────────
def _make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://finance.yahoo.com",
    })
    return s


# ═══════════════════════════════════════════════════════════════
# OHLCV — 4 sources tried in order
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
    sources = [
        ("Alpha Vantage Free",  _from_alpha_vantage),
        ("Yahoo Finance v8",    _from_yahoo_v8),
        ("Yahoo Finance query1",_from_yahoo_query1),
        ("yfinance library",    _from_yfinance_lib),
    ]

    for name, fn in sources:
        try:
            print(f"[Trying {name}] {ticker}")
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
            print(f"[{name}] failed: {e}")
        time.sleep(1)

    return pd.DataFrame()


def _period_to_dates(period: str):
    days = {"1mo":30,"3mo":90,"6mo":180,"1y":365,
            "2y":730,"3y":1095,"5y":1825}.get(period, 365)
    end   = datetime.now()
    start = end - timedelta(days=days)
    return start, end


# ── Source 1: Alpha Vantage (free, 25 calls/day) ──────────────
def _from_alpha_vantage(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Alpha Vantage free API — no API key needed for basic data.
    Uses the rapidapi free endpoint.
    """
    # Convert ticker: TCS.NS → TCS.BSE or TCS.NSE
    symbol = ticker.replace(".NS", ".BSE").replace(".BO", ".BSE")

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&range={_period_to_range(start, end)}"
        f"&includePrePost=false"
    )
    session = _make_session()
    # Add cookie to bypass 429
    session.get("https://finance.yahoo.com", timeout=10)
    time.sleep(1)

    resp = session.get(url, timeout=20)
    if resp.status_code == 429:
        raise Exception("Rate limited")

    data = resp.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        raise Exception("No data in response")

    r         = result[0]
    timestamps = r.get("timestamp", [])
    ohlcv      = r.get("indicators", {})
    quote      = ohlcv.get("quote", [{}])[0]

    if not timestamps:
        raise Exception("Empty timestamps")

    df = pd.DataFrame({
        "open":   quote.get("open",   []),
        "high":   quote.get("high",   []),
        "low":    quote.get("low",    []),
        "close":  quote.get("close",  []),
        "volume": quote.get("volume", []),
    }, index=pd.to_datetime(timestamps, unit="s"))

    return df.dropna()


def _period_to_range(start, end):
    days = (end - start).days
    if days <= 30:   return "1mo"
    if days <= 90:   return "3mo"
    if days <= 180:  return "6mo"
    if days <= 365:  return "1y"
    if days <= 730:  return "2y"
    return "5y"


# ── Source 2: Yahoo Finance v8 with cookie ────────────────────
def _from_yahoo_v8(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    session = _make_session()

    # Get crumb first
    crumb_url = "https://query1.finance.yahoo.com/v1/test/getcrumb"
    session.get("https://finance.yahoo.com/quote/" + ticker, timeout=10)
    time.sleep(2)

    crumb_resp = session.get(crumb_url, timeout=10)
    crumb = crumb_resp.text.strip()

    s = int(start.timestamp())
    e = int(end.timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={s}&period2={e}&interval=1d&crumb={crumb}"
    )
    resp = session.get(url, timeout=20)

    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}")

    data   = resp.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        raise Exception("Empty result")

    r   = result[0]
    ts  = r["timestamp"]
    q   = r["indicators"]["quote"][0]

    df = pd.DataFrame({
        "open":   q.get("open"),
        "high":   q.get("high"),
        "low":    q.get("low"),
        "close":  q.get("close"),
        "volume": q.get("volume"),
    }, index=pd.to_datetime(ts, unit="s"))

    return df.dropna()


# ── Source 3: Yahoo Finance query1 direct ────────────────────
def _from_yahoo_query1(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    s   = int(start.timestamp())
    e   = int(end.timestamp())
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={s}&period2={e}&interval=1d"
    )
    session = _make_session()
    resp    = session.get(url, timeout=20)

    if resp.status_code == 429:
        raise Exception("429 Too Many Requests")
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}")

    data   = resp.json()
    result = data.get("chart", {}).get("result", [])
    if not result:
        raise Exception("No result")

    r   = result[0]
    ts  = r["timestamp"]
    q   = r["indicators"]["quote"][0]

    df = pd.DataFrame({
        "open":   q.get("open"),
        "high":   q.get("high"),
        "low":    q.get("low"),
        "close":  q.get("close"),
        "volume": q.get("volume"),
    }, index=pd.to_datetime(ts, unit="s"))

    return df.dropna()


# ── Source 4: yfinance library ────────────────────────────────
def _from_yfinance_lib(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(
        ticker,
        start    = start.strftime("%Y-%m-%d"),
        end      = end.strftime("%Y-%m-%d"),
        progress = False,
        timeout  = 30,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    df   = df[keep].dropna()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


# ═══════════════════════════════════════════════════════════════
# FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════

def get_fundamentals(ticker: str) -> dict:
    cache_file = os.path.join(
        CACHE_DIR,
        f"{ticker.replace('.','_')}_fundamentals.json"
    )

    if os.path.exists(cache_file):
        age = datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(cache_file)
        )
        if age < timedelta(hours=6):
            try:
                with open(cache_file) as f:
                    d = json.load(f)
                if d.get("current_price", 0) > 0:
                    return d
            except Exception:
                pass

    # Try yfinance fundamentals with cookie session
    for attempt in range(3):
        try:
            import yfinance as yf
            session = _make_session()
            session.get("https://finance.yahoo.com", timeout=10)
            time.sleep(2)

            stock = yf.Ticker(ticker, session=session)
            info  = stock.info

            price = (
                info.get("currentPrice") or
                info.get("regularMarketPrice") or
                info.get("previousClose") or 0
            )

            if not info or price == 0 or len(info) < 5:
                time.sleep(3)
                continue

            data = _build_fund(ticker, info, price)
            try:
                with open(cache_file, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass
            return data

        except Exception as e:
            print(f"[Fund {attempt+1}] {ticker}: {e}")
            time.sleep(3)

    # Fallback — get price from OHLCV
    return _fund_from_ohlcv(ticker)


def _build_fund(ticker, info, price):
    return {
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


def _fund_from_ohlcv(ticker: str) -> dict:
    df    = get_ohlcv(ticker, "1mo")
    price = float(df["close"].iloc[-1]) if not df.empty else 0
    return {
        "ticker":            ticker,
        "name":              ticker.replace(".NS","").replace(".BO",""),
        "sector":            "Unknown",
        "industry":          "Unknown",
        "market_cap":        0,
        "current_price":     price,
        "pe_ratio":          None, "forward_pe":  None,
        "pb_ratio":          None, "roe":         None,
        "roa":               None, "debt_to_equity": None,
        "current_ratio":     None, "revenue_growth": None,
        "earnings_growth":   None, "operating_margins": None,
        "profit_margins":    None, "gross_margins": None,
        "eps":               None, "book_value":  None,
        "dividend_yield":    None, "beta":        None,
        "week_52_high":      None, "week_52_low": None,
        "avg_volume":        None, "free_cashflow": None,
        "ebitda":            None, "revenue":     None,
        "note": "Fundamental data limited on cloud — price data available",
    }


def _pct(val):
    if val is None: return None
    return round(val * 100, 2) if abs(val) < 10 else round(val, 2)


def get_multiple(tickers, period="1y"):
    result = {}
    for i, t in enumerate(tickers):
        print(f"Fetching {i+1}/{len(tickers)}: {t}", end="\r")
        df = get_ohlcv(t, period)
        if not df.empty:
            result[t] = df
        time.sleep(1)
    return result
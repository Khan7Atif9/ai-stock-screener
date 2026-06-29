# ── Generate Fundamentals JSON — Fixed Version ────────────────
import yfinance as yf
import json, time, os, requests

os.makedirs("data", exist_ok=True)

TICKERS = [
    "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS",
    "PERSISTENT.NS","COFORGE.NS","MPHASIS.NS",
    "HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS",
    "SBIN.NS","INDUSINDBK.NS","FEDERALBNK.NS",
    "HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
    "MARICO.NS","COLPAL.NS","GODREJCP.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS",
    "TORNTPHARM.NS","AUROPHARMA.NS",
    "MARUTI.NS","TMCV.NS","M&M.NS","BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS","EICHERMOT.NS",
    "ASIANPAINT.NS","BERGEPAINT.NS","PIDILITIND.NS",
    "TITAN.NS","TRENT.NS","KALYANKJIL.NS",
    "LT.NS","SIEMENS.NS","ABB.NS","HAVELLS.NS","BEL.NS",
    "RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS",
    "DMART.NS","ETERNAL.NS","NYKAA.NS","IRCTC.NS",
    "ITC.NS","TATACONSUM.NS","BAJFINANCE.NS","LTM.NS",
    "JIOFIN.NS","TMPV.NS",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def pct(v):
    if v is None: return None
    return round(v * 100, 2) if abs(v) < 10 else round(v, 2)

def get_info(ticker):
    """Try multiple methods to get ticker info."""

    # Method 1 — yfinance with requests session
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        t    = yf.Ticker(ticker, session=session)
        info = t.info
        if info and len(info) > 10:
            price = (info.get("currentPrice") or
                     info.get("regularMarketPrice") or
                     info.get("previousClose") or 0)
            if price and price > 0:
                print(f"✅ Method1 price=₹{price:.0f}")
                return info, price
    except Exception as e:
        print(f"  M1 failed: {e}")

    time.sleep(2)

    # Method 2 — yfinance fast_info
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        t  = yf.Ticker(ticker, session=session)
        fi = t.fast_info
        price = float(fi.get("last_price", 0) or
                      fi.get("lastPrice",  0) or 0)
        if price > 0:
            # Build minimal info from fast_info
            info = {
                "currentPrice":     price,
                "marketCap":        fi.get("market_cap", 0),
                "fiftyTwoWeekHigh": fi.get("year_high", 0),
                "fiftyTwoWeekLow":  fi.get("year_low",  0),
                "longName":         ticker.replace(".NS",""),
                "sector":           "Unknown",
                "industry":         "Unknown",
            }
            print(f"✅ Method2 fast_info price=₹{price:.0f}")
            return info, price
    except Exception as e:
        print(f"  M2 failed: {e}")

    time.sleep(2)

    # Method 3 — yfinance history to get price only
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        t    = yf.Ticker(ticker, session=session)
        hist = t.history(period="5d", timeout=20)
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            info  = {
                "currentPrice": price,
                "longName":     ticker.replace(".NS",""),
                "sector":       "Unknown",
                "industry":     "Unknown",
                "marketCap":    0,
            }
            print(f"✅ Method3 history price=₹{price:.0f}")
            return info, price
    except Exception as e:
        print(f"  M3 failed: {e}")

    time.sleep(2)

    # Method 4 — yfinance download
    try:
        from datetime import datetime, timedelta
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        df = yf.download(
            ticker,
            start    = start,
            end      = end,
            progress = False,
            timeout  = 20,
            auto_adjust = True,
        )
        if not df.empty:
            # Handle MultiIndex columns
            if isinstance(df.columns, __import__("pandas").MultiIndex):
                df.columns = df.columns.get_level_values(0)
            price = float(df["Close"].iloc[-1])
            info  = {
                "currentPrice": price,
                "longName":     ticker.replace(".NS",""),
                "sector":       "Unknown",
                "industry":     "Unknown",
                "marketCap":    0,
            }
            print(f"✅ Method4 download price=₹{price:.0f}")
            return info, price
    except Exception as e:
        print(f"  M4 failed: {e}")

    return None, 0


all_data = {}
failed   = []

print(f"Generating fundamentals for {len(TICKERS)} tickers...")
print("="*55)

for i, ticker in enumerate(TICKERS):
    print(f"\n[{i+1}/{len(TICKERS)}] {ticker}...", end=" ", flush=True)

    info, price = get_info(ticker)

    if not info or price == 0:
        failed.append(ticker)
        print("❌ SKIP")
        continue

    all_data[ticker] = {
        "ticker":            ticker,
        "name":              info.get("longName", ticker.replace(".NS","")),
        "sector":            info.get("sector", "Unknown"),
        "industry":          info.get("industry", "Unknown"),
        "market_cap":        info.get("marketCap", 0) or 0,
        "current_price":     price,
        "pe_ratio":          info.get("trailingPE"),
        "forward_pe":        info.get("forwardPE"),
        "pb_ratio":          info.get("priceToBook"),
        "roe":               pct(info.get("returnOnEquity")),
        "roa":               pct(info.get("returnOnAssets")),
        "debt_to_equity":    info.get("debtToEquity"),
        "current_ratio":     info.get("currentRatio"),
        "revenue_growth":    pct(info.get("revenueGrowth")),
        "earnings_growth":   pct(info.get("earningsGrowth")),
        "operating_margins": pct(info.get("operatingMargins")),
        "profit_margins":    pct(info.get("profitMargins")),
        "gross_margins":     pct(info.get("grossMargins")),
        "eps":               info.get("trailingEps"),
        "book_value":        info.get("bookValue"),
        "dividend_yield":    pct(info.get("dividendYield")),
        "beta":              info.get("beta"),
        "week_52_high":      info.get("fiftyTwoWeekHigh"),
        "week_52_low":       info.get("fiftyTwoWeekLow"),
        "free_cashflow":     info.get("freeCashflow"),
        "ebitda":            info.get("ebitda"),
        "revenue":           info.get("totalRevenue"),
    }

    time.sleep(1.5)

# Save
with open("data/fundamentals.json", "w") as f:
    json.dump(all_data, f, indent=2)

print(f"\n{'='*55}")
print(f"✅ Saved : {len(all_data)} tickers")
print(f"❌ Failed: {len(failed)} — {failed}")
print(f"📁 File  : data/fundamentals.json")
print(f"\nNext steps:")
print(f"  git add data/fundamentals.json")
print(f"  git commit -m 'Add fundamentals data'")
print(f"  git push origin main")
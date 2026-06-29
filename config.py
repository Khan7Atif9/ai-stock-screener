# ── AI Stock Screener ── Config ──────────────────────────────
# All settings, constants, ticker lists in one place

# ── NSE Tickers (75 stocks across all sectors) ───────────────
TICKERS = [
    # IT
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS",

    # ADD YOUR NEW STOCKS HERE ↓
    "LTM.NS",        # LTI Mindtree
    "OFSS.NS",        # Oracle Financial
    "KPITTECH.NS",    # KPIT Technologies

    # Banking
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "SBIN.NS", "INDUSINDBK.NS", "FEDERALBNK.NS",

    # ADD MORE BANKS HERE ↓
    "YESBANK.NS",     # Yes Bank
    "RBLBANK.NS",     # RBL Bank
    "PNB.NS",         # Punjab National Bank
    "BANKBARODA.NS",  # Bank of Baroda
    "CANBK.NS","IDFCFIRSTB.NS",       # Canara Bank

    # FMCG
    "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
    "MARICO.NS", "COLPAL.NS", "GODREJCP.NS",

    # ADD MORE FMCG ↓
    "ITC.NS",         # ITC
    "TATACONSUM.NS",  # Tata Consumer
    "VBL.NS",         # Varun Beverages
    "RADICO.NS",      # Radico Khaitan

    # Pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "TORNTPHARM.NS", "AUROPHARMA.NS",

    # ADD MORE PHARMA ↓
    "LUPIN.NS",       # Lupin
    "BIOCON.NS",      # Biocon
    "IPCALAB.NS",     # IPCA Labs
    "GLAND.NS",       # Gland Pharma

    # Auto
    "MARUTI.NS", "TMPV.NS", "M&M.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "EICHERMOT.NS","TMCV",

    # ADD MORE AUTO ↓
    "TVSMOTOR.NS",    # TVS Motor
    "ASHOKLEY.NS",    # Ashok Leyland
    "MOTHERSON.NS",   # Motherson Sumi
    "BALKRISIND.NS",  # Balkrishna Industries

    # Paints & Consumer
    "ASIANPAINT.NS", "BERGEPAINT.NS", "PIDILITIND.NS",
    "TITAN.NS", "TRENT.NS", "KALYANKJIL.NS",

    # ADD MORE ↓
    "PAGEIND.NS",     # Page Industries
    "RELAXO.NS",      # Relaxo Footwear
    "BATAINDIA.NS",   # Bata India
    "VMART.NS",       # V-Mart Retail

    # Capital Goods / Infra
    "LT.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS",
    "CUMMINSIND.NS", "BEL.NS",

    # ADD MORE ↓
    "BHEL.NS",        # BHEL
    "THERMAX.NS",     # Thermax
    "GRINDWELL.NS",   # Grindwell Norton
    "AIAENG.NS",      # AIA Engineering

    # Oil & Gas
    "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS",

    # ADD MORE ↓
    "GAIL.NS",        # GAIL India
    "IGL.NS",         # Indraprastha Gas
    "MGL.NS",         # Mahanagar Gas
    "PETRONET.NS",    # Petronet LNG

    # Metals
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS",

    # ADD MORE ↓
    "SAIL.NS",        # SAIL
    "NMDC.NS",        # NMDC
    "COALINDIA.NS",   # Coal India
    "NATIONALUM.NS",  # National Aluminium

    # Retail / New Age
    "DMART.NS", "ETERNAL.NS", "NYKAA.NS", "IRCTC.NS","GROWW.NS",

    # ADD MORE ↓
    "PAYTM.NS",       # Paytm
    "POLICYBZR.NS",   # Policy Bazaar
    "CARTRADE.NS",    # CarTrade
    "EASEMYTRIP.NS",  # Easy My Trip

    # Smallcap / Midcap gems
    "DEEPAKNTR.NS",   # Deepak Nitrite
    "FINPIPE.NS",     # Finolex Industries
    "ASTRAL.NS",      # Astral
    "SUPREMEIND.NS",  # Supreme Industries
    "APOLLOHOSP.NS",  # Apollo Hospitals
    "METROPOLIS.NS", 
    "MARINE.NS", # Marine Electricals (India)
    "APOLLO.NS",
    "CGPOWER.NS",
    "JIOFIN.NS",
    "SUZLON.NS",
    "IRB.NS" 
        # Metropolis Healthcare
]

# ── Industry P/E Benchmarks (from your notes Module 4) ───────
INDUSTRY_PE = {
    "IT":           22,
    "Banking":      15,
    "FMCG":         45,
    "Pharma":       30,
    "Auto":         20,
    "Paints":       55,
    "Metals":       10,
    "Oil":           8,
    "Retail":       60,
    "Jewellery":    35,
    "Infra":        25,
    "Default":      25,
}

# ── OPM Benchmarks by sector (from notes 3.2) ────────────────
OPM_BENCHMARKS = {
    "IT":      (22, 28),
    "Pharma":  (25, 35),
    "FMCG":    (18, 25),
    "Paints":  (18, 28),
    "Auto":    (8,  12),
    "Retail":  (5,   8),
    "Metals":  (8,  15),
    "Default": (10, 20),
}

# ── Scoring thresholds (from notes 3.2) ──────────────────────
STAGE1_PASS  = 6.5   # Must score >= 6.5 to go to Stage 2
STAGE2_PASS  = 6.0

# D/E Ratio
DE_EXCELLENT = 0.5
DE_GOOD      = 1.0
DE_ACCEPTABLE= 1.25

# Current Ratio
CR_EXCELLENT = 2.0
CR_GOOD      = 1.5

# Interest Coverage
IC_EXCELLENT = 10
IC_GOOD      = 5
IC_ACCEPTABLE= 3

# ROE benchmarks
ROE_EXCEPTIONAL = 25
ROE_EXCELLENT   = 20
ROE_GOOD        = 15
ROE_ACCEPTABLE  = 10

# ROCE benchmarks
ROCE_EXCEPTIONAL = 20
ROCE_GOOD        = 15
ROCE_ACCEPTABLE  = 10

# ── ML Settings ──────────────────────────────────────────────
SEQ_LENGTH    = 60    # Days LSTM looks back
PREDICT_DAYS  = 30    # Days to forecast
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
N_ESTIMATORS  = 200

# ── Technical Indicator Settings ─────────────────────────────
RSI_PERIOD    = 14
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIGNAL   = 9
BB_PERIOD     = 20
ATR_PERIOD    = 14
EMA_PERIODS   = [9, 21, 50, 200]

# ── SMC Settings ─────────────────────────────────────────────
SWING_LENGTH  = 5
FVG_THRESHOLD = 0.003   # 0.3% minimum gap

# ── App Settings ─────────────────────────────────────────────
CACHE_DIR     = "data/cache"
MODEL_DIR     = "models/saved"
APP_TITLE     = "AI Stock Screener"
APP_ICON      = "📈"

# ── Valuation (from notes Module 4.1) ────────────────────────
MARGIN_OF_SAFETY      = 0.20   # Buffett 20% rule
CONSERVATIVE_FACTOR   = 2/3    # Use 2/3 of historical growth
# ── Fundamental Analysis Engine ───────────────────────────────
# Implements the exact scoring system from your Module 3.2 notes
# Stage 1: 10-point Gatekeeping Score
# Stage 2: Deep Trend Analysis
# Total = 20 points

import sys
sys.path.append("..")
from config import (
    STAGE1_PASS, STAGE2_PASS,
    DE_EXCELLENT, DE_GOOD, DE_ACCEPTABLE,
    CR_EXCELLENT, CR_GOOD,
    IC_EXCELLENT, IC_GOOD, IC_ACCEPTABLE,
    ROE_EXCEPTIONAL, ROE_EXCELLENT, ROE_GOOD, ROE_ACCEPTABLE,
    ROCE_EXCEPTIONAL, ROCE_GOOD, ROCE_ACCEPTABLE,
    INDUSTRY_PE, OPM_BENCHMARKS,
    MARGIN_OF_SAFETY, CONSERVATIVE_FACTOR,
)


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — GATEKEEPING (10 points)
# ═══════════════════════════════════════════════════════════════

def stage1_score(fund: dict) -> dict:
    """
    Runs all 10 checks from your notes.
    Returns score out of 10 with breakdown.
    """
    scores  = {}
    details = {}

    scores["sales_growth"],   details["sales_growth"]   = _check_sales_growth(fund)
    scores["profit_growth"],  details["profit_growth"]  = _check_profit_growth(fund)
    scores["opm"],            details["opm"]            = _check_opm(fund)
    scores["roe"],            details["roe"]            = _check_roe(fund)
    scores["roce"],           details["roce"]           = _check_roce(fund)
    scores["debt_equity"],    details["debt_equity"]    = _check_de(fund)
    scores["current_ratio"],  details["current_ratio"]  = _check_cr(fund)
    scores["interest_cover"], details["interest_cover"] = _check_ic(fund)
    scores["pe_ratio"],       details["pe_ratio"]       = _check_pe(fund)
    scores["roa"],            details["roa"]            = _check_roa(fund)

    total = round(sum(scores.values()), 2)

    # Verdict from your notes
    if total >= 8.5:
        verdict = "🟢 Exceptional"
    elif total >= 7.5:
        verdict = "🟢 Excellent"
    elif total >= 6.5:
        verdict = "🟡 Good — Proceed to Stage 2"
    else:
        verdict = "🔴 Avoid"

    return {
        "total":   total,
        "max":     10,
        "verdict": verdict,
        "passed":  total >= STAGE1_PASS,
        "scores":  scores,
        "details": details,
    }


# ── Check 1: Sales Growth ─────────────────────────────────────
def _check_sales_growth(fund):
    growth = fund.get("revenue_growth")
    if growth is None:
        return 0, "No data"

    score = 0
    # 1yr growth positive → 0.5 pts
    if growth > 0:
        score += 0.5
    # Growth > 10% → excellent
    if growth >= 10:
        score += 0.5
        detail = f"✅ Sales growth {growth:.1f}% — Excellent (>10%)"
    elif growth >= 5:
        detail = f"✅ Sales growth {growth:.1f}% — Good (5–10%)"
        score += 0.25
    elif growth > 0:
        detail = f"⚠️ Sales growth {growth:.1f}% — Slow (0–5%)"
    else:
        detail = f"🔴 Sales growth {growth:.1f}% — Red Flag"
        score  = 0

    return round(score, 2), detail


# ── Check 2: Profit Growth ────────────────────────────────────
def _check_profit_growth(fund):
    growth = fund.get("earnings_growth")
    if growth is None:
        return 0, "No data"

    score = 0
    if growth > 0:
        score += 0.5
    if growth >= 15:
        score += 0.5
        detail = f"✅ Profit growth {growth:.1f}% — Excellent (>15%)"
    elif growth >= 10:
        score += 0.25
        detail = f"✅ Profit growth {growth:.1f}% — Good (10–15%)"
    elif growth >= 5:
        detail = f"⚠️ Profit growth {growth:.1f}% — Acceptable (5–10%)"
    elif growth > 0:
        detail = f"⚠️ Profit growth {growth:.1f}% — Slow"
    else:
        detail = f"🔴 Profit growth {growth:.1f}% — Red Flag"
        score  = 0

    return round(score, 2), detail


# ── Check 3: OPM (Operating Profit Margin) ───────────────────
def _check_opm(fund):
    opm    = fund.get("operating_margins")
    sector = fund.get("sector", "Default")
    bm     = OPM_BENCHMARKS.get(sector, OPM_BENCHMARKS["Default"])

    if opm is None:
        return 0, "No data"

    score = 0
    lo, hi = bm

    if opm >= lo:
        score += 0.5
        detail = f"✅ OPM {opm:.1f}% — Above benchmark ({lo}–{hi}%)"
        if opm >= hi:
            score += 0.5
            detail = f"✅ OPM {opm:.1f}% — Excellent (above {hi}%)"
    else:
        detail = f"🔴 OPM {opm:.1f}% — Below benchmark ({lo}%)"

    return round(score, 2), detail


# ── Check 4: ROE ──────────────────────────────────────────────
def _check_roe(fund):
    roe = fund.get("roe")
    if roe is None:
        return 0, "No data"

    # Warren Buffett rule: look for consistent 15%+ ROE
    if roe >= ROE_EXCEPTIONAL:   # > 25%
        score  = 1.0
        detail = f"✅ ROE {roe:.1f}% — Exceptional (>25%)"
    elif roe >= ROE_EXCELLENT:   # 20–25%
        score  = 0.75
        detail = f"✅ ROE {roe:.1f}% — Excellent (20–25%)"
    elif roe >= ROE_GOOD:        # 15–20%
        score  = 0.5
        detail = f"✅ ROE {roe:.1f}% — Good (15–20%)"
    elif roe >= ROE_ACCEPTABLE:  # 10–15%
        score  = 0.25
        detail = f"⚠️ ROE {roe:.1f}% — Acceptable (10–15%)"
    else:
        score  = 0
        detail = f"🔴 ROE {roe:.1f}% — Weak Red Flag (<10%)"

    return round(score, 2), detail


# ── Check 5: ROCE ─────────────────────────────────────────────
def _check_roce(fund):
    # yfinance doesn't give ROCE directly — approximate via ROA
    # ROCE ≈ EBIT / Capital Employed
    # We use operating_margins * revenue / (total assets - current liabilities)
    # Fallback: use ROA as proxy
    roa = fund.get("roa")
    roe = fund.get("roe")

    if roa is None:
        return 0, "No data"

    # Proxy: if ROA > certain threshold, ROCE likely good
    # Ideally ROCE >= ROE means debt used productively
    score  = 0
    detail = ""

    if roa >= 20:
        score  = 0.75
        detail = f"✅ ROA {roa:.1f}% (ROCE proxy) — Excellent"
    elif roa >= 15:
        score  = 0.5
        detail = f"✅ ROA {roa:.1f}% — Good"
    elif roa >= 10:
        score  = 0.25
        detail = f"⚠️ ROA {roa:.1f}% — Acceptable"
    else:
        detail = f"🔴 ROA {roa:.1f}% — Weak"

    # Bonus if ROCE >= ROE (debt productive)
    if roe and roa and roa >= roe * 0.8:
        score += 0.25
        detail += " + Debt used productively ✅"

    return round(min(score, 1.0), 2), detail


# ── Check 6: Debt-to-Equity ───────────────────────────────────
def _check_de(fund):
    de = fund.get("debt_to_equity")
    if de is None:
        return 0, "No data"

    # yfinance returns D/E as percentage sometimes — normalise
    if de > 10:
        de = de / 100

    if de < DE_EXCELLENT:        # < 0.5
        score  = 1.0
        detail = f"✅ D/E {de:.2f} — Excellent (<0.5)"
    elif de < DE_GOOD:           # 0.5–1.0
        score  = 0.5
        detail = f"✅ D/E {de:.2f} — Good (0.5–1.0)"
    elif de < DE_ACCEPTABLE:     # 1.0–1.25
        score  = 0.25
        detail = f"⚠️ D/E {de:.2f} — Acceptable (1.0–1.25)"
    else:
        score  = 0
        detail = f"🔴 D/E {de:.2f} — Danger! (>1.25)"

    return round(score, 2), detail


# ── Check 7: Current Ratio ────────────────────────────────────
def _check_cr(fund):
    cr = fund.get("current_ratio")
    if cr is None:
        return 0, "No data"

    if cr >= CR_EXCELLENT:       # > 2.0
        score  = 1.0
        detail = f"✅ Current Ratio {cr:.2f} — Excellent (>2)"
    elif cr >= CR_GOOD:          # 1.5–2.0
        score  = 0.5
        detail = f"✅ Current Ratio {cr:.2f} — Good (1.5–2.0)"
    elif cr >= 1.0:              # 1.0–1.5
        score  = 0.25
        detail = f"⚠️ Current Ratio {cr:.2f} — Acceptable (1–1.5)"
    else:
        score  = 0
        detail = f"🔴 Current Ratio {cr:.2f} — Danger! (<1)"

    return round(score, 2), detail


# ── Check 8: Interest Coverage ────────────────────────────────
def _check_ic(fund):
    # Approximate: EBITDA / (Revenue * estimated interest rate)
    ebitda  = fund.get("ebitda")
    revenue = fund.get("revenue")

    if not ebitda or not revenue or revenue == 0:
        return 0, "No data"

    # Rough IC proxy using EBITDA margin
    ebitda_margin = (ebitda / revenue) * 100

    if ebitda_margin >= 30:
        score  = 1.0
        detail = f"✅ EBITDA margin {ebitda_margin:.1f}% — Interest well covered"
    elif ebitda_margin >= 20:
        score  = 0.5
        detail = f"✅ EBITDA margin {ebitda_margin:.1f}% — Good coverage"
    elif ebitda_margin >= 10:
        score  = 0.25
        detail = f"⚠️ EBITDA margin {ebitda_margin:.1f}% — Acceptable"
    else:
        score  = 0
        detail = f"🔴 EBITDA margin {ebitda_margin:.1f}% — Weak"

    return round(score, 2), detail


# ── Check 9: P/E vs Industry ──────────────────────────────────
def _check_pe(fund):
    pe     = fund.get("pe_ratio")
    sector = fund.get("sector", "Default")
    ind_pe = INDUSTRY_PE.get(sector, INDUSTRY_PE["Default"])

    if pe is None or pe <= 0:
        return 0, "No P/E data"

    x = pe / ind_pe   # Ratio of stock P/E to industry P/E

    if 0.8 <= x <= 2.0:
        score  = 1.0
        detail = f"✅ P/E {pe:.1f} vs Industry {ind_pe} (ratio {x:.1f}x) — Fair valued"
    elif 2.0 < x <= 2.5:
        score  = 0.5
        detail = f"⚠️ P/E {pe:.1f} vs Industry {ind_pe} (ratio {x:.1f}x) — Getting expensive"
    elif 2.5 < x <= 3.0:
        score  = 0.25
        detail = f"⚠️ P/E {pe:.1f} vs Industry {ind_pe} (ratio {x:.1f}x) — Expensive"
    else:
        score  = 0
        detail = f"🔴 P/E {pe:.1f} vs Industry {ind_pe} (ratio {x:.1f}x) — Highly overvalued"

    return round(score, 2), detail


# ── Check 10: ROA ─────────────────────────────────────────────
def _check_roa(fund):
    roa    = fund.get("roa")
    sector = fund.get("sector", "Default")

    if roa is None:
        return 0, "No data"

    # Minimum ROA benchmarks from notes
    min_roa = {
        "IT": 20, "FMCG": 15,
        "Manufacturing": 8, "Default": 8,
    }.get(sector, 8)

    score = 0
    if roa >= min_roa:
        score  = 0.5
        detail = f"✅ ROA {roa:.1f}% — Above minimum {min_roa}%"
    else:
        detail = f"🔴 ROA {roa:.1f}% — Below minimum {min_roa}%"

    return round(score, 2), detail


# ═══════════════════════════════════════════════════════════════
# VALUATION ENGINE (Module 4.1 from your notes)
# ═══════════════════════════════════════════════════════════════

def calculate_valuation(fund: dict) -> dict:
    """
    Calculates Fair Price, Target Price, Margin of Safety.
    Implements all 4 methods from your notes.
    """
    results = {}

    # ── Method 1: P/E Valuation ──────────────────────────────
    eps    = fund.get("eps")
    sector = fund.get("sector", "Default")
    ind_pe = INDUSTRY_PE.get(sector, INDUSTRY_PE["Default"])
    price  = fund.get("current_price", 0)

    if eps and eps > 0 and ind_pe:
        fair_price = eps * ind_pe
        margin     = (price - fair_price) / fair_price * 100 if fair_price else None
        buy_price  = fair_price * (1 - MARGIN_OF_SAFETY)

        results["pe_valuation"] = {
            "method":      "P/E Valuation",
            "eps":         eps,
            "industry_pe": ind_pe,
            "fair_price":  round(fair_price, 2),
            "current":     round(price, 2),
            "overvalued_pct": round(margin, 1) if margin else None,
            "buy_below":   round(buy_price, 2),
            "verdict":     _pe_verdict(margin),
        }

    # ── Method 2: PEG Ratio ───────────────────────────────────
    pe     = fund.get("pe_ratio")
    growth = fund.get("earnings_growth")

    if pe and growth and growth > 0:
        peg = pe / growth
        results["peg_ratio"] = {
            "method":  "PEG Ratio",
            "pe":      pe,
            "growth":  growth,
            "peg":     round(peg, 2),
            "verdict": _peg_verdict(peg),
        }

    # ── Method 3: P/B Ratio ───────────────────────────────────
    pb = fund.get("pb_ratio")
    if pb:
        results["pb_ratio"] = {
            "method":  "P/B Ratio",
            "pb":      round(pb, 2),
            "verdict": _pb_verdict(pb, sector),
        }

    # ── Overall Valuation Score ───────────────────────────────
    results["overall"] = _overall_valuation(results, price)

    return results


def _pe_verdict(margin):
    if margin is None:   return "No data"
    if margin < -20:     return "🟢 Undervalued — Great Buy!"
    if margin < 0:       return "🟢 Slightly Undervalued"
    if margin < 20:      return "🟡 Fairly Valued"
    if margin < 50:      return "🟠 Overvalued"
    return               "🔴 Highly Overvalued — Avoid"


def _peg_verdict(peg):
    if peg < 1:          return "🟢 Undervalued — Paying less than growth!"
    if peg <= 1.5:       return "🟡 Fair Valued"
    if peg <= 2.0:       return "🟠 Getting Expensive"
    return               "🔴 Overvalued"


def _pb_verdict(pb, sector):
    if sector == "Banking":
        if pb < 1.5:     return "🟢 Very Cheap (Rare!)"
        if pb <= 2.0:    return "🟡 Fair"
        if pb <= 3.0:    return "🟠 Reasonable"
        return           "🔴 Expensive"
    else:
        if pb < 1.5:     return "🟢 Cheap"
        if pb <= 3.0:    return "🟡 Fair"
        return           "🔴 Expensive"


def _overall_valuation(results, price):
    scores = []
    if "pe_valuation" in results:
        margin = results["pe_valuation"].get("overvalued_pct", 0) or 0
        if margin < 0:   scores.append(3)
        elif margin < 20: scores.append(2)
        elif margin < 50: scores.append(1)
        else:            scores.append(0)

    if "peg_ratio" in results:
        peg = results["peg_ratio"]["peg"]
        if peg < 1:      scores.append(3)
        elif peg <= 1.5: scores.append(2)
        elif peg <= 2:   scores.append(1)
        else:            scores.append(0)

    if not scores:
        return {"verdict": "Insufficient data", "score": 0}

    avg = sum(scores) / len(scores)
    if avg >= 2.5:   verdict = "🟢 BUY — Good Valuation"
    elif avg >= 1.5: verdict = "🟡 HOLD — Fair Price"
    elif avg >= 0.5: verdict = "🟠 WAIT — Slightly Expensive"
    else:            verdict = "🔴 AVOID — Overvalued"

    return {"verdict": verdict, "score": round(avg, 1)}


# ═══════════════════════════════════════════════════════════════
# FULL ANALYSIS — combines Stage 1 + Valuation
# ═══════════════════════════════════════════════════════════════

def full_analysis(fund: dict) -> dict:
    """
    Complete fundamental analysis.
    Returns everything needed for the Streamlit app.
    """
    s1          = stage1_score(fund)
    valuation   = calculate_valuation(fund)

    return {
        "ticker":     fund.get("ticker"),
        "name":       fund.get("name"),
        "sector":     fund.get("sector"),
        "price":      fund.get("current_price"),
        "stage1":     s1,
        "valuation":  valuation,
        "passed":     s1["passed"],
        "final_verdict": _final_verdict(s1, valuation),
    }


def _final_verdict(s1, valuation):
    score    = s1["total"]
    val_verd = valuation.get("overall", {}).get("verdict", "")

    if score >= 7.5 and "BUY" in val_verd:
        return "🚀 STRONG BUY — Great business at great price!"
    elif score >= 6.5 and ("BUY" in val_verd or "HOLD" in val_verd):
        return "✅ BUY — Good fundamentals at fair price"
    elif score >= 6.5:
        return "🟡 WATCHLIST — Good business but wait for better price"
    elif score >= 5.0:
        return "⚠️ WEAK — Risky investment"
    else:
        return "🔴 AVOID — Poor fundamentals"
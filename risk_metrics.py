# src/hedging/risk_metrics.py
import numpy as np

def var_es(pnl, alpha: float = 0.05):
    """Return (VaR_alpha, ES_alpha) for a 1D PnL array."""
    pnl = np.asarray(pnl)
    q = np.quantile(pnl, alpha)
    es = pnl[pnl <= q].mean() if np.any(pnl <= q) else q
    return float(q), float(es)

def var_es_summary(pnl, name: str = "Series", alpha: float = 0.05):
    """Small dict with mean/std/VaR/ES, handy for printing/logging."""
    pnl = np.asarray(pnl)
    q, es = var_es(pnl, alpha=alpha)
    return {
        "name": name,
        "mean": float(np.mean(pnl)),
        "stdev": float(np.std(pnl, ddof=1)),
        f"VaR_{int(alpha*100)}%": q,
        f"ES_{int(alpha*100)}%": es,
    }

def hedge_effectiveness_var(unhedged, hedged):
    """HE_var = 1 - Var(hedged)/Var(unhedged), with ddof=1 and safety guards."""
    unhedged = np.asarray(unhedged, dtype=float)
    hedged   = np.asarray(hedged,   dtype=float)
    if unhedged.size < 2 or hedged.size < 2:
        raise ValueError("Need at least 2 scenarios to compute sample variance.")
    var_u = np.var(unhedged, ddof=1)
    var_h = np.var(hedged,   ddof=1)
    if var_u <= 0:
        return np.nan  # undefined if unhedged variance is zero or negative (numeric)
    return 1.0 - (var_h / var_u)

def hedge_effectiveness_es_from_stats(stats_u, stats_h, key="ES_5%"):
    """
    Compute HE_ES given summary dicts (e.g., from var_es_summary).
    Assumes the ES values are positive expected losses.
    """
    es_u = float(stats_u[key])
    es_h = float(stats_h[key])
    if es_u <= 0:
        return np.nan
    return 1.0 - (es_h / es_u)

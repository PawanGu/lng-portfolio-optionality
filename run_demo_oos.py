# run_demo_oos.py
import numpy as np

# If your files are in a package, adjust these imports accordingly.
from simulators import MultiFactorOU
from swing import SwingSpec
from lsmc import lsmc_swing_value

# If in hedging/ subpackage, use: from hedging.risk_metrics import var_es_summary
from risk_metrics import var_es_summary
from level_strip_hedge import hedge_level_strip, apply_strip_positions

def hedg_eff(u, h):
    # Variance reduction (higher is better)
    return 1.0 - (np.var(h, ddof=1) / np.var(u, ddof=1))

def main():
    rng = np.random.default_rng(0)
    T = 30
    n_paths = 6000

    # --- simulate paths ---
    sim = MultiFactorOU(  # defaults are fine; make explicit if you like
        mu=[0.00, 0.00, 0.00],
        sigma=[0.80, 0.60, 0.70],
        corr=[
            [1.0, 0.35, 0.30],
            [0.35, 1.0, 0.60],
            [0.30, 0.60, 1.0]
        ],
        dt=1/252,
        rng=42
    )
    paths = sim.simulate_paths(
        S0={'HH': 3.0, 'TTF': 9.0, 'JKM': 11.0},
        B0={'B_JKM_TTF': 2.0},
        T=T,
        n_paths=n_paths
    )

    # --- swing spec ---
    spec = SwingSpec(
        T=T, q_min=0.5, q_max=1.5, Q_min=20.0, Q_max=30.0,
        index='HH', spread_addon=1.2, destination='TTF', fee=0.1
    )

    # --- LSMC valuation; we only need cf (cashflows) ---
    out = lsmc_swing_value(paths, spec)
    value, q, cf = out[:3]  # tolerant if your function returns 4 values (deltas)

    # --- OOS split ---
    idx = rng.permutation(n_paths)
    tr, te = idx[: n_paths//2], idx[n_paths//2 :]

    paths_tr = {k: v[tr] for k, v in paths.items()}
    paths_te = {k: v[te] for k, v in paths.items()}
    cf_tr, cf_te = cf[tr], cf[te]

    # --- Train: fit level→strip hedge (two-asset) ---
    pnl_hedge_tr, H = hedge_level_strip(
        paths_tr, spec, cf_tr,
        two_assets=True, ridge=1e-4, tc_bps=0.0, lot=None
    )
    pnl_unhedged_tr = cf_tr.sum(axis=1)
    pnl_hedged_tr   = pnl_unhedged_tr + pnl_hedge_tr

    # --- Test: apply positions to test set with costs/rounding if you want ---
    pnl_unhedged_te = cf_te.sum(axis=1)
    pnl_hedge_te    = apply_strip_positions(paths_te, spec, H, tc_bps=1.0, lot=None)
    pnl_hedged_te   = pnl_unhedged_te + pnl_hedge_te

    # --- Print stats ---
    print("--- TRAIN ---")
    print(var_es_summary(pnl_unhedged_tr, "Unhedged"))
    print(var_es_summary(pnl_hedged_tr,   "Hedged (train)"))
    print({"HedgeEffectiveness_var_train": hedg_eff(pnl_unhedged_tr, pnl_hedged_tr)})

    print("\n--- TEST (OOS) ---")
    print(var_es_summary(pnl_unhedged_te, "Unhedged"))
    print(var_es_summary(pnl_hedged_te,   "Hedged (test, tc=1bp)"))
    print({"HedgeEffectiveness_var_test": hedg_eff(pnl_unhedged_te, pnl_hedged_te)})

if __name__ == "__main__":
    # ensure local imports work if you run from repo root
    import os, sys
    ROOT = os.path.dirname(os.path.abspath(__file__))
    if ROOT not in sys.path: sys.path.append(ROOT)
    main()

import numpy as np

def cvar(pnl, alpha=0.05):
    # Expected shortfall estimator
    q = np.quantile(pnl, alpha)
    tail = pnl[pnl <= q]
    return tail.mean()

def mean_cvar_optimize(weights_init, pnl_matrix, lam=2.0):
    """
    Maximize E[PnL] - λ * ES_5%
    pnl_matrix: shape (n_paths, n_assets) where assets = [unhedged_contract, HH_hedge, TTF_hedge, ...]
    Solve a small unconstrained quadratic-like problem or grid-search for demo.
    """
    W = weights_init
    # simple projected gradient or grid search in examples/run_demo.py
    ...

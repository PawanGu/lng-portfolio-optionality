# delta_hedge_grad.py (or modify your existing file)
import numpy as np

def hedge_with_pathwise_deltas(paths, spec, lsmc_pricer):
    """
    Daily re-hedge using Δ from LSMC gradients. Position = -Δ_t (classic delta hedge).
    """
    _, _, _, deltas = lsmc_pricer(paths, spec)   # shape (n_paths, T)
    S = paths[spec.index]                        # shape (n_paths, T+1)
    pos = -deltas                                # hedge position per day
    dS = S[:,1:] - S[:,:-1]
    pnl_hedge = (pos * dS).sum(axis=1)
    return pnl_hedge

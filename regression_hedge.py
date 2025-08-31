# hedging/regression_hedge.py
import numpy as np

def hedge_regression_daily(paths, spec, cf, two_assets=True):
    """
    Choose daily hedge positions to minimize Var( cf_t + h_t · dS_t )
    Returns: (pnl_hedge, hedge_weights) where pnl_hedge is per-path sum over t
    """
    S_idx  = paths[spec.index]         # (n_paths, T+1)
    dS_idx = S_idx[:, 1:] - S_idx[:, :-1]

    if two_assets:
        S_dst  = paths[spec.destination]
        dS_dst = S_dst[:, 1:] - S_dst[:, :-1]

    n, T = dS_idx.shape
    pnl_hedge = np.zeros(n)
    H = []

    for t in range(T):
        y = cf[:, t]                              # current day's cashflow
        if two_assets:
            X = np.column_stack([dS_idx[:, t], dS_dst[:, t]])  # (n,2)
        else:
            X = dS_idx[:, t:t+1]                               # (n,1)

        # Solve h_t = argmin_h || y + X h ||_2  => least-squares on (-y)
        h_t, *_ = np.linalg.lstsq(X, -y, rcond=None)
        pnl_hedge += X @ h_t
        H.append(h_t)

    return pnl_hedge, np.array(H)  # shapes: (n,), (T, 2 or 1)

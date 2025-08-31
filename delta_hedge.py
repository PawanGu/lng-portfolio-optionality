import numpy as np

def finite_diff_delta(price_paths, spec, lsmc_pricer, bump=0.01):
    """
    Pathwise delta to underlying index (e.g., HH) at each t by bump-and-reprice.
    Expensive but simple & defensible in interviews.
    """
    base_val, _, _ = lsmc_pricer(price_paths, spec)
    bumped = {k:v.copy() for k,v in price_paths.items()}
    bumped[spec.index] = bumped[spec.index] * (1 + bump)
    bump_val, _, _ = lsmc_pricer(bumped, spec)
    return (bump_val - base_val) / (price_paths[spec.index].mean()*bump)

def rolling_delta_hedge(paths, spec, lsmc_pricer, rebalance_days, tc_bps=0.5):
    """
    Rebalance a futures hedge on index (HH) and/or destination (TTF/JKM).
    PnL_t = Δ_{t-1} * (S_t - S_{t-1}) - costs; compare unhedged vs hedged.
    """
    HH = paths[spec.index]
    n, T = HH.shape[0], HH.shape[1]-1

    # Compute a simple time-constant delta for demo (or recompute every k days)
    base_val, _, _ = lsmc_pricer(paths, spec)
    deltas = []
    hedge_pos = np.zeros((n, T+1))
    for t in range(T+1):
        if t % rebalance_days == 0:
            d = finite_diff_delta(paths, spec, lsmc_pricer, bump=0.01)
        deltas.append(d)
        hedge_pos[:, t] = d

    # Hedge PnL across paths
    dS = HH[:,1:] - HH[:,:-1]
    pnl_hedge = (hedge_pos[:,:-1] * dS).sum(axis=1)
    costs = tc_bps*1e-4 * np.abs(np.diff(hedge_pos, axis=1)).sum(axis=1) * HH[:, :-1].mean(axis=1)
    return pnl_hedge - costs, np.array(deltas)

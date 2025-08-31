# level_strip_hedge.py
import numpy as np

def _ols(X, y, ridge=0.0):
    if ridge > 0:
        k = X.shape[1]
        A = X.T @ X + ridge * np.eye(k)
        b = X.T @ y
        return np.linalg.solve(A, b)
    return np.linalg.lstsq(X, y, rcond=None)[0]

def _round_positions(pos_vec, lot):
    if lot is None or lot <= 0:
        return pos_vec
    return np.round(pos_vec / lot) * lot

def hedge_level_strip(paths, spec, cf, two_assets=True, ridge=1e-6, tc_bps=0.0, lot=None):
    """
    Fit a 'level → futures strip' hedge on TRAIN data.
    1) For each day t, regress cashflow_t on *levels* with intercept:
         cf_t ≈ a_t + b_idx[t]*S_idx_t (+ b_dst[t]*S_dst_t)
    2) Convert level betas to futures strip via backward cumulative sums.
    3) Positions are NEGATIVE of those strips to offset exposure.
    4) Compute training hedge PnL (optionally with rounding & transaction costs).

    Returns: (pnl_hedge_train, H)
      - pnl_hedge_train: shape (n_train_paths,)
      - H: per-day positions (T x k), k=1 (idx) or 2 (idx,dst)
    """
    S_idx  = paths[spec.index]                  # (n, T+1)
    dS_idx = S_idx[:, 1:] - S_idx[:, :-1]       # (n, T)
    n, T   = cf.shape

    if two_assets:
        S_dst  = paths[spec.destination]
        dS_dst = S_dst[:, 1:] - S_dst[:, :-1]

    # --- estimate per-day level betas (with intercept) ---
    b_idx = np.zeros(T)
    b_dst = np.zeros(T) if two_assets else None
    for t in range(T):
        y = cf[:, t]
        if two_assets:
            X = np.column_stack([np.ones(n), S_idx[:, t], S_dst[:, t]])
            a, bi, bd = _ols(X, y, ridge=ridge)
            b_idx[t], b_dst[t] = bi, bd
        else:
            X = np.column_stack([np.ones(n), S_idx[:, t]])
            a, bi = _ols(X, y, ridge=ridge)
            b_idx[t] = bi

    # --- backward cumulative sums of betas; positions are NEGATIVE of these strips ---
    H_idx = -np.flip(np.cumsum(np.flip(b_idx))).astype(float)   # shape (T,)
    if two_assets:
        H_dst = -np.flip(np.cumsum(np.flip(b_dst))).astype(float)  # shape (T,)

    # Optional rounding to lots (stabilizes)
    H_idx = _round_positions(H_idx, lot)
    if two_assets:
        H_dst = _round_positions(H_dst, lot)

    # --- training hedge PnL ---
    pnl_hedge = (H_idx[None, :] * dS_idx).sum(axis=1)
    if two_assets:
        pnl_hedge += (H_dst[None, :] * dS_dst).sum(axis=1)

    # Add turnover costs (single mid proxy across paths)
    if tc_bps and tc_bps > 0:
        dpos_idx = np.diff(np.r_[0.0, H_idx])
        mid_idx  = S_idx[:, :-1].mean(axis=0)  # (T,)
        tc = np.sum(np.abs(dpos_idx) * mid_idx) * (tc_bps * 1e-4)
        if two_assets:
            dpos_dst = np.diff(np.r_[0.0, H_dst])
            mid_dst  = S_dst[:, :-1].mean(axis=0)
            tc += np.sum(np.abs(dpos_dst) * mid_dst) * (tc_bps * 1e-4)
        pnl_hedge = pnl_hedge - tc

    # stack daily positions as (T, k)
    if two_assets:
        H = np.column_stack([H_idx, H_dst])
    else:
        H = H_idx[:, None]
    return pnl_hedge, H

def apply_strip_positions(paths, spec, H, tc_bps=0.0, lot=None):
    """
    Apply a fixed per-day position strip H (T x k) to TEST data and return hedge PnL.
    H[:,0] = index leg; H[:,1] (optional) = destination leg.
    """
    H = np.asarray(H)
    T = H.shape[0]
    S_idx  = paths[spec.index]
    dS_idx = S_idx[:, 1:] - S_idx[:, :-1]
    assert dS_idx.shape[1] == T, "H has wrong length vs horizon T"

    pnl_hedge = (H[:, 0][None, :] * dS_idx).sum(axis=1)

    if H.shape[1] == 2:
        S_dst  = paths[spec.destination]
        dS_dst = S_dst[:, 1:] - S_dst[:, :-1]
        pnl_hedge += (H[:, 1][None, :] * dS_dst).sum(axis=1)

    if tc_bps and tc_bps > 0:
        # Single cost computed off average path; conservative and simple.
        tc = 0.0
        dpos_idx = np.diff(np.r_[0.0, H[:, 0]])
        mid_idx  = S_idx[:, :-1].mean(axis=0)
        tc += np.sum(np.abs(dpos_idx) * mid_idx) * (tc_bps * 1e-4)
        if H.shape[1] == 2:
            dpos_dst = np.diff(np.r_[0.0, H[:, 1]])
            mid_dst  = S_dst[:, :-1].mean(axis=0)
            tc += np.sum(np.abs(dpos_dst) * mid_dst) * (tc_bps * 1e-4)
        pnl_hedge = pnl_hedge - tc

    return pnl_hedge

import numpy as np
from swing import SwingState, SwingSpec

def basis_functions(state_vec):
    # [1, x0(index), x1(dest), x2(rem_quota_frac), x0*x1, x1*x2, x0^2, x1^2]
    x0, x1, x2, tf, b = state_vec
    return np.array([1.0, x0, x1, x2, x0*x1, x1*tf, x0**2, x1**2])

def dbasis_dindex(state_vec):
    x0, x1, x2, tf, b = state_vec
    # d/dx0 of the basis above
    return np.array([0.0, 1.0, 0.0, 0.0, x1, 0.0, 2.0*x0, 0.0])

def lsmc_swing_value(paths, spec: SwingSpec):
    n, T = paths['HH'].shape[0], paths['HH'].shape[1]-1
    q   = np.zeros((n, T))
    cf  = np.zeros((n, T))
    dlt = np.zeros((n, T))  # pathwise deltas wrt index

    rem_quota = np.full(n, spec.Q_max)
    cont_val  = np.zeros(n)

    for t in reversed(range(T)):
        idx  = paths[spec.index][:, t]
        dest = paths[spec.destination][:, t]
        basis_spread = dest - idx
        time_frac = (t+1)/T

        state = np.vstack([idx, dest, rem_quota/spec.Q_max,
                           np.full(n, time_frac), basis_spread]).T

        # feasible bounds
        lo, hi = [], []
        for i in range(n):
            s = SwingState(spec); s.cum = spec.Q_max - rem_quota[i]
            a, b = s.feasible_bounds(t)
            lo.append(a); hi.append(b)
        lo  = np.array(lo); hi = np.array(hi); mid = 0.5*(lo+hi)

        spread = dest - (idx + spec.spread_addon) - spec.fee
        payoff_lo  = spread * lo
        payoff_mid = spread * mid
        payoff_hi  = spread * hi

        X = np.stack([basis_functions(sv) for sv in state])
        beta = np.linalg.lstsq(X, cont_val, rcond=None)[0]

        def cont_next(rem_after):
            st = state.copy()
            st[:,2] = rem_after/spec.Q_max
            return np.stack([basis_functions(sv) for sv in st]) @ beta

        val_lo  = payoff_lo  + cont_next(rem_quota - lo)
        val_mid = payoff_mid + cont_next(rem_quota - mid)
        val_hi  = payoff_hi  + cont_next(rem_quota - hi)

        take = np.argmax(np.stack([val_lo, val_mid, val_hi]), axis=0)
        chosen_q = np.where(take==0, lo, np.where(take==1, mid, hi))

        # cashflow + state update
        q[:, t]  = chosen_q
        cf[:, t] = spread * chosen_q
        rem_quota = rem_quota - chosen_q
        cont_val = np.maximum.reduce([val_lo, val_mid, val_hi])

        # ---- analytic delta wrt index ----
        # immediate payoff derivative wrt index is -q_t (since spread = dest - (idx + k) - fee)
        d_immediate = -chosen_q
        # continuation derivative via regression gradient:
        grad = np.stack([dbasis_dindex(sv) for sv in state]) @ beta
        dlt[:, t] = d_immediate + grad

    value_est = cf.sum(axis=1).mean()
    return value_est, q, cf, dlt

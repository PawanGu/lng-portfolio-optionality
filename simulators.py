# simulators.py
import numpy as np

class MultiFactorOU:
    """
    Correlated 3-factor GBM for HH/TTF/JKM (simple, robust).
    Returns paths for HH, TTF, JKM and the basis B_JKM_TTF = JKM - TTF.
    """
    def __init__(self, mu=None, sigma=None, corr=None, dt=1/252, rng=None, **kwargs):
        # Annualized drifts (log space). Defaults ~0 for simplicity.
        self.mu = np.array(mu if mu is not None else [0.00, 0.00, 0.00], dtype=float)
        # Annualized vols for (HH, TTF, JKM)
        self.sigma = np.array(sigma if sigma is not None else [0.80, 0.60, 0.70], dtype=float)
        self.dt = float(dt)

        if corr is None:
            corr = np.array([
                [1.0, 0.35, 0.30],
                [0.35, 1.0, 0.60],
                [0.30, 0.60, 1.0]
            ], dtype=float)
        self.corr = np.array(corr, dtype=float)
        self.L = np.linalg.cholesky(self.corr)

        # RNG can be an int seed or a Generator
        self.rng = np.random.default_rng(rng)

    def simulate_paths(self, S0, B0=None, T=30, n_paths=1000):
        """
        S0: dict with 'HH','TTF','JKM' starting prices
        B0: optional dict with 'B_JKM_TTF' (so JKM0 ≈ TTF0 + basis)
        """
        HH0 = float(S0['HH'])
        TTF0 = float(S0['TTF'])
        if B0 is not None and 'B_JKM_TTF' in B0:
            JKM0 = TTF0 + float(B0['B_JKM_TTF'])
        else:
            JKM0 = float(S0.get('JKM', TTF0 + 2.0))

        n = int(n_paths)
        Tn = int(T)

        HH  = np.zeros((n, Tn + 1), dtype=float)
        TTF = np.zeros_like(HH)
        JKM = np.zeros_like(HH)

        HH[:, 0]  = HH0
        TTF[:, 0] = TTF0
        JKM[:, 0] = JKM0

        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        vol_sqrt_dt = self.sigma * np.sqrt(self.dt)

        for t in range(Tn):
            z = self.rng.standard_normal(size=(n, 3))
            corr_z = z @ self.L.T
            incr = drift + corr_z * vol_sqrt_dt  # shape (n, 3)

            HH[:,  t+1] = HH[:,  t] * np.exp(incr[:, 0])
            TTF[:, t+1] = TTF[:, t] * np.exp(incr[:, 1])
            JKM[:, t+1] = JKM[:, t] * np.exp(incr[:, 2])

        Bjt = JKM - TTF
        return {"HH": HH, "TTF": TTF, "JKM": JKM, "B_JKM_TTF": Bjt}

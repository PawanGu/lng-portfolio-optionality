# swing.py
from dataclasses import dataclass

@dataclass
class SwingSpec:
    T: int                   # number of delivery days
    q_min: float             # daily min lift
    q_max: float             # daily max lift
    Q_min: float             # cumulative min over T
    Q_max: float             # cumulative max over T
    index: str               # e.g., 'HH'
    spread_addon: float      # e.g., +k over index
    destination: str         # e.g., 'TTF' or 'JKM'
    fee: float = 0.0         # per-unit fee

class SwingState:
    def __init__(self, spec: SwingSpec):
        self.spec = spec
        self.cum = 0.0  # cumulative lifted so far

    def feasible_bounds(self, t: int):
        """
        Feasible daily bounds at time t given cumulative lifted so far.
        Enforces both daily [q_min, q_max] and possibility to still meet Q_min/Q_max
        over the remaining days.
        """
        T = self.spec.T
        qmin, qmax = self.spec.q_min, self.spec.q_max
        Qmin, Qmax = self.spec.Q_min, self.spec.Q_max

        rem_days = T - t
        # Max possible you could still take if you go max each remaining day (excluding today after choosing)
        max_future = qmax * max(rem_days - 1, 0)
        min_future = qmin * max(rem_days - 1, 0)

        # To not violate Qmax today, upper bound:
        hi_cap = Qmax - self.cum - min_future
        # To still be able to reach Qmin by the end, lower bound:
        lo_cap = Qmin - self.cum - max_future

        lo = max(qmin, lo_cap)
        hi = min(qmax, hi_cap)

        # Clamp to non-negative and ensure lo <= hi
        lo = max(0.0, lo)
        hi = max(0.0, hi)
        if lo > hi:
            # If constraints collide, pick the closest feasible (set both to a safe bound)
            lo = hi
        return lo, hi

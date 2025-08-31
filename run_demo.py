"""
1) Simulate joint HH/TTF/JKM paths
2) Price a swing contract via LSMC
3) Compute a rolling delta hedge on HH
4) Compare unhedged vs hedged PnL; print ES_5% & Hedge effectiveness
5) Save plots to /figs
"""
from simulators import MultiFactorOU
from swing import SwingSpec
from lsmc import lsmc_swing_value
from delta_hedge import rolling_delta_hedge
from risk_metrics import var_es_summary, hedge_effectiveness_var, hedge_effectiveness_es_from_stats
from plots import plot_pnl_hist
from delta_hedge_grad import hedge_with_pathwise_deltas
from regression_hedge import hedge_regression_daily
from level_strip_hedge import hedge_level_strip
import numpy as np

def main():
    # 0) config
    T, n_paths = 30, 5000   # one month
    spec = SwingSpec(T=T, q_min=0.5, q_max=1.5, Q_min=20, Q_max=30,
                     index='HH', spread_addon=1.2, destination='TTF', fee=0.1)

    # 1) simulate prices (fill with synthetic params)
    sim = MultiFactorOU(rng=42)  # uses default mu/sigma/corr, dt=1/252
    paths = sim.simulate_paths(
    S0={'HH': 3.0, 'TTF': 9.0, 'JKM': 11.0},
    B0={'B_JKM_TTF': 2.0},
    T=T,
    n_paths=n_paths
)
    # 2) LSMC price & cashflows
    value, q, cf, deltas = lsmc_swing_value(paths, spec)

    # 3) Hedge & PnL
    #pnl_hedge, deltas = rolling_delta_hedge(paths, spec, lsmc_pricer=lsmc_swing_value, rebalance_days=5)
    #pnl_hedge = hedge_with_pathwise_deltas(paths, spec, lsmc_swing_value)
    #pnl_hedge, H = hedge_regression_daily(paths, spec, cf, two_assets=True)
    pnl_hedge, H = hedge_level_strip(paths, spec, cf, two_assets=True, ridge=1e-6)

    # 4) Unhedged path PnL is cf.sum(axis=1)
    pnl_unhedged = cf.sum(axis=1)

    # 5) Risk stats & plots
    stats_u = var_es_summary(pnl_unhedged, name="Unhedged", alpha=0.05)
    stats_h = var_es_summary(pnl_unhedged + pnl_hedge, name="Delta-Hedged", alpha=0.05)
    
    print(stats_u)
    print(stats_h)
 
    he_var = hedge_effectiveness_var(pnl_unhedged, pnl_unhedged + pnl_hedge)
    he_es  = hedge_effectiveness_es_from_stats(stats_u, stats_h, key="ES_5%")

    print("Hedge effectiveness (variance):", he_var)
    print("Hedge effectiveness (ES 5%):", he_es)
    
    plot_pnl_hist([pnl_unhedged, pnl_unhedged+pnl_hedge], labels=["Unhedged","Hedged"], outfile="pnl_hist.png")

    u, h = pnl_unhedged, pnl_hedge
    cov, varh = np.cov(u, h, ddof=1)[0,1], np.var(h, ddof=1)
    lam_star = - cov / varh
    he_star  = hedge_effectiveness_var(u, u + lam_star*h)
    print(lam_star, he_star)

if __name__ == "__main__":
    main()

# LNG Portfolio Optionality (LSMC + Hedging)

Price an **LNG swing contract** and show how to **hedge its cash‑flow risk**.
Correlated **HH/TTF/JKM** drivers → **LSMC** pricing → two hedges:

1. **Level → futures strip** (targets realized cashflows)
2. **M2M delta** from LSMC gradients (targets mark‑to‑market value)

Includes **out‑of‑sample** validation and **VaR/ES** reporting.

---

## Why this repo

* Demonstrates how LNG desks monetize **volumetric flexibility** and control risk.
* Bridges **quant modelling** with **trading PnL & hedging**—not just pricing.
* Clean, self‑contained code you can discuss at interview.

---

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=.
python run_demo.py      # basic pricing + PnL histogram
python run_demo_oos.py  # train/test hedge with VaR/ES + effectiveness
```

Outputs go to console and (optionally) `figs/`.

---

## What you’ll see

* **Pricing:** LSMC computes optimal daily liftings within min/max and monthly bounds.
* **Hedging (cash‑flow):** regress daily cashflows on **levels**, convert betas to a **backward cumulative strip** of futures positions; compare unhedged vs hedged **cashflow sums**.
* **Risk:** mean/stdev/**VaR₅%/ES₅%** and **HedgeEffectiveness (variance reduction)**.

Example console snippet:

```text
--- TEST (OOS) ---
Unhedged: {"mean": ..., "stdev": ..., "VaR_5%": ..., "ES_5%": ...}
Hedged  : {"mean": ..., "stdev": ..., "VaR_5%": ..., "ES_5%": ...}
HedgeEffectiveness_var_test: 0.xx
```

---

## Repository layout

```
.
├─ run_demo.py              # price swing, simple plot
├─ run_demo_oos.py          # out‑of‑sample hedge validation
├─ simulators.py            # correlated GBM for HH/TTF/JKM
├─ swing.py                 # SwingSpec + feasibility (daily & cumulative bounds)
├─ lsmc.py                  # LSMC pricer (+ optional analytic deltas)
├─ level_strip_hedge.py     # cash‑flow hedge via level→futures strip
├─ hedging/
│  ├─ risk_metrics.py       # mean/stdev/VaR/ES helpers
│  └─ regression_hedge.py   # (optional) ΔS regression hedge baseline
└─ src/viz/plots.py         # PnL histogram helper
```

---

## Method in 30 seconds

**Pricing (LSMC).**

* State: `(Index, Destination, remaining_quota_frac, t/T, basis_spread)`
* Actions: discrete volumes within feasible bounds (from `SwingState`).
* Backward induction with a compact polynomial basis for continuation value.

**Hedge A — Level → futures strip (cash‑flow metric).**

* Regress `CF_t ≈ a_t + b_t^idx S_t^idx + b_t^dst S_t^dst`.
* Positions: `H_u^· = − Σ_{t≥u} b_t^·` (backward cumulative); apply to ΔS.
* Evaluate variance/ES of **realized cashflows** (unhedged vs hedged).

**Hedge B — M2M delta (value metric).**

* Differentiate LSMC continuation: `Δ_t ≈ ∂V_t/∂S`.
* Hedge changes in **option value**; analyze replication error variance.

---

## Configuration knobs

* **Simulation:** vols/corr, horizon `T`, paths, seed.
* **Contract:** `q_min/q_max`, `Q_min/Q_max`, `index`, `destination`, `spread_addon`, `fee`.
* **Hedging:** `two_assets`, `ridge`, transaction costs `tc_bps`, `lot` rounding.

---

## Extensions

* Diversion optionality (choose TTF vs JKM each day).
* Shipping / regas capacity and lags.
* Historical calibration (seasonality, term structure).
* Explicit basis hedge (JKM–TTF spread).
* Inventory & storage; PnL attribution and stress tests.

---

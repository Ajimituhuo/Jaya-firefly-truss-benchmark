# Jaya–Firefly hybrid for truss sizing: code and data

Supplementary material for "Does Firefly Attraction Improve the Jaya Algorithm? A Benchmark
Evaluation of a Jaya–Firefly Hybrid for Truss Sizing Optimisation".

Requires Python 3.9+, NumPy, SciPy and Matplotlib.

| File | Purpose |
| --- | --- |
| `trusses.py` | Direct-stiffness truss solver and the 10-bar (cases 1 and 2), 25-bar and 72-bar benchmarks |
| `validate.py` | Re-analyses published optimum designs to check the solver (10-bar case 1 gives 5,060.85 lb) |
| `algos.py` | Jaya, standard Firefly Algorithm and the Jaya–Firefly hybrid; Deb (2000) feasibility rules |
| `experiments.py` | `python experiments.py main` runs 30 runs per algorithm per problem; `python experiments.py sens` runs the γ/α₀ sensitivity grid |
| `analyse.py` | Produces the statistics in Table 4 (`summary.json`) and Fig. 2 |
| `fig_geom.py` | Produces Fig. 1 |
| `results_main.pkl`, `results_sens.pkl` | Raw results (best weights and convergence/diversity traces for every run) |

Seeds are fixed (1000–1029 for the main study, 5000–5014 for the sensitivity grid), so results
are reproducible. Units are inches, kips and ksi; weights are in lb.

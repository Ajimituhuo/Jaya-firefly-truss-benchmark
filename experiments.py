import json, pickle, time, sys, numpy as np
from trusses import *; from algos import run
N = 20
problems = {"10c1": (ten_bar(1), 20000), "10c2": (ten_bar(2), 20000),
            "25": (twentyfive_bar(), 15000), "72": (seventytwo_bar(), 20000)}
algs = {"Jaya": ("jaya", {}), "FA": ("fa", {}),
        "Hybrid": ("hybrid", {}),
        "Hybrid (unscaled γ)": ("hybrid", dict(gamma=0.5, alpha0=0.2, raw_units=True))}
part = sys.argv[1]
out = {}
t0 = time.time()
if part == "main":
    for pk, (prob, nfe) in problems.items():
        for ak, (alg, kw) in algs.items():
            res = [run(alg, prob, N, nfe, 1000 + s, log_every=5, **kw) for s in range(30)]
            out[(pk, ak)] = res
            b = [r["best"] for r in res]
            print(pk, ak, np.min(b), np.mean(b), np.std(b), round(time.time()-t0), flush=True)
else:
    grid = [(g, a) for g in (0.1, 1.0, 10.0) for a in (0.05, 0.2, 0.5)]
    for pk in ("10c1", "72"):
        prob, nfe = problems[pk]
        for alg in ("fa", "hybrid"):
            for g, a in grid:
                res = [run(alg, prob, N, nfe, 5000 + s, gamma=g, alpha0=a, log_every=1000) for s in range(15)]
                out[(pk, alg, g, a)] = [r["best"] for r in res]
                b = out[(pk, alg, g, a)]
                print(pk, alg, g, a, np.mean(b), np.std(b), round(time.time()-t0), flush=True)
pickle.dump(out, open(f"results_{part}.pkl", "wb"))
print("DONE", flush=True)

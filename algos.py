"""Jaya, Firefly (standard, Yang 2010) and the Jaya-Firefly hybrid.
All search in normalised space z in [0,1]^n; x = lb + z*(ub-lb).
Constraints: Deb (2000) feasibility rules (no penalty coefficient).
Each candidate is evaluated once per iteration, so every algorithm spends N analyses per iteration.
"""
import numpy as np

FEAS_TOL = 1e-4  # sum of normalised violations treated as zero

class Evaluator:
    def __init__(self, truss):
        self.t = truss; self.nfe = 0
        self.best_w = np.inf; self.hist = []   # (nfe, best feasible weight)
    def __call__(self, z):
        x = self.t.lb + z * (self.t.ub - self.t.lb)
        w = self.t.weight(x); v = self.t.violation(x)
        if v <= FEAS_TOL: v = 0.0
        self.nfe += 1
        if v == 0.0 and w < self.best_w:
            self.best_w = w
        return w, v

def better(w1, v1, w2, v2):
    """Deb's rules: is (w1,v1) better than (w2,v2)?"""
    if v1 == 0 and v2 == 0: return w1 < w2
    if v1 == 0 or v2 == 0: return v1 == 0
    return v1 < v2

def rank_key(W, V):
    # sort: feasible by weight first, then infeasible by violation
    return np.lexsort((np.where(V == 0, W, V), (V > 0).astype(int)))

def diversity(Z):
    return float(np.mean(np.linalg.norm(Z - Z.mean(0), axis=1)))

def run(alg, truss, N, max_nfe, seed, gamma=1.0, alpha0=0.2, beta0=1.0, raw_units=False,
        log_every=1):
    rng = np.random.default_rng(seed)
    n = truss.ng; span = truss.ub - truss.lb
    ev = Evaluator(truss)
    Z = rng.random((N, n))
    W = np.empty(N); V = np.empty(N)
    for k in range(N): W[k], V[k] = ev(Z[k])
    T = max_nfe // N - 1
    trace = []; t = 0
    while ev.nfe + N <= max_nfe:
        order = rank_key(W, V)
        ib, iw = order[0], order[-1]
        alpha = alpha0 * (0.01 ** (t / max(T, 1)))
        Znew = Z.copy()
        if alg == "jaya":
            r1 = rng.random((N, n)); r2 = rng.random((N, n))
            Znew = Z + r1 * (Z[ib] - np.abs(Z)) - r2 * (Z[iw] - np.abs(Z))
        elif alg == "fa":
            rank = np.empty(N, int); rank[order] = np.arange(N)
            for i in range(N):
                zi = Z[i].copy()
                brighter = [j for j in order if rank[j] < rank[i]]
                if not brighter:
                    zi = zi + alpha * (rng.random(n) - 0.5)
                for j in brighter:
                    r2d = np.sum((zi - Z[j]) ** 2)
                    beta = beta0 * np.exp(-gamma * r2d)
                    zi = zi + beta * (Z[j] - zi) + alpha * (rng.random(n) - 0.5)
                Znew[i] = zi
        elif alg == "hybrid":
            rank = np.empty(N, int); rank[order] = np.arange(N)
            for i in range(N):
                zi = Z[i]
                brighter = np.array([j for j in range(N) if rank[j] < rank[i]], int)
                if raw_units:   # manuscript settings: distance and step in raw in^2
                    if len(brighter):
                        d2 = np.sum(((Z[brighter] - zi) * span) ** 2, axis=1)
                        j = brighter[np.argmin(d2)]; beta = beta0 * np.exp(-gamma * d2.min())
                        zpp = zi + beta * (Z[j] - zi)
                    else:
                        zpp = zi.copy()
                    zpp = zpp + alpha * (rng.random(n) - 0.5) / span
                else:
                    if len(brighter):
                        d2 = np.sum((Z[brighter] - zi) ** 2, axis=1)
                        j = brighter[np.argmin(d2)]; beta = beta0 * np.exp(-gamma * d2.min())
                        zpp = zi + beta * (Z[j] - zi)
                    else:
                        zpp = zi.copy()
                    zpp = zpp + alpha * (rng.random(n) - 0.5)
                r1 = rng.random(n); r2 = rng.random(n)
                Znew[i] = zpp + r1 * (Z[ib] - np.abs(zpp)) - r2 * (Z[iw] - np.abs(zpp))
        Znew = np.clip(Znew, 0.0, 1.0)
        for k in range(N):
            w, v = ev(Znew[k])
            if alg == "fa":      # standard FA accepts the move
                Z[k], W[k], V[k] = Znew[k], w, v
            elif better(w, v, W[k], V[k]):
                Z[k], W[k], V[k] = Znew[k], w, v
        t += 1
        if t % log_every == 0:
            trace.append((ev.nfe, ev.best_w, diversity(Z)))
    return dict(best=ev.best_w, trace=np.array(trace), nfe=ev.nfe)

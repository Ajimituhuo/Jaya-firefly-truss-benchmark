"""Truss benchmark definitions and a direct-stiffness solver (2D/3D pin-jointed)."""
import numpy as np

class Truss:
    def __init__(self, nodes, members, fixed, loads, E, rho, groups, lb, ub,
                 sig_t, sig_c, disp_lim, disp_nodes=None, disp_dofs=None, name=""):
        self.nodes = np.asarray(nodes, float)          # (nn, dim)
        self.members = np.asarray(members, int)        # (nm, 2) 0-based
        self.dim = self.nodes.shape[1]
        self.nn = len(self.nodes); self.nm = len(self.members)
        self.ndof = self.nn * self.dim
        fixed_dofs = [n * self.dim + d for n in fixed for d in range(self.dim)]
        self.free = np.setdiff1d(np.arange(self.ndof), fixed_dofs)
        self.loads = [np.asarray(L, float) for L in loads]  # list of (ndof,) vectors
        self.E, self.rho = E, rho
        self.groups = groups                     # list of lists of member indices
        self.ng = len(groups)
        self.member_group = np.empty(self.nm, int)
        for g, mem in enumerate(groups):
            self.member_group[mem] = g
        self.lb = np.full(self.ng, lb, float) if np.isscalar(lb) else np.asarray(lb, float)
        self.ub = np.full(self.ng, ub, float) if np.isscalar(ub) else np.asarray(ub, float)
        self.sig_t = np.broadcast_to(np.asarray(sig_t, float), (self.nm,)).copy()
        self.sig_c = np.broadcast_to(np.asarray(sig_c, float), (self.nm,)).copy()  # positive magnitudes
        self.disp_lim = disp_lim
        # displacement-constrained dofs (default: all free dofs)
        if disp_nodes is None:
            self.dcon = self.free
        else:
            dd = disp_dofs if disp_dofs is not None else range(self.dim)
            self.dcon = np.array([n * self.dim + d for n in disp_nodes for d in dd])
        self.name = name
        # geometry
        a, b = self.members[:, 0], self.members[:, 1]
        dvec = self.nodes[b] - self.nodes[a]
        self.L = np.linalg.norm(dvec, axis=1)
        self.c = dvec / self.L[:, None]
        # member dof map
        self.mdofs = np.hstack([a[:, None] * self.dim + np.arange(self.dim),
                                b[:, None] * self.dim + np.arange(self.dim)])
        # unit stiffness blocks (EA/L = 1)
        cc = np.einsum('mi,mj->mij', self.c, self.c)
        self.k_unit = np.block([[cc, -cc], [-cc, cc]]) / self.L[:, None, None]
        self.Lsum_by_group = np.array([self.L[g].sum() for g in groups])
        nd = 2 * self.dim
        rows = np.repeat(self.mdofs, nd, axis=1)
        cols = np.tile(self.mdofs, (1, nd))
        self.flat_idx = (rows * self.ndof + cols).ravel()
        fr = self.free
        self.free_idx = (fr[:, None] * self.ndof + fr[None, :]).ravel()
        self.k_unit_flat = self.k_unit.reshape(self.nm, -1)
        self.F = np.column_stack([L[self.free] for L in self.loads])

    def weight(self, x):
        return self.rho * float(self.Lsum_by_group @ x)

    def analyse(self, x):
        A = x[self.member_group]
        vals = (self.k_unit_flat * (self.E * A)[:, None]).ravel()
        K = np.bincount(self.flat_idx, weights=vals, minlength=self.ndof * self.ndof)
        nf = len(self.free)
        Kf = K[self.free_idx].reshape(nf, nf)
        F = self.F
        Uf = np.linalg.solve(Kf, F)
        U = np.zeros((self.ndof, F.shape[1])); U[self.free] = Uf
        du = U[self.mdofs[:, self.dim:]] - U[self.mdofs[:, :self.dim]]  # (nm, dim, ncase)
        strain = np.einsum('mi,mic->mc', self.c, du) / self.L[:, None]
        sigma = self.E * strain
        return sigma, U

    def violation(self, x):
        """Sum of normalised constraint violations (0 if feasible)."""
        sigma, U = self.analyse(x)
        vt = np.maximum(0, sigma / self.sig_t[:, None] - 1)
        vc = np.maximum(0, -sigma / self.sig_c[:, None] - 1)
        vd = np.maximum(0, np.abs(U[self.dcon]) / self.disp_lim - 1)
        return float(vt.sum() + vc.sum() + vd.sum())


def ten_bar(case=1):
    # Units: in, kips, ksi. Nodes 1..6 (1-based) ; supports 5,6
    nodes = [(720, 360), (720, 0), (360, 360), (360, 0), (0, 360), (0, 0)]
    mem = [(3, 5), (1, 3), (4, 6), (2, 4), (3, 4), (1, 2), (4, 5), (3, 6), (2, 3), (1, 4)]
    mem = [(a - 1, b - 1) for a, b in mem]
    F = np.zeros(12)
    if case == 1:
        P1, P2 = 100.0, 0.0
    else:
        P1, P2 = 150.0, 50.0
    F[2 * 1 + 1] = -P1; F[2 * 3 + 1] = -P1      # nodes 2,4 downward
    F[2 * 0 + 1] = P2;  F[2 * 2 + 1] = P2       # nodes 1,3 upward
    return Truss(nodes, mem, fixed=[4, 5], loads=[F], E=1.0e4, rho=0.1,
                 groups=[[i] for i in range(10)], lb=0.1, ub=35.0,
                 sig_t=25.0, sig_c=25.0, disp_lim=2.0, name=f"10-bar case {case}")


def twentyfive_bar():
    nodes = [(-37.5, 0, 200), (37.5, 0, 200), (-37.5, 37.5, 100), (37.5, 37.5, 100),
             (37.5, -37.5, 100), (-37.5, -37.5, 100), (-100, 100, 0), (100, 100, 0),
             (100, -100, 0), (-100, -100, 0)]
    mem = [(1, 2), (1, 4), (2, 3), (1, 5), (2, 6), (2, 4), (2, 5), (1, 3), (1, 6), (3, 6),
           (4, 5), (3, 4), (5, 6), (3, 10), (6, 7), (4, 9), (5, 8), (4, 7), (3, 8), (5, 10),
           (6, 9), (6, 10), (3, 7), (4, 8), (5, 9)]
    mem = [(a - 1, b - 1) for a, b in mem]
    groups = [[0], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10], [11, 12], [13, 14, 15, 16],
              [17, 18, 19, 20], [21, 22, 23, 24]]
    comp = [35.092, 11.590, 17.305, 35.092, 35.092, 6.759, 6.959, 11.082]
    sig_c = np.empty(25)
    for g, m in enumerate(groups):
        sig_c[m] = comp[g]
    L1 = np.zeros(30); L2 = np.zeros(30)
    def put(L, n, v):
        L[3 * (n - 1):3 * (n - 1) + 3] += v
    put(L1, 1, (1.0, 10.0, -5.0)); put(L1, 2, (0.0, 10.0, -5.0))
    put(L1, 3, (0.5, 0.0, 0.0)); put(L1, 6, (0.5, 0.0, 0.0))
    put(L2, 1, (0.0, 20.0, -5.0)); put(L2, 2, (0.0, -20.0, -5.0))
    return Truss(nodes, mem, fixed=[6, 7, 8, 9], loads=[L1, L2], E=1.0e4, rho=0.1,
                 groups=groups, lb=0.01, ub=3.4, sig_t=40.0, sig_c=sig_c,
                 disp_lim=0.35, disp_nodes=list(range(6)), name="25-bar")


def seventytwo_bar():
    # 4 storeys of 60 in, 120x120 in plan. Levels 0..4 ; level 0 fixed.
    corners = [(0, 0), (120, 0), (120, 120), (0, 120)]
    nodes = []
    for lev in range(5):
        for (x, y) in corners:
            nodes.append((x, y, 240 - 60 * lev))   # level 0 = top (nodes 1-4) ... level 4 = base
    nid = lambda lev, k: lev * 4 + k
    mem, groups = [], []
    for s in range(4):  # storey s between level s (upper) and s+1 (lower)
        up, lo = s, s + 1
        gv, gd, gh, ghd = [], [], [], []
        for k in range(4):                      # verticals
            gv.append(len(mem)); mem.append((nid(up, k), nid(lo, k)))
        for k in range(4):                      # face diagonals (2 per face)
            k2 = (k + 1) % 4
            gd.append(len(mem)); mem.append((nid(up, k), nid(lo, k2)))
            gd.append(len(mem)); mem.append((nid(up, k2), nid(lo, k)))
        for k in range(4):                      # horizontal perimeter at upper level
            gh.append(len(mem)); mem.append((nid(up, k), nid(up, (k + 1) % 4)))
        ghd.append(len(mem)); mem.append((nid(up, 0), nid(up, 2)))   # horizontal diagonals
        ghd.append(len(mem)); mem.append((nid(up, 1), nid(up, 3)))
        groups += [gv, gd, gh, ghd]
    # standard grouping order is bottom storey first; reorder so group 1 = bottom verticals
    groups = groups[12:16] + groups[8:12] + groups[4:8] + groups[0:4]
    L1 = np.zeros(60); L2 = np.zeros(60)
    L1[0:3] = (5.0, 5.0, -5.0)                         # node 1 (top)
    for k in range(4):
        L2[3 * k + 2] = -5.0
    return Truss(nodes, mem, fixed=[16, 17, 18, 19], loads=[L1, L2], E=1.0e4, rho=0.1,
                 groups=groups, lb=0.1, ub=4.0, sig_t=25.0, sig_c=25.0, disp_lim=0.25,
                 disp_nodes=[0, 1, 2, 3], disp_dofs=[0, 1], name="72-bar")

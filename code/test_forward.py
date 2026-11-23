# -*- coding: utf-8 -*-
import sys, time, json
sys.path.insert(0, "code")
import numpy as np
import geom, cases, dimenet_np as dn

X1 = geom.reconstruct(cases.D1); X2 = geom.reconstruct(cases.D2)
Z = cases.Z_C4
mdl = dn.DimeNet(emb_size=128, num_blocks=6, num_bilinear=8, num_spherical=7, num_radial=6, cutoff=5.0, seed=1)
g1 = dn.Graph(Z, X1, cutoff=5.0)
g2 = dn.Graph(Z, X2, cutoff=5.0)
print("n_atoms", g1.n_atoms, "edges", g1.n_edges, "triplets", g1.n_triplets)
t0 = time.time(); e1 = float(mdl.forward(g1)); t1 = time.time()
e2 = float(mdl.forward(g2))
print("E1 =", e1, " E2 =", e2, " |dE| =", abs(e1-e2), "  fwd time %.3fs" % (t1-t0))
# 旋转/平移不变性
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.normal(size=(3,3)))
if np.linalg.det(Q) < 0: Q[:, 0] *= -1
X1r = X1 @ Q.T + np.array([3.0, -1.0, 2.0])
gr = dn.Graph(Z, X1r, cutoff=5.0)
print("rotation/translation diff =", abs(float(mdl.forward(gr)) - e1))
# 置换等变性(总能量不变)
perm = [2, 0, 3, 1]
gp = dn.Graph(np.array(Z)[perm], X1[perm], cutoff=5.0)
print("permutation diff =", abs(float(mdl.forward(gp)) - e1))
# 镜像
Xm = X1.copy(); Xm[:, 2] *= -1
gm = dn.Graph(Z, Xm, cutoff=5.0)
print("mirror diff =", abs(float(mdl.forward(gm)) - e1))
json.dump({"e1": e1, "e2": e2, "dE": abs(e1-e2), "edges": int(g1.n_edges), "triplets": int(g1.n_triplets)}, open("results/smoke_forward.json", "w"), indent=1)

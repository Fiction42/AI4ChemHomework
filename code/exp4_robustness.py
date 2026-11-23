# -*- coding: utf-8 -*-
"""实验四：不变性、等变性、截断半径与数值稳定性测试。"""
import sys, json
sys.path.insert(0, "code")
import numpy as np
import geom, cases, dimenet_np as dn

CFG = dict(emb_size=128, num_blocks=6, num_bilinear=8, num_spherical=7, num_radial=6, cutoff=5.0)
Z = cases.Z_C4
X1 = geom.reconstruct(cases.D1); X2 = geom.reconstruct(cases.D2)
mdl = dn.DimeNet(**CFG, seed=3)
res = {}

def E(X, cutoff=None):
    g = dn.Graph(Z, X, cutoff or CFG["cutoff"])
    return float(mdl.forward(g))

e0 = E(X1)
# 旋转不变性: 随机旋转 20 次
rng = np.random.default_rng(0)
errs = []
for _ in range(20):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    errs.append(abs(E(X1 @ Q.T) - e0))
res["rotation_max_abs_err"] = float(max(errs))
res["reflection_abs_err"] = abs(E(X1 * np.array([1, 1, -1])) - e0)
res["translation_abs_err"] = abs(E(X1 + np.array([10.0, -7.0, 3.5])) - e0)
perm_errs = []
import itertools
for p in itertools.permutations(range(4)):
    perm_errs.append(abs(E(X1[list(p)]) - e0))
res["permutation_max_abs_err"] = float(max(perm_errs))

# 连续性: 沿一条边小幅改变距离
eps = np.array([1e-4, 1e-3, 1e-2, 5e-2, 1e-1])
errs = []
for e in eps:
    Y = X1.copy()
    Y[0] = Y[0] + e * np.array([1.0, 0, 0])
    errs.append(abs(E(Y) - e0))
res["perturbation"] = {"eps": eps.tolist(), "abs_dE": errs}

# 截断半径对两案例可区分性的影响
cut_rows = []
for cut in [2.2, 2.4, 2.6, 3.0, 3.3, 4.0, 5.0]:
    g1 = dn.Graph(Z, X1, cut); g2 = dn.Graph(Z, X2, cut)
    _, h1, _ = mdl.forward(g1, return_hidden=True)
    _, h2, _ = mdl.forward(g2, return_hidden=True)
    h1, h2 = np.asarray(h1), np.asarray(h2)
    sep = min(np.linalg.norm(h1 - h2[list(p)]) for p in itertools.permutations(range(4))) / (
        np.linalg.norm(h1) + np.linalg.norm(h2))
    cut_rows.append((cut, int(g1.n_edges), int(g1.n_triplets), float(sep)))
res["cutoff_sweep"] = cut_rows
np.savetxt("results/cutoff_sweep.csv", np.array(cut_rows), delimiter=",", header="cutoff,n_edges,n_triplets,rel_sep", comments="")

# float32 数值噪声水平（作为"可区分"判据的下限）
h1 = np.asarray(mdl.forward(dn.Graph(Z, X1, 5.0), return_hidden=True)[1])
res["float32_noise_est"] = float(np.abs(h1.astype(np.float32).astype(np.float64) - h1).max() / np.abs(h1).max())

json.dump(res, open("results/exp4_robustness.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))

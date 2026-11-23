# -*- coding: utf-8 -*-
"""小规模 DimeNet 端到端训练（更快配置），逐 epoch 写 CSV。"""
import sys, time, json
sys.path.insert(0, "code")
import itertools
import numpy as np
import autograd.numpy as anp
from autograd import grad
import geom, cases, descriptors as ds, dimenet_np as dn

np.random.seed(11)
rc = 5.0
def fc(r):
    return np.where(r < rc, 0.5 * (1 + np.cos(np.pi * r / rc)), 0.0)
def V2_term(X):
    r = np.array(ds.pair_list(X))
    return float(np.sum(0.2 * (1 - np.exp(-1.2 * (r - 2.6))) ** 2 - 0.2))
def V3_term(X):
    d = ds.distances(X); n = len(X); tot = 0.0
    for i in range(n):
        others = [a for a in range(n) if a != i]
        for j, k in itertools.combinations(others, 2):
            v1 = X[j] - X[i]; v2 = X[k] - X[i]
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            tot += 0.35 * (c + 1.0 / 3.0) ** 2 * fc(d[i, j]) * fc(d[i, k])
    return float(tot)
def E_ref(X):
    return V2_term(X) + V3_term(X)

def make_dataset(n):
    Xs = []
    while len(Xs) < n:
        X = np.random.uniform(0, 3.6, size=(4, 3))
        d = geom.dist_matrix(X); iu = np.triu_indices(4, 1)
        if d[iu].min() > 1.5 and d[iu].max() < 4.6:
            Xs.append(X)
    return Xs

CFG = dict(emb_size=32, num_blocks=2, num_bilinear=4, num_spherical=5, num_radial=4, cutoff=5.0,
           num_dense_output=2, num_targets=1)
mdl = dn.DimeNet(**CFG, seed=0)
Z = cases.Z_C4
N_TRAIN = 400
pool = make_dataset(N_TRAIN + 100)
graphs = [dn.Graph(Z, X, 5.0) for X in pool]
E = np.array([E_ref(X) for X in pool])
mu, sd = E[:N_TRAIN].mean(), E[:N_TRAIN].std()
ytr = (E - mu) / sd
print("train E std (eV) =", sd, flush=True)

def loss(params, idx):
    tot = 0.0
    for i in idx:
        tot = tot + (mdl.forward(graphs[i], params=params) - ytr[i]) ** 2
    return tot / len(idx)

gloss = grad(loss)
m = {k: np.zeros_like(v) for k, v in mdl.params.items()}
v = {k: np.zeros_like(x) for k, x in mdl.params.items()}
b1, b2, lr, eps = 0.9, 0.999, 3e-3, 1e-8
t0 = time.time()
with open("results/dimenet_training.csv", "w") as fcsv:
    fcsv.write("epoch,train_rmse,test_rmse\n")
    test_idx = list(range(N_TRAIN, N_TRAIN + 100))
    for epoch in range(40):
        perm = np.random.permutation(N_TRAIN)
        for s in range(0, N_TRAIN, 20):
            idx = perm[s:s + 20]
            g = gloss(mdl.params, idx)
            for k in mdl.params:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** (epoch + 1))
                vh = v[k] / (1 - b2 ** (epoch + 1))
                mdl.params[k] = mdl.params[k] - lr * mh / (np.sqrt(vh) + eps)
        tr = float(np.sqrt(np.mean([(float(mdl.forward(graphs[i])) - ytr[i]) ** 2 for i in range(N_TRAIN)]))) * sd
        te = float(np.sqrt(np.mean([(float(mdl.forward(graphs[i])) - ytr[i]) ** 2 for i in test_idx]))) * sd
        fcsv.write("%d,%.6f,%.6f\n" % (epoch + 1, tr, te)); fcsv.flush()
        print("epoch %3d train %.4f test %.4f (%.1fs)" % (epoch + 1, tr, te, time.time() - t0), flush=True)

X1 = geom.reconstruct(cases.D1); X2 = geom.reconstruct(cases.D2)
p1 = float(mdl.forward(dn.Graph(Z, X1, 5.0))) * sd + mu
p2 = float(mdl.forward(dn.Graph(Z, X2, 5.0))) * sd + mu
out = {"pred1": p1, "pred2": p2, "pred_dE": p2 - p1, "ref_dE": E_ref(X2) - E_ref(X1),
       "final_test_rmse": te, "cfg": {k: (v if not isinstance(v, int) else int(v)) for k, v in CFG.items()},
       "N_TRAIN": N_TRAIN, "seconds": time.time() - t0, "init_test_rmse": None}
json.dump(out, open("results/dimenet_training.json", "w"), indent=1)
print("PRED", p1, p2, p2 - p1, "REF", E_ref(X2) - E_ref(X1), "SECONDS", time.time() - t0, flush=True)

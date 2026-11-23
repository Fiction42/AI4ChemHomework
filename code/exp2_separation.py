# -*- coding: utf-8 -*-
"""实验二：表示的完备性/可区分性测试。
对两个案例计算 (i) 2 体不变描述符 (ii) 3 体描述符 (iii) DimeNet 表示，
并给出与随机 4 原子构型背景分布的对比。"""
import sys, json
sys.path.insert(0, "code")
import numpy as np
import geom, cases, descriptors as ds, dimenet_np as dn

rng = np.random.default_rng(42)
X1 = geom.reconstruct(cases.D1); X2 = geom.reconstruct(cases.D2)
Z = cases.Z_C4
res = {}

# ---- 2 体类 ----
res["distance_multiset_max_abs_diff"] = float(np.abs(ds.distance_multiset(X1) - ds.distance_multiset(X2)).max())
f1, f2 = ds.rbf_pair_sum(X1), ds.rbf_pair_sum(X2)
res["rbf_pair_sum_rel_diff"] = float(np.linalg.norm(f1 - f2) / np.linalg.norm(f1))
f1, f2 = ds.inverse_power_pair_sum(X1), ds.inverse_power_pair_sum(X2)
res["inverse_power_rel_diff"] = float(np.linalg.norm(f1 - f2) / np.linalg.norm(f1))
e1, e2 = ds.coulomb_eigs(X1, Z), ds.coulomb_eigs(X2, Z)
res["coulomb_eigs_rel_diff"] = float(np.linalg.norm(e1 - e2) / np.linalg.norm(e1))
# BP 径向: 全局求和相同, 逐原子不同
g1, g2 = ds.bp_radial(X1), ds.bp_radial(X2)
res["bp_radial_global_sum_rel_diff"] = float(np.linalg.norm(g1.sum(0) - g2.sum(0)) / np.linalg.norm(g1.sum(0)))
res["bp_radial_peratom_rel_diff"] = float(np.linalg.norm(np.sort(g1, 0) - np.sort(g2, 0)) / np.linalg.norm(g1))

# ---- 3 体类 ----
res["angle_multiset_deg"] = [ds.angle_list(X1).tolist(), ds.angle_list(X2).tolist()]
res["angle_max_abs_diff_deg"] = float(np.abs(ds.angle_list(X1) - ds.angle_list(X2)).max())
res["angle_multiset_rel_diff"] = float(np.linalg.norm(ds.angle_list(X1) - ds.angle_list(X2)) / np.linalg.norm(ds.angle_list(X1)))
a1, a2 = ds.bp_angular(X1), ds.bp_angular(X2)
res["bp_angular_global_sum_rel_diff"] = float(np.linalg.norm(a1.sum(0) - a2.sum(0)) / np.linalg.norm(a1.sum(0)))

# ---- DimeNet 表示 ----
CFG = dict(emb_size=128, num_blocks=6, num_bilinear=8, num_spherical=7, num_radial=6, cutoff=5.0)
def features(X, model):
    g = dn.Graph(Z, X, CFG["cutoff"])
    _, x, y = model.forward(g, return_hidden=True)
    return np.asarray(x), np.asarray(y).ravel()

perms = list(__import__("itertools").permutations(range(4)))
def perm_min_rel(Xa, Xb):
    return min(np.linalg.norm(Xa - Xb[list(p)]) for p in perms) / (np.linalg.norm(Xa) + np.linalg.norm(Xb))

# 随机背景构型
def rand_tetra(mins=1.4):
    while True:
        X = rng.uniform(0, 3.4, size=(4, 3))
        d = geom.dist_matrix(X)
        iu = np.triu_indices(4, 1)
        if d[iu].min() > mins and d[iu].max() < 5.0:
            return X

seeds = [0, 1, 2, 3, 4]
sep_case_hidden, sep_case_out = [], []
sep_bg = []
for seed in seeds:
    mdl = dn.DimeNet(**CFG, seed=seed)
    h1, y1_ = features(X1, mdl)
    h2, y2_ = features(X2, mdl)
    sep_case_hidden.append(perm_min_rel(h1, h2))
    sep_case_out.append(np.linalg.norm(y1_ - y2_) / (np.linalg.norm(y1_) + np.linalg.norm(y2_)))
    if seed == 0:
        H = [h1]
        for _ in range(120):
            H.append(features(rand_tetra(), mdl)[0])
        for i in range(len(H)):
            for j in range(i + 1, len(H)):
                sep_bg.append(perm_min_rel(H[i], H[j]))
res["dimenet_hidden_rel_sep_per_seed"] = sep_case_hidden
res["dimenet_output_rel_sep_per_seed"] = sep_case_out
res["dimenet_hidden_rel_sep_mean"] = float(np.mean(sep_case_hidden))
res["bg_rel_sep_median"] = float(np.median(sep_bg))
res["bg_rel_sep_p05"] = float(np.percentile(sep_bg, 5))
res["bg_rel_sep_p95"] = float(np.percentile(sep_bg, 95))
res["case_rel_sep_percentile"] = float(100.0 * np.mean(np.array(sep_bg) < np.mean(sep_case_hidden)))

json.dump(res, open("results/exp2_separation.json", "w"), indent=1, ensure_ascii=False)
np.save("results/bg_sep.npy", np.array(sep_bg))
print(json.dumps({k: v for k, v in res.items() if k not in ("angle_multiset_deg",)}, indent=1, ensure_ascii=False))

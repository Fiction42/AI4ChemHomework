# -*- coding: utf-8 -*-
"""实验三：可学习性实验。
参考势能面 E_ref = V2 + V3（V2 为 2 体项, V3 为 3 体角向项, 角度依赖性占显著比例），
比较仅用 2 体特征的模型与含 3 体特征的模型。"""
import sys, json
sys.path.insert(0, "code")
import itertools
import numpy as np
import geom, cases, descriptors as ds

np.random.seed(7)
rc = 5.0

def fc(r):
    return np.where(r < rc, 0.5 * (1 + np.cos(np.pi * r / rc)), 0.0)

def V2_term(X):
    r = np.array(ds.pair_list(X))
    # Morse 型 2 体: 平滑、无量纲化到 eV 量级
    return float(np.sum(0.2 * (1 - np.exp(-1.2 * (r - 2.6))) ** 2 - 0.2))

def V3_term(X):
    d = ds.distances(X)
    n = len(X)
    tot = 0.0
    for i in range(n):
        others = [a for a in range(n) if a != i]
        for j, k in itertools.combinations(others, 2):
            v1 = X[j] - X[i]; v2 = X[k] - X[i]
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            # 偏离 ideal 角 109.47deg (cos=-1/3) 的惩罚
            tot += 0.35 * (c + 1.0 / 3.0) ** 2 * fc(d[i, j]) * fc(d[i, k])
    return float(tot)

def E_ref(X):
    return V2_term(X) + V3_term(X)

def feat_2body(X):
    return np.concatenate([ds.rbf_pair_sum(X), ds.inverse_power_pair_sum(X)])

def feat_3body(X):
    return np.concatenate([feat_2body(X), ds.angle_list(X), ds.bp_angular(X).sum(0)])

def make_dataset(n):
    Xs = []
    while len(Xs) < n:
        X = np.random.uniform(0, 3.6, size=(4, 3))
        d = geom.dist_matrix(X)
        iu = np.triu_indices(4, 1)
        if d[iu].min() > 1.5 and d[iu].max() < 4.6:
            Xs.append(X)
    return Xs

def ridge(F, y, lam=1e-6):
    mu, sd = F.mean(0), F.std(0) + 1e-12
    Zs = (F - mu) / sd
    A = Zs.T @ Zs + lam * len(y) * np.eye(Zs.shape[1])
    w = np.linalg.solve(A, Zs.T @ (y - y.mean()))
    return lambda G: ((G - mu) / sd) @ w + y.mean()

X1 = geom.reconstruct(cases.D1); X2 = geom.reconstruct(cases.D2)
pool = make_dataset(1400)
E2 = np.array([V2_term(X) for X in pool])
E3 = np.array([V3_term(X) for X in pool])
E = E2 + E3
res = {"V2_std": float(E2.std()), "V3_std": float(E3.std()), "E_std": float(E.std()),
       "E_mean": float(E.mean())}
res["V3_var_fraction"] = float(E3.var() / E.var())

split = np.arange(1000, 1400)
rows = []
for ntrain in [50, 100, 200, 400, 800]:
    idx = np.arange(ntrain)
    for name, feat in (("2body", feat_2body), ("3body", feat_3body)):
        Ftr = np.array([feat(pool[i]) for i in idx])
        Fte = np.array([feat(pool[i]) for i in split])
        m = ridge(Ftr, E[idx])
        rmse = float(np.sqrt(np.mean((m(Fte) - E[split]) ** 2)))
        rows.append((name, ntrain, rmse))
res["learning_curve"] = rows
with open("results/learning_curve.csv", "w") as f:
    f.write("feature,ntrain,rmse\n")
    for a, b, c_ in rows:
        f.write(f"{a},{b},{c_}\n")

# 两案例加入训练集: 2 体模型存在特征碰撞, 无法同时拟合
allX = pool[:800] + [X1, X2]
ally = np.append(E[:800], [E_ref(X1), E_ref(X2)])
pair_res = {}
for name, feat in (("2body", feat_2body), ("3body", feat_3body)):
    F = np.array([feat(X) for X in allX])
    m = ridge(F, ally)
    pred = m(np.array([feat(X1), feat(X2)]))
    pair_res[name] = {
        "pred_dE": float(pred[1] - pred[0]),
        "true_dE": float(E_ref(X2) - E_ref(X1)),
        "train_rmse": float(np.sqrt(np.mean((m(F) - ally) ** 2))),
        "test_rmse": float(np.sqrt(np.mean((m(np.array([feat(pool[i]) for i in split])) - E[split]) ** 2))),
    }
res["pair"] = pair_res
dE_true = (cases.E2_HARTREE - cases.E1_HARTREE) * cases.HARTREE2KCAL
res["B3LYP_dE_kcal"] = float(dE_true)
res["2body_min_max_error_kcal"] = float(abs(dE_true) / 2)
res["2body_min_rmse_kcal"] = float(abs(dE_true) / np.sqrt(2))
res["2body_feature_rel_diff_on_cases"] = float(
    np.linalg.norm(feat_2body(X1) - feat_2body(X2)) / np.linalg.norm(feat_2body(X1)))
res["3body_feature_rel_diff_on_cases"] = float(
    np.linalg.norm(feat_3body(X1) - feat_3body(X2)) / np.linalg.norm(feat_3body(X1)))

# 逐项分解: 两案例的 V2/V3 值
res["V2_case1"], res["V2_case2"] = V2_term(X1), V2_term(X2)
res["V3_case1"], res["V3_case2"] = V3_term(X1), V3_term(X2)
json.dump(res, open("results/exp3_learning.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "learning_curve"}, indent=1, ensure_ascii=False))
print("learning curve:", rows)

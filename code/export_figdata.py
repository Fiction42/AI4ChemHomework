# -*- coding: utf-8 -*-
import sys, json, os
sys.path.insert(0, "code")
import numpy as np
import geom, cases, descriptors as ds, dimenet_np as dn, basis_eval as be

os.makedirs("figures", exist_ok=True)
X1 = geom.reconstruct(cases.D1); X2 = geom.reconstruct(cases.D2)
Z = cases.Z_C4

# ---- 距离多重集 ----
d1 = ds.distance_multiset(X1); d2 = ds.distance_multiset(X2)
with open("figures/dist_multiset.csv", "w") as f:
    f.write("idx,d1,d2\n")
    for i, (a, b) in enumerate(zip(d1, d2), 1):
        f.write(f"{i},{a:.6f},{b:.6f}\n")

# ---- 角度多重集 ----
a1 = ds.angle_list(X1); a2 = ds.angle_list(X2)
with open("figures/angles.csv", "w") as f:
    f.write("idx,a1,a2\n")
    for i, (a, b) in enumerate(zip(a1, a2), 1):
        f.write(f"{i},{a:.4f},{b:.4f}\n")

# ---- 背景分布直方图 ----
bg = np.load("results/bg_sep.npy")
res2 = json.load(open("results/exp2_separation.json"))
case_sep = res2["dimenet_hidden_rel_sep_mean"]
counts, edges = np.histogram(bg, bins=25)
with open("figures/bgsep_hist.csv", "w") as f:
    f.write("lo,hi,count\n")
    for i in range(len(counts)):
        f.write(f"{edges[i]:.6f},{edges[i+1]:.6f},{counts[i]}\n")

# ---- 2D 径向基与 3D 方向基 ----
B = be.Basis(7, 6, cutoff=5.0)
x = np.linspace(1e-4, 1.0, 240)
rbf = np.asarray(B.rbf(x * 5.0))
with open("figures/basis_rbf.csv", "w") as f:
    f.write("x," + ",".join([f"b{i+1}" for i in range(6)]) + "\n")
    for k in range(len(x)):
        f.write(f"{x[k]:.6f}," + ",".join([f"{rbf[k, i]:.6f}" for i in range(6)]) + "\n")
th = np.linspace(1e-3, np.pi - 1e-3, 180)
yb = np.stack([np.asarray(be.real_sph_harm_l0(l, np.cos(th))) for l in range(7)], axis=1)
with open("figures/basis_y.csv", "w") as f:
    f.write("theta_deg," + ",".join([f"y{l}" for l in range(7)]) + "\n")
    for k in range(len(th)):
        f.write(f"{np.degrees(th[k]):.4f}," + ",".join([f"{yb[k, l]:.6f}" for l in range(7)]) + "\n")
# 3D 基整体 (l=2, n=1..4)
sbf = np.asarray(B.sbf(np.full(len(th), 0.5) * 5.0, th))
with open("figures/basis_sbf.csv", "w") as f:
    f.write("theta_deg," + ",".join([f"s{i}" for i in range(sbf.shape[1])]) + "\n")
    for k in range(len(th)):
        f.write(f"{np.degrees(th[k]):.4f}," + ",".join([f"{sbf[k, i]:.6f}" for i in range(sbf.shape[1])]) + "\n")

# ---- 案例结构: 投影到 2D 供 TikZ 作图 ----
def proj(X, az=35, el=18):
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Y = X @ Rz.T @ Rx.T
    return Y[:, 0], Y[:, 1]

def tikz_case(X, name, az, el, scale=1.35):
    px, py = proj(X, az, el)
    px = px - px.mean(); py = py - py.mean()
    m = max(np.abs(px).max(), np.abs(py).max())
    px = px / m * scale; py = py / m * scale
    lines = []
    for i in range(4):
        lines.append(f"\\coordinate ({name}{i}) at ({px[i]:.3f},{py[i]:.3f});")
    return "\n".join(lines), px, py

c1, px1, py1 = tikz_case(X1, "A", 30, 15)
c2, px2, py2 = tikz_case(X2, "B", 30, 15)
labels = {}
for i in range(4):
    for j in range(i + 1, 4):
        labels[(i, j)] = X1[i, j] if False else None
d1m = geom.dist_matrix(X1); d2m = geom.dist_matrix(X2)
edges_tex = []
for i in range(4):
    for j in range(i + 1, 4):
        edges_tex.append(f"\\draw[edge] (A{i}) -- (A{j}) node[midway, above, font=\\scriptsize] {{{d1m[i, j]:.3f}}};")
        edges_tex.append(f"\\draw[edge] (B{i}) -- (B{j}) node[midway, above, font=\\scriptsize] {{{d2m[i, j]:.3f}}};")
with open("figures/cases_tikz.tex", "w") as f:
    f.write("% 自动生成: 两个案例的三维坐标投影 (TikZ)\n")
    f.write(c1 + "\n" + c2 + "\n")
    f.write("\n".join(edges_tex) + "\n")

# ---- 关键数字 ----
nparam = sum(np.size(v) for v in dn.DimeNet(emb_size=128, num_blocks=6, cutoff=5.0, seed=0).params.values())
g = dn.Graph(Z, X1, 5.0)
res = {
  "n_params_official_cfg": int(nparam),
  "n_edges": int(g.n_edges),
  "n_triplets": int(g.n_triplets),
  "case_sep": case_sep,
  "bg_median": res2["bg_rel_sep_median"],
  "bg_p05": res2["bg_rel_sep_p05"],
  "percentile": res2["case_rel_sep_percentile"],
  "VOL1": float(geom.cm_volume(cases.D1)), "VOL2": float(geom.cm_volume(cases.D2)),
  "rmsd": float(geom.unlabeled_rmsd(X1, X2)[0]),
  "cou12": ds.coulomb_eigs(X1, Z).tolist(),
  "cou22": ds.coulomb_eigs(X2, Z).tolist(),
}
json.dump(res, open("results/key_numbers.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
print("done")

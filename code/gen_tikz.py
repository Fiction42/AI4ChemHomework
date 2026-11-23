# -*- coding: utf-8 -*-
import sys, numpy as np
sys.path.insert(0, "code")
import geom, cases

X1 = geom.reconstruct(cases.D1); X2 = geom.reconstruct(cases.D2)
d1 = geom.dist_matrix(X1); d2 = geom.dist_matrix(X2)

def proj(X, az=28, el=16):
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Y = X @ Rz.T @ Rx.T
    return Y[:, 0], Y[:, 1]

def place(X, name, dx, scale=1.45):
    px, py = proj(X)
    px = px - px.mean(); py = py - py.mean()
    m = max(np.abs(px).max(), np.abs(py).max())
    px = px / m * scale + dx; py = py / m * scale
    out = []
    for i in range(4):
        out.append(f"\\coordinate ({name}{i}) at ({px[i]:.3f},{py[i]:.3f});")
    return out, (px, py)

la, (pa_x, pa_y) = place(X1, "A", -2.4)
lb, (pb_x, pb_y) = place(X2, "B", 2.4)
lines = ["% 自动生成：两个案例的三维结构投影（TikZ），单位 cm"]
lines += la + lb
for i in range(4):
    for j in range(i + 1, 4):
        lines.append(f"\\draw[gray!70,line width=0.9pt] (A{i}) -- (A{j});")
        lines.append(f"\\draw[gray!70,line width=0.9pt] (B{i}) -- (B{j});")
# 顶点
for k, (x, y) in enumerate(zip(pa_x, pa_y)):
    lines.append(f"\\shade[ball color=gray!40] (A{k}) circle (0.11);")
for k, (x, y) in enumerate(zip(pb_x, pb_y)):
    lines.append(f"\\shade[ball color=gray!40] (B{k}) circle (0.11);")
# 距离标注(仅标两条不同的边, 其余见表格)
lines.append("\\node[font=\\scriptsize] at ($ (A0)!0.5!(A1) + (0,0.18) $) {2.532};")
lines.append("\\node[font=\\scriptsize] at ($ (B0)!0.5!(B1) + (0,0.18) $) {2.532};")
lines.append("\\node[font=\\scriptsize,fill=white,inner sep=1pt] at ($ (A0)!0.55!(A2) + (0.05,0.02) $) {2.177};")
lines.append("\\node[font=\\scriptsize,fill=white,inner sep=1pt] at ($ (B0)!0.55!(B2) + (0.05,0.02) $) {2.661};")
lines.append("\\node[font=\\scriptsize,fill=white,inner sep=1pt] at ($ (A0)!0.55!(A3) + (-0.15,0.0) $) {2.661};")
lines.append("\\node[font=\\scriptsize,fill=white,inner sep=1pt] at ($ (B0)!0.55!(B3) + (-0.15,0.0) $) {2.177};")
open("figures/cases_tikz.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "code")
import numpy as np
import geom, cases, dimenet_np as dn

X1 = geom.reconstruct(cases.D1); X2 = geom.reconstruct(cases.D2)
Z = cases.Z_C4
mdl = dn.DimeNet(emb_size=128, num_blocks=6, cutoff=5.0, seed=1)
g1 = dn.Graph(Z, X1, 5.0); g2 = dn.Graph(Z, X2, 5.0)
e1, x1, y1 = mdl.forward(g1, return_hidden=True)
e2, x2, y2 = mdl.forward(g2, return_hidden=True)
x1 = np.asarray(x1); x2 = np.asarray(x2); y1 = np.asarray(y1); y2 = np.asarray(y2)
print("x1 norm", np.linalg.norm(x1), "x2 norm", np.linalg.norm(x2))
print("rel diff hidden =", np.linalg.norm(x1 - x2) / np.linalg.norm(x1))
print("rel diff output =", np.linalg.norm(y1 - y2) / np.linalg.norm(y1))
print("y1", np.ravel(y1), "y2", np.ravel(y2))
# 基函数层面
b1 = mdl.basis.rbf(g1.Dij); b2 = mdl.basis.rbf(g2.Dij)
print("rbf sorted diff:", np.abs(np.sort(b1, 0) - np.sort(b2, 0)).max())
s1 = np.asarray(mdl.basis.sbf(g1.d_kj, g1.theta)); s2 = np.asarray(mdl.basis.sbf(g2.d_kj, g2.theta))
print("sbf shapes", s1.shape, s2.shape)
print("sbf norm ratio", np.linalg.norm(s1), np.linalg.norm(s2))
print("theta1[:6] (deg)", np.degrees(g1.theta[:6]))
print("theta2[:6] (deg)", np.degrees(g2.theta[:6]))
# 检查 sbf 的角向灵敏度: 固定 d, 改变 theta
th = np.linspace(0.05, np.pi - 0.05, 7)
sv = np.asarray(mdl.basis.sbf(np.full(7, 0.4) * 5.0, th))
print("sbf vs theta variation (rel)", np.abs(sv - sv[0]).max() / np.abs(sv).max())

# -*- coding: utf-8 -*-
import sys, json
import numpy as np
sys.path.insert(0, "code")
import basis_utils as bu
import basis_eval as be
import sympy as sym

n_l, n_n = 7, 6
B = be.Basis(n_l, n_n)
out = {}
z = B.zeros

# 1) 零点检验: j_l(z_ln) 应为 0
res = []
for l in range(n_l):
    for n in range(n_n):
        res.append(abs(float(be.sph_bessel_j(l, z[l, n]))))
out["zeros_max_residual"] = max(res)

# 2) 数值球贝塞尔 vs 官方 sympy 公式
forms = bu.bessel_basis(n_l, n_n)
x = sym.symbols('x')
xs = np.linspace(0.02, 0.999, 9)
errs_b = []
worst = None
for l in range(n_l):
    for n in range(n_n):
        f = sym.lambdify(x, forms[l][n], "numpy")
        ref = np.asarray(f(xs), dtype=float)
        mine = np.asarray(B.norm[l, n] * be.sph_bessel_j(l, z[l, n] * xs))
        e = float(np.max(np.abs(ref - mine)))
        errs_b.append(e)
        if worst is None or e > worst[0]:
            worst = (e, l, n)
out["bessel_vs_official_max_err"] = max(errs_b)
out["bessel_worst"] = worst

# 3) 实球谐
Yforms = bu.real_sph_harm(n_l)
th = sym.symbols('theta')
ths = np.linspace(0.02, 3.12, 13)
errs_y = []
for l in range(n_l):
    f = sym.lambdify(th, Yforms[l][0], "numpy")
    ref = np.asarray(f(ths), dtype=float)
    mine = np.asarray(be.real_sph_harm_l0(l, np.cos(ths)))
    errs_y.append(float(np.max(np.abs(ref - mine))))
out["sph_harm_vs_official_max_err"] = max(errs_y)

# 4) 3D 基正交归一性: int_0^1 B_ln B_lm x^2 dx = delta_nm
xg = np.linspace(0, 1, 400001)
wx = xg ** 2
errs_o = []
diag = []
for l in range(n_l):
    Zl = np.array([np.asarray(B.norm[l, n] * be.sph_bessel_j(l, z[l, n] * xg)) for n in range(n_n)])
    for n in range(n_n):
        for m in range(n, n_n):
            v = float(np.trapezoid(Zl[n] * Zl[m] * wx, xg))
            tgt = 1.0 if n == m else 0.0
            if n == m:
                diag.append(v)
            errs_o.append(abs(v - tgt))
out["bessel_orthonormality_max_err"] = max(errs_o)
out["bessel_diag_values"] = diag

# 5) 2D 径向基(正弦基) 正交性: int_0^1 sin(n pi x) sin(m pi x) dx
errs_r = []
for n in range(1, n_n + 1):
    for m in range(n, n_n + 1):
        v = float(np.trapezoid(np.sin(n * np.pi * xg) * np.sin(m * np.pi * xg), xg))
        tgt = 0.5 if n == m else 0.0
        errs_r.append(abs(v - tgt))
out["rbf_orthogonality_max_err"] = max(errs_r)

json.dump(out, open("results/basis_validation.json", "w"), indent=1)
print(json.dumps(out, indent=1, ensure_ascii=False)[:1500])

# -*- coding: utf-8 -*-
"""dimenet/model/layers/basis_utils.py 的忠实移植。
原实现依赖 scipy.optimize.brentq 与 scipy.special.jv；此处用 mpmath + 二分法替代，
其余（sympy 公式生成、归一化、球谐）逐行对应官方源码。"""
import numpy as np
import sympy as sym
import mpmath as mp


def Jn(r, n):
    """数值球贝塞尔函数 (与官方相同: sqrt(pi/2r) J_{n+1/2}(r))"""
    return np.sqrt(np.pi / (2 * r)) * float(mp.besselj(n + 0.5, r))


def Jn_zeros(n, k):
    """球贝塞尔函数 j_0..j_{n-1} 的前 k 个零点（官方用 brentq，此处用二分法）"""
    zerosj = np.zeros((n, k), dtype="float64")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float64")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            lo, hi = points[j], points[j + 1]
            flo = Jn(lo, i)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fmid = Jn(mid, i)
                if (fmid > 0) == (flo > 0):
                    lo, flo = mid, fmid
                else:
                    hi = mid
                if hi - lo < 1e-14 * max(1.0, abs(mid)):
                    break
            racines[j] = 0.5 * (lo + hi)
        points = racines.copy()
        zerosj[i][:k] = racines[:k]
    return zerosj


def spherical_bessel_formulas(n):
    """sympy 形式的球贝塞尔函数 j_0..j_{n-1}"""
    x = sym.symbols('x')
    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    """归一化/缩放后的球贝塞尔基 sympy 公式"""
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]
    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(
                normalizer[order][i] * f[order].subs(x, zeros[order, i] * x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l, m):
    from math import factorial
    return ((2 * l + 1) * factorial(l - abs(m)) / (4 * np.pi * factorial(l + abs(m)))) ** 0.5


def associated_legendre_polynomials(l, zero_m_only=True):
    z = sym.symbols('z')
    P_l_m = [[0] * (j + 1) for j in range(l)]
    P_l_m[0][0] = 1
    if l > 0:
        P_l_m[1][0] = z
        for j in range(2, l):
            P_l_m[j][0] = sym.simplify(
                ((2 * j - 1) * z * P_l_m[j - 1][0] - (j - 1) * P_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, l):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < l:
                    P_l_m[i + 1][i] = sym.simplify((2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, l):
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] - (i + j - 1) * P_l_m[j - 2][i]) / (j - i))
    return P_l_m


def real_sph_harm(l, zero_m_only=True, spherical_coordinates=True):
    """官方 real_sph_harm 的移植（DimeNet 仅用 m=0 的实球谐）"""
    P_l_m = associated_legendre_polynomials(l, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if not isinstance(P_l_m[i][j], int):
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
    Y_func_l_m = [['0'] * (2 * j + 1) for j in range(l)]
    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])
    return Y_func_l_m

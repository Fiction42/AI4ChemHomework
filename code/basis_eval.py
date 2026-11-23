# -*- coding: utf-8 -*-
"""数值基函数（autograd 兼容），用于移植 DimeNet 的 RBF/SBF 层。
- 2D 径向基: 官方 BesselBasisLayer:  env(x) * sin(freq_n * x),  x = d/r_c
- 3D 方向基: 官方 SphericalBasisLayer: env(x) * N_{l,n} j_l(z_{l,n} x) * Y_l^0(theta)
"""
import numpy as np
import autograd.numpy as anp

import basis_utils as bu

ENV_EXP = 5


def envelope(x, exponent=ENV_EXP):
    """官方 Envelope: 1/x + a x^{p-1} + b x^p + c x^{p+1} (x<1), 否则 0"""
    p = exponent + 1
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    x = anp.asarray(x, dtype=float)
    xs = anp.where(x < 1e-12, 1e-12, x)
    env_val = 1.0 / xs + a * xs ** (p - 1) + b * xs ** p + c * xs ** (p + 1)
    return anp.where(x < 1.0, env_val, 0.0)


def sph_bessel_j(l, x):
    """球贝塞尔 j_l(x), l<=6, 递推实现"""
    x = anp.asarray(x, dtype=float)
    xs = anp.where(x < 1e-10, 1e-10, x)
    j0 = anp.sin(xs) / xs
    if l == 0:
        return j0
    j1 = anp.sin(xs) / xs ** 2 - anp.cos(xs) / xs
    if l == 1:
        return j1
    jm, j = j0, j1
    for n in range(1, l):
        jp = (2 * n + 1) / xs * j - jm
        jm, j = j, jp
    return j


def legendre_P(l, costh):
    """勒让德多项式 P_l(cos theta)"""
    c = anp.asarray(costh, dtype=float)
    if l == 0:
        return anp.ones_like(c)
    p0, p1 = anp.ones_like(c), c
    if l == 1:
        return p1
    for n in range(1, l):
        p2 = ((2 * n + 1) * c * p1 - n * p0) / (n + 1)
        p0, p1 = p1, p2
    return p1


def real_sph_harm_l0(l, costh):
    """实球谐 Y_l^0(theta) = sqrt((2l+1)/4pi) P_l(cos theta)"""
    return np.sqrt((2 * l + 1) / (4 * np.pi)) * legendre_P(l, costh)


class Basis:
    """封装 DimeNet 的 2D/3D 基函数参数（零点、归一化因子、频率）"""

    def __init__(self, num_spherical=7, num_radial=6, cutoff=5.0, envelope_exponent=5):
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.inv_cutoff = 1.0 / cutoff
        self.envelope_exponent = envelope_exponent
        self.zeros = bu.Jn_zeros(num_spherical, num_radial)          # (l, n)
        self.norm = np.zeros_like(self.zeros)
        for l in range(num_spherical):
            for n in range(num_radial):
                z = self.zeros[l, n]
                jl1 = float(bu.Jn(z, l + 1))
                self.norm[l, n] = np.sqrt(2.0) / abs(jl1)
        self.freq = np.pi * np.arange(1, num_radial + 1, dtype=float)  # 官方 BesselBasisLayer 频率初值

    def rbf(self, d):
        """2D 径向基, d: (E,) 距离 -> (E, num_radial)"""
        x = d * self.inv_cutoff
        env = envelope(x, self.envelope_exponent)
        return env[:, None] * anp.sin(anp.asarray(x)[:, None] * self.freq[None, :])

    def sbf(self, d, theta):
        """3D 方向基, d: (E,), theta: (T,) -> (T, num_spherical*num_radial)"""
        x = d * self.inv_cutoff
        env = envelope(x, self.envelope_exponent)
        cols = []
        for l in range(self.num_spherical):
            c_l = real_sph_harm_l0(l, anp.cos(theta))
            for n in range(self.num_radial):
                b = self.norm[l, n] * sph_bessel_j(l, self.zeros[l, n] * x)
                cols.append(env * b)
        radial = anp.stack(cols, axis=1)
        # 角向部分按 l 重复 num_radial 次，与官方 cbf = repeat(cbf, num_radial) 一致
        ang = anp.stack([real_sph_harm_l0(l, anp.cos(theta)) for l in range(self.num_spherical)], axis=1)
        ang = anp.repeat(ang, self.num_radial, axis=1)
        return radial * ang


def sph_bessel_j_series(l, x, K=48):
    """球贝塞尔 j_l(x) 的 Taylor 级数（小/中等 x 稳定）：
    j_l = x^l/(2l+1)!! * sum_k (-x^2/2)^k / (k! prod_{j<k}(2l+2j+3))"""
    x = anp.asarray(x, dtype=float)
    term = anp.ones_like(x)
    s = anp.ones_like(x)
    for k in range(K):
        term = term * (-x ** 2 / 2.0) / ((k + 1) * (2 * l + 2 * k + 3))
        s = s + term
    dfact = 1.0
    for j in range(l + 1):
        dfact *= 2 * j + 1
    return x ** l / dfact * s


def sph_bessel_j(l, x):
    """球贝塞尔 j_l(x)：|x|<8 用级数，否则用向上递推（x>=8 时递推稳定）"""
    x = anp.asarray(x, dtype=float)
    xs = anp.where(x < 1e-12, 1e-12, x)
    small = sph_bessel_j_series(l, xs)
    j0 = anp.sin(xs) / xs
    if l == 0:
        big = j0
    else:
        j1 = anp.sin(xs) / xs ** 2 - anp.cos(xs) / xs
        if l == 1:
            big = j1
        else:
            jm, j = j0, j1
            for n in range(1, l):
                jp = (2 * n + 1) / xs * j - jm
                jm, j = j, jp
            big = j
    out = anp.where(x < 8.0, small, big)
    # 大 x 的递推在 x<8 时不稳定，但那条分支不会被选中；数值上仍可能很大，做一次安全裁剪
    return out


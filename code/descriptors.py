# -*- coding: utf-8 -*-
"""分子表示描述符：2体（原子对距离）类与 3体（角度）类。"""
import itertools
import numpy as np


def distances(X):
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


def pair_list(X):
    n = len(X)
    return [distances(X)[i, j] for i, j in itertools.combinations(range(n), 2)]


# ---------- 2 体不变描述符（只依赖原子对距离的多重集） ----------
def rbf_pair_sum(X, centers=np.linspace(0.5, 5.0, 40), sigma=0.3):
    r = np.array(pair_list(X))
    return np.exp(-0.5 * ((r[:, None] - centers[None, :]) / sigma) ** 2).sum(0)


def inverse_power_pair_sum(X, powers=(1, 2, 3, 4, 6)):
    r = np.array(pair_list(X))
    return np.array([np.sum(r ** (-p)) for p in powers])


def distance_multiset(X):
    return np.sort(pair_list(X))


def coulomb_matrix(X, Z):
    n = len(X)
    d = distances(X)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = Z[i] * Z[j] / d[i, j]
    return M


def coulomb_eigs(X, Z):
    return np.sort(np.linalg.eigvalsh(coulomb_matrix(X, Z)))


# ---------- 3 体/角度描述符 ----------
def angle_list(X):
    n = len(X)
    out = []
    for i in range(n):
        others = [a for a in range(n) if a != i]
        for j, k in itertools.combinations(others, 2):
            v1 = X[j] - X[i]
            v2 = X[k] - X[i]
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            out.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.sort(out)


def bp_radial(X, etas=(0.05, 0.2, 0.5, 1.0, 2.0), rs=(0.0, 0.5, 1.0, 1.5), rc=5.0):
    """Behler-Parrinello 径向对称函数 G^2, 返回 (n_atoms, n_eta*n_rs)"""
    d = distances(X)
    n = len(X)
    fc = np.where(d < rc, 0.5 * (1 + np.cos(np.pi * d / rc)), 0.0)
    feats = []
    for eta in etas:
        for r_s in rs:
            g = np.sum(np.exp(-eta * (d - r_s) ** 2) * fc, axis=1)
            feats.append(g)
    return np.stack(feats, axis=1)


def bp_angular(X, zetas=(1.0, 4.0), lambdas=(1.0, -1.0), rc=5.0):
    """Behler-Parrinello 角向对称函数 G^4"""
    n = len(X)
    d = distances(X)
    fc = np.where(d < rc, 0.5 * (1 + np.cos(np.pi * d / rc)), 0.0)
    out = []
    for zeta in zetas:
        for lam in lambdas:
            g = np.zeros(n)
            for i in range(n):
                s = 0.0
                others = [a for a in range(n) if a != i]
                for j, k in itertools.combinations(others, 2):
                    v1 = X[j] - X[i]
                    v2 = X[k] - X[i]
                    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    base = 1 + lam * c
                    if abs(base) < 1e-12:
                        continue
                    s += (base ** zeta) * fc[i, j] * fc[i, k] * np.exp(-(d[i, j] ** 2 + d[i, k] ** 2))
                g[i] = 2.0 ** (1 - zeta) * s
            out.append(g)
    return np.stack(out, axis=1)

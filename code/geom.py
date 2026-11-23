# -*- coding: utf-8 -*-
"""几何工具：距离几何重建、Kabsch 叠合、键角、Cayley-Menger。纯 numpy。"""
import itertools
import numpy as np


def reconstruct(D):
    """由 4x4 距离矩阵解析重建三维坐标 (4,3)，并精确复现所有 d_ij。"""
    D = np.asarray(D, dtype=float)
    assert D.shape == (4, 4)
    r01, r02, r03 = D[0, 1], D[0, 2], D[0, 3]
    r12, r13, r23 = D[1, 2], D[1, 3], D[2, 3]
    x0 = np.zeros(3)
    x1 = np.array([r01, 0.0, 0.0])
    cx = (r02 ** 2 + r01 ** 2 - r12 ** 2) / (2 * r01)
    cy2 = r02 ** 2 - cx ** 2
    if cy2 < -1e-9:
        raise ValueError("三角不等式不满足 (atom 0,1,2)")
    x2 = np.array([cx, np.sqrt(max(cy2, 0.0)), 0.0])
    ax = (r03 ** 2 + r01 ** 2 - r13 ** 2) / (2 * r01)
    ay = (r03 ** 2 - r23 ** 2 + cx ** 2 + x2[1] ** 2 - 2 * ax * cx) / (2 * x2[1])
    az2 = r03 ** 2 - ax ** 2 - ay ** 2
    if az2 < -1e-7:
        raise ValueError("三维不可实现 (Cayley-Menger < 0): z^2 = %g" % az2)
    x3 = np.array([ax, ay, np.sqrt(max(az2, 0.0))])
    X = np.vstack([x0, x1, x2, x3])
    err = np.abs(dist_matrix(X) - D).max()
    if err > 1e-9:
        raise ValueError("重建误差过大 %.2e" % err)
    return X


def mirror(X, atom=3):
    Y = X.copy()
    Y[:, 2] = 0.0
    Y[atom, 2] = np.linalg.norm(X[atom] - Y[atom])
    return Y


def dist_matrix(X):
    X = np.asarray(X, dtype=float)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def cm_det(D):
    n = D.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[0, 1:] = 1.0
    M[1:, 0] = 1.0
    M[1:, 1:] = D ** 2
    return float(np.linalg.det(M))


def cm_volume(D):
    return np.sqrt(max(cm_det(D) / 288.0, 0.0))


def kabsch_rmsd(P, Q):
    Pc = np.asarray(P, float) - np.mean(P, 0)
    Qc = np.asarray(Q, float) - np.mean(Q, 0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    Pr = (Rm @ Pc.T).T
    return float(np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))))


def unlabeled_rmsd(X, Y):
    best = (1e9, None)
    for perm in itertools.permutations(range(len(X))):
        r = kabsch_rmsd(X[list(perm)], Y)
        if r < best[0]:
            best = (r, perm)
    return best


def bond_angles(X):
    n = len(X)
    out = []
    for i in range(n):
        others = [a for a in range(n) if a != i]
        for j, k in itertools.combinations(others, 2):
            v1 = X[j] - X[i]
            v2 = X[k] - X[i]
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            out.append((j, i, k, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))))
    return out

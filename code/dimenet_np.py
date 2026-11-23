# -*- coding: utf-8 -*-
"""DimeNet (Gasteiger et al., ICLR 2020) 的 NumPy/autograd 移植。
结构与官方 TF 实现 (dimenet/model/*.py) 逐层对应，用 autograd 自动求导替代 TF 反向传播。"""
import numpy as np
import autograd.numpy as anp

import basis_eval as be

EMBED_MAX_Z = 95


def swish(x):
    return x / (1.0 + anp.exp(-x))


def glorot_orthogonal(rng, shape, scale=2.0):
    """GlorotOrthogonal 初始化（QR 正交 + sqrt(2/(fan_in+fan_out)) 缩放）。
    官方实现对非方阵直接取 reduced QR 的 q，形状会退化；这里按 fan_in/fan_out
    的大小关系选转置，保证返回形状为 shape。"""
    fan_in, fan_out = shape[0], shape[-1]
    if len(shape) == 2:
        if fan_in >= fan_out:
            A = rng.normal(size=(fan_in, fan_out))
            Q, R = np.linalg.qr(A)
            Q = Q * np.sign(np.diag(R))[None, :]
            W = Q
        else:
            A = rng.normal(size=(fan_out, fan_in))
            Q, R = np.linalg.qr(A)
            Q = Q * np.sign(np.diag(R))[None, :]
            W = Q.T
        return np.asarray(W * np.sqrt(scale / (fan_in + fan_out)), dtype=float)
    N = int(np.prod(shape))
    W = rng.normal(size=(N, 1))
    W = W / np.linalg.norm(W)
    return np.asarray(W.reshape(shape) * np.sqrt(scale / (2.0 * fan_in)), dtype=float)


def one_hot(idx, n):
    idx = np.asarray(idx, dtype=int)
    M = np.zeros((n, len(idx)))
    for e, i in enumerate(idx):
        M[i, e] = 1.0
    return M


class Graph:
    """邻接/三元组索引，对应官方 data provider 产生的输入张量"""

    def __init__(self, Z, R, cutoff):
        self.Z = np.asarray(Z, dtype=int)
        self.R = np.asarray(R, dtype=float)
        n = len(self.Z)
        self.n_atoms = n
        d = np.linalg.norm(self.R[:, None, :] - self.R[None, :, :], axis=-1)
        self.D = d
        nbr = [[j for j in range(n) if j != i and d[i, j] < cutoff] for i in range(n)]
        self.nbr = nbr
        edges_i, edges_j = [], []
        edge_id = {}
        for i in range(n):
            for j in nbr[i]:
                edge_id[(i, j)] = len(edges_i)
                edges_i.append(i)
                edges_j.append(j)
        self.idnb_i = np.array(edges_i)
        self.idnb_j = np.array(edges_j)
        self.edge_id = edge_id
        self.Dij = d[self.idnb_i, self.idnb_j]
        # 三元组 (k, j, i): j 为中心, 消息 m_kj -> m_ji
        t_k, t_j, t_i = [], [], []
        for i in range(n):
            for j in nbr[i]:
                for k in nbr[j]:
                    if k == i:
                        continue
                    t_k.append(k)
                    t_j.append(j)
                    t_i.append(i)
        self.t_k = np.array(t_k, dtype=int)
        self.t_j = np.array(t_j, dtype=int)
        self.t_i = np.array(t_i, dtype=int)
        # 角度 theta_kji: 顶点在 j, 两条边 jk 与 ji
        v1 = self.R[self.t_k] - self.R[self.t_j]
        v2 = self.R[self.t_i] - self.R[self.t_j]
        cosang = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-30)
        self.theta = np.arccos(np.clip(cosang, -1.0, 1.0))
        # 索引映射
        self.edge_kj = np.array([edge_id[(j, k)] for k, j in zip(t_k, t_j)], dtype=int)
        self.edge_ji = np.array([edge_id[(j, i)] for i, j in zip(t_i, t_j)], dtype=int)
        self.d_kj = d[self.t_k, self.t_j]
        self.n_edges = len(edges_i)
        self.n_triplets = len(t_k)
        self._M_edge = one_hot(self.idnb_i, n)          # (n, E)
        self._M_triplet = one_hot(self.edge_ji, self.n_edges)  # (E, T)

    def segment_sum_edges(self, x):
        return self._M_edge @ x

    def segment_sum_triplets(self, x):
        return self._M_triplet @ x


class DimeNet:
    def __init__(self, emb_size=128, num_blocks=6, num_bilinear=8, num_spherical=7,
                 num_radial=6, cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                 num_after_skip=2, num_dense_output=3, num_targets=1, seed=0):
        self.cfg = dict(emb_size=emb_size, num_blocks=num_blocks, num_bilinear=num_bilinear,
                        num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
                        envelope_exponent=envelope_exponent, num_before_skip=num_before_skip,
                        num_after_skip=num_after_skip, num_dense_output=num_dense_output,
                        num_targets=num_targets)
        self.basis = be.Basis(num_spherical, num_radial, cutoff, envelope_exponent)
        rng = np.random.default_rng(seed)
        E, B = emb_size, num_bilinear
        P = {}
        P["emb.embeddings"] = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(EMBED_MAX_Z, E))
        P["emb.W"] = glorot_orthogonal(rng, (3 * E, E))
        P["emb.b"] = np.zeros(E)
        P["emb.Wrbf"] = glorot_orthogonal(rng, (num_radial, E))
        P["emb.brbf"] = np.zeros(E)
        for blk in range(num_blocks):
            p = f"int{blk}."
            P[p + "Wji"] = glorot_orthogonal(rng, (E, E)); P[p + "bji"] = np.zeros(E)
            P[p + "Wkj"] = glorot_orthogonal(rng, (E, E)); P[p + "bkj"] = np.zeros(E)
            P[p + "Wrbf"] = glorot_orthogonal(rng, (num_radial, E))
            P[p + "Wsbf"] = glorot_orthogonal(rng, (num_spherical * num_radial, B))
            P[p + "Wbilin"] = rng.normal(0.0, 2.0 / E, size=(E, B, E))
            for name, count in (("before", num_before_skip), ("after", num_after_skip)):
                for r in range(count):
                    P[f"{p}{name}{r}.W1"] = glorot_orthogonal(rng, (E, E))
                    P[f"{p}{name}{r}.b1"] = np.zeros(E)
                    P[f"{p}{name}{r}.W2"] = glorot_orthogonal(rng, (E, E))
                    P[f"{p}{name}{r}.b2"] = np.zeros(E)
            P[p + "final.W"] = glorot_orthogonal(rng, (E, E)); P[p + "final.b"] = np.zeros(E)
        for ob in range(num_blocks + 1):
            p = f"out{ob}."
            P[p + "Wrbf"] = glorot_orthogonal(rng, (num_radial, E))
            for d in range(num_dense_output):
                P[p + f"dense{d}.W"] = glorot_orthogonal(rng, (E, E))
                P[p + f"dense{d}.b"] = np.zeros(E)
            P[p + "Wfinal"] = glorot_orthogonal(rng, (E, num_targets))
        self.params = P

    # ---------- 前向 ----------
    def _dense(self, P, name, x, bias=True):
        out = anp.dot(x, P[name + ".W"]) if x.shape[-1] == P[name + ".W"].shape[0] \
            else anp.dot(x, P[name + ".W"].T)
        return out + P[name + ".b"] if bias else out

    def forward(self, g: Graph, params=None, return_hidden=False):
        P = self.params if params is None else params
        c = self.cfg
        Z, R = g.Z, g.R
        rbf = self.basis.rbf(g.Dij)                       # (E, num_radial)
        sbf = self.basis.sbf(g.d_kj, g.theta)             # (T, num_spherical*num_radial)
        # embedding block
        rbf_e = swish(anp.dot(rbf, P["emb.Wrbf"]) + P["emb.brbf"])
        x_i = P["emb.embeddings"][Z[g.idnb_i]]
        x_j = P["emb.embeddings"][Z[g.idnb_j]]
        x = swish(anp.dot(anp.concatenate([x_i, x_j, rbf_e], axis=-1), P["emb.W"]) + P["emb.b"])
        # output block 0
        y = self._output(P, "out0.", x, rbf, g)
        for blk in range(c["num_blocks"]):
            x = self._interaction(P, f"int{blk}.", x, rbf, sbf, g)
            y = y + self._output(P, f"out{blk + 1}.", x, rbf, g)
        energy = anp.sum(y)
        if return_hidden:
            # 逐原子池化表示: 把边级消息按目标原子求和 (DimeNet 的 per-atom embedding)
            h_atom = g.segment_sum_edges(x)
            return energy, h_atom, y
        return energy

    def _output(self, P, p, x, rbf, g):
        g_ = anp.dot(rbf, P[p + "Wrbf"])
        x = g_ * x
        x = g.segment_sum_edges(x)
        for d in range(self.cfg["num_dense_output"]):
            x = swish(anp.dot(x, P[p + f"dense{d}.W"]) + P[p + f"dense{d}.b"])
        return anp.dot(x, P[p + "Wfinal"])

    def _interaction(self, P, p, x, rbf, sbf, g):
        x_ji = swish(anp.dot(x, P[p + "Wji"]) + P[p + "bji"])
        x_kj = swish(anp.dot(x, P[p + "Wkj"]) + P[p + "bkj"])
        g_ = anp.dot(rbf, P[p + "Wrbf"])
        x_kj = x_kj * g_
        sbf_ = anp.dot(sbf, P[p + "Wsbf"])
        x_kj = x_kj[g.edge_kj]
        x_kj = anp.einsum("wj,wl,ijl->wi", sbf_, x_kj, P[p + "Wbilin"])
        x_kj = g.segment_sum_triplets(x_kj)
        x2 = x_ji + x_kj
        for r in range(self.cfg["num_before_skip"]):
            x2 = x2 + swish(anp.dot(swish(anp.dot(x2, P[f"{p}before{r}.W1"]) + P[f"{p}before{r}.b1"]),
                                    P[f"{p}before{r}.W2"]) + P[f"{p}before{r}.b2"])
        x2 = swish(anp.dot(x2, P[p + "final.W"]) + P[p + "final.b"])
        x = x + x2
        for r in range(self.cfg["num_after_skip"]):
            x = x + swish(anp.dot(swish(anp.dot(x, P[f"{p}after{r}.W1"]) + P[f"{p}after{r}.b1"]),
                                  P[f"{p}after{r}.W2"]) + P[f"{p}after{r}.b2"])
        return x

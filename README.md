# Chapter 3 作业：分子表示完备性与振动光谱到结构

本仓库是两个实验的代码与结果：

1. **作业 1：距离几何与分子表示的完备性** —— 把 DimeNet（ICLR 2020）的官方 TensorFlow 实现逐层移植为
   NumPy/autograd 版本，用两个“原子对距离多重集相同、但键角不同”的四原子案例检验其表示的完备性，
   并与 2 体描述符（距离多重集、RDF、2 体 ACE 项等）对比。
2. **作业 2：振动光谱到分子结构的统一深度学习框架** —— 复现 Vib2Mol（*Chem. Sci.* 2026, 17, 20395；
   arXiv:2503.07014）的官方代码与权重，在 ViBench SDBS 基准上复现论文指标，并在一个文献外的独立实验 IR
   数据集（Hugging Face: `Lamblador/IRSpectra`）上做跨库测试、案例剖析与改进实验。

## 目录结构

```
report/    作业 1 报告（LaTeX 源 + PDF）
report2/   作业 2 报告（LaTeX 源 + PDF）
code/      全部实验代码
results/   实验输出（JSON/CSV/日志），报告中每个数字可追溯
figures/   作业 1 作图数据
figs2/     作业 2 作图数据
scripts/   环境搭建脚本
```

## 主要结果

### 作业 1

| 项目 | 结果 |
|---|---|
| 两案例结构 | 距离多重集完全相同；最小不可标号 RMSD 0.311 Å，体积差 13%，B3LYP 能量差 24.6 mHa = 15.44 kcal/mol |
| 2 体多重集型描述符 | 两案例上严格相等（相对差 ≤ 3e-16）→ 无法区分，误差下界 7.72 kcal/mol（最大误差）/ 10.92 kcal/mol（RMSE） |
| DimeNet 表示 | 可区分（相对分离度 1.6e-3，高于数值噪声 19 个数量级），但处于随机背景分布第 1.1 百分位 |
| 显式 3 体特征 | 角度多重集相对差 6.5%，BP 角向对称函数全局和 53% |
| 可学习性 | 2 体特征模型在两案例上预测差 4.4e-16（特征碰撞），3 体特征恢复约 71% 能量差；小规模 DimeNet 端到端训练成功 |

### 作业 2

| 任务 / 模型 | 论文 Recall@1 | 本仓库复现 |
|---|---|---|
| SDBS-IR（Vib2Mol） | 86.17% | **86.53%** |
| SDBS-IR（Vib2Mol-MMM） | 93.26% | **93.97%** |
| SDBS-Raman（Vib2Mol） | 90.43% | **90.78%** |

- 关键复现细节：论文指标为“初筛 → 化学式过滤 → 匹配头重排”三段式；仅初筛时同一权重 SDBS-IR Recall@1 只有 66.31%。
- 文献外数据集（1786 化合物）上 Recall@1 仅 2.41%；同分子跨库嵌入余弦 0.376，定位为峰形/基线等实验域偏移。
- 改进：化学式全局过滤 90.46%（候选库仅 1786，需谨慎解读）；少样本投影头域自适应 2.70% → 9.79%（3.6 倍）。

## 复现说明

### 数据与权重（未包含在仓库中）

- 作业 1：剧本为 `code/validate_basis.py`、`verify_cases.py`、`exp2_separation.py`、`exp3_learning.py`、
  `exp4_robustness.py`、`train_dimenet.py`，仅需 NumPy + autograd + sympy/mpmath。
- 作业 2：ViBench 数据与 Vib2Mol 权重较大，需自行下载：
  - 数据：`hf download xinyulu/vibench --repo-type=dataset --local-dir datasets/vibench`
    （本实验使用 `sdbs/` 子集：train 2393 / eval 140 / test 282）
  - 权重：`hf download xinyulu/vib2mol --local-dir checkpoints`
  - 独立测试集：Hugging Face `Lamblador/IRSpectra` 的 test split
- 第三方代码（官方 DimeNet / Vib2Mol 仓库）未纳入本仓库，请从其 GitHub 获取。

### 环境

作业 1 依赖：Python 3.12、NumPy、autograd、sympy、mpmath。

作业 2 另需 PyTorch（CPU 即可）、rdkit、transformers==4.44.2、lmdb、timm、pyarrow。
在 Apple Silicon 上，若运行时的 Python 带代码签名/库验证限制，pip 二进制扩展无法直接加载；
本仓库的 `scripts/setup_env.sh` 给出了“对解释器副本重新签名并禁用库验证”的做法。

```bash
bash scripts/setup_env.sh            # 生成 pyrenv/python3 与 pylibs/ 目录
PYTHONPATH=$PWD/pylibs ./pyrenv/python3 code/exp2_separation.py
```

### 官方复现命令（作业 2）

```bash
cd <Vib2Mol 官方仓库>
python scripts/infer_retrieval.py --model vib2mol --ds sdbs --spectral-types ir \
  --test-model-path <ckpt> --rerank --topk 5 --rank-model-path <ckpt> --use-formula
```

## 引用

- J. Gasteiger, J. Groß, S. Günnemann. *Directional Message Passing for Molecular Graphs.* ICLR 2020.
- X. Lu et al. *Vib2Mol: from vibrational spectra to molecular structures—a unified deep learning framework.*
  Chem. Sci. 2026, 17, 20395–20405. arXiv:2503.07014.
- SDBS: Spectral Database for Organic Compounds, AIST. NIST Chemistry WebBook.

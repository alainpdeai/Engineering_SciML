# -*- coding: utf-8 -*-
"""M1.5 解析基准库：矩形膜本征值与模态。

V3 §6 给 M1.5 的核心实验是「矩形膜解析模态谱」，本文件是该基准的**单一来源**：

- 解析谱        —— $\\lambda_{mn} = \\pi^2\\left[(m/a)^2+(n/b)^2\\right]$
- 模态函数      —— $\\phi_{mn}(x,y)=\\sin(m\\pi x/a)\\sin(n\\pi y/b)$
- 正方形简并    —— $a=b$ 时 $\\lambda_{12}=\\lambda_{21}$（重根，子空间可任意旋转）
- 自由边界零模  —— Neumann 下 $\\lambda_1\\approx0$ 是刚体平移模，不是基频

**边界与用途**：本模块只提供**解析基准**，不含离散求解。离散侧（Q1 双线性有限元）
的唯一实现在 ``tests/test_membrane_spectrum.py``；``code/solution.py`` 直接复用它，
三处共用同一套数字，避免各自漂移。

自检：``python benchmarks/rectangular_membrane.py``
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rectangular_spectrum",
    "analytic_lambda",
    "mode_function",
    "degeneracy_pairs",
    "m_orthonormal_modes",
    "is_square",
]

# 圆周率，避免到处硬编码
PI = np.pi


def rectangular_spectrum(a: float = 1.0, b: float = 1.0, m_max: int = 3, n_max: int = 3):
    """矩形 $[0,a]\\times[0,b]$ 四边 Dirichlet 下的前若干阶解析本征值。

    返回 ``[(lambda, m, n), ...]``，按 $\\lambda$ 升序。
    """
    if a <= 0 or b <= 0:
        raise ValueError("边长 a、b 必须为正")
    if m_max < 1 or n_max < 1:
        raise ValueError("m_max、n_max 至少为 1")
    rows = []
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            rows.append((analytic_lambda(m, n, a, b), m, n))
    return sorted(rows)


def analytic_lambda(m: int, n: int, a: float = 1.0, b: float = 1.0) -> float:
    """第 $(m,n)$ 阶解析本征值 $\\lambda_{mn}=\\pi^2[(m/a)^2+(n/b)^2]$。"""
    if m < 1 or n < 1:
        raise ValueError("模态序号 m、n 从 1 起")
    return PI ** 2 * ((m / a) ** 2 + (n / b) ** 2)


def mode_function(x, y, m: int, n: int, a: float = 1.0, b: float = 1.0):
    """模态振型 $\\phi_{mn}(x,y)=\\sin(m\\pi x/a)\\sin(n\\pi y/b)$。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sin(m * PI * x / a) * np.sin(n * PI * y / b)


def is_square(a: float, b: float, rtol: float = 1e-12) -> bool:
    """是否正方形（正方形才有 $\\lambda_{12}=\\lambda_{21}$ 的简并）。"""
    return abs(a - b) <= rtol * max(a, b)


def degeneracy_pairs(a: float, b: float, m_max: int = 4, n_max: int = 4, rtol: float = 1e-9):
    """返回解析上简并的模态对：``[((m1,n1),(m2,n2)), ...]``。

    简并的直接后果：重根的特征向量只在二维子空间内有意义，
    数值解返回哪一组基是任意的——**逐点比较振型没有意义**（F4）。
    """
    pairs = []
    rows = rectangular_spectrum(a, b, m_max, n_max)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            l1, m1, n1 = rows[i]
            l2, m2, n2 = rows[j]
            if abs(l1 - l2) <= rtol * max(l1, l2):
                pairs.append(((m1, n1), (m2, n2)))
    return pairs


def m_orthonormal_modes(modes, a: float = 1.0, b: float = 1.0, n_quad: int = 400):
    """$M$-归一化的模态矩阵 $Q$（$q_i^TMq_j=\\delta_{ij}$，$M$ 为一致质量矩阵）。

    归一化口径必须先声明：只有 $M$-归一化下 $Q^TMQ\\approx I$ 才成立；
    若改用欧氏范数，$Q^TMQ$ 的**对角**会偏离 1（坏的是对角，非对角仍正交），
    且只在**跨整个谱**抽样时才暴露。

    注意：算 Gram 矩阵时必须带求积权重 $hx\\cdot hy$，
    即 $Q^TMQ=\\sum_{ij}q_iq_j\\,hx\\,hy$；直接用 $QQ^T$ 会差一个网格点数因子。
    """
    modes = list(modes)
    x = (np.arange(n_quad) + 0.5) * a / n_quad
    y = (np.arange(n_quad) + 0.5) * b / n_quad
    X, Y = np.meshgrid(x, y, indexing="ij")
    Q = np.empty((len(modes), n_quad * n_quad))
    for k, (m, n) in enumerate(modes):
        phi = mode_function(X, Y, m, n, a, b).ravel()
        Q[k] = phi / np.sqrt(np.sum(phi ** 2) * (a / n_quad) * (b / n_quad))
    return Q


if __name__ == "__main__":
    a, b = 1.0, 1.2
    print("[M1.5 解析基准自检]")
    print(f"  矩形 {a} x {b}，前 4 阶解析谱：")
    for lam, m, n in rectangular_spectrum(a, b, 3, 3)[:4]:
        print(f"    (m,n)=({m},{n})  lambda={lam:.8f}  omega={np.sqrt(lam):.6f}")
    lam11 = analytic_lambda(1, 1, a, b)
    print(f"  lambda_11 = {lam11:.8f}")

    print("  简并检查（正方形 a=b=1）：")
    for p in degeneracy_pairs(1.0, 1.0, 3, 3):
        print(f"    lambda({p[0]}) == lambda({p[1]})")
    print("  简并检查（长方形 a=1.0, b=1.2，应为空）：", degeneracy_pairs(1.0, 1.2, 3, 3))

    Q = m_orthonormal_modes([(1, 1), (2, 1), (1, 2), (3, 2)], a, b, n_quad=400)
    # Gram 矩阵必须带求积权重：q_i^T M q_j = Σ_ij q_i q_j · hx·hy
    w = (a / 400) * (b / 400)
    G = (Q * w) @ Q.T
    print(f"  M-归一化下 Q^T M Q 与 I 的最大偏差 = {np.max(np.abs(G - np.eye(len(Q)))):.3e}")

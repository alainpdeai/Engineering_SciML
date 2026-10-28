# -*- coding: utf-8 -*-
"""M2.1 环境与算法自检：五点格式稀疏矩阵 + 制造解收敛（网页运行与「下载」同源）。

用法：
    python code/env_check.py          # 本地终端直接跑
    网页「运行」按钮                    # Pyodide 内运行同一份源码

自检三件套（对应演示页⑥「验证三件套」）：
    1. 解：与制造解 u* = sin(pi x) sin(pi y) 逐点对照
    2. 残差：‖A u - b‖（直接法应降到机器精度量级）
    3. 误差：真 L2 口径 sqrt(h^2 * sum(e^2)) 与 Linf，并给出收敛阶

口径备注（2026-10-16 统一）：
    * 本脚本与源稿④/⑤、CSV、tests/test_mms.py 一致，用「真 L2」范数；
      含端点 N×N 网格上真 L2/RMS=N/(N-1)=1+1/(N-1)，这个网格因子把 RMS 阶压低约 0.03，
      历史数字 1.97-1.99 即来源于此，双向留档于 diagnostics/M2.1_F4_cards.md。
    * 矩阵用 kron 内点版（5N^2 非零元、除 h^2）；与 code/solution.py 的
      全节点版 assemble_poisson_system 在内点块上逐元素一致，差异仅在边界行
      形式与 h^2 放缩（见源稿④「双装配口径」长注）。
"""
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve


def poisson_matrix(n):
    """内点五点格式：kron 构造 n^2 x n^2，除 h^2（源稿④口径）。"""
    h = 1.0 / (n + 1)
    T = diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    A = (kron(eye(n), T) + kron(T, eye(n))) / h ** 2
    return A.tocsr(), h


def manufactured_rhs(n):
    """制造解 u* = sin(pi x) sin(pi y)，右端 f = 2 pi^2 u*。"""
    h = 1.0 / (n + 1)
    x = np.arange(1, n + 1) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    return X, Y, 2 * np.pi ** 2 * u_exact, u_exact


def main():
    print("=" * 66)
    print("M2.1 自检：五点格式稀疏矩阵 + 制造解（MMS）收敛")
    print("=" * 66)
    prev = None
    for n in (15, 31, 63):
        A, h = poisson_matrix(n)
        X, Y, b_grid, u_ex = manufactured_rhs(n)
        u = spsolve(A, b_grid.ravel()).reshape(n, n)
        err_l2 = float(np.sqrt(h * h * np.sum((u - u_ex) ** 2)))
        err_inf = float(np.max(np.abs(u - u_ex)))
        residual = float(np.linalg.norm(A @ u.ravel() - b_grid.ravel()))
        order = float("nan") if prev is None else float(np.log2(prev / err_l2))
        print("n=%2d, h=%.4e, L2=%.4e, Linf=%.4e, residual=%.2e, p=%.3f"
              % (n, h, err_l2, err_inf, residual, order))
        prev = err_l2
    # 对称性体检：CG 类迭代法要求对称正定；这里顺带给出判据
    A31, _ = poisson_matrix(31)
    print("max|A-A^T| = %.1e（kron 内点版恒为 0；只划行不清列会变成 1.0，见 F4 卡二）"
          % float(np.abs(A31 - A31.T).max()))


if __name__ == "__main__":
    main()

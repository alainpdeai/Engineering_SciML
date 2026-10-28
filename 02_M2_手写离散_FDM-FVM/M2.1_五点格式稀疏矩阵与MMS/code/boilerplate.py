"""
M2.1 学生练习脚手架: 二维泊松方程五点差分稀疏矩阵组装
请根据提示补全带有 TODO 标记的代码区域
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def solve_poisson_2d_student(N=51, L=1.0):
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    
    # 制造真实源项: 对应真实解 sin(pi*x)*sin(pi*y)
    f_val = 2.0 * (np.pi**2) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    b = (dx**2) * f_val.flatten()
    
    total_nodes = N * N
    main_diag = 4.0 * np.ones(total_nodes)
    off_diag_x = -1.0 * np.ones(total_nodes - 1)
    off_diag_y = -1.0 * np.ones(total_nodes - N)
    
    A = sp.diags(
        [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y],
        [0, 1, -1, N, -N], shape=(total_nodes, total_nodes), format='lil'
    )
    
    # TODO 1: 遍历四周物理边界节点，对矩阵 A 划行划列置 1，并将右端项向量 b 对应分量置 0
    # -------------------------------------------------------------
    # 提示:
    # for i in range(N):
    #     for j in range(N):
    #         if i == 0 or i == N - 1 or j == 0 or j == N - 1:
    #             idx = i * N + j
    #             A[idx, :] = 0.0
    #             ...
    # -------------------------------------------------------------
    
    # TODO 2: 将 A 转换为 CSR 稀疏矩阵格式并调用 spla.spsolve 求解
    # u_vec = ...
    # return X, Y, u_vec.reshape((N, N))
    pass

# -*- coding: utf-8 -*-
"""
solution.py —— M1.2 定解条件与适定性标准参考求解器与数值实验套件

涵盖核心功能：
1. 1D 泊松/拉普拉斯方程四类边界装配与矩阵条件数/奇异性分析 (Dirichlet / Mixed / Neumann)
2. Robin 边界条件对流传热极限扫描 (支持一般 k、T_inf 参数并与解析解对照)
3. 逆热传导 (Backward Heat Equation) 算子不稳定性与噪声放大/谱截断实验
4. 纯 Neumann 边界制造解法 (MMS) 兼容性检验与二阶收敛阶验证
"""
import numpy as np


def assemble_1d_poisson(n=11, bc="dirichlet"):
    """
    装配一维负拉普拉斯算子 -d^2/dx^2 的差分矩阵 A
    
    参数:
        n: 离散网格节点数
        bc: 边界条件类型 ('dirichlet', 'mixed', 'neumann')
    返回:
        A: 离散微分算子矩阵 (已除以 dx^2)
        dx: 空间网格步长
    """
    dx = 1.0 / (n - 1)
    A = np.zeros((n, n))
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = -1.0, 2.0, -1.0
        
    if bc != "neumann":
        A[0, 0] = 1.0                  # 左端 Dirichlet
    else:
        A[0, 0], A[0, 1] = 1.0, -1.0  # 左端纯 Neumann
        
    if bc == "dirichlet":
        A[-1, -1] = 1.0                # 右端 Dirichlet
    else:
        A[-1, -2], A[-1, -1] = -1.0, 1.0 # 右端 Neumann
        
    return A / dx**2, dx


def solve_robin_convection(n=51, k=1.0, T_inf=0.0, h=1.0):
    """
    求解带左端 Dirichlet 与右端 Robin 对流换热边界的一维稳态热传导:
        -k u''(x) = 0, x in (0, 1)
        u(0) = 1.0,  -k u'(1) = h * (u(1) - T_inf)
    解析解:
        u(x) = 1 - h * (1 - T_inf) / (k + h) * x
        u(1) = (k * 1 + h * T_inf) / (k + h)
    """
    dx = 1.0 / (n - 1)
    x = np.linspace(0.0, 1.0, n)
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = -1.0, 2.0, -1.0
        
    A[0, 0] = 1.0
    b[0] = 1.0
    A[-1, -2], A[-1, -1] = -k / dx, k / dx + h
    b[-1] = h * T_inf
    
    u = np.linalg.solve(A, b)
    exact_right = (k * 1.0 + h * T_inf) / (k + h)
    return x, u, exact_right


def backward_heat_noise_amplification(n=64, sigma=1e-3, seed=0, n_modes=8, rng=None):
    """
    逆热传导 (Backward Heat) 高频噪声放大与 Hadamard 不稳定性验证:
    通过离散导数算子 L 演示解对观测噪声的剧烈敏感性以及低通模态截断稳定化。
    默认按 seed 创建随机流；传入 rng 时使用调用者提供的共享随机流。
    """
    dx = 1.0 / (n + 1)
    x = np.linspace(dx, 1 - dx, n)
    L = (np.diag(2 * np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)) / dx**2
    u_true = np.sin(np.pi * x)
    q_true = L @ u_true
    w, V = np.linalg.eigh(L)
    if rng is None:
        rng = np.random.default_rng(seed)
    cond_L = float(np.linalg.cond(L))
    
    u_obs = u_true + sigma * rng.standard_normal(n)
    ru = float(np.linalg.norm(u_obs - u_true) / np.linalg.norm(u_true))
    rq = float(np.linalg.norm(L @ u_obs - q_true) / np.linalg.norm(q_true))
    
    # 截断低通滤波稳定化 (只保留前 n_modes 个特征模态)
    qc = L @ (V[:, :n_modes] @ (V.T @ u_obs)[:n_modes])
    rc = float(np.linalg.norm(qc - q_true) / np.linalg.norm(q_true))
    
    return {
        "cond_L": cond_L,
        "amplification": rq / ru,
        "direct_err": rq,
        "truncated_err": rc
    }


def solve_neumann_mms(n=6, fix_ref=True):
    """
    纯 Neumann 边界制造解法 (MMS):
        -u''(x) = f(x) = 6x - 2
        u'(0) = 0, u'(1) = -1
    制造精确解:
        u^*(x) = x^2 - x^3
    满足自洽兼容条件:
        int_0^1 f(x) dx = [3x^2 - 2x]_0^1 = 1 = - (u'(1) - u'(0))
    """
    dx = 1.0 / (n - 1)
    x = np.linspace(0.0, 1.0, n)
    u_star = x**2 - x**3
    f = 6.0 * x - 2.0
    g0, g1 = 0.0, -1.0
    
    A = np.zeros((n, n))
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = -1.0, 2.0, -1.0
    A[0, 0], A[0, 1] = 1.0, -1.0
    A[-1, -2], A[-1, -1] = -1.0, 1.0
    
    b = dx**2 * f
    b[0] = 0.5 * dx**2 * f[0] - dx * g0
    b[-1] = 0.5 * dx**2 * f[-1] + dx * g1
    
    rank = int(np.linalg.matrix_rank(A))
    sigma_min = float(np.linalg.svd(A, compute_uv=False)[-1])
    sum_b = float(np.sum(b))
    
    if not fix_ref:
        return x, None, u_star, rank, sigma_min, sum_b
        
    A_ref, b_ref = A.copy(), b.copy()
    A_ref[0, :] = 0.0
    A_ref[0, 0] = 1.0
    b_ref[0] = u_star[0]
    u = np.linalg.solve(A_ref, b_ref)
    err = float(np.max(np.abs(u - u_star)))
    res = float(np.linalg.norm(A_ref @ u - b_ref))
    return x, u, u_star, err, res


def compute_neumann_convergence(n_list=(17, 33, 65)):
    """测试纯 Neumann 钉住参考点后的 MMS 二阶收敛阶"""
    errs = []
    for n in n_list:
        _, _, _, err, _ = solve_neumann_mms(n=n, fix_ref=True)
        errs.append(err)
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    return orders, errs


def main():
    print("=" * 60)
    print("M1.2 定解条件与适定性 —— 核心数值实验展示")
    print("=" * 60)
    
    print("\n[1] 边界条件与矩阵秩/条件数分析 (assemble_1d_poisson):")
    for bc in ["dirichlet", "mixed", "neumann"]:
        A, _ = assemble_1d_poisson(n=11, bc=bc)
        rank = np.linalg.matrix_rank(A)
        cond = np.linalg.cond(A)
        s_min = np.linalg.svd(A, compute_uv=False)[-1]
        print(f"  - {bc:>9}: 秩 = {rank:>2}/11, 最小奇异值 = {s_min:.2e}, 条件数 = {cond:.2e}")
        
    print("\n[2] Robin 边界对流换热系数极限扫描 (solve_robin_convection):")
    for h in [0.0, 0.1, 1.0, 10.0, 1000.0]:
        _, u, u_exact = solve_robin_convection(n=51, h=h)
        print(f"  - h = {h:>6.1f} | 数值 u(1) = {u[-1]:.6f} | 解析 1/(1+h) = {u_exact:.6f} | 误差 = {abs(u[-1]-u_exact):.2e}")
        
    print("\n[3] 逆热传导噪声放大与谱截断滤波 (backward_heat_noise_amplification):")
    noise_rng = np.random.default_rng(0)
    for sg in [1e-4, 1e-3, 1e-2]:
        res = backward_heat_noise_amplification(n=64, sigma=sg, rng=noise_rng)
        print(f"  - 噪声 σ={sg:.0e} | 放大倍率 = {res['amplification']:>6.1f}x | 直接反演误差 = {res['direct_err']:.2e} | 截断8模误差 = {res['truncated_err']:.2e}")
        
    print("\n[4] 纯 Neumann 边界制造解法与二阶收敛阶 (solve_neumann_mms):")
    orders, errs = compute_neumann_convergence()
    print(f"  - 网格加密 N=17/33/65 误差: {errs}")
    print(f"  - 观测数值收敛阶: {orders} (理论期望: 2.00)")
    print("=" * 60)


if __name__ == "__main__":
    main()

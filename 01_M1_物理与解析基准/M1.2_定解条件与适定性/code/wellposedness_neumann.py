# -*- coding: utf-8 -*-
"""
wellposedness_neumann.py —— M1.2 纯 Neumann 边界条件与适定性最小可运行检查脚本

对应 M1.2 讲义第 ⑪ 页 (最小可运行检查) 与第 ⑫ 页 (验证三件套)：
1. 奇异性与秩检查 (零空间常数零模判断，F3 典型特征)
2. 离散源项与通量积分兼容性条件检验 (存在性判断，F4 守恒检验)
3. 制造解法 (Method of Manufactured Solutions, MMS) 精度检验
4. 钉住参考点后的二阶空间收敛阶验证 (O(dx^2))
"""
import numpy as np


def check_neumann_system(n=6, verbose=True):
    """
    对一维纯 Neumann 泊松方程进行最小可运行检查。
    真解选取为 u*(x) = x^2 - x^3
    f(x) = -u*''(x) = 6x - 2
    g0 = u*'(0) = 0.0, g1 = u*'(1) = -1.0
    """
    dx = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    u_star = x**2 - x**3
    f = 6 * x - 2
    g0, g1 = 0.0, -1.0
    
    # 1. 装配离散矩阵 A (行和恒为 0，代表常数零模)
    A = np.zeros((n, n))
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = -1.0, 2.0, -1.0
    # 两端 Neumann 边界
    A[0, 0], A[0, 1] = 1.0, -1.0
    A[-1, -2], A[-1, -1] = -1.0, 1.0
    
    # 2. 装配右端载荷向量 b (包含源项 f 与边界通量 g)
    b = dx**2 * f
    b[0] = 0.5 * dx**2 * f[0] - dx * g0
    b[-1] = 0.5 * dx**2 * f[-1] + dx * g1
    
    rank = np.linalg.matrix_rank(A)
    s_min = np.linalg.svd(A, compute_uv=False)[-1]
    ones_residual = np.linalg.norm(A @ np.ones(n))
    sum_b = np.sum(b)
    
    if verbose:
        print(f"--- 纯 Neumann 离散系统自检 (n={n}) ---")
        print(f"1. 矩阵维度: {n}x{n} | 矩阵的秩: {rank}/{n} (缺 1 行独立方程，属于 F3 秩亏)")
        print(f"2. 最小奇异值: {s_min:.2e} (机器零，表明矩阵严格奇异)")
        print(f"3. 常数零空间检验 ||A · 1||: {ones_residual:.2e} (表明常数解属于内核)")
        print(f"4. 积分兼容性检验 Σ b_i: {sum_b:.2e} (等于零，表明系统存在解)")
        
    # 3. 尝试直接求解 (预期捕获奇异矩阵报错)
    direct_solve_error = None
    try:
        np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        direct_solve_error = str(e)
        if verbose:
            print(f"5. 直接求解 np.linalg.solve(A, b) 捕获异常 -> {direct_solve_error}")
            
    # 4. 修复方式：钉住一个参考点 u(0) = u*(0) 消除刚体自由度
    A_pinned = A.copy()
    b_pinned = b.copy()
    A_pinned[0, :] = 0.0
    A_pinned[0, 0] = 1.0
    b_pinned[0] = u_star[0]
    
    u_h = np.linalg.solve(A_pinned, b_pinned)
    max_err = np.max(np.abs(u_h - u_star))
    res_pinned = np.linalg.norm(A_pinned @ u_h - b_pinned)
    
    if verbose:
        print(f"6. 钉住参考点 u(0) 消除常数漂移后:")
        print(f"   - 最大物理误差 max|u_h - u*|: {max_err:.4e}")
        print(f"   - 线性残差 ||A_pinned @ u - b_pinned||: {res_pinned:.2e}")
        
    return {
        "n": n, "rank": rank, "s_min": s_min,
        "sum_b": sum_b, "direct_error": direct_solve_error,
        "max_err": max_err, "residual": res_pinned
    }


def verify_convergence():
    """验证 MMS 二阶空间精度收敛阶"""
    print("\n--- 验证三件套：MMS 网格减半收敛阶验证 ---")
    n_list = [9, 17, 33, 65]
    errors = []
    for n in n_list:
        res = check_neumann_system(n=n, verbose=False)
        errors.append(res["max_err"])
        
    orders = []
    for i in range(len(errors) - 1):
        order = np.log2(errors[i] / errors[i + 1])
        orders.append(order)
        print(f"网格 n={n_list[i]:>2} -> {n_list[i+1]:>2} (dx={1/(n_list[i]-1):.4f} -> {1/(n_list[i+1]-1):.4f}): "
              f"误差 = {errors[i]:.2e} -> {errors[i+1]:.2e} | 观测阶 = {order:.2f}")
              
    mean_order = np.mean(orders)
    print(f"平均收敛阶: {mean_order:.2f} (理论目标: 2.00 阶精度)")
    assert abs(mean_order - 2.00) < 0.05, f"收敛阶与二阶期望偏离过大: {mean_order}"
    print(">>> [PASS] 纯 Neumann 适定性检查与 MMS 验证全部通过！\n")


if __name__ == "__main__":
    check_neumann_system(n=6, verbose=True)
    verify_convergence()

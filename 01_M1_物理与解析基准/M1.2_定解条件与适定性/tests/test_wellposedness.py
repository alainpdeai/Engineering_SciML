# -*- coding: utf-8 -*-
"""
test_wellposedness.py —— M1.2 定解条件与适定性自动化门禁测试套件

涵盖：
1. 四类边界条件离散微分矩阵的秩与奇异性测试 (纯 Neumann 刚体自由度与零空间)
2. Robin 对流换热边界在渐进极限下的解析解吻合度验证
3. 逆热传导算子病态条件数与噪声放大/低通截断滤波有效性验证
4. 纯 Neumann 边界制造解法 (MMS) 兼容条件与二阶收敛阶验证
5. Robin 一般参数解析解与讲义噪声实验复现验证
"""
import os
import sys
import pytest
import numpy as np

# 将 code/ 目录加入系统路径
here = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(os.path.dirname(here), "code")
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from solution import (
    assemble_1d_poisson,
    solve_robin_convection,
    backward_heat_noise_amplification,
    solve_neumann_mms,
    compute_neumann_convergence
)


def test_boundary_ranks_and_singularity():
    """验证边界条件类型对矩阵满秩性与零空间的影响 (F3 失效模式识别)"""
    n = 11
    A_dir, _ = assemble_1d_poisson(n=n, bc="dirichlet")
    A_mix, _ = assemble_1d_poisson(n=n, bc="mixed")
    A_neu, _ = assemble_1d_poisson(n=n, bc="neumann")
    
    # Dirichlet 与 Mixed 为满秩矩阵，解唯一
    assert np.linalg.matrix_rank(A_dir) == n, "Dirichlet 边界应满秩"
    assert np.linalg.matrix_rank(A_mix) == n, "Mixed 边界应满秩"
    
    # 纯 Neumann 边界存在一维常数零空间，秩亏 1
    assert np.linalg.matrix_rank(A_neu) == n - 1, "纯 Neumann 边界矩阵秩应为 n - 1"
    assert np.linalg.cond(A_neu) > 1e14, "纯 Neumann 矩阵条件数应发散趋近无穷 (奇异矩阵)"
    
    # 常数向量位于零空间中: A @ 1 = 0
    ones_vec = np.ones(n)
    null_res = np.linalg.norm(A_neu @ ones_vec)
    assert null_res < 1e-12, f"常数基底未落入零空间，残差: {null_res}"


def test_robin_asymptotic_convergence():
    """验证 Robin 边界条件在不同对流系数 h 下向解析渐进极限收敛"""
    # 理论解析解: u(1) = 1 / (1 + h) (当 k=1, T_inf=0, u(0)=1 时)
    for h in [0.0, 0.1, 1.0, 10.0, 1000.0]:
        _, u, exact = solve_robin_convection(n=101, h=h)
        assert np.isclose(u[-1], exact, atol=1e-4), f"Robin 解在 h={h} 处未收敛至解析解: 数值={u[-1]}, 解析={exact}"


def test_robin_general_parameters_match_analytic_solution():
    """Robin 解析边界值应适用于非默认导热率与环境温度。"""
    _, u, exact = solve_robin_convection(n=51, k=2.0, T_inf=3.0, h=1.0)
    expected = (2.0 * 1.0 + 1.0 * 3.0) / (2.0 + 1.0)
    assert np.isclose(exact, expected, atol=1e-12)
    assert np.isclose(u[-1], expected, atol=1e-12)


def test_backward_heat_ill_posedness_and_truncation():
    """验证逆热传导算子的 Hadamard 不稳定性及模态截断稳定化"""
    res = backward_heat_noise_amplification(n=64, sigma=1e-3, seed=42, n_modes=8)
    
    # 离散微分算子逆问题病态，条件数很大
    assert res["cond_L"] > 1e3, f"离散导数算子条件数过小: {res['cond_L']}"
    
    # 高频噪声放大倍数超过 500 倍 (不适定性核心特征)
    assert res["amplification"] > 500.0, f"逆问题噪声放大倍率不足: {res['amplification']}"
    
    # 模态截断能有效滤除高频噪声，误差降低至少 1 个数量级
    assert res["truncated_err"] < res["direct_err"] * 0.1, \
        f"模态截断未能压低逆问题误差: 直接={res['direct_err']}, 截断={res['truncated_err']}"


def test_backward_heat_shared_rng_reproduces_slide_experiment():
    """一个共享随机流应复现讲义三档噪声实验的放大倍率。"""
    rng = np.random.default_rng(0)
    amplifications = [
        backward_heat_noise_amplification(n=64, sigma=sigma, rng=rng)["amplification"]
        for sigma in (1e-4, 1e-3, 1e-2)
    ]
    assert amplifications == pytest.approx(
        [927.0111, 1069.6613, 1050.7054], rel=1e-5
    )


def test_solution_main_uses_shared_noise_stream(capsys):
    """交付脚本的用户入口应复现讲义三档结果，而非每档重置种子。"""
    import solution

    solution.main()
    output = capsys.readouterr().out
    assert "放大倍率 =  927.0x" in output
    assert "放大倍率 = 1069.7x" in output
    assert "放大倍率 = 1050.7x" in output


def test_neumann_compatibility_and_second_order_mms():
    """验证纯 Neumann 制造解的自洽兼容条件及二阶网格收敛阶"""
    # 1. 离散载荷自洽兼容性检查 (sum(b) == 0)
    _, _, _, rank, s_min, sum_b = solve_neumann_mms(n=17, fix_ref=False)
    assert np.isclose(sum_b, 0.0, atol=1e-14), f"制造解纯 Neumann 载荷不满足相容性条件: sum(b)={sum_b}"
    
    # 2. 钉住参考点后线性方程具备唯一解，且满足严格二阶收敛阶 O(dx^2)
    orders, errs = compute_neumann_convergence(n_list=(17, 33, 65))
    assert (orders >= 1.90).all(), f"纯 Neumann MMS 收敛阶未达到二阶: {orders}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

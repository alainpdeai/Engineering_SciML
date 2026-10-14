# -*- coding: utf-8 -*-
"""
solution.py —— M1.3 分离变量与解析基准库标准参考求解器

涵盖核心功能:
1. 1D 热传导方程分离变量级数解析解 (方波初温、指数光滑衰减、能量单调耗散)
2. 1D 波动方程分离变量级数解析解 (三角波释放、驻波往复、机械能守恒)
3. 谱截断能量计算与 Gibbs 振铃病态分析
4. 与 benchmarks/ 解析基准库 (heat_1d_square_wave / wave_string_modes) 严格对齐
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 允许从上一级目录导入 benchmarks 模块
here = os.path.dirname(os.path.abspath(__file__))
unit_dir = os.path.dirname(here)
if unit_dir not in sys.path:
    sys.path.insert(0, unit_dir)

try:
    from benchmarks.heat_1d_square_wave import (
        heat_square_wave_series,
        compute_spectral_energy as heat_spectral_energy,
        convergence_study as heat_convergence_study
    )
    from benchmarks.wave_string_modes import (
        wave_string_solution,
        compute_exact_modal_energy as wave_modal_energy,
        compute_discrete_energy as wave_discrete_energy
    )
except ImportError:
    # 单元独立运行时降级自包含实现
    heat_square_wave_series = None
    wave_string_solution = None


def analytical_heat_1d(x, t, a=0.5, L=1.0, n_terms=50):
    """
    1D 热传导方程方波初温分离变量解（本单元标准问题）:
        u_t = a^2 u_xx,  x in (0, L), t > 0
        u(0, t) = u(L, t) = 0
        u(x, 0) = +1 (0 < x < L/2),  -1 (L/2 < x < L)

    ⚠ 系数必须是 b_n = 8/(n*pi)，且仅在 n = 4k+2（即 n ≡ 2 mod 4）时非零。
      这正是讲义 ⑪ 页诊断题 1 的标准答案 C 所指的错误：
      若误用全区间常数展开 b_n = 4/(n*pi)（仅奇数项），右半域会整体反号，
      x > L/2 处误差恒为 2.0。本函数与 benchmarks/heat_1d_square_wave.py 同口径。
    """
    x = np.asarray(x, dtype=float)
    u = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        if n % 4 != 2:                 # 仅 n ≡ 2 (mod 4) 非零
            continue
        k_n = n * np.pi / L
        b_n = 8.0 / (n * np.pi)
        decay = np.exp(-(a * k_n)**2 * t)
        u += b_n * np.sin(k_n * x) * decay
    return u


def analytical_wave_1d(x, t, c=1.0, L=1.0, n_terms=30, x0=0.3):
    """
    1D 弦振动初位移拨弦解 (三角波释放):
        u_tt = c^2 u_xx,  x in (0, L), t > 0
        u(0, t) = u(L, t) = 0
        u(x, 0) = x/x0 (x<x0), (L-x)/(L-x0) (x>x0)   —— x0 以 L 为单位的比例

    系数闭式: a_n = 2 L^2 sin(n*pi*x0_val/L) / (n^2 pi^2 x0_val (L - x0_val))
    当 x0 = L/2 时退化为 a_n = 8 sin(n*pi/2)/(n^2 pi^2)，即仅奇数项非零。
    本单元（讲义 ⑨ 页、benchmarks/）用的是 x0 = 0.3L，偶数阶一般非零。
    """
    x = np.asarray(x, dtype=float)
    x0_val = x0 * L
    u = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        k_n = n * np.pi / L
        a_n = (2.0 * L**2 * np.sin(n * np.pi * x0_val / L)
               / (n**2 * np.pi**2 * x0_val * (L - x0_val)))
        u += a_n * np.sin(k_n * x) * np.cos(c * k_n * t)
    return u


def solve_heat_square_wave(x, t, L=1.0, alpha=1.0, N=51):
    """基准库同口径热传导方波接口"""
    if heat_square_wave_series is not None:
        return heat_square_wave_series(x, t, L=L, alpha=alpha, N=N)
    return analytical_heat_1d(x, t, a=np.sqrt(alpha), L=L, n_terms=N)


def solve_wave_string_modes(x, t, L=1.0, c=1.0, N=31, x0=0.3):
    """基准库同口径弦振动驻波接口"""
    if wave_string_solution is not None:
        return wave_string_solution(x, t, L=L, c=c, N=N, x0=x0)
    return analytical_wave_1d(x, t, c=c, L=L, n_terms=N, x0=x0)


def run_demo(save_path=None):
    """运行物理场演化对比绘图并保存云图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    L = 1.0
    x = np.linspace(0, L, 200)
    plt.figure(figsize=(12, 5))
    
    # 1. 热传导耗散平滑
    plt.subplot(1, 2, 1)
    for t_val in [0.0, 0.005, 0.02, 0.1]:
        u_heat = analytical_heat_1d(x, t=t_val, a=0.5, L=L)
        plt.plot(x, u_heat, label=f"t = {t_val} s", lw=2)
    plt.title("1D 热传导分离变量解: 方波初温衰减平滑")
    plt.xlabel("x")
    plt.ylabel("温度 u(x,t)")
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend()
    
    # 2. 波动方程驻波往复
    plt.subplot(1, 2, 2)
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        u_wave = analytical_wave_1d(x, t=t_val, c=1.0, L=L)
        plt.plot(x, u_wave, label=f"t = {t_val} s", lw=2)
    plt.title("1D 弦振动三角波解析解: 驻波周期往复")
    plt.xlabel("x")
    plt.ylabel("位移 u(x,t)")
    plt.grid(True, ls='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(here, "analytical_demo.png")
    plt.savefig(save_path, dpi=150)
    print(f"[OK] Analytical demo figure saved to: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("M1.3 分离变量与解析基准库 —— 标准参考求解器")
    print("=" * 60)
    x_test = np.linspace(0.0, 1.0, 5)
    u_h = solve_heat_square_wave(x_test, t=0.01)
    u_w = solve_wave_string_modes(x_test, t=0.25)
    print("热传导方波测试点 u(x, 0.01):", np.round(u_h, 4))
    print("波动方程驻波测试点 u(x, 0.25):", np.round(u_w, 4))
    run_demo()
    print("=" * 60)


if __name__ == "__main__":
    main()

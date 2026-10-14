# -*- coding: utf-8 -*-
"""
Unit M1.3 基准 1: 一维热传导方程方波初温分离变量解析解 (G1 解析基准库组件)

数学物理模型:
    PDE:  u_t = alpha * u_xx,  x in (0, L), t > 0
    BC:   u(0, t) = 0,  u(L, t) = 0
    IC:   u(x, 0) = +1 (0 < x < L/2),  -1 (L/2 < x < L)

傅里叶正弦级数推导:
    本征函数: X_n(x) = sin(n * pi * x / L)
    投影积分: b_n = (2/L) * int_0^L f(x) * sin(n*pi*x/L) dx
    当且仅当 n = 2, 6, 10, ... (即 n = 4k + 2) 时非零:
        b_n = 8 / (n * pi)
    时空解析解:
        u(x, t) = sum_{n=4k+2}^{N} b_n * sin(n*pi*x/L) * exp(-alpha * (n*pi/L)^2 * t)
"""
import numpy as np


def heat_square_wave_series(x, t, L=1.0, alpha=1.0, N=51):
    """
    计算一维热传导方波分离变量级数解

    参数:
        x: 空间坐标数组 (一维 ndarray)
        t: 时间点 (标量 float)
        L: 空间区间长度 (默认 1.0)
        alpha: 热扩散率 (默认 1.0)
        N: 傅里叶截断阶数 (默认 51)

    返回:
        u: 温度场数组，形状与 x 一致
    """
    x = np.asarray(x, dtype=float)
    u = np.zeros_like(x)
    for n in range(1, N + 1):
        if n % 4 == 2:
            b_n = 8.0 / (n * np.pi)
            decay = np.exp(-alpha * (n * np.pi / L) ** 2 * t)
            u += b_n * np.sin(n * np.pi * x / L) * decay
    return u


def compute_spectral_energy(t, L=1.0, alpha=1.0, N=101):
    """
    计算热传导方波在时间 t 的谱能量和:
        E_spectral(t) = sum_{n=4k+2}^N b_n^2 * exp(-2 * alpha * (n*pi/L)^2 * t)
    注意: 连续场总积分能量为 int_0^L u^2 dx = (L / 2) * E_spectral(t)
    """
    energy = 0.0
    for n in range(1, N + 1):
        if n % 4 == 2:
            b_n = 8.0 / (n * np.pi)
            decay = np.exp(-2.0 * alpha * (n * np.pi / L) ** 2 * t)
            energy += b_n**2 * decay
    return energy


def convergence_study(t=0.01, L=1.0, alpha=1.0, N_list=(3, 6, 11), N_ref=101):
    """
    在指定平滑时刻 t > 0 下研究傅里叶级数随截断阶数 N 的收敛阶 (L_inf 误差)
    """
    x_grid = np.linspace(0, L, 2001)
    u_ref = heat_square_wave_series(x_grid, t=t, L=L, alpha=alpha, N=N_ref)
    errors = {}
    for N in N_list:
        u_N = heat_square_wave_series(x_grid, t=t, L=L, alpha=alpha, N=N)
        err = float(np.max(np.abs(u_N - u_ref)))
        errors[N] = err
    return errors


def main():
    print("=" * 60)
    print(" Unit M1.3 Benchmark 1: 一维热传导方波初温分离变量解析解")
    print("=" * 60)

    # 1. t=0 纯 Gibbs 振铃与空间收敛检验
    x = np.linspace(0, 1.0, 2001)
    away = (np.abs(x - 0.5) > 0.03) & (x > 0.03) & (x < 0.97)
    target = np.where(x < 0.5, 1.0, -1.0)

    print("\n[1] t = 0 (Gibbs 效应: 离开跳变区最大误差随 N 变化):")
    for N in (3, 11, 51):
        u_0 = heat_square_wave_series(x, t=0.0, N=N)
        err_gibbs = np.max(np.abs(u_0[away] - target[away]))
        print(f"    N = {N:2d} | 远离跳变点最大误差 = {err_gibbs:.4f}")

    # 2. t=0.01 指数收敛阶验证
    print("\n[2] t = 0.01 (物理耗散消除高频后，谱指数收敛验证):")
    conv_errs = convergence_study(t=0.01, N_list=(3, 6, 11))
    for N, err in conv_errs.items():
        print(f"    N = {N:2d} | L_inf 截断误差 = {err:.3e}")

    # 3. 谱能量单调耗散验证
    print("\n[3] 谱能量时间耗散:")
    for t_val in (0.0, 0.001, 0.01, 0.05):
        e_spec = compute_spectral_energy(t=t_val, N=101)
        print(f"    t = {t_val:.3f} s | 谱能量和 = {e_spec:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

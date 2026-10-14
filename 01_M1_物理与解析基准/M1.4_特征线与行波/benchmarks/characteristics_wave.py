# -*- coding: utf-8 -*-
"""M1.4 解析基准库：特征线、d'Alembert 行波与波动能量。

本模块是 M1.4 单元唯一的解析基准来源，供 code/solution.py、tests/ 与
notebooks/episode_lab.ipynb 共用。只依赖 NumPy，可在普通笔记本 CPU 上运行。

提供的基准
----------
characteristic_feet  目标点 (x0, t) 的依赖域端点 x0 ± c t
gaussian_pulse       光滑高斯初值（可指定中心与宽度）
d_alembert           无限直线上的 d'Alembert 解（含初速度积分项）
d_alembert_finite    有限区间 [0, L] 上的镜像延拓解（固定端 / 自由端）
leapfrog_wave        二阶显式 leapfrog 数值解
discrete_energy      leapfrog 的守恒型离散能量（半步格式）
pde_residual         把解析解代回 u_tt - c^2 u_xx 的残差

符号与约定
----------
- x 轴向右为正，t >= 0，波速 c > 0，长度与时间单位自洽。
- 特征线族：x - c t = const（右行），x + c t = const（左行）。
- 依赖域（domain of dependence）：目标点 (x0, t) 的解只由 [x0 - c t, x0 + c t] 内的初值决定。
- 影响域（domain of influence）：t0 时刻位于 x0 的扰动，在 t > t0 时只影响 |x - x0| <= c (t - t0)。
- 固定端指 u = 0；自由端指 u_x = 0。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "characteristic_feet",
    "gaussian_pulse",
    "d_alembert",
    "d_alembert_finite",
    "leapfrog_wave",
    "discrete_energy",
    "pde_residual",
]


def _trapz(y, x):
    """兼容 NumPy 1.x (trapz) 与 2.x (trapezoid) 的梯形积分。"""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def characteristic_feet(x0, t, c):
    """返回目标点 (x0, t) 的依赖域端点 (x0 - c t, x0 + c t)。

    这两个点就是过 (x0, t) 的两条特征线与 t = 0 轴的交点。
    """
    if c <= 0:
        raise ValueError("波速 c 必须为正")
    if t < 0:
        raise ValueError("时间 t 必须非负")
    return x0 - c * t, x0 + c * t


def gaussian_pulse(x, center=0.0, width=0.05):
    """高斯脉冲初值 f(x) = exp(-((x - center)/width)^2)。"""
    x = np.asarray(x, dtype=float)
    return np.exp(-((x - center) / width) ** 2)


# Gauss-Legendre 求积节点，用于 d'Alembert 公式里的初速度积分项
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(96)


def _integrate_velocity(x, t, g, c):
    """向量化计算积分 ∫_{x-ct}^{x+ct} g(s) ds（Gauss-Legendre 求积）。"""
    a = x - c * t
    b = x + c * t
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    s = mid[:, None] + half[:, None] * _GL_NODES[None, :]
    return half * np.sum(_GL_WEIGHTS[None, :] * g(s), axis=1)


def d_alembert(x, t, f, g=None, c=1.0):
    """无限直线上的 d'Alembert 解。

    u(x, t) = [f(x - c t) + f(x + c t)] / 2 + (1 / 2c) ∫_{x-ct}^{x+ct} g(s) ds

    参数
    ----
    x : 数组或标量，空间坐标
    t : 标量，时间（t >= 0）
    f : 可调用对象，初位移 f(x)
    g : 可调用对象或 None，初速度 g(x)；None 表示零初速度
    c : 波速
    """
    if c <= 0:
        raise ValueError("波速 c 必须为正")
    x = np.asarray(x, dtype=float)
    out = 0.5 * (f(x - c * t) + f(x + c * t))
    if g is not None:
        out = out + _integrate_velocity(x, t, g, c) / (2.0 * c)
    return out


def _mirror_extend(z, f, L, parity):
    """把 [0, L] 上的初值按 2L 周期镜像延拓到整条直线。

    parity = "fixed" 用奇延拓（u = 0 边界），"free" 用偶延拓（u_x = 0 边界）。
    """
    zz = np.mod(z, 2.0 * L)
    folded = np.where(zz <= L, zz, 2.0 * L - zz)
    val = f(folded)
    if parity == "fixed":
        # 每穿过一条边界，奇延拓改变一次符号
        sign = np.where(np.floor(z / L) % 2 == 0, 1.0, -1.0)
        val = val * sign
    return val


def d_alembert_finite(x, t, f, c, L, boundary="fixed"):
    """有限区间 [0, L] 上的镜像延拓解析解（零初速度）。

    固定端（u = 0）用奇延拓，自由端（u_x = 0）用偶延拓；两种情形都只需要
    初位移 f。这是 M1.3 正弦模态解的"特征线视角"等价形式。
    """
    if boundary not in ("fixed", "free"):
        raise ValueError("boundary 只能是 'fixed' 或 'free'")
    parity = "fixed" if boundary == "fixed" else "free"
    x = np.asarray(x, dtype=float)
    left = _mirror_extend(x - c * t, f, L, parity)
    right = _mirror_extend(x + c * t, f, L, parity)
    return 0.5 * (left + right)


def _first_step(u0, g0, dt, nu2, boundary):
    """二阶起步 u^1 = u^0 + dt g + nu2/2 * u^0_xx（自由端用镜像鬼点）。"""
    if boundary not in ("fixed", "free"):
        raise ValueError("boundary 只能是 'fixed' 或 'free'")
    u1 = u0 + dt * g0
    u1[1:-1] += 0.5 * nu2 * (u0[2:] - 2.0 * u0[1:-1] + u0[:-2])
    if boundary == "fixed":
        u1[0] = 0.0
        u1[-1] = 0.0
    else:
        # 鬼点镜像 u_{-1} = u_1, u_N = u_{N-2}，等价于 u_x = 0
        u1[0] = u0[0] + dt * g0[0] + nu2 * (u0[1] - u0[0])
        u1[-1] = u0[-1] + dt * g0[-1] + nu2 * (u0[-2] - u0[-1])
    return u1


def _next_step(prev, cur, nu2, boundary):
    """leapfrog 单步推进；自由端用镜像鬼点，而不是直接复制端点值。

    直接令 u[0] = u[1] 只是把边界条件"贴"在解上，会破坏离散能量守恒；
    鬼点格式把 u_x = 0 写进差分模板，才能保持守恒。
    """
    nxt = np.empty_like(cur)
    nxt[1:-1] = 2.0 * cur[1:-1] - prev[1:-1] + nu2 * (cur[2:] - 2.0 * cur[1:-1] + cur[:-2])
    if boundary == "fixed":
        nxt[0] = 0.0
        nxt[-1] = 0.0
    elif boundary == "free":
        nxt[0] = 2.0 * cur[0] - prev[0] + 2.0 * nu2 * (cur[1] - cur[0])
        nxt[-1] = 2.0 * cur[-1] - prev[-1] + 2.0 * nu2 * (cur[-2] - cur[-1])
    else:
        raise ValueError("boundary 只能是 'fixed' 或 'free'")
    return nxt


def leapfrog_wave(x, t_end, c, cfl=0.9, f=None, g=None, boundary="fixed"):
    """二阶显式 leapfrog 求解 u_tt = c^2 u_xx。

    返回 (t, U, dt)：t 为时间数组，U[n] 为第 n 步的解，dt 为实际时间步长。
    固定端 u = 0 直接置零；自由端 u_x = 0 用镜像鬼点，保证离散能量守恒。
    """
    x = np.asarray(x, dtype=float)
    if f is None:
        f = lambda z: np.zeros_like(z, dtype=float)
    if g is None:
        g = lambda z: np.zeros_like(z, dtype=float)
    if boundary not in ("fixed", "free"):
        raise ValueError("boundary 只能是 'fixed' 或 'free'")

    dx = float(x[1] - x[0])
    if cfl <= 0:
        raise ValueError("cfl 必须为正")
    dt = cfl * dx / c
    nt = max(1, int(round(t_end / dt)))
    dt = t_end / nt  # 让最后一步恰好落在 t_end

    u0 = np.asarray(f(x), dtype=float).copy()
    g0 = np.asarray(g(x), dtype=float)
    nu2 = (c * dt / dx) ** 2
    u1 = _first_step(u0, g0, dt, nu2, boundary)

    U = [u0, u1]
    for _ in range(nt - 1):
        U.append(_next_step(U[-2], U[-1], nu2, boundary))
    t = np.arange(len(U)) * dt
    return t, np.asarray(U), dt


def discrete_energy(u_prev, u_next, c, dt, dx, weights=None):
    """leapfrog 的守恒型离散能量（半步格式 + 梯形权重）。

    E^{n+1/2} = dx/2 * [ Σ_j w_j ((u^{n+1}_j - u^n_j)/dt)^2
                        + c^2 Σ_edges (Δu^{n+1}/dx)(Δu^n/dx) ]

    其中 w 是梯形权重：两端点为 1/2、内部点为 1。自由端（u_x = 0）必须用
    梯形权重，否则端点动能被计了两次，能量会出现 O(1e-2) 的假漂移。
    对固定端 / 自由端，该量都在舍入误差范围内守恒。
    """
    kinetic = ((u_next - u_prev) / dt) ** 2
    if weights is None:
        weights = np.ones_like(kinetic)
        weights[0] = 0.5
        weights[-1] = 0.5
    grad_next = np.diff(u_next) / dx
    grad_prev = np.diff(u_prev) / dx
    potential = c ** 2 * grad_next * grad_prev
    return 0.5 * dx * (np.sum(weights * kinetic) + np.sum(potential))


def pde_residual(x, t, u_func, c, dx=1e-3, dt=1e-3):
    """把解析解 u_func(x, t) 代回 u_tt - c^2 u_xx，返回最大绝对残差。

    残差应随 dx, dt 减小而趋于 0；它检验的是"表达式满足方程"，
    不检验边界条件与初始条件是否匹配物理问题。
    """
    x = np.asarray(x, dtype=float)
    u = lambda xx, tt: np.asarray(u_func(xx, tt), dtype=float)
    utt = (u(x, t + dt) - 2.0 * u(x, t) + u(x, t - dt)) / dt ** 2
    uxx = (u(x + dx, t) - 2.0 * u(x, t) + u(x - dx, t)) / dx ** 2
    return float(np.max(np.abs(utt - c ** 2 * uxx)))


if __name__ == "__main__":  # 单文件自检：python benchmarks/characteristics_wave.py
    _x = np.linspace(-4.0, 4.0, 801)
    _c = 1.0
    _f = lambda z: gaussian_pulse(z, center=-1.0, width=0.2)
    print("[基准库自检]")
    print("  依赖域端点 x0=0, t=1:", characteristic_feet(0.0, 1.0, _c))
    print("  d'Alembert 残差(t=0.5):", pde_residual(_x, 0.5, lambda xx, tt: d_alembert(xx, tt, _f, c=_c), _c))
    _t, _U, _dt = leapfrog_wave(_x, 1.0, _c, cfl=0.9, f=_f)
    _err = np.max(np.abs(_U[-1] - d_alembert(_x, _t[-1], _f, c=_c)))
    print("  leapfrog L∞ 误差:", _err)
    _e = [discrete_energy(_U[i], _U[i + 1], _c, _dt, _x[1] - _x[0]) for i in range(len(_U) - 1)]
    print("  能量相对漂移:", abs(max(_e) - min(_e)) / abs(_e[0]))

# -*- coding: utf-8 -*-
"""M1.4 单元测试：特征线、d'Alembert 解、反射边界、离散能量与 CFL。

运行：pytest tests -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_UNIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _UNIT not in sys.path:
    sys.path.insert(0, _UNIT)

from benchmarks.characteristics_wave import (  # noqa: E402
    characteristic_feet,
    d_alembert,
    d_alembert_finite,
    discrete_energy,
    gaussian_pulse,
    leapfrog_wave,
    pde_residual,
)


def compact_bump(x, center=0.5, width=0.15):
    z = (np.asarray(x, dtype=float) - center) / width
    return np.where(np.abs(z) < 1.0, (1.0 - z ** 2) ** 2, 0.0)


def test_characteristic_feet_positions():
    """依赖域端点就是过目标点的两条特征线与 t = 0 的交点。"""
    assert characteristic_feet(0.0, 1.0, 1.0) == (-1.0, 1.0)
    assert characteristic_feet(2.0, 0.5, 2.0) == (1.0, 3.0)


def test_characteristic_feet_rejects_bad_input():
    with pytest.raises(ValueError):
        characteristic_feet(0.0, 1.0, -1.0)
    with pytest.raises(ValueError):
        characteristic_feet(0.0, -1.0, 1.0)


def test_dalembert_satisfies_wave_equation():
    """解析解代回 u_tt - c^2 u_xx 的残差应远小于场量本身。"""
    x = np.linspace(-4.0, 4.0, 401)
    c = 1.0
    f = lambda z: gaussian_pulse(z, center=-1.0, width=0.2)
    u = lambda xx, tt: d_alembert(xx, tt, f, c=c)
    assert pde_residual(x, 0.5, u, c) < 1e-8


def test_dalembert_zero_velocity_splits_into_two_half_pulses():
    """零初速度时 u = [f(x-ct) + f(x+ct)] / 2。"""
    x = np.linspace(-4.0, 4.0, 201)
    c, t = 1.0, 0.7
    f = lambda z: gaussian_pulse(z, center=0.0, width=0.2)
    expected = 0.5 * (f(x - c * t) + f(x + c * t))
    assert np.allclose(d_alembert(x, t, f, c=c), expected)


def test_causality_before_wavefront_arrival():
    """紧支集初值：波前未到之处，解析解必须严格为零。"""
    c, x0, width, x_far = 1.0, -1.0, 0.2, 3.0
    f = lambda z: compact_bump(z, center=x0, width=width)
    t_phys = (x_far - (x0 + width)) / c
    for t in (0.5 * t_phys, 0.9 * t_phys):
        assert abs(float(d_alembert(np.array([x_far]), t, f, c=c)[0])) == 0.0


def test_numerical_cone_is_not_narrower_than_physical_cone():
    """数值依赖域比物理锥更宽，数值解不会晚于物理到达时间才响应。"""
    c, x0, width, x_far = 1.0, -1.0, 0.2, 3.0
    t_phys = (x_far - (x0 + width)) / c
    x = np.linspace(-4.0, 4.0, 801)
    f = lambda z: compact_bump(z, center=x0, width=width)
    t, U, _ = leapfrog_wave(x, 4.2, c, cfl=0.5, f=f)
    i = int(np.argmin(np.abs(x - x_far)))
    hit = np.abs(U[:, i]) > 1e-6
    assert np.any(hit)
    assert float(t[np.argmax(hit)]) < t_phys


def test_fixed_boundary_holds_zero():
    L, c = 1.0, 1.0
    f = lambda z: compact_bump(z, center=0.5, width=0.15)
    for t in np.linspace(0.0, 2.0, 41):
        assert abs(float(d_alembert_finite(np.array([0.0]), t, f, c, L, "fixed")[0])) < 1e-12
        assert abs(float(d_alembert_finite(np.array([L]), t, f, c, L, "fixed")[0])) < 1e-12


def test_free_boundary_holds_zero_slope():
    L, c, d = 1.0, 1.0, 1e-6
    f = lambda z: compact_bump(z, center=0.5, width=0.15)
    for t in np.linspace(0.0, 2.0, 21):
        hi = float(d_alembert_finite(np.array([L + d]), t, f, c, L, "free")[0])
        lo = float(d_alembert_finite(np.array([L - d]), t, f, c, L, "free")[0])
        assert abs((hi - lo) / (2 * d)) < 1e-6


@pytest.mark.parametrize("boundary", ["fixed", "free"])
def test_discrete_energy_is_conserved(boundary):
    """无阻尼时离散能量应在舍入误差量级守恒（含自由端梯形权重）。"""
    c = 1.0
    x = np.linspace(0.0, 1.0, 401)
    dx = x[1] - x[0]
    f = lambda z: compact_bump(z, center=0.5, width=0.15)
    t, U, dt = leapfrog_wave(x, 4.0, c, cfl=0.9, f=f, boundary=boundary)
    energies = np.array([discrete_energy(U[i], U[i + 1], c, dt, dx) for i in range(len(U) - 1)])
    drift = np.max(np.abs(energies - energies[0])) / abs(energies[0])
    assert drift < 1e-12, f"{boundary} 边界能量漂移过大: {drift:.3e}"


def test_second_order_convergence():
    """CFL 固定时，L∞ 误差随 dx 减半约降 4 倍（观测阶 ≈ 2）。"""
    c, t_end, cfl = 1.0, 1.0, 0.9
    f = lambda z: gaussian_pulse(z, center=-1.0, width=0.2)
    errors = []
    for N in (200, 400):
        x = np.linspace(-4.0, 4.0, N)
        t, U, _ = leapfrog_wave(x, t_end, c, cfl=cfl, f=f)
        errors.append(float(np.max(np.abs(U[-1] - d_alembert(x, t[-1], f, c=c)))))
    order = np.log2(errors[0] / errors[1])
    assert order > 1.7, f"观测收敛阶 {order:.2f} 低于二阶预期"


def test_cfl_above_one_is_unstable():
    """CFL > 1 时 leapfrog 失稳：最大值应远超初始幅值（F2 证据）。"""
    c, t_end = 1.0, 2.0
    x = np.linspace(0.0, 1.0, 201)
    f = lambda z: compact_bump(z, center=0.5, width=0.15)
    with np.errstate(over="ignore", invalid="ignore"):
        t, U, _ = leapfrog_wave(x, t_end, c, cfl=1.05, f=f)
        peak = float(np.max(np.abs(U)))
    assert (not np.isfinite(peak)) or peak > 1e3


def test_cfl_at_one_is_marginally_stable():
    """CFL = 1 是稳定边界：幅值不应增长。"""
    c, t_end = 1.0, 4.0
    x = np.linspace(0.0, 1.0, 201)
    f = lambda z: compact_bump(z, center=0.5, width=0.15)
    t, U, _ = leapfrog_wave(x, t_end, c, cfl=1.0, f=f)
    assert np.max(np.abs(U)) < 1.01

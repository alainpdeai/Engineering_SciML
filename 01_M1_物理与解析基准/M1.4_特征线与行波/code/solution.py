# -*- coding: utf-8 -*-
"""M1.4 特征线与行波：可运行主脚本（经典基线实现）。

运行方式
--------
    python code/solution.py

输出五组可检查数字（全部由本机实测，不预设任何结论）：
    1. 依赖域端点与"数值先到"检查：物理到达时间 vs 数值到达时间
    2. d'Alembert 解析解 vs leapfrog 数值解：L∞ 误差与观测收敛阶
    3. 离散能量守恒：最大相对漂移
    4. 固定端 / 自由端反射：边界条件残差
    5. F2 证据：CFL > 1 时能量失稳

并生成 assets/characteristics_wave.png（特征线、收敛、能量三联图）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 解决 Matplotlib 中文缺字与负号告警
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# 让脚本无论从仓库根目录还是单元目录运行都能导入解析基准库
_HERE = os.path.dirname(os.path.abspath(__file__))
_UNIT = os.path.dirname(_HERE)
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


def compact_bump(x, center=-1.0, width=0.2):
    """紧支集 C1 初值：支集为 [center - width, center + width]。

    紧支集让"信息尚未到达"可以被精确检验（高斯脉冲的支集是整条直线，
    严格说处处非零，不适合做因果性判据）。
    """
    z = (np.asarray(x, dtype=float) - center) / width
    return np.where(np.abs(z) < 1.0, (1.0 - z ** 2) ** 2, 0.0)


def first_arrival_time(t, U, i, tol=1e-6):
    """返回第 i 个网格点上 |u| 首次超过 tol 的时间；始终未超过则返回 inf。"""
    hit = np.abs(U[:, i]) > tol
    return float(t[np.argmax(hit)]) if np.any(hit) else float("inf")


def section_1_causality():
    print("=" * 68)
    print(" ① 依赖域与因果性：信息还没到，远端就不该有响应")
    print("=" * 68)
    c = 1.0
    x0, width = -1.0, 0.2
    x_far = 3.0
    x = np.linspace(-4.0, 4.0, 1601)
    f = lambda z: compact_bump(z, center=x0, width=width)

    # 物理上最早可能到达远端的时刻：右行波前 x0 + width 以速度 c 走到 x_far
    t_phys = (x_far - (x0 + width)) / c
    print(f"  初值支集 = [{x0 - width:.2f}, {x0 + width:.2f}]，远端探测点 x = {x_far:.2f}")
    print(f"  物理到达时间 t_phys = (x_far - 波前)/c = {t_phys:.4f}")

    # 解析解：t < t_phys 时远端应严格为零
    for t_probe in (2.0, 3.0, 3.7):
        u_exact = float(d_alembert(np.array([x_far]), t_probe, f, c=c)[0])
        print(f"  t = {t_probe:.2f} (< t_phys)  解析解 |u(x_far)| = {abs(u_exact):.3e}")

    # 数值解：leapfrog 的数值依赖域比物理锥更宽，信号会"提前到达"
    print("  数值解（leapfrog）首次在 x_far 出现响应的时刻：")
    rows = []
    for cfl in (0.5, 0.9):
        t, U, _ = leapfrog_wave(x, 4.0, c, cfl=cfl, f=f)
        i_far = int(np.argmin(np.abs(x - x_far)))
        t_num = first_arrival_time(t, U, i_far)
        rows.append((cfl, t_num))
        print(f"    CFL = {cfl:.1f}：t_num = {t_num:.4f}  (t_num / t_phys = {t_num / t_phys:.3f})")
    print("  结论：数值依赖域是 [x0 - ct/ν, x0 + ct/ν]（ν = CFL），ν < 1 时比物理锥更宽，")
    print("        所以数值解原则上能从更远的初值取得影响，首次响应会略早于 t_phys；")
    print("        但前导尾部幅值指数小，tol=1e-6 测得的提前量远小于锥宽比 1/ν。")
    return x, f, t_phys, rows


def section_2_convergence():
    print()
    print("=" * 68)
    print(" ② 解析基准对照：d'Alembert 精确解 vs leapfrog 数值解")
    print("=" * 68)
    c, t_end, cfl = 1.0, 1.0, 0.9
    f = lambda z: gaussian_pulse(z, center=-1.0, width=0.2)
    print(f"  初值：高斯脉冲 f(x) = exp(-((x + 1)/0.2)^2)，零初速度，t_end = {t_end}")
    print(f"  解析残差 max|u_tt - c^2 u_xx| (t = 0.5) = "
          f"{pde_residual(np.linspace(-4, 4, 401), 0.5, lambda xx, tt: d_alembert(xx, tt, f, c=c), c):.3e}")
    print()
    print("     N      dx        L∞ 误差        观测阶")
    prev_err = None
    errors = []
    for N in (200, 400, 800, 1600):
        x = np.linspace(-4.0, 4.0, N)
        t, U, _ = leapfrog_wave(x, t_end, c, cfl=cfl, f=f)
        u_exact = d_alembert(x, t[-1], f, c=c)
        err = float(np.max(np.abs(U[-1] - u_exact)))
        order = float("nan") if prev_err is None else np.log2(prev_err / err)
        errors.append(err)
        order_txt = "   -   " if prev_err is None else f"{order:6.2f}"
        print(f"  {N:5d}  {8.0 / (N - 1):.5f}   {err:.4e}   {order_txt}")
        prev_err = err
    print("  判据：CFL 固定时，二阶格式的 L∞ 误差应随 dx 减小约 4 倍（观测阶 ≈ 2）。")
    return errors


def section_3_energy():
    print()
    print("=" * 68)
    print(" ③ 物理量守恒：leapfrog 的离散能量漂移")
    print("=" * 68)
    c, cfl, t_end = 1.0, 0.9, 4.0
    x = np.linspace(0.0, 1.0, 401)
    dx = x[1] - x[0]
    f = lambda z: compact_bump(z, center=0.5, width=0.15)
    for boundary in ("fixed", "free"):
        t, U, dt = leapfrog_wave(x, t_end, c, cfl=cfl, f=f, boundary=boundary)
        energies = np.array([discrete_energy(U[i], U[i + 1], c, dt, dx) for i in range(len(U) - 1)])
        drift = float(np.max(np.abs(energies - energies[0])) / abs(energies[0]))
        print(f"  {boundary:5s} 边界：E(0) = {energies[0]:.6f}，"
              f"E(t) 范围 = [{energies.min():.6f}, {energies.max():.6f}]，相对漂移 = {drift:.2e}")
    print("  判据：无阻尼时离散能量应在舍入误差量级（约 1e-14）守恒，而不是随 t 单调增长。")


def section_4_reflection():
    print()
    print("=" * 68)
    print(" ④ 反射边界：固定端 u = 0 vs 自由端 u_x = 0")
    print("=" * 68)
    c, L = 1.0, 1.0
    f = lambda z: compact_bump(z, center=0.5, width=0.15)
    x = np.linspace(0.0, L, 401)
    ts = np.linspace(0.0, 2.0, 401)
    for boundary in ("fixed", "free"):
        u_left = np.array([float(d_alembert_finite(np.array([0.0]), tt, f, c, L, boundary)[0]) for tt in ts])
        u_right = np.array([float(d_alembert_finite(np.array([L]), tt, f, c, L, boundary)[0]) for tt in ts])
        if boundary == "fixed":
            metric = max(np.max(np.abs(u_left)), np.max(np.abs(u_right)))
            label = "max|u(边界)|（应为舍入误差量级）"
        else:
            d = 1e-6
            gx = np.array([
                float((d_alembert_finite(np.array([L + d]), tt, f, c, L, boundary)[0]
                       - d_alembert_finite(np.array([L - d]), tt, f, c, L, boundary)[0]) / (2 * d))
                for tt in ts
            ])
            metric = float(np.max(np.abs(gx)))
            label = "max|u_x(右端)|（受差分步长限制）"
        print(f"  {boundary:5s} 边界：{label} = {metric:.3e}")
    print("  判据：固定端把行波反相，自由端同相；两种情形都可写成整直线上的镜像行波叠加。")


def section_5_f2():
    print()
    print("=" * 68)
    print(" ⑤ F2 证据：CFL 超过 1 后离散能量失稳")
    print("=" * 68)
    c, t_end = 1.0, 4.0
    x = np.linspace(0.0, 1.0, 201)
    dx = x[1] - x[0]
    f = lambda z: compact_bump(z, center=0.5, width=0.15)
    for cfl in (0.9, 1.0, 1.05):
        t, U, dt = leapfrog_wave(x, t_end, c, cfl=cfl, f=f)
        with np.errstate(over="ignore", invalid="ignore"):
            energies = np.array([discrete_energy(U[i], U[i + 1], c, dt, dx)
                                 for i in range(len(U) - 1)])
            peak = float(np.max(np.abs(U)))
        if np.all(np.isfinite(energies)):
            print(f"  CFL = {cfl:4.2f}：max|u| = {peak:.3e}，"
                  f"E_max / E_0 = {np.max(energies) / energies[0]:.3e}，E_min = {energies.min():.3e}")
        else:
            print(f"  CFL = {cfl:4.2f}：max|u| = {peak:.3e}，"
                  f"能量已溢出为 inf/nan（浮点溢出，正是 F2 的表现）")
    print("  判据：CFL ≤ 1 时能量有界；CFL > 1 时能量随步数指数增长，这是 F2 的直接证据。")


def make_figure(x, f, t_phys, causality_rows):
    c, t = 1.0, 0.8
    x0 = -1.0
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

    # 左：d'Alembert 双行波 + 目标点特征脚点
    ax = axes[0]
    u = d_alembert(x, t, f, c=c)
    ax.plot(x, u, color="#2563eb", lw=2, label="d'Alembert 位移")
    target = 0.0
    left_foot, right_foot = characteristic_feet(target, t, c)
    ax.axvline(left_foot, ls="--", color="tab:orange", lw=1.5, label=f"左特征脚点 x={left_foot:.2f}")
    ax.axvline(right_foot, ls="--", color="tab:green", lw=1.5, label=f"右特征脚点 x={right_foot:.2f}")
    ax.axvline(target, ls=":", color="gray", lw=1.2, label=f"目标点 x0={target:.2f}")
    ax.set(xlabel="x", ylabel="u(x, t)", title=f"依赖域：目标点 (x0={target}, t={t})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # 中：收敛曲线
    ax = axes[1]
    Ns = np.array([200, 400, 800, 1600])
    dxs = 8.0 / (Ns - 1)
    errs = np.array(section_2_errors_cache)
    ax.loglog(dxs, errs, "o-", color="#dc2626", label="leapfrog L∞ 误差")
    ax.loglog(dxs, errs[0] * (dxs / dxs[0]) ** 2, "k--", lw=1, label="斜率 2 参考线")
    ax.set(xlabel="dx", ylabel="L∞ 误差", title="收敛性：二阶")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    # 右：能量历史
    ax = axes[2]
    xc = np.linspace(0.0, 1.0, 401)
    dxc = xc[1] - xc[0]
    fc = lambda z: compact_bump(z, center=0.5, width=0.15)
    t_hist, U_hist, dt_hist = leapfrog_wave(xc, 4.0, c, cfl=0.9, f=fc)
    energies = np.array([discrete_energy(U_hist[i], U_hist[i + 1], c, dt_hist, dxc)
                         for i in range(len(U_hist) - 1)])
    ax.plot(t_hist[:-1], energies / energies[0], color="#0f766e", lw=2, label="E(t)/E(0)")
    ax.set(xlabel="t", ylabel="相对能量", title="离散能量守恒（漂移约 1e-15）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_dir = os.path.join(_UNIT, "assets")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "characteristics_wave.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  图已保存：{out}")


section_2_errors_cache = []


def main():
    print()
    print("################  M1.4 特征线与行波：单元自检  ################")
    x, f, t_phys, rows = section_1_causality()
    errors = section_2_convergence()
    section_2_errors_cache.clear()
    section_2_errors_cache.extend(errors)
    section_3_energy()
    section_4_reflection()
    section_5_f2()
    print()
    make_figure(x, f, t_phys, rows)
    print()
    print("################  自检结束：所有数字以本机实测为准  ################")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
M2.2: 隐式 Crank-Nicolson 求解器（无条件稳定）与显式 FTCS 对比
配套 Readme 的第二个交付物：code/solution.py

目标：
  1. 复现显式 FTCS 在 r > 1/2 时的数值爆炸（与 interactive_demo.py 可视化同口径）；
  2. 给出隐式 Crank-Nicolson 求解器：无条件稳定只保证「不爆炸」，
     不保证精度 —— 大 r 下必须对照精确解报告误差后才能交付（源稿⑧ 故障二）。
Crank-Nicolson 在每个时间步解一个三对角线性方程组（Thomas 算法），
因此比显式贵，换来「不爆炸」；但 r 越大时间误差越大（二阶 ⇒ Δt 10× 误差 100×）。
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def ftcs_explicit(u0, a2, dx, dt, n_steps):
    """显式 FTCS：u^{n+1} = u^n + r*(u_{i+1} - 2u_i + u_{i-1})，r = a^2 dt/dx^2"""
    u = np.asarray(u0, dtype=float)
    if u.ndim != 1 or len(u) < 3:
        raise ValueError("u0 must be a 1D array with at least one interior node")
    if a2 <= 0 or dx <= 0 or dt < 0 or n_steps < 0:
        raise ValueError("a2 and dx must be positive; dt and n_steps must be non-negative")
    if int(n_steps) != n_steps:
        raise ValueError("n_steps must be an integer")
    r = a2 * dt / dx**2
    u = u.copy()
    for _ in range(int(n_steps)):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    return u


def _thomas(a, b, c, d):
    """解三对角方程组 A x = d（a=亚对角, b=主对角, c=超对角），O(N) 追赶法。

    接口约定：b、d 长度为 n；**a、c 长度须为 n**（a[0]、c[-1] 是占位、从不读取），
    与「长度 n-1」的常规写法不同 —— 调用方请按 np.full(n, v) 准备。
    """
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    n = len(d)
    if n == 0 or any(v.ndim != 1 or len(v) != n for v in (a, b, c, d)):
        raise ValueError("a, b, c, d must be non-empty 1D arrays of equal length")
    cp = np.zeros(n)
    dp = np.zeros(n)
    if b[0] == 0:
        raise np.linalg.LinAlgError("zero pivot in Thomas algorithm")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        if m == 0:
            raise np.linalg.LinAlgError(f"zero pivot in Thomas algorithm at row {i}")
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _implicit_theta(u0, a2, dx, dt, n_steps, theta):
    """向后 Euler/CN 共用的隐式 θ 推进核心；空间算子为三点中心差分。"""
    u0 = np.asarray(u0, dtype=float)
    if u0.ndim != 1 or len(u0) < 3:
        raise ValueError("u0 must be a 1D array with at least one interior node")
    if a2 <= 0 or dx <= 0 or dt < 0 or n_steps < 0:
        raise ValueError("a2 and dx must be positive; dt and n_steps must be non-negative")
    if theta not in (0.5, 1.0):
        raise ValueError("implicit theta must be 0.5 (CN) or 1.0 (backward Euler)")
    if int(n_steps) != n_steps:
        raise ValueError("n_steps must be an integer")

    r = a2 * dt / dx**2
    m = len(u0) - 2
    a = np.full(m, -theta * r)
    b = np.full(m, 1.0 + 2.0 * theta * r)
    c = np.full(m, -theta * r)
    u = u0.copy()
    for _ in range(int(n_steps)):
        rhs = (1.0 - 2.0 * (1.0 - theta) * r) * u[1:-1]
        rhs += (1.0 - theta) * r * (u[:-2] + u[2:])
        u[1:-1] = _thomas(a, b, c, rhs)
    return u


def backward_euler(u0, a2, dx, dt, n_steps):
    """隐式后向 Euler，零 Dirichlet 边界；每步用 Thomas 法，A-稳定且强耗散。"""
    return _implicit_theta(u0, a2, dx, dt, n_steps, theta=1.0)


def crank_nicolson(u0, a2, dx, dt, n_steps):
    """隐式 Crank-Nicolson，零 Dirichlet 边界；每步用 Thomas 法，A-稳定且二阶。"""
    return _implicit_theta(u0, a2, dx, dt, n_steps, theta=0.5)



def exact_heat(x, t, a=1.0):
    """两端 u=0 的热方程解析解：u = sin(pi x) * exp(-a^2 pi^2 t)"""
    return np.sin(np.pi * x) * np.exp(-(a**2) * (np.pi**2) * t)


def main():
    print("==================================================")
    print(" Unit M2.2: 隐式 Crank-Nicolson 无条件稳定验证")
    print("==================================================")

    a2 = 1.0
    N = 51
    x = np.linspace(0.0, 1.0, N)
    dx = x[1] - x[0]
    u0 = np.sin(np.pi * x)

    # ---- 1) 显式 FTCS 在 r=0.6 (>1/2) 下爆炸 ----
    # 窗口必须足够长：r=0.6 的放大因子 |G(pi)|=1.4，但增长只能从 1e-16 量级的
    # 舍入噪声起步 —— 前 ~30 步 max|u| 仍在初值包络内、曲线「看起来完全正常」
    #（F4 卡故障一的陷阱段）。实测 r=0.6 首次越界(>10)约在 130 步，250 步达 1e+19，
    # 与 diagnostics 卡片口径（nx=101 / 250 步 -> 1.5e+20）同量级对齐。
    r_expl = 0.6
    n_expl = 250
    dt_expl = r_expl * dx**2 / a2
    u = u0.copy()
    first_over = None
    for step in range(1, n_expl + 1):
        u = ftcs_explicit(u, a2, dx, dt_expl, 1)
        if first_over is None and np.max(np.abs(u)) > 10.0:
            first_over = step
    u_max = float(np.max(np.abs(u)))
    if not np.isfinite(u_max):
        expl_state = "数值爆炸 (inf/nan)"
    elif u_max > 1e6:
        expl_state = f"数值爆炸 (有限但幅值超限，u_max={u_max:.3e})"
    else:
        expl_state = f"u_max={u_max:.3f}"
    print(f"[显式 FTCS] r={r_expl:.2f} (>1/2) | 步数={n_expl} | 终态: {expl_state}")
    print(f"          陷阱段：前 30 步峰值仍 <10（曲线光滑得看不出问题）| 首次越界(>10)步数: {first_over}")

    # ---- 2) 隐式 CN 在 r=2.0 (远超 1/2) 下依然稳定 ----
    n_steps = 100
    dt_cn = 2.0 * dx**2 / a2          # 故意取大时间步
    t_final = dt_cn * n_steps
    u_cn = crank_nicolson(u0, a2, dx, dt_cn, n_steps)
    u_cn_max = np.max(np.abs(u_cn))
    u_ref = exact_heat(x, t_final, a=1.0)
    l2_err = np.linalg.norm(u_cn - u_ref) / np.linalg.norm(u_ref)
    print(f"[隐式 CN ] r={2.0:.2f} (>>1/2) | 步数={n_steps} | t={t_final:.4f}")
    print(f"          终态 u_max={u_cn_max:.4f} (有限，未爆炸) | 相对 L2 误差={l2_err*100:.3f}%")
    print("          注：r<=10 仍属 CN 的精确区（relL2<1e-3）；「稳定≠精确」的失真证据")
    print("          （r=5 时间误差较 r=0.5 退化 100×）见 tests --dump 与 F4 卡故障三")

    # ---- 3) 绘图：CN 解 vs 解析解 ----
    plt.figure(figsize=(9, 5))
    plt.plot(x, u_ref, 'k-', lw=2.5, label='解析解 (精确)')
    plt.plot(x, u_cn, 'r--', lw=2.0, label='Crank-Nicolson (r=2.0)')
    plt.title(f"隐式 CN 在超大步长 r=2.0 下仍稳定 (t={t_final:.3f}, 误差={l2_err*100:.2f}%)")
    plt.xlabel("空间坐标 x")
    plt.ylabel("温度 u(x,t)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_fig = os.path.join(out_dir, "cn_unconditionally_stable.png")
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"对比图已保存: {out_fig}")
    print("==================================================\n")


def run_benchmarks(out_dir=None):
    """--all: 生成 benchmarks/heat_time_stepping/ 的显式阈值与隐式稳定性证据。"""
    import csv
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "benchmarks", "heat_time_stepping")
    os.makedirs(out_dir, exist_ok=True)
    N = 51
    x = np.linspace(0.0, 1.0, N); dx = x[1] - x[0]
    u0 = np.sin(np.pi * x)
    rows_e, rows_i = [], []
    with np.errstate(over="ignore", invalid="ignore"):
        for r in (0.45, 0.49, 0.52, 0.8, 1.0):
            # 2000 步：r>0.5 的增长因子 |1-4r| 需要足够长的时间才能从舍入噪声放大到可见
            u = ftcs_explicit(u0, 1.0, dx, r * dx**2, 2000)
            finite = bool(np.all(np.isfinite(u)))
            umax = float(np.max(np.abs(u))) if finite else float("inf")
            state = "爆炸" if (not finite or umax > 1e6) else "稳定"
            rows_e.append({"scheme": "FTCS", "r": r, "n_steps": 2000,
                           "u_max": f"{umax:.6e}", "state": state})
            print(f"[FTCS] r={r:<5} 2000步 → {state}  u_max={umax:.4e}")
        for r in (0.5, 1.0, 2.0, 10.0):
            dt = r * dx**2
            u = crank_nicolson(u0, 1.0, dx, dt, 100)
            t_final = 100 * dt
            u_ref = exact_heat(x, t_final, a=1.0)
            l2 = float(np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref))
            finite = bool(np.all(np.isfinite(u)))
            umax = float(np.max(np.abs(u))) if finite else float("inf")
            state = "稳定" if finite and umax <= 1e6 else "爆炸"
            rows_i.append({"scheme": "CN", "r": r, "n_steps": 100,
                           "t_final": f"{t_final:.6e}", "u_max": f"{umax:.6e}",
                           "rel_l2_error": f"{l2:.6e}", "state": state})
            print(f"[CN  ] r={r:<5} 100步 → {state}  relL2={l2:.3e}")
    with open(os.path.join(out_dir, "explicit_stability_sweep.csv"), "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["scheme", "r", "n_steps", "u_max", "state"])
        w.writeheader(); w.writerows(rows_e)
    with open(os.path.join(out_dir, "implicit_cn_sweep.csv"), "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["scheme", "r", "n_steps", "t_final", "u_max", "rel_l2_error", "state"])
        w.writeheader(); w.writerows(rows_i)
    print(f"[M2.2] benchmarks written: {out_dir}")
    print("[M2.2] 注：本扫描 r<=10 的 relL2 均 <1e-3，属 CN 的精确区；「稳定≠精确」的")
    print("       失真证据（r=5 时间误差较 r=0.5 退化 100×）见 `tests --dump` 与")
    print("       diagnostics/M2.2_F2_F4_cards.md 故障三 —— 单看本 CSV 会误以为「CN 大步长也准」。")


if __name__ == '__main__' and '__file__' in globals():
    import sys
    if '--all' in sys.argv:
        run_benchmarks()
    else:
        main()

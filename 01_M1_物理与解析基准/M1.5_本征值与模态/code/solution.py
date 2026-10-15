# -*- coding: utf-8 -*-
"""
Unit M1.5: 本征值与模态
求解矩形膜的解析本征值，并与 Q1 双线性有限元的离散谱对照验证。
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros

# 离散本征求解器直接复用 tests/test_membrane_spectrum.py 的 Q1 FEM 实现，
# 避免在交付脚本里再写第二套 —— 两套实现会各自漂移（本文件曾因此报出与失效卡
# 不一致的谱误差）。失效卡与 pytest 用的就是下面这一个实现。
_HERE = os.path.dirname(os.path.abspath(__file__))
_UNIT = os.path.dirname(_HERE)
_spec = importlib.util.spec_from_file_location(
    "_membrane_tests", os.path.join(_UNIT, "tests", "test_membrane_spectrum.py"))
_tests = importlib.util.module_from_spec(_spec)
sys.dont_write_bytecode = True
_spec.loader.exec_module(_tests)
solve_spectrum = _tests.solve_spectrum
analytic_lambda = _tests.analytic_lambda

# 设置中文字体兼容性
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def solve_rectangular_membrane(a=2.0, b=1.0, max_m=4, max_n=4):
    modes = []
    for m in range(1, max_m + 1):
        for n in range(1, max_n + 1):
            val = (np.pi**2) * ((m / a)**2 + (n / b)**2)
            modes.append((val, m, n))
    modes.sort(key=lambda item: item[0])
    return modes

def main():
    print("==================================================")
    print(" Unit M1.5: 本征值与模态（解析谱 ↔ Q1 有限元离散谱）")
    print("==================================================")

    # 1. 矩形膜计算 (a=2, b=1)
    a, b = 2.0, 1.0
    modes = solve_rectangular_membrane(a, b)

    # 可检查数字①：Q1 有限元离散谱 → 解析谱，二阶收敛
    lam_exact = analytic_lambda(1, 1, 1.0, 1.2)
    print(f"\n[离散谱 → 解析谱收敛 (Q1 FEM, a=1.0, b=1.2, λ₁ 解析 = {lam_exact:.6f})]")
    print(f"{'网格':<12}{'λ₁ 数值':<14}{'相对误差 r₁':<14}{'观测阶':<8}")
    prev = None
    for nx, ny in ((8, 10), (16, 20), (32, 40)):
        vals, _, _, _ = solve_spectrum(nx, ny, a=1.0, b=1.2, k=1)
        r = abs(vals[0] - lam_exact) / lam_exact
        order = "-" if prev is None else f"{np.log2(prev / r):.2f}"
        print(f"{f'{nx}x{ny}':<12}{vals[0]:<14.6f}{r:<14.3e}{order:<8}")
        prev = r
    print("  判据：二阶格式的 r₁ 应随网格加密约降 4 倍（观测阶 ≈ 2）。")
    print("  与失效卡 / pytest --dump 同源：两者应逐位相同。")

    print(f"\n[矩形薄膜 (a={a}, b={b}) 前 8 阶理论本征值]:")
    print(f"{'阶数':<6}{'(m, n)':<10}{'λ理论值':<15}{'ω = sqrt(λ)':<15}")
    print("-" * 46)
    for idx, (lam, m, n) in enumerate(modes[:8], start=1):
        omega = np.sqrt(lam)
        print(f"{idx:<6}({m}, {n})     {lam:<15.4f}{omega:<15.4f}")

    # 2. 绘制前 6 阶矩形膜振型云图
    x = np.linspace(0, a, 100)
    y = np.linspace(0, b, 80)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i in range(6):
        ax = axes[i // 3, i % 3]
        lam, m, n = modes[i]
        Z = np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
        cf = ax.contourf(X, Y, Z, 25, cmap='RdBu_r')
        ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=1.5, linestyles='--')
        ax.set_title(f"模态 {i+1}: (m={m}, n={n})\nλ = {lam:.2f}, ω = {np.sqrt(lam):.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        fig.colorbar(cf, ax=ax)

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(output_dir, exist_ok=True)
    out_rect = os.path.join(output_dir, "membrane_modes.png")
    plt.savefig(out_rect, dpi=200)
    plt.close()
    print(f"\n[可视化结果已保存]: {out_rect}")

    # 3. 圆形薄膜 Bessel 模态快速计算展示
    print(f"\n[圆形薄膜 (R=1.0) 前 4 阶贝塞尔根与本征值]:")
    print(f"{'m (周向角向)':<12}{'n (径向零点)':<12}{'零点 α_mn':<15}{'λ = (α_mn/R)^2':<15}")
    print("-" * 54)
    for m in range(0, 3):
        zeros = jn_zeros(m, 2)
        for n_idx, z in enumerate(zeros, start=1):
            print(f"{m:<12}{n_idx:<12}{z:<15.4f}{z**2:<15.4f}")
    print("==================================================\n")

if __name__ == '__main__':
    main()

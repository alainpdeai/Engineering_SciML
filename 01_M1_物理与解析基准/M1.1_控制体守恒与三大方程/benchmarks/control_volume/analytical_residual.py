# -*- coding: utf-8 -*-
"""
analytical_residual.py —— M1.1 验证三件套 · ② 解析残差

把热核基本解与达朗贝尔行波代入相应 PDE 算子，用 SymPy 验证残差严格为零；
并对三大方程的判别式 Δ = B² − 4AC 做符号分类。
残差为零只说明代入一致，物理建模仍要靠守恒与边界条件。

结果以单一来源原则写入本目录 results.json（键 analytical_residual）。
"""
import json
import os

import sympy as sp


def check_residuals():
    """符号验证热核与达朗贝尔解的解析残差，返回可 JSON 化的记录字典。"""
    x, t = sp.symbols("x t", real=True)
    a, c = sp.symbols("a c", positive=True)

    # 1) 热核基本解 Φ = exp(-x²/(4αt)) / √(4παt)，满足 ∂Φ/∂t = α ∂²Φ/∂x²
    Phi = sp.exp(-x**2 / (4 * a**2 * t)) / sp.sqrt(4 * sp.pi * a**2 * t)
    heat_res = sp.simplify(sp.diff(Phi, t) - a**2 * sp.diff(Phi, x, 2))
    print("[通过] 热核残差 =", heat_res)

    # 2) 达朗贝尔通解 u = f(x−ct) + g(x+ct)，满足 ∂²u/∂t² = c² ∂²u/∂x²
    f, g = sp.Function("f"), sp.Function("g")
    u = f(x - c * t) + g(x + c * t)
    wave_res = sp.simplify(sp.diff(u, t, 2) - c**2 * sp.diff(u, x, 2))
    print("[通过] 达朗贝尔残差 =", wave_res)

    # 3) PDE 判别式分类
    disc = {}
    for name, (A, B, C) in {"heat": (-a**2, 0, 0), "wave": (-c**2, 0, 1),
                            "Laplace": (1, 0, 1)}.items():
        disc[name] = sp.simplify(B**2 - 4 * A * C)

    print()
    print("[分类] 热方程   Δ =", disc["heat"], "→ 抛物型")
    print("[分类] 波动方程 Δ =", disc["wave"], "→ 双曲型")
    print("[分类] Laplace   Δ =", disc["Laplace"], "→ 椭圆型")

    assert heat_res == 0, f"热核残差不为零: {heat_res}"
    assert wave_res == 0, f"达朗贝尔残差不为零: {wave_res}"

    return {
        "heat_kernel_residual": str(heat_res),
        "dalembert_residual": str(wave_res),
        "discriminant": {k: str(v) for k, v in disc.items()},
    }


if __name__ == "__main__":
    record = {"analytical_residual": check_residuals()}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    existing = {}
    if os.path.exists(out):
        try:
            existing = json.load(open(out, encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(record)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print("已写入", out)

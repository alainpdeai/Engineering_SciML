# -*- coding: utf-8 -*-
"""
dimension_check.py —— M1.1 验证三件套 · ① 量纲检查

对控制体守恒推导出的两个关键衍生量做 [M, L, T, Θ] 量纲指数核验：
  * 热扩散率  α = k/(ρ c_p)   期望 [L²/T]
  * 波速      c = √(T/μ)      期望 [L/T]

最常见的 F4 失效是把 α（L²/T）当波速（L/T）用——量纲检查是最便宜的防线。
结果以单一来源原则写入本目录 results.json（键 dim_check）。
"""
import json
import os

# 本单元涉及的基础量纲 (M, L, T, Θ)
DIM_K = (1, 1, -3, -1)        # 导热系数 k     W/(m·K)
DIM_RHO = (1, -3, 0, 0)       # 密度 ρ         kg/m³
DIM_CP = (0, 2, -2, -1)       # 比热容 c_p     J/(kg·K)
DIM_TENSION = (1, 1, -2, 0)   # 弦张力 T       N
DIM_MU = (1, -1, 0, 0)        # 线密度 μ       kg/m

EXPECTED_ALPHA = (0, 2, -1, 0)   # [L²/T]
EXPECTED_C = (0.0, 1.0, -1.0, 0.0)  # [L/T]（sqrt 引入 0.5 步长，用浮点）


def dim_div(a, b):
    """量纲相除：a / b。"""
    return tuple(x - y for x, y in zip(a, b))


def dim_sqrt(a):
    """量纲开平方：√a。"""
    return tuple(x / 2.0 for x in a)


def check_dims():
    """核验 α 与 c 的量纲指数，返回可 JSON 化的记录字典。"""
    alpha = dim_div(DIM_K, tuple(a + b for a, b in zip(DIM_RHO, DIM_CP)))
    c = dim_sqrt(dim_div(DIM_TENSION, DIM_MU))

    ok_alpha = tuple(alpha) == EXPECTED_ALPHA
    ok_c = tuple(c) == EXPECTED_C

    print("[通过] 热扩散率 α 量纲 =", alpha)
    print("       期望量纲       =", EXPECTED_ALPHA)
    print("[通过] 波速 c 量纲     =", c)
    print("       期望量纲       =", EXPECTED_C)
    assert ok_alpha, f"热扩散率量纲错误: {alpha}"
    assert ok_c, f"波速量纲错误: {c}"

    return {
        "alpha": {"dims": list(alpha), "expected": list(EXPECTED_ALPHA), "pass": ok_alpha},
        "c": {"dims": list(c), "expected": list(EXPECTED_C), "pass": ok_c},
    }


if __name__ == "__main__":
    record = {"dim_check": check_dims()}
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

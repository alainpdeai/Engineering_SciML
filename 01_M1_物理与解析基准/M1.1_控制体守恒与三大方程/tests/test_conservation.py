# -*- coding: utf-8 -*-
"""
test_conservation.py —— M1.1 单元测试与质量门禁套件
涵盖：
1. 量纲一致性检查（热扩散率 α 与波速 c）
2. 解析残差自洽性检查（SymPy 热核与达朗贝尔行波）
3. 稳态控制体守恒性与边界净通量闭环检查
4. 交付符号推导脚本直接调用
"""
import pytest
import numpy as np
import sympy as sp
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "solution.py"
SPEC = spec_from_file_location("m11_solution", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
m11_solution = module_from_spec(SPEC)
SPEC.loader.exec_module(m11_solution)


def test_delivery_solution_script(capsys):
    """直接调用交付求解脚本，确保符号推导和残差验证可运行。"""
    m11_solution.verify_pde_derivations()
    output = capsys.readouterr().out
    assert "基本解 (热核) 代入残差: 0" in output
    assert "任意行波形式" in output and "代入残差: 0" in output


def test_dimensional_consistency():
    """验证热扩散率与波速的量纲推导"""
    def div(a, b):
        return tuple(x - y for x, y in zip(a, b))

    def sqrt_dim(a):
        return tuple(x / 2.0 for x in a)

    # 基础物理量量纲 (M, L, T, Theta)
    k = (1, 1, -3, -1)           # 导热系数 W/(m·K)
    rho = (1, -3, 0, 0)          # 密度 kg/m^3
    cp = (0, 2, -2, -1)          # 比热容 J/(kg·K)
    tension = (1, 1, -2, 0)      # 弦张力 N
    line_density = (1, -1, 0, 0) # 线密度 kg/m

    # 计算衍生量量纲
    alpha = div(k, tuple(a + b for a, b in zip(rho, cp)))
    c = sqrt_dim(div(tension, line_density))

    # 断言
    assert alpha == (0, 2, -1, 0), f"热扩散率量纲错误: {alpha}"
    assert c == (0.0, 1.0, -1.0, 0.0), f"波速量纲错误: {c}"


def test_analytical_residuals():
    """验证解析解对偏微分算子的精确自洽性（残差为 0）"""
    x, t = sp.symbols("x t", real=True)
    a, c = sp.symbols("a c", positive=True)

    # 1. 热核基本解
    Phi = sp.exp(-x**2 / (4 * a**2 * t)) / sp.sqrt(4 * sp.pi * a**2 * t)
    heat_res = sp.simplify(sp.diff(Phi, t) - a**2 * sp.diff(Phi, x, 2))
    assert heat_res == 0, f"热核残差不为零: {heat_res}"

    # 2. 达朗贝尔通解
    f = sp.Function("f")
    g = sp.Function("g")
    u = f(x - c * t) + g(x + c * t)
    wave_res = sp.simplify(sp.diff(u, t, 2) - c**2 * sp.diff(u, x, 2))
    assert wave_res == 0, f"达朗贝尔残差不为零: {wave_res}"


def test_steady_conservation_residual():
    """验证离散稳态控制体守恒性与边界净通量闭合"""
    x = np.linspace(0.0, 1.0, 101)
    k = 2.0
    Q = 3.0

    # 物理正确的一维稳态温度场 (Q>0 内部隆起)
    T = 20.0 + 80.0 * x + 0.5 * Q / k * x * (1.0 - x)
    # 傅里叶定律 q = -k dT/dx
    # T(x) 是二次多项式，边界必须用二阶单侧差分（edge_order=2）；
    # 默认的 edge_order=1 会在两端各吃掉 O(dx) 的通量，总账出现约 1% 假缺口。
    q = -k * np.gradient(T, x, edge_order=2)
    # 控制体稳态能量守恒残差: -dq/dx + Q = 0
    residual = -np.gradient(q, x, edge_order=2) + Q

    # 内部点残差应在机器精度级别为 0
    inner_residual_max = np.max(np.abs(residual[2:-2]))
    assert inner_residual_max < 1e-8, f"内部守恒残差过大: {inner_residual_max}"

    # 边界净流出应严格等于内部总生成量 Q：
    # 二次场 + 二阶边界差分下该恒等式精确成立，只应剩浮点舍入。
    net_flux = q[-1] - q[0]
    assert np.isclose(net_flux, Q, rtol=1e-10, atol=1e-12), \
        f"净流出通量 {net_flux} 不匹配内部总源项 {Q}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

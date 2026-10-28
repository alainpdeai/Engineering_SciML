# -*- coding: utf-8 -*-
"""M1.6 解析基准库：柱/球坐标与特殊函数。

V3 §6 给 M1.6 的核心实验是「Bessel 解、偏心圆柱」。本包是该基准的**单一来源**：

柱坐标 —— 径向方程 $r^2R''+rR'+(k^2r^2-m^2)R=0$ 的两个分支：
    * $J_m(kr)$   —— 内问题，$r\\to0$ 有界（**正则**）
    * $H_m^{(1)}(kr)$ —— 外问题，满足 Sommerfeld 出射条件（**出射**）
    二者在 $r=1$ 上可给同一组边界数据；**区分它们靠的是 $r=0$ 的正则性，
    不是截断阶数**（F3 vs F4 的分界）。

球坐标 —— 球对称扩散 $u_t=\\alpha r^{-2}\\partial_r(r^2u_r)$，令 $v=ru$ 化为
    $v_t=\\alpha v_{rr}$，$v(0,t)=0$（几何约束，非外加 Dirichlet）。
    系数 $b_n=-2a\\,u_0\\,(-1)^n/(n\\pi)$（= $(2/a)\\int_0^a r\\,u_0\\sin(n\\pi r/a)\\,dr$）。

**与离散侧的关系**：本包只给解析基准。带通量对账、R 扫描、角向截断表的
完整实现在 ``tests/test_cylindrical_spherical.py``（``--dump`` 可复现失效卡全表）。

自检：``python benchmarks/cylindrical_spherical/__init__.py`` 或
      ``python -c "import benchmarks.cylindrical_spherical as b; b.selfcheck()"``
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1, jn, jn_zeros

__all__ = [
    "bessel_radial_branches",
    "regularity_selects_branch",
    "sphere_v_coefficients",
    "sphere_temperature",
    "sommerfeld_residual",
    "selfcheck",
]

PI = np.pi


# ---------------------------------------------------------------- 柱坐标
def bessel_radial_branches(m: int, kr, mode: str = "regular"):
    """第 $m$ 阶径向分支。

    mode='regular' → $J_m(kr)$（内问题，$r\\to0$ 有界）
    mode='outgoing' → $H_m^{(1)}(kr)$（外问题，出射）
    """
    kr = np.asarray(kr, dtype=complex)
    if mode == "regular":
        return jn(m, kr)
    if mode == "outgoing":
        return hankel1(m, kr)
    raise ValueError("mode 只能是 'regular' 或 'outgoing'")


def regularity_selects_branch(m: int, r, r_probe: float = 1e-4):
    """轴心正则性能不能两个分支，返回 ``(正则幅值, 奇异幅值)``。

    $r^m$（正则）与 $r^{-m}$（奇异）在 $r=1$ 上给出**同一组边界数据**，
    但在 $r\\to0$ 处差 $r^{-2m}$ 倍。排除奇异分支的**唯一**依据是 $r=0$ 的正则性；
    增大截断阶数 $M$ 对此毫无帮助——这是 F3 与 F4 的分界。
    """
    r = np.asarray(r, dtype=float)
    return np.abs(r ** m), np.abs(r ** (-m))


# ---------------------------------------------------------------- 球坐标
def sphere_v_coefficients(n_max: int = 400, a: float = 1.0, u0: float = 1.0):
    """球对称扩散中 $v=ru$ 在 $\\sin(n\\pi r/a)$ 上的系数 $b_n$。

    $v(r,0)=r\\,u_0$ ⇒ $b_n=\\frac{2}{a}\\int_0^a r\\,u_0\\sin(n\\pi r/a)\\,dr
              =-\\frac{2a\\,u_0\\,(-1)^n}{n\\pi}$。

    **注意**：$b_n$ 里不能再多除一个 $n\\pi$（这是本单元历史上出过的 P0）。
    """
    n = np.arange(1, n_max + 1)
    return -2.0 * a * u0 * (-1.0) ** n / (n * PI)


def sphere_temperature(r, t: float, alpha: float, a: float, u0: float = 1.0,
                       n_max: int = 400, u_env: float = 0.0):
    """球 $0\\le r\\le a$ 内温度；球心按 $r\\to0$ 极限稳定求值。

    表面 Dirichlet $u(a,t)=u_{\\rm env}$，初始内部温度为 $u_0$。在球心，
    $\\sin(n\\pi r/a)/r$ 的极限是 $n\\pi/a$；用 ``numpy.sinc`` 计算该比值，
    避免直接 ``0/0``。对于 $t=0$ 的球心，显式返回初值 $u_0$。
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = np.arange(1, n_max + 1)
    b = sphere_v_coefficients(n_max, a, u0 - u_env)
    lam = (n * PI / a) ** 2
    # sin(n*pi*r/a)/r = (n*pi/a) * sinc(n*r/a), including its r=0 limit.
    shape = (n * PI / a)[None, :] * np.sinc(np.outer(r / a, n))
    temperature = u_env + shape @ (b * np.exp(-alpha * lam * t))
    if t == 0:
        temperature[r == 0] = u0
    return temperature


def sommerfeld_residual(r, p_func, k, dr: float = 1e-6):
    """Sommerfeld 出射条件残差 $\\partial_rp-ikp$（应在 $r\\to\\infty$ 趋于 0）。"""
    rp = (p_func(r + dr) - p_func(r - dr)) / (2 * dr)
    return rp - 1j * k * p_func(r)


# ---------------------------------------------------------------- 自检
def selfcheck() -> None:
    print("[M1.6 解析基准自检]")

    # 球：t→0 必须回到初值 u0
    a, alpha, u0 = 1.0, 1.0, 100.0
    r = np.array([0.25, 0.5, 0.75])
    u = sphere_temperature(r, 0.0, alpha, a, u0=u0)
    print("  球对称瞬态 t→0 回归（应处处 = u0）：")
    for ri, ui in zip(r, u):
        print(f"    r={ri:.2f}  u={ui:8.3f}  相对误差 {abs(ui - u0) / u0 * 100:.3f}%")

    # 柱：正则 vs 奇异分支在轴心
    m = 3
    reg, sing = regularity_selects_branch(m, np.array([1.0, 1e-2, 1e-4]))
    print(f"  轴心正则性（m={m}）：r=1  正则 {reg[0]:.3e} / 奇异 {sing[0]:.3e}"
          f"（边界数据相同）")
    print(f"                  r=1e-4 正则 {reg[2]:.3e} / 奇异 {sing[2]:.3e}"
          f"（奇异放大 {sing[2] / reg[2]:.1e} 倍）")

    # Bessel 零点（圆膜基准的通用件）
    z = jn_zeros(0, 1)
    print(f"  J_0 首个零点 = {z[0]:.8f}  (lambda = {z[0] ** 2:.8f})")


if __name__ == "__main__":
    selfcheck()

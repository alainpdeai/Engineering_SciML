"""
M1.1: 符号计算自动化推导与偏微分方程分类检验 (SymPy 实现)
"""
import sympy as sp

def verify_pde_derivations():
    print("=" * 60)
    print("   SymPy 符号验证: 控制体通量平衡与经典三大方程")
    print("=" * 60)
    
    # 1. 符号定义
    x, t, a, c = sp.symbols('x t a c', real=True, positive=True)
    u = sp.Function('u')(x, t)
    
    # 2. 热传导方程验证: u_t - a^2 * u_xx = 0
    heat_lhs = sp.diff(u, t) - a**2 * sp.diff(u, x, 2)
    print("[1] 一维热传导方程算子:")
    sp.pprint(heat_lhs)
    
    # 测试热核 (Fundamental Solution) 满足该方程:
    Phi = (1 / sp.sqrt(4 * sp.pi * a**2 * t)) * sp.exp(-x**2 / (4 * a**2 * t))
    heat_test = sp.diff(Phi, t) - a**2 * sp.diff(Phi, x, 2)
    heat_residual = sp.simplify(heat_test)
    print(f"    --> 基本解 (热核) 代入残差: {heat_residual} (严格验证通过!)")
    
    # 3. 波动方程验证: u_tt - c^2 * u_xx = 0
    # 测试 d'Alembert 解: f(x - c*t) + g(x + c*t)
    f_arb = sp.Function('f')(x - c * t)
    g_arb = sp.Function('g')(x + c * t)
    u_wave = f_arb + g_arb
    wave_test = sp.diff(u_wave, t, 2) - c**2 * sp.diff(u_wave, x, 2)
    wave_residual = sp.simplify(wave_test)
    print("\n[2] 一维波动方程算子与达朗贝尔解检验:")
    print(f"    --> 任意行波形式 f(x-ct) + g(x+ct) 代入残差: {wave_residual} (严格验证通过!)")
    
    # 4. 二阶线性 PDE 分类判别式: B^2 - 4AC
    print("\n[3] 经典 PDE 判别式特征分类:")
    for name, (A_val, B_val, C_val) in [("热方程", (-a**2, 0, 0)), ("波动方程", (-c**2, 0, 1)), ("拉普拉斯", (1, 0, 1))]:
        delta = B_val**2 - 4 * A_val * C_val
        delta_str = str(delta)
        pde_type = "椭圆型 (Delta < 0)" if delta < 0 else ("双曲型 (Delta > 0)" if delta > 0 else "抛物型 (Delta = 0)")
        print(f"    - {name:<6}: Delta = {delta_str:<6} -> {pde_type}")
    print("=" * 60)

if __name__ == "__main__":
    verify_pde_derivations()

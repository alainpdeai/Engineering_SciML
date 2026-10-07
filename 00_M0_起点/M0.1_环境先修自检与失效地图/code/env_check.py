"""
工科 Python 科学计算与 AI4Science 环境健康度自检脚本 (v2)

用法（本地终端）:  python code/env_check.py
输出:              终端报告 + assets/env_status.png（环境状态图）

在 Jupyter / 网页里调用:
    import env_check as S
    S.check_environment()                     # 真实检测
    S.check_environment(simulate={"torch"})   # 演示模式：假装 PyTorch 已装好
"""
import importlib
import os
import sys

PACKAGES = [
    ("NumPy", "numpy"),
    ("SciPy", "scipy"),
    ("Matplotlib", "matplotlib"),
    ("SymPy", "sympy"),
    ("PyTorch", "torch"),
]


def _probe(pkg, simulate):
    """返回 (状态, 详情)。状态 ∈ {'PASS', 'SIM', 'FAIL'}"""
    if pkg in simulate:
        return "SIM", "模拟通过（仅演示，不代表真实环境）"
    try:
        mod = importlib.import_module(pkg)
    except ModuleNotFoundError:
        return "FAIL", "尚未安装! 请运行 pip install " + pkg
    except Exception as exc:  # 装了但导入出错：常见于 DLL / 路径 / 编码问题
        return "FAIL", "导入时报错 %s: %s（排查见「三大暗坑」）" % (type(exc).__name__, exc)
    return "PASS", "版本: " + str(getattr(mod, "__version__", "已安装"))


def _draw_status(rows, save_path, show):
    """画环境状态图（标签用英文，避免缺中文字体时出现方块）。"""
    try:
        from matplotlib.patches import Rectangle
        if show:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6.4, 0.55 * len(rows) + 1.0))
        else:  # 命令行模式：不经过 pyplot，避免弹出窗口
            from matplotlib.figure import Figure
            fig = Figure(figsize=(6.4, 0.55 * len(rows) + 1.0))
    except Exception:
        return None

    colors = {"PASS": "#16a34a", "SIM": "#d97706", "FAIL": "#e11d48"}
    words = {"PASS": "PASS", "SIM": "SIMULATED", "FAIL": "FAIL"}
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows))
    ax.axis("off")
    for i, (name, status, detail) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.add_patch(Rectangle((0.1, y + 0.12), 9.8, 0.76, color=colors[status], alpha=0.14, lw=0))
        ax.add_patch(Rectangle((0.1, y + 0.12), 0.16, 0.76, color=colors[status], lw=0))
        ax.text(0.6, y + 0.5, name, va="center", fontsize=12, fontweight="bold")
        ax.text(3.3, y + 0.5, detail, va="center", fontsize=9.5, family="monospace")
        ax.text(9.7, y + 0.5, words[status], va="center", ha="right",
                fontsize=11, fontweight="bold", color=colors[status])
    ax.set_title("Environment self-check (env_check.py)", fontsize=12)
    fig.tight_layout()
    try:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    except Exception:
        pass
    return fig


def check_environment(simulate=(), show=True, save_path="assets/env_status.png"):
    """检测 5 个核心库。全部通过（含模拟）返回 True，否则 False。"""
    simulate = set(simulate)
    print("=" * 60)
    print("      工科数学物理方程与 SciML 科学计算环境自检报告")
    print("=" * 60)
    print("Python 解释器版本:", sys.version.split()[0])

    rows, ok = [], True
    for label, pkg in PACKAGES:
        status, detail = _probe(pkg, simulate)
        tag = "PASS" if status in ("PASS", "SIM") else "FAIL"
        print("  [%s] %-12s -> %s" % (tag, label, detail))
        # 状态图里的详情用英文/数字，避免中文字体缺失
        short = {"PASS": detail.replace("版本: ", "v"),
                 "SIM": "simulated (demo only)",
                 "FAIL": "not installed / import error"}[status]
        rows.append((label, status, short))
        ok = ok and status != "FAIL"

    print("-" * 60)
    if not ok:
        print(">>> 注意: 请补充安装标记为 [FAIL] 的软件包后再继续。")
    elif simulate:
        print(">>> 演示模式: 上面有 %d 项是【模拟】通过，不代表你的真实环境。" % len(simulate))
    else:
        print(">>> 恭喜! 你的 Python 科学计算环境完全就绪，可以开始本课程的所有实验。")
    print("=" * 60)

    fig = _draw_status(rows, save_path, show)
    if fig is not None and not show:
        print("状态图已保存: " + save_path)
    return ok


if __name__ == "__main__":
    sys.exit(0 if check_environment(show=False) else 1)

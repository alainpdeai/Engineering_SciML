"""`python -m benchmarks.cylindrical_spherical` 入口：跑完整自检。

基准模块本体在 __init__.py，这里只做可执行入口，保证 metadata 里
`deliverables: benchmarks/cylindrical_spherical/` 声明的交付物能一条命令跑通。
"""
from . import selfcheck

if __name__ == "__main__":
    selfcheck()

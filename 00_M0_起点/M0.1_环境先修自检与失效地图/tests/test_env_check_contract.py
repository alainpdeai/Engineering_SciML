"""Prevent the M0.1 local environment checker from drifting from the lesson API."""

from importlib.util import module_from_spec, spec_from_file_location
from inspect import signature
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "code" / "env_check.py"
SPEC = spec_from_file_location("m0_env_check", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
env_check = module_from_spec(SPEC)
SPEC.loader.exec_module(env_check)


def test_classroom_api_and_status_chart(tmp_path):
    """The local source supports the calls shown in the interactive lecture."""
    assert list(signature(env_check.check_environment).parameters) == [
        "simulate",
        "show",
        "save_path",
    ]

    chart = tmp_path / "env_status.png"
    completed = env_check.check_environment(
        simulate={"numpy", "scipy", "matplotlib", "sympy", "torch"},
        show=False,
        save_path=str(chart),
    )

    assert completed is True
    assert chart.is_file()
    assert chart.stat().st_size > 0

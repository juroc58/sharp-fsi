"""Pytest wrappers for sharpFSI verification checks (F1-F19).

Each check_* function in verify_solver.py becomes one pytest test.
"""
from __future__ import annotations

import pytest

import verify_solver

_CHECK_FUNCS = [
    (
        name,
        getattr(verify_solver, name),
        pytest.mark.needs_jit if name.startswith("check_f10") else None,
    )
    for name in sorted(dir(verify_solver))
    if name.startswith("check_") and callable(getattr(verify_solver, name))
]


@pytest.mark.parametrize(
    "check_name,check_fn,marker",
    _CHECK_FUNCS,
    ids=lambda v: v[0] if isinstance(v, tuple) else str(v),
)
def test_check(check_name, check_fn, marker):
    """Run a single F-check and assert it passes."""
    assert check_fn(), f"Check {check_name} failed"

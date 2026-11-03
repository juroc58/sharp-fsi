"""Pytest wrappers for sharpFSI verification checks (F1-F19).

Each check_* function in scripts/verify_solver.py becomes one pytest test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

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

def test_bem_kinematics_skips_without_reference(monkeypatch):
    """F10 skips gracefully when sharp-vawt-lab is not importable."""
    import verify_solver

    monkeypatch.setattr(verify_solver.ref_solver, 'find', lambda: None)

    import sys
    sys.path = [p for p in sys.path if 'sharp-vawt-lab' not in p and 'sharp_proj' not in p]

    result = verify_solver.test_bem_kinematics()
    assert result is None, f'Expected None (skip), got {result}'

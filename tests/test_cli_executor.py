"""Tests for CLI script executor."""

import tempfile
from pathlib import Path

import pytest

from strata_fdtd.cli.executor import (
    RestrictedImportError,
    execute_simulation_script,
    validate_solver_object,
)


def test_execute_valid_script():
    """Test executing a valid simulation script."""
    script_content = """
from strata_fdtd import FDTDSolver

solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3)
test_value = 42
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        script_path = Path(f.name)

    try:
        namespace = execute_simulation_script(script_path, script_content)

        assert "solver" in namespace
        assert "test_value" in namespace
        assert namespace["test_value"] == 42
    finally:
        script_path.unlink()


def test_execute_script_with_numpy():
    """Test that numpy imports are allowed."""
    script_content = """
import numpy as np

arr = np.array([1, 2, 3])
result = arr.sum()
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        script_path = Path(f.name)

    try:
        namespace = execute_simulation_script(script_path, script_content)

        assert "result" in namespace
        assert namespace["result"] == 6
    finally:
        script_path.unlink()


def test_execute_script_restricted_import():
    """Test that restricted imports are blocked."""
    script_content = """
import os  # Not allowed!

files = os.listdir('.')
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        script_path = Path(f.name)

    try:
        with pytest.raises(RestrictedImportError) as exc_info:
            execute_simulation_script(script_path, script_content)

        assert "os" in str(exc_info.value)
    finally:
        script_path.unlink()


def test_execute_script_syntax_error():
    """Test that syntax errors are propagated."""
    script_content = """
this is not valid python syntax!
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        script_path = Path(f.name)

    try:
        with pytest.raises(SyntaxError):
            execute_simulation_script(script_path, script_content)
    finally:
        script_path.unlink()


def test_validate_solver_valid():
    """Test validating a valid solver object."""
    from strata_fdtd import FDTDSolver

    solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3)
    namespace = {"solver": solver}

    result = validate_solver_object(namespace)
    assert result is solver


def test_validate_solver_missing():
    """Test validation when solver is missing."""
    namespace = {"other_var": 42}

    with pytest.raises(ValueError) as exc_info:
        validate_solver_object(namespace)

    assert "must define a 'solver'" in str(exc_info.value)


def test_validate_solver_invalid():
    """Test validation when solver is not a valid object."""
    namespace = {"solver": "not a solver"}

    with pytest.raises(ValueError) as exc_info:
        validate_solver_object(namespace)

    assert "missing required methods" in str(exc_info.value)

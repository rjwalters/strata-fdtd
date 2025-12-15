"""Script execution sandbox for FDTD simulations.

This module provides secure execution of user-provided simulation scripts
with restricted imports and controlled namespace.
"""

import sys
from pathlib import Path
from typing import Any


class RestrictedImportError(ImportError):
    """Raised when a disallowed module import is attempted."""

    pass


def execute_simulation_script(
    script_path: Path, script_content: str, verbose: bool = False
) -> dict[str, Any]:
    """Execute simulation script in controlled namespace.

    The script is executed with restricted imports - only specific scientific
    computing and strata_fdtd modules are allowed. This prevents execution of
    potentially harmful code while allowing legitimate simulation scripts.

    Args:
        script_path: Path to the script file (for __file__ and relative imports)
        script_content: Content of the script to execute
        verbose: If True, print debug information

    Returns:
        Namespace dict containing all variables defined by the script

    Raises:
        RestrictedImportError: If script attempts to import disallowed module
        SyntaxError: If script has syntax errors
        Exception: Any exception raised by the script during execution
    """
    # Allowed module prefixes (first component of import path)
    allowed_modules = {
        "strata_fdtd",
        "numpy",
        "np",  # Common alias
        "scipy",
        "math",
        "pathlib",
    }

    # Create restricted namespace
    namespace = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "__builtins__": __builtins__,
    }

    # Patch __import__ to restrict imports
    # __builtins__ can be either a dict or module depending on context
    if isinstance(__builtins__, dict):
        original_import = __builtins__["__import__"]
    else:
        original_import = __builtins__.__import__

    def restricted_import(name, *args, **kwargs):
        """Restricted import that only allows specific modules."""
        # Get the top-level module name
        top_level = name.split(".")[0]

        if top_level not in allowed_modules:
            raise RestrictedImportError(
                f"Import of '{name}' is not allowed in simulation scripts. "
                f"Allowed modules: {', '.join(sorted(allowed_modules))}"
            )

        return original_import(name, *args, **kwargs)

    # Temporarily patch __import__ in namespace
    # Handle both dict and module cases
    if isinstance(namespace["__builtins__"], dict):
        namespace["__builtins__"]["__import__"] = restricted_import
    else:
        namespace["__builtins__"].__import__ = restricted_import

    try:
        # Add script directory to path for relative imports
        script_dir = str(script_path.parent)
        sys.path.insert(0, script_dir)

        if verbose:
            print(f"Executing script: {script_path}")
            print(f"Script directory added to path: {script_dir}")

        # Execute script
        exec(script_content, namespace)

        if verbose:
            defined_vars = [k for k in namespace.keys() if not k.startswith("__")]
            print(f"Script defined variables: {', '.join(defined_vars)}")

    finally:
        # Restore original import
        if isinstance(namespace["__builtins__"], dict):
            namespace["__builtins__"]["__import__"] = original_import
        else:
            namespace["__builtins__"].__import__ = original_import

        # Remove script directory from path
        if script_dir in sys.path:
            sys.path.remove(script_dir)

    return namespace


def validate_solver_object(namespace: dict[str, Any]) -> Any:
    """Validate that namespace contains a valid solver object.

    Args:
        namespace: Namespace dict from script execution

    Returns:
        The solver object

    Raises:
        ValueError: If no solver found or solver is invalid
    """
    solver = namespace.get("solver")

    if solver is None:
        raise ValueError(
            "Script must define a 'solver' variable. "
            "Example: solver = FDTDSolver(shape=(100,100,100), resolution=1e-3)"
        )

    # Basic validation that it looks like a solver
    required_methods = ["run", "step"]
    missing_methods = [m for m in required_methods if not hasattr(solver, m)]

    if missing_methods:
        raise ValueError(
            f"'solver' object is missing required methods: {', '.join(missing_methods)}. "
            f"Make sure it's an FDTDSolver instance."
        )

    return solver

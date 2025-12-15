"""Pytest configuration for strata-fdtd test suite.

This conftest.py handles critical initialization that must occur before
any test imports, including OpenMP conflict resolution.
"""

import os

# =============================================================================
# OpenMP Library Conflict Resolution (Issue #10)
# =============================================================================
# Both PyTorch and our native C++ FDTD kernels link against OpenMP. When both
# are imported in the same Python process, multiple OpenMP runtimes are
# initialized, causing the error:
#
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
#   initialized.
#
# Setting KMP_DUPLICATE_LIB_OK=TRUE allows multiple OpenMP runtimes to coexist.
# This is safe in test environments where:
#   1. We're not relying on OpenMP for numerical accuracy across libraries
#   2. Tests run in isolated processes anyway
#   3. Performance isn't critical during testing
#
# This MUST be set before importing torch or the native extension.
# =============================================================================
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def pytest_configure(config):
    """Called after command line options have been parsed and all plugins
    and initial conftest files been loaded.
    """
    # Ensure the environment variable is set even if conftest imports happen
    # after os.environ is already read by some libraries
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

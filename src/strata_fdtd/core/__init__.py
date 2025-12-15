"""Core FDTD solver components."""

from strata_fdtd.core.grid import NonuniformGrid, UniformGrid
from strata_fdtd.core.solver import (
    POLAR_PATTERNS,
    FDTDSolver,
    GaussianPulse,
    Microphone,
    Probe,
    has_native_kernels,
)
from strata_fdtd.core.solver_gpu import (
    BatchedGPUFDTDSolver,
    GPUFDTDSolver,
    has_gpu_support,
)

__all__ = [
    "FDTDSolver",
    "GPUFDTDSolver",
    "BatchedGPUFDTDSolver",
    "GaussianPulse",
    "Probe",
    "Microphone",
    "POLAR_PATTERNS",
    "UniformGrid",
    "NonuniformGrid",
    "has_native_kernels",
    "has_gpu_support",
]

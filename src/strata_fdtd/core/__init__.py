"""Core FDTD solver components."""

from strata_fdtd.core.solver import (
    FDTDSolver,
    GaussianPulse,
    Probe,
    Microphone,
    POLAR_PATTERNS,
    has_native_kernels,
)
from strata_fdtd.core.solver_gpu import (
    GPUFDTDSolver,
    BatchedGPUFDTDSolver,
    has_gpu_support,
)
from strata_fdtd.core.grid import UniformGrid, NonuniformGrid

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

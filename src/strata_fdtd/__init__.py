"""
Strata FDTD - FDTD acoustic simulation toolchain.

Main exports:
- FDTDSolver: Core 3D acoustic FDTD solver
- GPUFDTDSolver: GPU-accelerated solver
- GaussianPulse: Acoustic source
- PML: Absorbing boundary
- Materials: Acoustic material library
"""

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
from strata_fdtd.boundaries import PML, RigidBoundary, RadiationImpedance

# Submodules for more specific imports
from . import materials
from . import geometry
from . import manufacturing
from . import io
from . import analysis

__version__ = "0.1.0"

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
    "PML",
    "RigidBoundary",
    "RadiationImpedance",
    "has_native_kernels",
    "has_gpu_support",
    "materials",
    "geometry",
    "manufacturing",
    "io",
    "analysis",
]

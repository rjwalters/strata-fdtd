"""
Strata FDTD - FDTD acoustic simulation toolchain.

Main exports:
- FDTDSolver: Core 3D acoustic FDTD solver
- GPUFDTDSolver: GPU-accelerated solver
- GaussianPulse: Point/plane acoustic source
- MembraneSource: Spatially-distributed source with modal patterns
- PML: Absorbing boundary
- Materials: Acoustic material library
"""

# Re-export common analysis functions
from strata_fdtd.analysis import (
    apply_weighting,
    calculate_leq,
    calculate_spl,
)
from strata_fdtd.boundaries import PML, RadiationImpedance, RigidBoundary
from strata_fdtd.core.grid import NonuniformGrid, UniformGrid
from strata_fdtd.core.solver import (
    POLAR_PATTERNS,
    CircularMembraneSource,
    FDTDSolver,
    GaussianPulse,
    MembraneSource,
    Microphone,
    Probe,
    has_native_kernels,
)
from strata_fdtd.core.solver_gpu import (
    BatchedGPUFDTDSolver,
    GPUFDTDSolver,
    has_gpu_support,
)

# Re-export common geometry classes for convenience
from strata_fdtd.geometry import (
    Box,
    Cone,
    Cylinder,
    Difference,
    HelmholtzResonator,
    Horn,
    Intersection,
    LoudspeakerEnclosure,
    Rotate,
    Scale,
    SDFPrimitive,
    SmoothDifference,
    SmoothIntersection,
    SmoothUnion,
    Sphere,
    Translate,
    Union,
)

# Submodules for more specific imports
from . import analysis, geometry, io, manufacturing, materials

__version__ = "0.1.0"

__all__ = [
    # Core solver
    "FDTDSolver",
    "GPUFDTDSolver",
    "BatchedGPUFDTDSolver",
    "GaussianPulse",
    "MembraneSource",
    "CircularMembraneSource",
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
    # Geometry
    "Box",
    "Sphere",
    "Cylinder",
    "Cone",
    "Horn",
    "Union",
    "Intersection",
    "Difference",
    "SmoothUnion",
    "SmoothIntersection",
    "SmoothDifference",
    "Translate",
    "Rotate",
    "Scale",
    "SDFPrimitive",
    "LoudspeakerEnclosure",
    "HelmholtzResonator",
    # Analysis
    "apply_weighting",
    "calculate_spl",
    "calculate_leq",
    # Submodules
    "materials",
    "geometry",
    "manufacturing",
    "io",
    "analysis",
]

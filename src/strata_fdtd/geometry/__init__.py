"""Geometric modeling for FDTD simulations."""

# SDF primitives and CSG operations
from strata_fdtd.geometry.sdf import (
    # Base SDF class
    SDF,
    # Primitives
    Box,
    Sphere,
    Cylinder,
    Cone,
    Horn,
    HelicalTube,
    SpiralHorn,
    # CSG operations
    Union,
    Intersection,
    Difference,
    SmoothUnion,
    SmoothIntersection,
    SmoothDifference,
    # Transformations
    Translation,
    Rotation,
    Scale,
)

# Metamaterial primitives
from strata_fdtd.geometry.primitives import (
    HelmholtzCell,
    Channel,
    TaperedChannel,
    SerpentineChannel,
    SphereCell,
    SphereLattice,
)

# Path generation
from strata_fdtd.geometry.paths import (
    HelixPath,
    SpiralPath,
)

# Loudspeaker enclosures
from strata_fdtd.geometry.loudspeaker import (
    LoudspeakerEnclosure,
    tower_speaker,
    bookshelf_speaker,
)

# Helmholtz resonators
from strata_fdtd.geometry.resonator import (
    HelmholtzResonator,
    helmholtz_array,
)

# Parametric geometry
from strata_fdtd.geometry.parametric import (
    Parameter,
    ParametricPrimitive,
    ParametricHorn,
    GeometryOptimizer,
    ConstraintSet,
)

# Material assignment
from strata_fdtd.geometry.material_assignment import (
    MaterializedGeometry,
    MaterialVolume,
)

__all__ = [
    # SDF
    "SDF",
    "Box",
    "Sphere",
    "Cylinder",
    "Cone",
    "Horn",
    "HelicalTube",
    "SpiralHorn",
    "Union",
    "Intersection",
    "Difference",
    "SmoothUnion",
    "SmoothIntersection",
    "SmoothDifference",
    "Translation",
    "Rotation",
    "Scale",
    # Primitives
    "HelmholtzCell",
    "Channel",
    "TaperedChannel",
    "SerpentineChannel",
    "SphereCell",
    "SphereLattice",
    # Paths
    "HelixPath",
    "SpiralPath",
    # Loudspeakers
    "LoudspeakerEnclosure",
    "tower_speaker",
    "bookshelf_speaker",
    # Resonators
    "HelmholtzResonator",
    "helmholtz_array",
    # Parametric
    "Parameter",
    "ParametricPrimitive",
    "ParametricHorn",
    "GeometryOptimizer",
    "ConstraintSet",
    # Material assignment
    "MaterializedGeometry",
    "MaterialVolume",
]

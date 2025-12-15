"""Geometric modeling for FDTD simulations."""

# SDF primitives and CSG operations
# Loudspeaker enclosures
from strata_fdtd.geometry.loudspeaker import (
    LoudspeakerEnclosure,
    bookshelf_speaker,
    tower_speaker,
)

# Material assignment
from strata_fdtd.geometry.material_assignment import (
    MaterializedGeometry,
    MaterialVolume,
)

# Parametric geometry
from strata_fdtd.geometry.parametric import (
    ConstraintSet,
    GeometryOptimizer,
    Parameter,
    ParametricHorn,
    ParametricPrimitive,
)

# Path generation
from strata_fdtd.geometry.paths import (
    HelixPath,
    SpiralPath,
)

# Metamaterial primitives
from strata_fdtd.geometry.primitives import (
    Channel,
    HelmholtzCell,
    SerpentineChannel,
    SphereCell,
    SphereLattice,
    TaperedChannel,
)

# Helmholtz resonators
from strata_fdtd.geometry.resonator import (
    HelmholtzResonator,
    helmholtz_array,
)
from strata_fdtd.geometry.sdf import (
    # Primitives
    Box,
    Cone,
    Cylinder,
    Difference,
    HelicalTube,
    Horn,
    Intersection,
    Rotate,
    Scale,
    # Base SDF class
    SDFPrimitive,
    SmoothDifference,
    SmoothIntersection,
    SmoothUnion,
    Sphere,
    SpiralHorn,
    # Transformations
    Translate,
    # CSG operations
    Union,
)

__all__ = [
    # SDF
    "SDFPrimitive",
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
    "Translate",
    "Rotate",
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

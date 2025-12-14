"""Frequency-dependent acoustic materials for FDTD simulation.

This module provides a complete system for defining and using
frequency-dependent acoustic materials in FDTD simulations using
the Auxiliary Differential Equation (ADE) method.

Material Types:
    - SimpleMaterial: Non-dispersive (constant properties)
    - PorousMaterial: Fibrous absorbers (JCA model)
    - ViscoelasticSolid: Wood panels (Zener model)
    - DampingMaterial: High-loss damping (extended Zener)
    - HelmholtzResonator: Tuned cavity absorbers
    - MembraneAbsorber: Panel bass traps
    - PerforatedPanel: Distributed resonant absorbers

Example:
    >>> from strata_fdtd.materials import FIBERGLASS_48, MDF_MEDIUM
    >>> from strata_fdtd import FDTDSolver
    >>>
    >>> solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)
    >>> solver.register_material(FIBERGLASS_48, material_id=1)
    >>> solver.register_material(MDF_MEDIUM, material_id=2)
    >>>
    >>> # Set regions
    >>> solver.set_material_region(absorber_mask, material_id=1)
    >>> solver.set_material_region(enclosure_mask, material_id=2)

Material Library:
    Pre-defined materials are available for common speaker construction:

    >>> from strata_fdtd.materials import list_materials, get_material
    >>> print(list_materials("fiberglass"))
    ['fiberglass_24kg', 'fiberglass_32kg', 'fiberglass_48kg', 'fiberglass_96kg']
    >>> mat = get_material("fiberglass_48kg")
"""

# Base classes
from .base import (
    AcousticMaterial,
    Material,
    Pole,
    PoleType,
    SimpleMaterial,
)

# Pre-defined material library
from .library import (
    ACOUSTIC_FOAM_DENSE,
    # Foam
    ACOUSTIC_FOAM_OPEN,
    # Fluids
    AIR_15C,
    AIR_20C,
    AIR_25C,
    # Damping
    BUTYL_RUBBER,
    COTTON_BATTING,
    EPDM_RUBBER,
    # Fiberglass
    FIBERGLASS_24,
    FIBERGLASS_32,
    FIBERGLASS_48,
    FIBERGLASS_96,
    HARDWOOD_OAK,
    # Registry functions
    MATERIALS,
    MDF_HIGH,
    # MDF
    MDF_LIGHT,
    MDF_MEDIUM,
    # Mineral wool
    MINERAL_WOOL_40,
    MINERAL_WOOL_60,
    MINERAL_WOOL_100,
    NEOPRENE,
    OSB,
    # Other wood
    PARTICLE_BOARD,
    # Plywood
    PLYWOOD_BIRCH,
    PLYWOOD_MARINE,
    PLYWOOD_SOFTWOOD,
    # Fibrous
    POLYESTER_BATTING,
    POLYURETHANE_FOAM,
    SILICONE_RUBBER,
    SORBOTHANE,
    WATER_20C,
    WOOL_FELT,
    get_material,
    list_categories,
    list_materials,
    material_summary,
)

# Porous materials (JCA model)
from .porous import PorousMaterial

# Resonant absorbers
from .resonators import (
    HelmholtzResonator,
    MembraneAbsorber,
    PerforatedPanel,
    QuarterWaveResonator,
)

# Viscoelastic solids (Zener model)
from .solids import (
    DampingMaterial,
    ImpedanceTube,
    ViscoelasticSolid,
)

__all__ = [
    # Base classes
    "Pole",
    "PoleType",
    "AcousticMaterial",
    "Material",
    "SimpleMaterial",
    # Material types
    "PorousMaterial",
    "ViscoelasticSolid",
    "DampingMaterial",
    "HelmholtzResonator",
    "MembraneAbsorber",
    "PerforatedPanel",
    "QuarterWaveResonator",
    # Utilities
    "ImpedanceTube",
    # Library - Fluids
    "AIR_15C",
    "AIR_20C",
    "AIR_25C",
    "WATER_20C",
    # Library - Fiberglass
    "FIBERGLASS_24",
    "FIBERGLASS_32",
    "FIBERGLASS_48",
    "FIBERGLASS_96",
    # Library - Mineral wool
    "MINERAL_WOOL_40",
    "MINERAL_WOOL_60",
    "MINERAL_WOOL_100",
    # Library - Foam
    "ACOUSTIC_FOAM_OPEN",
    "ACOUSTIC_FOAM_DENSE",
    "POLYURETHANE_FOAM",
    # Library - Fibrous
    "POLYESTER_BATTING",
    "WOOL_FELT",
    "COTTON_BATTING",
    # Library - MDF
    "MDF_LIGHT",
    "MDF_MEDIUM",
    "MDF_HIGH",
    # Library - Plywood
    "PLYWOOD_BIRCH",
    "PLYWOOD_MARINE",
    "PLYWOOD_SOFTWOOD",
    # Library - Other wood
    "PARTICLE_BOARD",
    "OSB",
    "HARDWOOD_OAK",
    # Library - Damping
    "BUTYL_RUBBER",
    "NEOPRENE",
    "SORBOTHANE",
    "EPDM_RUBBER",
    "SILICONE_RUBBER",
    # Registry
    "MATERIALS",
    "get_material",
    "list_materials",
    "list_categories",
    "material_summary",
]

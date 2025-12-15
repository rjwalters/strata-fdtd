"""Pre-defined acoustic material library for speaker construction.

This module provides a library of common acoustic materials with
parameters derived from literature and measurement data. Materials
are organized by category:

- Fluids: Air at various temperatures
- Porous absorbers: Fiberglass, mineral wool, foam, batting
- Enclosure materials: MDF, plywood, particle board
- Damping materials: Butyl rubber, neoprene, sorbothane

All materials are ready to use with the FDTD solver:

    >>> from strata_fdtd.materials import FIBERGLASS_48
    >>> solver.register_material(FIBERGLASS_48, material_id=1)

Material parameters are based on:
- ISO 10534 impedance tube measurements (porous materials)
- DIN EN 29052 flow resistivity measurements
- Published acoustic property databases
"""

from .base import SimpleMaterial
from .porous import PorousMaterial
from .solids import DampingMaterial, ViscoelasticSolid

# =============================================================================
# Fluids
# =============================================================================

AIR_20C = SimpleMaterial(
    name="air_20C",
    _rho=1.204,
    _c=343.0,
)
"""Air at 20°C, 1 atm."""

AIR_25C = SimpleMaterial(
    name="air_25C",
    _rho=1.184,
    _c=346.3,
)
"""Air at 25°C, 1 atm."""

AIR_15C = SimpleMaterial(
    name="air_15C",
    _rho=1.225,
    _c=340.3,
)
"""Air at 15°C, 1 atm (standard atmosphere)."""

WATER_20C = SimpleMaterial(
    name="water_20C",
    _rho=998.2,
    _c=1482.0,
)
"""Fresh water at 20°C."""

# =============================================================================
# Porous Absorbers - Fiberglass
# =============================================================================

FIBERGLASS_24 = PorousMaterial(
    name="fiberglass_24kg",
    flow_resistivity=8000,
    porosity=0.99,
    tortuosity=1.02,
    viscous_length=200e-6,
    thermal_length=400e-6,
    num_poles=6,
)
"""Fiberglass insulation, 24 kg/m³ density (light density, e.g., R-11)."""

FIBERGLASS_32 = PorousMaterial(
    name="fiberglass_32kg",
    flow_resistivity=14000,
    porosity=0.98,
    tortuosity=1.03,
    viscous_length=150e-6,
    thermal_length=300e-6,
    num_poles=6,
)
"""Fiberglass insulation, 32 kg/m³ density (medium density)."""

FIBERGLASS_48 = PorousMaterial(
    name="fiberglass_48kg",
    flow_resistivity=25000,
    porosity=0.98,
    tortuosity=1.04,
    viscous_length=100e-6,
    thermal_length=200e-6,
    num_poles=6,
)
"""Fiberglass insulation, 48 kg/m³ density (high density, e.g., OC 703)."""

FIBERGLASS_96 = PorousMaterial(
    name="fiberglass_96kg",
    flow_resistivity=60000,
    porosity=0.96,
    tortuosity=1.06,
    viscous_length=60e-6,
    thermal_length=120e-6,
    num_poles=6,
)
"""Fiberglass board, 96 kg/m³ density (rigid board, e.g., OC 705)."""

# =============================================================================
# Porous Absorbers - Mineral Wool
# =============================================================================

MINERAL_WOOL_40 = PorousMaterial(
    name="mineral_wool_40kg",
    flow_resistivity=20000,
    porosity=0.97,
    tortuosity=1.04,
    viscous_length=80e-6,
    thermal_length=160e-6,
    num_poles=6,
)
"""Mineral wool (rock wool), 40 kg/m³ density."""

MINERAL_WOOL_60 = PorousMaterial(
    name="mineral_wool_60kg",
    flow_resistivity=35000,
    porosity=0.96,
    tortuosity=1.05,
    viscous_length=60e-6,
    thermal_length=120e-6,
    num_poles=6,
)
"""Mineral wool (rock wool), 60 kg/m³ density."""

MINERAL_WOOL_100 = PorousMaterial(
    name="mineral_wool_100kg",
    flow_resistivity=70000,
    porosity=0.94,
    tortuosity=1.08,
    viscous_length=40e-6,
    thermal_length=80e-6,
    num_poles=6,
)
"""Mineral wool board, 100 kg/m³ density (high density board)."""

# =============================================================================
# Porous Absorbers - Foam
# =============================================================================

ACOUSTIC_FOAM_OPEN = PorousMaterial(
    name="acoustic_foam_open",
    flow_resistivity=12000,
    porosity=0.97,
    tortuosity=1.10,
    viscous_length=150e-6,
    thermal_length=300e-6,
    num_poles=6,
)
"""Open-cell acoustic foam (e.g., melamine foam, egg crate)."""

ACOUSTIC_FOAM_DENSE = PorousMaterial(
    name="acoustic_foam_dense",
    flow_resistivity=30000,
    porosity=0.95,
    tortuosity=1.15,
    viscous_length=80e-6,
    thermal_length=160e-6,
    num_poles=6,
)
"""Dense acoustic foam (e.g., pyramid foam)."""

POLYURETHANE_FOAM = PorousMaterial(
    name="polyurethane_foam",
    flow_resistivity=15000,
    porosity=0.98,
    tortuosity=1.08,
    viscous_length=120e-6,
    thermal_length=240e-6,
    num_poles=6,
)
"""Polyurethane foam, open cell (furniture foam grade)."""

# =============================================================================
# Porous Absorbers - Fibrous Materials
# =============================================================================

POLYESTER_BATTING = PorousMaterial(
    name="polyester_batting",
    flow_resistivity=5000,
    porosity=0.99,
    tortuosity=1.01,
    viscous_length=300e-6,
    thermal_length=600e-6,
    num_poles=6,
)
"""Polyester fiberfill batting (speaker cabinet stuffing)."""

WOOL_FELT = PorousMaterial(
    name="wool_felt",
    flow_resistivity=30000,
    porosity=0.95,
    tortuosity=1.30,
    viscous_length=80e-6,
    thermal_length=160e-6,
    num_poles=6,
)
"""Wool felt (dense natural fiber)."""

COTTON_BATTING = PorousMaterial(
    name="cotton_batting",
    flow_resistivity=8000,
    porosity=0.98,
    tortuosity=1.05,
    viscous_length=200e-6,
    thermal_length=400e-6,
    num_poles=6,
)
"""Cotton batting (natural fiber)."""

# =============================================================================
# Enclosure Materials - MDF
# =============================================================================

MDF_LIGHT = ViscoelasticSolid(
    name="MDF_light",
    density=600,
    youngs_modulus=2.8e9,
    loss_factor=0.035,
    poissons_ratio=0.25,
    num_poles=3,
)
"""Light density MDF, ~600 kg/m³ (furniture grade)."""

MDF_MEDIUM = ViscoelasticSolid(
    name="MDF_medium",
    density=720,
    youngs_modulus=3.5e9,
    loss_factor=0.030,
    poissons_ratio=0.25,
    num_poles=3,
)
"""Medium density fiberboard, ~720 kg/m³ (standard grade)."""

MDF_HIGH = ViscoelasticSolid(
    name="MDF_high",
    density=850,
    youngs_modulus=4.2e9,
    loss_factor=0.025,
    poissons_ratio=0.25,
    num_poles=3,
)
"""High density MDF, ~850 kg/m³ (high quality speaker cabinets)."""

# =============================================================================
# Enclosure Materials - Plywood
# =============================================================================

PLYWOOD_BIRCH = ViscoelasticSolid(
    name="plywood_birch",
    density=680,
    youngs_modulus=12.0e9,
    loss_factor=0.015,
    poissons_ratio=0.35,
    num_poles=3,
)
"""Baltic birch plywood, void-free."""

PLYWOOD_MARINE = ViscoelasticSolid(
    name="plywood_marine",
    density=700,
    youngs_modulus=10.0e9,
    loss_factor=0.020,
    poissons_ratio=0.30,
    num_poles=3,
)
"""Marine grade plywood (water resistant adhesive)."""

PLYWOOD_SOFTWOOD = ViscoelasticSolid(
    name="plywood_softwood",
    density=550,
    youngs_modulus=8.0e9,
    loss_factor=0.020,
    poissons_ratio=0.35,
    num_poles=3,
)
"""Softwood construction plywood (CDX grade)."""

# =============================================================================
# Enclosure Materials - Other Wood Products
# =============================================================================

PARTICLE_BOARD = ViscoelasticSolid(
    name="particle_board",
    density=650,
    youngs_modulus=2.5e9,
    loss_factor=0.040,
    poissons_ratio=0.25,
    num_poles=3,
)
"""Particle board (chipboard)."""

OSB = ViscoelasticSolid(
    name="OSB",
    density=620,
    youngs_modulus=4.5e9,
    loss_factor=0.025,
    poissons_ratio=0.30,
    num_poles=3,
)
"""Oriented strand board."""

HARDWOOD_OAK = ViscoelasticSolid(
    name="hardwood_oak",
    density=750,
    youngs_modulus=12.5e9,
    loss_factor=0.010,
    poissons_ratio=0.35,
    num_poles=3,
)
"""Red oak hardwood."""

# =============================================================================
# Damping Materials
# =============================================================================

BUTYL_RUBBER = DampingMaterial(
    name="butyl_rubber",
    density=1200,
    youngs_modulus=1.0e6,
    loss_factor=0.40,
    poissons_ratio=0.49,
    num_poles=5,
)
"""Butyl rubber damping sheet (e.g., Dynamat, Second Skin)."""

NEOPRENE = DampingMaterial(
    name="neoprene",
    density=1300,
    youngs_modulus=2.0e6,
    loss_factor=0.15,
    poissons_ratio=0.49,
    num_poles=5,
)
"""Neoprene rubber."""

SORBOTHANE = DampingMaterial(
    name="sorbothane",
    density=1100,
    youngs_modulus=0.5e6,
    loss_factor=0.50,
    poissons_ratio=0.49,
    num_poles=5,
)
"""Sorbothane viscoelastic polymer (maximum damping)."""

EPDM_RUBBER = DampingMaterial(
    name="EPDM_rubber",
    density=1150,
    youngs_modulus=3.0e6,
    loss_factor=0.12,
    poissons_ratio=0.49,
    num_poles=5,
)
"""EPDM rubber (weather sealing, gaskets)."""

SILICONE_RUBBER = DampingMaterial(
    name="silicone_rubber",
    density=1100,
    youngs_modulus=1.5e6,
    loss_factor=0.08,
    poissons_ratio=0.49,
    num_poles=5,
)
"""Silicone rubber (moderate damping, temperature stable)."""


# =============================================================================
# Material Registry
# =============================================================================

# Organized dictionary of all materials by category
MATERIALS = {
    "fluids": {
        "air_20C": AIR_20C,
        "air_25C": AIR_25C,
        "air_15C": AIR_15C,
        "water_20C": WATER_20C,
    },
    "fiberglass": {
        "fiberglass_24kg": FIBERGLASS_24,
        "fiberglass_32kg": FIBERGLASS_32,
        "fiberglass_48kg": FIBERGLASS_48,
        "fiberglass_96kg": FIBERGLASS_96,
    },
    "mineral_wool": {
        "mineral_wool_40kg": MINERAL_WOOL_40,
        "mineral_wool_60kg": MINERAL_WOOL_60,
        "mineral_wool_100kg": MINERAL_WOOL_100,
    },
    "foam": {
        "acoustic_foam_open": ACOUSTIC_FOAM_OPEN,
        "acoustic_foam_dense": ACOUSTIC_FOAM_DENSE,
        "polyurethane_foam": POLYURETHANE_FOAM,
    },
    "fibrous": {
        "polyester_batting": POLYESTER_BATTING,
        "wool_felt": WOOL_FELT,
        "cotton_batting": COTTON_BATTING,
    },
    "mdf": {
        "MDF_light": MDF_LIGHT,
        "MDF_medium": MDF_MEDIUM,
        "MDF_high": MDF_HIGH,
    },
    "plywood": {
        "plywood_birch": PLYWOOD_BIRCH,
        "plywood_marine": PLYWOOD_MARINE,
        "plywood_softwood": PLYWOOD_SOFTWOOD,
    },
    "other_wood": {
        "particle_board": PARTICLE_BOARD,
        "OSB": OSB,
        "hardwood_oak": HARDWOOD_OAK,
    },
    "damping": {
        "butyl_rubber": BUTYL_RUBBER,
        "neoprene": NEOPRENE,
        "sorbothane": SORBOTHANE,
        "EPDM_rubber": EPDM_RUBBER,
        "silicone_rubber": SILICONE_RUBBER,
    },
}


def get_material(name: str):
    """Look up a material by name.

    Args:
        name: Material name (case-insensitive)

    Returns:
        AcousticMaterial instance

    Raises:
        KeyError: If material not found
    """
    name_lower = name.lower()

    # Search all categories
    for category in MATERIALS.values():
        for mat_name, material in category.items():
            if mat_name.lower() == name_lower:
                return material

    raise KeyError(f"Material '{name}' not found. Use list_materials() to see available materials.")


def list_materials(category: str | None = None) -> list[str]:
    """List available materials.

    Args:
        category: Optional category filter (e.g., "fiberglass", "mdf")

    Returns:
        List of material names
    """
    if category is not None:
        if category not in MATERIALS:
            raise KeyError(f"Unknown category '{category}'. Available: {list(MATERIALS.keys())}")
        return list(MATERIALS[category].keys())

    # All materials
    all_materials = []
    for cat_materials in MATERIALS.values():
        all_materials.extend(cat_materials.keys())
    return all_materials


def list_categories() -> list[str]:
    """List available material categories."""
    return list(MATERIALS.keys())


def material_summary(name: str) -> str:
    """Get a summary of material properties.

    Args:
        name: Material name

    Returns:
        Multi-line string with material properties
    """
    material = get_material(name)
    return material.summary()

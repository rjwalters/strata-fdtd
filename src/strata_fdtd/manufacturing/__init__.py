"""Manufacturing tools for lamination, constraints, and export."""

# Lamination (Slice/Stack geometry)
from strata_fdtd.manufacturing.lamination import (
    Slice,
    Stack,
    Violation,
)

# Manufacturing constraints
from strata_fdtd.manufacturing.constraints import (
    check_connectivity,
    check_min_wall,
    check_min_gap,
    check_slice_alignment,
)

# SDF ↔ Stack conversion
from strata_fdtd.manufacturing.conversion import (
    VoxelSDF,
    sdf_to_stack,
    stack_to_sdf,
)

# CAD export (DXF, STL, JSON)
from strata_fdtd.manufacturing.export import (
    export_dxf,
    export_stl,
    export_json,
)

__all__ = [
    # Lamination
    "Slice",
    "Stack",
    "Violation",
    # Constraints
    "check_connectivity",
    "check_min_wall",
    "check_min_gap",
    "check_slice_alignment",
    # Conversion
    "VoxelSDF",
    "sdf_to_stack",
    "stack_to_sdf",
    # Export
    "export_dxf",
    "export_stl",
    "export_json",
]

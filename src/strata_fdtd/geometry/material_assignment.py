"""Material region assignment for FDTD geometries.

This module provides classes for assigning different acoustic materials to
regions within SDF geometries, enabling simulation of heterogeneous acoustic
materials (air, absorbers, rigid boundaries).

Classes:
    MaterializedGeometry: SDF geometry with material assignments
    MaterialVolume: Convenience class for box-shaped material regions

Example:
    >>> from strata_fdtd import MaterializedGeometry, MaterialVolume
    >>> from strata_fdtd.sdf import Box, Difference
    >>> from strata_fdtd.materials import AIR_20C, FIBERGLASS_48
    >>>
    >>> # Create enclosure geometry
    >>> outer = Box(center=(0.15, 0.15, 0.2), size=(0.3, 0.3, 0.4))
    >>> inner = Box(center=(0.15, 0.15, 0.2), size=(0.26, 0.26, 0.36))
    >>> enclosure = Difference(outer, inner)
    >>>
    >>> # Add material assignments
    >>> mat_geo = MaterializedGeometry(enclosure, default_material=AIR_20C)
    >>>
    >>> # Absorber region at back of enclosure
    >>> absorber_region = Box(center=(0.15, 0.28, 0.2), size=(0.2, 0.04, 0.3))
    >>> mat_geo.set_material(absorber_region, FIBERGLASS_48)
    >>>
    >>> # Voxelize with materials
    >>> geometry_mask, material_ids = mat_geo.voxelize_with_materials(grid)
    >>> material_table = mat_geo.get_material_table()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from strata_fdtd.grid import NonuniformGrid, UniformGrid
    from strata_fdtd.materials.base import AcousticMaterial
    from strata_fdtd.geometry.sdf import Box, SDFPrimitive


@dataclass
class MaterializedGeometry:
    """Geometry with material assignments.

    Wraps an SDF geometry and adds material regions. The geometry defines
    the air/solid boundary, while material regions assign acoustic properties
    to different parts of the air volume.

    Material assignments are applied in the order they are added, with later
    assignments overriding earlier ones. Material ID 0 is always reserved for
    the default material (typically air).

    Args:
        geometry: SDF primitive defining the overall geometry
        default_material: Material for unassigned regions (default: air at 20°C)

    Attributes:
        geometry: The underlying SDF geometry
        default_material: Default material properties
        _material_regions: List of (region SDF, material, material_id) tuples
        _material_id_counter: Counter for assigning unique material IDs

    Example:
        >>> # Create geometry with absorber region
        >>> box = Box(center=(0.1, 0.1, 0.1), size=(0.2, 0.2, 0.2))
        >>> mat_geo = MaterializedGeometry(box)
        >>>
        >>> # Add absorber in one region
        >>> absorber_region = Box(center=(0.05, 0.1, 0.1), size=(0.05, 0.15, 0.15))
        >>> mat_id = mat_geo.set_material(absorber_region, FIBERGLASS_48)
        >>>
        >>> # Voxelize
        >>> mask, mat_ids = mat_geo.voxelize_with_materials(grid)
    """

    geometry: SDFPrimitive
    default_material: AcousticMaterial | None = None

    def __post_init__(self):
        """Initialize material tracking."""
        self._material_regions: list[tuple[SDFPrimitive, AcousticMaterial, int]] = []
        self._material_id_counter = 1  # 0 reserved for default

        # Import default air material if not provided
        if self.default_material is None:
            from strata_fdtd.materials.library import AIR_20C

            self.default_material = AIR_20C

    def set_material(
        self,
        region: SDFPrimitive,
        material: AcousticMaterial,
    ) -> int:
        """Assign material to a region.

        The region is defined by an SDF primitive. Only cells that are both
        inside the overall geometry AND inside the region will be assigned
        this material.

        Later assignments override earlier ones in overlapping regions.

        Args:
            region: SDF defining the material region
            material: Acoustic material properties

        Returns:
            Material ID assigned to this region (1-255)

        Raises:
            ValueError: If maximum material count (255) is exceeded

        Example:
            >>> mat_geo = MaterializedGeometry(box)
            >>> absorber = Box(center=(0.05, 0.1, 0.1), size=(0.05, 0.15, 0.15))
            >>> mat_id = mat_geo.set_material(absorber, FIBERGLASS_48)
            >>> print(f"Assigned material ID: {mat_id}")
        """
        if self._material_id_counter > 255:
            raise ValueError("Maximum of 255 material regions exceeded")

        material_id = self._material_id_counter
        self._material_id_counter += 1
        self._material_regions.append((region, material, material_id))
        return material_id

    def voxelize_with_materials(
        self,
        grid: UniformGrid | NonuniformGrid,
    ) -> tuple[NDArray[np.bool_], NDArray[np.uint8]]:
        """Voxelize geometry and material assignments to grid.

        Returns both the geometry mask (air/solid boundary) and material IDs
        for each cell. Material ID 0 represents the default material.

        Args:
            grid: Simulation grid (UniformGrid or NonuniformGrid)

        Returns:
            Tuple of:
            - geometry_mask: bool array (True=air/material, False=rigid solid)
            - material_ids: uint8 array of material IDs (0 = default material)

        Example:
            >>> grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)
            >>> geometry_mask, material_ids = mat_geo.voxelize_with_materials(grid)
            >>> print(f"Cells with absorber: {np.sum(material_ids > 0)}")
        """
        # Get base geometry mask (defines air vs solid boundary)
        geometry_mask = self.geometry.voxelize(grid)

        # Initialize material IDs to default (0)
        material_ids = np.zeros(grid.shape, dtype=np.uint8)

        # Assign materials in order (later assignments override earlier)
        for region, _material, mat_id in self._material_regions:
            region_mask = region.voxelize(grid)
            # Only assign within air regions (not in solids)
            valid = geometry_mask & region_mask
            material_ids[valid] = mat_id

        return geometry_mask, material_ids

    def get_material_table(self) -> dict[int, AcousticMaterial]:
        """Get mapping from material ID to material properties.

        Returns a dictionary mapping material IDs (0-255) to their associated
        acoustic material properties. ID 0 is always the default material.

        Returns:
            Dict mapping material_id -> AcousticMaterial

        Example:
            >>> table = mat_geo.get_material_table()
            >>> for mat_id, material in table.items():
            ...     print(f"ID {mat_id}: {material.name}")
        """
        table = {0: self.default_material}
        for _region, material, mat_id in self._material_regions:
            table[mat_id] = material
        return table

    def material_count(self) -> int:
        """Get number of distinct materials (including default).

        Returns:
            Number of materials (minimum 1 for default material)
        """
        return len(self._material_regions) + 1  # +1 for default

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MaterializedGeometry("
            f"geometry={self.geometry.__class__.__name__}, "
            f"materials={self.material_count()})"
        )


@dataclass
class MaterialVolume:
    """A box-shaped region filled with a specific material.

    Convenience class for the common case of rectangular material regions.
    Automatically creates a Box SDF from corner coordinates or center+size.

    Args:
        bounds: Tuple of (min_corner, max_corner) arrays, or None if using center/size
        material: Acoustic material properties
        center: (x, y, z) center position (alternative to bounds)
        size: (width, height, depth) dimensions (alternative to bounds)

    Attributes:
        bounds: (min_corner, max_corner) defining the volume
        material: The acoustic material
        region: The Box SDF primitive for this volume

    Example:
        >>> # Create from corners
        >>> volume = MaterialVolume(
        ...     bounds=(np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1])),
        ...     material=FIBERGLASS_48
        ... )
        >>>
        >>> # Create from center and size
        >>> volume = MaterialVolume.from_center_size(
        ...     center=(0.05, 0.05, 0.05),
        ...     size=(0.1, 0.1, 0.1),
        ...     material=FIBERGLASS_48
        ... )
        >>>
        >>> # Use with MaterializedGeometry
        >>> mat_geo.set_material(volume.region, volume.material)
    """

    material: AcousticMaterial
    bounds: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None
    center: tuple[float, float, float] | None = None
    size: tuple[float, float, float] | None = None

    def __post_init__(self):
        """Validate and compute bounds."""
        if self.bounds is None:
            if self.center is None or self.size is None:
                raise ValueError("Must provide either bounds or (center, size)")

            # Compute bounds from center and size
            center_arr = np.array(self.center, dtype=np.float64)
            size_arr = np.array(self.size, dtype=np.float64)
            half_size = size_arr / 2
            self.bounds = (center_arr - half_size, center_arr + half_size)
        else:
            # Validate bounds
            if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
                raise ValueError("bounds must be tuple of (min_corner, max_corner)")

            min_corner, max_corner = self.bounds
            min_corner = np.asarray(min_corner, dtype=np.float64)
            max_corner = np.asarray(max_corner, dtype=np.float64)

            if min_corner.shape != (3,) or max_corner.shape != (3,):
                raise ValueError("bounds corners must be 3-element arrays")

            if np.any(min_corner >= max_corner):
                raise ValueError("min_corner must be less than max_corner in all dimensions")

            self.bounds = (min_corner, max_corner)

    @property
    def region(self) -> Box:
        """Get SDF primitive for this volume.

        Returns:
            Box SDF primitive matching this volume's bounds
        """
        from strata_fdtd.geometry.sdf import Box

        min_corner, max_corner = self.bounds
        return Box(min_corner=tuple(min_corner), max_corner=tuple(max_corner))

    @classmethod
    def from_center_size(
        cls,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        material: AcousticMaterial,
    ) -> MaterialVolume:
        """Create volume from center position and size.

        Args:
            center: (x, y, z) center position
            size: (width, height, depth) dimensions
            material: Acoustic material

        Returns:
            MaterialVolume with specified center and size

        Example:
            >>> volume = MaterialVolume.from_center_size(
            ...     center=(0.15, 0.15, 0.2),
            ...     size=(0.1, 0.1, 0.05),
            ...     material=FIBERGLASS_48
            ... )
        """
        return cls(material=material, center=center, size=size)

    @classmethod
    def from_corners(
        cls,
        min_corner: tuple[float, float, float] | NDArray[np.floating],
        max_corner: tuple[float, float, float] | NDArray[np.floating],
        material: AcousticMaterial,
    ) -> MaterialVolume:
        """Create volume from corner coordinates.

        Args:
            min_corner: (x_min, y_min, z_min) corner
            max_corner: (x_max, y_max, z_max) corner
            material: Acoustic material

        Returns:
            MaterialVolume with specified corners

        Example:
            >>> volume = MaterialVolume.from_corners(
            ...     min_corner=(0.1, 0.1, 0.175),
            ...     max_corner=(0.2, 0.2, 0.225),
            ...     material=FIBERGLASS_48
            ... )
        """
        min_arr = np.asarray(min_corner, dtype=np.float64)
        max_arr = np.asarray(max_corner, dtype=np.float64)
        return cls(material=material, bounds=(min_arr, max_arr))

    def __repr__(self) -> str:
        """String representation."""
        min_c, max_c = self.bounds
        return (
            f"MaterialVolume("
            f"bounds=({min_c}, {max_c}), "
            f"material={self.material.name})"
        )

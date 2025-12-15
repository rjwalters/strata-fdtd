"""
SDF to Stack conversion for manufacturing integration.

This module bridges simulation geometry (SDFs) with fabrication (laser-cut MDF slices)
by providing bidirectional conversion between SDF primitives and Stack/Slice format.

Functions:
    sdf_to_stack: Convert SDF geometry to manufacturing Stack
    sdf_to_slices_generator: Memory-efficient generator version
    stack_to_sdf: Convert Stack to SDF geometry
    validate_slices_manufacturable: Check manufacturing constraints

Classes:
    VoxelSDF: SDF backed by 3D voxel array

Example:
    >>> from strata_fdtd.sdf import Horn
    >>> from strata_fdtd.manufacturing.conversion import sdf_to_stack
    >>>
    >>> # Design horn for simulation
    >>> horn = Horn(
    ...     throat_position=(0.05, 0.05, 0),
    ...     mouth_position=(0.05, 0.05, 0.15),
    ...     throat_radius=0.015,
    ...     mouth_radius=0.045,
    ...     profile="exponential",
    ... )
    >>>
    >>> # Convert to manufacturing slices
    >>> stack = sdf_to_stack(
    ...     horn,
    ...     z_range=(0, 0.15),
    ...     slice_thickness=6e-3,  # 6mm MDF
    ...     xy_resolution=0.5e-3,  # 0.5mm laser precision
    ... )
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import ndimage

from strata_fdtd.geometry.sdf import SDFPrimitive

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from strata_fdtd.manufacturing.lamination import Slice, Stack, Violation


def sdf_to_stack(
    sdf: SDFPrimitive,
    z_range: tuple[float, float],
    slice_thickness: float = 6e-3,
    xy_resolution: float = 0.5e-3,
    xy_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> Stack:
    """Convert SDF geometry to manufacturing Stack.

    Slices the SDF at regular Z intervals and creates 2D masks
    suitable for laser cutting.

    Args:
        sdf: Source geometry
        z_range: (z_min, z_max) range to slice
        slice_thickness: Thickness of each slice (default 6mm MDF)
        xy_resolution: Resolution of 2D slice masks in meters
        xy_bounds: Optional ((x_min, y_min), (x_max, y_max)) bounds in meters.
                   If None, derived from SDF bounding box

    Returns:
        Stack of Slices ready for DXF export

    Example:
        >>> from strata_fdtd.sdf import Box
        >>> box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        >>> stack = sdf_to_stack(box, z_range=(0.04, 0.06))
    """
    from strata_fdtd.manufacturing.lamination import Slice, Stack

    # Get bounding box if not specified
    if xy_bounds is None:
        bb_min, bb_max = sdf.bounding_box
        xy_bounds = ((bb_min[0], bb_min[1]), (bb_max[0], bb_max[1]))

    z_min, z_max = z_range

    # Calculate grid dimensions
    x_min, y_min = xy_bounds[0]
    x_max, y_max = xy_bounds[1]

    nx = int(np.ceil((x_max - x_min) / xy_resolution))
    ny = int(np.ceil((y_max - y_min) / xy_resolution))

    # Generate slices
    slices = []
    z = z_min + slice_thickness / 2  # Sample at slice center
    z_index = 0

    while z < z_max:
        # Create 2D grid at this z level
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = np.full_like(X, z)

        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # Evaluate SDF
        distances = sdf.sdf(points)
        mask = (distances <= 0).reshape(ny, nx)

        # Convert to Slice
        # Note: Slice uses True=air, SDF uses negative=inside
        # Transpose to match (height, width) expected by Slice
        slice_ = Slice(
            mask=mask.T,  # Transpose from (ny, nx) to (nx, ny) then store as (height, width)
            z_index=z_index,
            resolution=xy_resolution * 1000,  # Slice uses mm
        )
        slices.append(slice_)

        z += slice_thickness
        z_index += 1

    return Stack(
        slices=slices,
        resolution_xy=xy_resolution * 1000,  # mm
        thickness_z=slice_thickness * 1000,  # mm
    )


def sdf_to_slices_generator(
    sdf: SDFPrimitive,
    z_range: tuple[float, float],
    slice_thickness: float = 6e-3,
    xy_resolution: float = 0.5e-3,
    xy_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> Iterator[Slice]:
    """Memory-efficient generator version for large geometries.

    Yields one Slice at a time instead of building full Stack in memory.

    Args:
        sdf: Source geometry
        z_range: (z_min, z_max) range to slice
        slice_thickness: Thickness of each slice (default 6mm MDF)
        xy_resolution: Resolution of 2D slice masks in meters
        xy_bounds: Optional ((x_min, y_min), (x_max, y_max)) bounds in meters

    Yields:
        Slice objects one at a time

    Example:
        >>> from strata_fdtd.sdf import Sphere
        >>> sphere = Sphere(center=(0.05, 0.05, 0.05), radius=0.02)
        >>> for slice_ in sdf_to_slices_generator(sphere, z_range=(0.03, 0.07)):
        ...     print(f"Slice {slice_.z_index}: {slice_.mask.shape}")
    """
    from strata_fdtd.manufacturing.lamination import Slice

    # Get bounding box if not specified
    if xy_bounds is None:
        bb_min, bb_max = sdf.bounding_box
        xy_bounds = ((bb_min[0], bb_min[1]), (bb_max[0], bb_max[1]))

    z_min, z_max = z_range

    # Calculate grid dimensions
    x_min, y_min = xy_bounds[0]
    x_max, y_max = xy_bounds[1]

    nx = int(np.ceil((x_max - x_min) / xy_resolution))
    ny = int(np.ceil((y_max - y_min) / xy_resolution))

    # Create coordinate arrays once
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    # Generate slices
    z = z_min + slice_thickness / 2  # Sample at slice center
    z_index = 0

    while z < z_max:
        # Create 2D grid at this z level
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = np.full_like(X, z)

        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # Evaluate SDF
        distances = sdf.sdf(points)
        mask = (distances <= 0).reshape(ny, nx)

        # Create and yield slice
        slice_ = Slice(
            mask=mask.T,  # Transpose to match Slice convention
            z_index=z_index,
            resolution=xy_resolution * 1000,  # Slice uses mm
        )
        yield slice_

        z += slice_thickness
        z_index += 1


def stack_to_sdf(
    stack: Stack,
    method: Literal["extrude", "interpolate"] = "extrude",
) -> SDFPrimitive:
    """Convert manufacturing Stack to SDF geometry.

    Args:
        stack: Source Stack
        method: Conversion method
            - "extrude": Each slice extruded to full thickness (blocky)
            - "interpolate": Smooth interpolation between slices

    Returns:
        SDF primitive representing the stack geometry

    Example:
        >>> from strata_fdtd.manufacturing.lamination import Stack
        >>> # ... create stack ...
        >>> sdf = stack_to_sdf(stack, method="extrude")
    """
    if method == "extrude":
        return _stack_to_sdf_extrude(stack)
    elif method == "interpolate":
        return _stack_to_sdf_interpolate(stack)
    else:
        raise ValueError(f"Invalid method '{method}', must be 'extrude' or 'interpolate'")


def _stack_to_sdf_extrude(stack: Stack) -> VoxelSDF:
    """Convert Stack to VoxelSDF using simple extrusion.

    Each slice is extruded to its full thickness, creating blocky geometry.
    """
    # Convert to 3D voxel array
    voxels = stack.to_3d_mask()

    # Voxel coordinates
    # Assume slices are uniform in XY
    if not stack.slices:
        raise ValueError("Cannot convert empty stack to SDF")

    first_slice = stack.slices[0]
    height, width = first_slice.mask.shape

    # Create coordinate arrays
    # Resolution in meters
    xy_res = stack.resolution_xy / 1000  # mm to meters
    z_res = stack.thickness_z / 1000  # mm to meters

    # Calculate z origin from first slice z_index
    z_origin = stack.slices[0].z_index * z_res

    # Origin at (0, 0, z_origin)
    origin = np.array([0.0, 0.0, z_origin])

    return VoxelSDF(
        voxels=voxels,
        resolution=(xy_res, xy_res, z_res),
        origin=origin,
    )


def _stack_to_sdf_interpolate(stack: Stack) -> VoxelSDF:
    """Convert Stack to VoxelSDF using interpolation between slices.

    This creates smoother transitions between slices but is more complex.
    For now, this falls back to extrusion - full implementation would
    require upsampling in Z and trilinear interpolation.
    """
    # TODO: Implement proper interpolation
    # For now, use extrusion as fallback
    return _stack_to_sdf_extrude(stack)


def validate_slices_manufacturable(
    sdf: SDFPrimitive,
    z_range: tuple[float, float],
    slice_thickness: float = 6e-3,
    xy_resolution: float = 0.5e-3,
    min_wall: float = 3e-3,
    min_gap: float = 2e-3,
    xy_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> list[Violation]:
    """Check if SDF geometry can be manufactured as slices.

    Validates:
    - Minimum wall thickness at each slice
    - Minimum gap/slot width at each slice
    - Connectivity (no floating pieces)
    - Vertical alignment between adjacent slices

    Args:
        sdf: Source geometry
        z_range: (z_min, z_max) range to slice
        slice_thickness: Thickness of each slice (default 6mm MDF)
        xy_resolution: Resolution of 2D slice masks in meters
        min_wall: Minimum wall thickness in meters
        min_gap: Minimum gap/slot width in meters
        xy_bounds: Optional XY bounds

    Returns:
        List of Violation objects (empty if valid)

    Example:
        >>> from strata_fdtd.sdf import Cylinder
        >>> cyl = Cylinder((0.05, 0.05, 0), (0.05, 0.05, 0.1), radius=0.01)
        >>> violations = validate_slices_manufacturable(cyl, z_range=(0, 0.1))
        >>> if violations:
        ...     for v in violations:
        ...         print(v)
    """
    # Convert to stack
    stack = sdf_to_stack(
        sdf,
        z_range=z_range,
        slice_thickness=slice_thickness,
        xy_resolution=xy_resolution,
        xy_bounds=xy_bounds,
    )

    # Check manufacturability (convert meters to mm for stack.check_manufacturable)
    return stack.check_manufacturable(
        min_wall=min_wall * 1000,  # m to mm
        min_gap=min_gap * 1000,  # m to mm
    )


class VoxelSDF(SDFPrimitive):
    """SDF backed by 3D voxel array.

    Used for Stack → SDF conversion. Provides approximate signed distance
    function based on voxel occupancy.

    Args:
        voxels: 3D boolean array (True = inside/air)
        resolution: (dx, dy, dz) voxel spacing in meters
        origin: (x, y, z) position of voxel[0, 0, 0] in meters

    Example:
        >>> voxels = np.zeros((100, 100, 100), dtype=bool)
        >>> voxels[25:75, 25:75, 25:75] = True  # Cube of air
        >>> sdf = VoxelSDF(
        ...     voxels=voxels,
        ...     resolution=(1e-3, 1e-3, 1e-3),
        ...     origin=(0, 0, 0)
        ... )
    """

    def __init__(
        self,
        voxels: NDArray[np.bool_],
        resolution: tuple[float, float, float],
        origin: NDArray[np.floating] | tuple[float, float, float],
    ):
        if voxels.ndim != 3:
            raise ValueError(f"voxels must be 3D array, got {voxels.ndim}D")

        self.voxels = voxels.astype(np.bool_)
        self.resolution = np.array(resolution, dtype=np.float64)
        self.origin = np.array(origin, dtype=np.float64)

        if self.resolution.shape != (3,):
            raise ValueError(f"resolution must be 3-element tuple, got shape {self.resolution.shape}")

        if self.origin.shape != (3,):
            raise ValueError(f"origin must be 3-element tuple, got shape {self.origin.shape}")

        if np.any(self.resolution <= 0):
            raise ValueError(f"resolution must be positive, got {self.resolution}")

        # Precompute distance transform for better SDF approximation
        self._compute_distance_transform()

    def _compute_distance_transform(self) -> None:
        """Precompute distance transforms for inside and outside regions."""
        # Distance from inside points to nearest outside point (in voxel units)
        # Use spacing parameter to get distances in world units
        inside_dist = ndimage.distance_transform_edt(self.voxels, sampling=self.resolution)

        # Distance from outside points to nearest inside point (in voxel units)
        outside_dist = ndimage.distance_transform_edt(~self.voxels, sampling=self.resolution)

        # Combine: negative inside, positive outside
        self._distance_field = outside_dist - inside_dist

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Approximate SDF from voxel grid.

        Uses trilinear interpolation of precomputed distance field.

        Args:
            points: (N, 3) array of (x, y, z) coordinates

        Returns:
            (N,) array of approximate signed distances
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be Nx3 array, got shape {points.shape}")

        # Convert points to voxel coordinates (fractional indices)
        voxel_coords = (points - self.origin) / self.resolution

        # Trilinear interpolation
        from scipy.ndimage import map_coordinates

        distances = map_coordinates(
            self._distance_field,
            voxel_coords.T,
            order=1,  # Linear interpolation
            mode="constant",
            cval=np.max(self._distance_field),  # Points outside grid are far outside
        )

        return distances

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return axis-aligned bounding box.

        Returns:
            Tuple of (min_corner, max_corner)
        """
        max_corner = self.origin + self.voxels.shape * self.resolution
        return self.origin.copy(), max_corner


__all__ = [
    "sdf_to_stack",
    "sdf_to_slices_generator",
    "stack_to_sdf",
    "validate_slices_manufacturable",
    "VoxelSDF",
]

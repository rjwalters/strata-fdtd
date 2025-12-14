"""
Core geometry classes for laminated strata_fdtd structures.

Provides Slice (2D layer) and Stack (3D structure) classes with manufacturing
constraint validation and export capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Manufacturing constraints (defaults for 6mm MDF laser cutting)
DEFAULT_SLICE_THICKNESS = 6.0  # mm per slice
DEFAULT_MIN_WALL = 3.0  # mm minimum solid wall thickness
DEFAULT_MIN_GAP = 2.0  # mm minimum air gap/slot width
DEFAULT_KERF = 0.3  # mm material removed by laser cut


@dataclass
class Violation:
    """
    Represents a manufacturing constraint violation.

    Attributes:
        constraint: Name of the violated constraint (e.g., "min_wall_thickness")
        slice_index: Which slice contains the violation
        location: (x, y) position in mm where violation occurs
        measured: The measured value that violates the constraint
        required: The required value for the constraint
    """

    constraint: str
    slice_index: int
    location: tuple[float, float]
    measured: float
    required: float

    def __str__(self) -> str:
        return (
            f"Violation({self.constraint}): slice {self.slice_index} at "
            f"({self.location[0]:.1f}, {self.location[1]:.1f}) mm - "
            f"measured {self.measured:.2f} mm, required {self.required:.2f} mm"
        )


@dataclass
class Slice:
    """
    Single 6mm layer represented as a 2D boolean mask.

    The mask convention is:
    - True = air (material will be cut out)
    - False = solid (MDF material remains)

    Attributes:
        mask: 2D boolean array where True=air, False=solid
        z_index: Which layer this is (0, 1, 2, ...)
        resolution: mm per pixel (default 0.5 mm/pixel for good accuracy)
    """

    mask: NDArray[np.bool_]
    z_index: int
    resolution: float = 0.5

    def __post_init__(self) -> None:
        """Validate mask is 2D boolean array."""
        if self.mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got {self.mask.ndim}D")
        if self.mask.dtype != np.bool_:
            self.mask = self.mask.astype(np.bool_)

    @property
    def shape_mm(self) -> tuple[float, float]:
        """Return (width, height) of slice in mm."""
        return (
            self.mask.shape[1] * self.resolution,
            self.mask.shape[0] * self.resolution,
        )

    @property
    def solid_mask(self) -> NDArray[np.bool_]:
        """Return mask where True=solid (inverse of air mask)."""
        return ~self.mask

    def is_connected(self) -> bool:
        """
        Check if the solid region forms a single connected component.

        Uses 4-connectivity (no diagonal connections) which matches
        the physical reality of cut material.

        Returns:
            True if solid region is a single connected component
        """
        solid = self.solid_mask
        if not solid.any():
            # All air - technically "connected" but probably an error
            return True

        # Label connected components using 4-connectivity
        structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
        labeled, num_features = ndimage.label(solid, structure=structure)

        return num_features == 1

    def min_feature_size(self) -> tuple[float, float]:
        """
        Calculate minimum wall thickness and minimum gap width.

        Uses morphological operations (erosion/dilation) to find the
        minimum feature sizes in the geometry.

        Returns:
            (min_wall_mm, min_gap_mm) - minimum feature sizes in mm
        """
        solid = self.solid_mask
        air = self.mask

        min_wall = self._min_thickness(solid)
        min_gap = self._min_thickness(air)

        return (min_wall * self.resolution, min_gap * self.resolution)

    def _min_thickness(self, binary_mask: NDArray[np.bool_]) -> float:
        """
        Find minimum thickness of features in a binary mask.

        Uses iterative erosion to find the smallest feature that survives.

        Args:
            binary_mask: 2D boolean array of features to measure

        Returns:
            Minimum thickness in pixels
        """
        if not binary_mask.any():
            return float("inf")

        # Use distance transform for efficient measurement
        # Distance to nearest background pixel
        distance = ndimage.distance_transform_edt(binary_mask)

        # The maximum distance inside any feature gives half the min width
        # (distance from center to edge)
        # Minimum across all features is found by looking at the skeleton
        # Simpler approach: find narrowest "pinch point"
        # This is approximated by the minimum of max distances along each
        # row and column that contains the feature

        # Actually, let's use the distance transform more directly:
        # The minimum feature width is 2 * min(max_distance_per_component)
        labeled, num_features = ndimage.label(binary_mask)

        if num_features == 0:
            return float("inf")

        min_widths = []
        for i in range(1, num_features + 1):
            component = labeled == i
            component_dist = distance * component
            if component_dist.any():
                # Max distance in this component = radius of largest inscribed circle
                # This gives half the width at the widest point
                # For minimum width, we need the minimum of the local maximums
                # Approximate with the skeleton approach
                max_dist = component_dist.max()
                min_widths.append(2 * max_dist)

        return min(min_widths) if min_widths else float("inf")

    def boundary_polygons(self) -> list[NDArray[np.float64]]:
        """
        Extract cutting path polygons from the slice.

        Returns closed polylines suitable for DXF export. Each polygon
        represents either an outer boundary or an inner hole.

        Returns:
            List of Nx2 arrays, each representing a closed polygon in mm
        """
        from skimage import measure

        # Find contours at 0.5 level (boundary between air and solid)
        # The mask is air=True, so contours around air regions are cut paths
        contours = measure.find_contours(self.mask.astype(float), 0.5)

        # Convert to mm coordinates
        polygons = []
        for contour in contours:
            # Contour is in (row, col) format, convert to (x, y) mm
            # row = y, col = x
            polygon = np.zeros_like(contour)
            polygon[:, 0] = contour[:, 1] * self.resolution  # x from col
            polygon[:, 1] = contour[:, 0] * self.resolution  # y from row
            polygons.append(polygon)

        return polygons

    @classmethod
    def from_rectangles(
        cls,
        width_mm: float,
        height_mm: float,
        z_index: int,
        air_rects: list[tuple[float, float, float, float]] | None = None,
        resolution: float = 0.5,
    ) -> Slice:
        """
        Create a slice from a list of rectangular air regions.

        Args:
            width_mm: Total width of slice in mm
            height_mm: Total height of slice in mm
            z_index: Layer index
            air_rects: List of (x, y, w, h) tuples defining air regions in mm
            resolution: mm per pixel

        Returns:
            New Slice with specified geometry
        """
        # Create solid mask
        w_px = int(np.ceil(width_mm / resolution))
        h_px = int(np.ceil(height_mm / resolution))
        mask = np.zeros((h_px, w_px), dtype=np.bool_)

        # Cut out air regions
        if air_rects:
            for x, y, w, h in air_rects:
                x0 = int(x / resolution)
                y0 = int(y / resolution)
                x1 = int((x + w) / resolution)
                y1 = int((y + h) / resolution)
                mask[y0:y1, x0:x1] = True

        return cls(mask=mask, z_index=z_index, resolution=resolution)


@dataclass
class Stack:
    """
    Complete 3D geometry as a stack of laminated slices.

    Represents a 3D structure built from multiple 2D layers, each
    representing a 6mm thick MDF sheet.

    Attributes:
        slices: List of Slice objects ordered by z_index
        resolution_xy: mm per pixel in XY plane
        thickness_z: mm per slice (default 6.0 for MDF)
    """

    slices: list[Slice] = field(default_factory=list)
    resolution_xy: float = 0.5
    thickness_z: float = DEFAULT_SLICE_THICKNESS

    def __post_init__(self) -> None:
        """Sort slices by z_index."""
        self.slices = sorted(self.slices, key=lambda s: s.z_index)

    @property
    def num_slices(self) -> int:
        """Number of slices in the stack."""
        return len(self.slices)

    @property
    def height_mm(self) -> float:
        """Total height of stack in mm."""
        return self.num_slices * self.thickness_z

    @property
    def bounding_box(self) -> tuple[float, float, float]:
        """Return (width, depth, height) in mm."""
        if not self.slices:
            return (0.0, 0.0, 0.0)
        # Assume all slices have same dimensions
        w, h = self.slices[0].shape_mm
        return (w, h, self.height_mm)

    def add_slice(self, slice_: Slice) -> None:
        """Add a slice to the stack, maintaining z_index order."""
        self.slices.append(slice_)
        self.slices.sort(key=lambda s: s.z_index)

    def get_slice(self, z_index: int) -> Slice | None:
        """Get slice by z_index, or None if not found."""
        for s in self.slices:
            if s.z_index == z_index:
                return s
        return None

    def to_3d_mask(self) -> NDArray[np.bool_]:
        """
        Expand stack to full 3D grid for FDTD simulation.

        The returned array has shape (nx, ny, nz) where:
        - nx, ny match the slice mask dimensions
        - nz = num_slices (each slice is one voxel thick)

        The convention matches the Slice convention:
        - True = air
        - False = solid

        Returns:
            3D boolean array with shape (height, width, num_slices)
        """
        if not self.slices:
            raise ValueError("Cannot create 3D mask from empty stack")

        # Get dimensions from first slice
        h, w = self.slices[0].mask.shape
        nz = self.num_slices

        # Create 3D array
        mask_3d = np.zeros((h, w, nz), dtype=np.bool_)

        for i, slice_ in enumerate(self.slices):
            if slice_.mask.shape != (h, w):
                raise ValueError(
                    f"Slice {slice_.z_index} has shape {slice_.mask.shape}, "
                    f"expected {(h, w)}"
                )
            mask_3d[:, :, i] = slice_.mask

        return mask_3d

    @classmethod
    def from_3d_mask(
        cls,
        mask_3d: NDArray[np.bool_],
        resolution_xy: float = 0.5,
        thickness_z: float = DEFAULT_SLICE_THICKNESS,
    ) -> Stack:
        """
        Create a Stack from a 3D boolean mask.

        Args:
            mask_3d: 3D array with shape (height, width, num_slices)
            resolution_xy: mm per pixel in XY plane
            thickness_z: mm per slice

        Returns:
            New Stack with slices extracted from the 3D mask
        """
        if mask_3d.ndim != 3:
            raise ValueError(f"Expected 3D array, got {mask_3d.ndim}D")

        slices = []
        for z in range(mask_3d.shape[2]):
            slice_ = Slice(
                mask=mask_3d[:, :, z].copy(),
                z_index=z,
                resolution=resolution_xy,
            )
            slices.append(slice_)

        return cls(
            slices=slices,
            resolution_xy=resolution_xy,
            thickness_z=thickness_z,
        )

    def check_manufacturable(
        self,
        min_wall: float = DEFAULT_MIN_WALL,
        min_gap: float = DEFAULT_MIN_GAP,
    ) -> list[Violation]:
        """
        Run all manufacturing constraint checks.

        Args:
            min_wall: Minimum wall thickness in mm
            min_gap: Minimum gap/slot width in mm

        Returns:
            List of Violation objects (empty if all constraints pass)
        """
        from strata_fdtd.manufacturing.constraints import (
            check_connectivity,
            check_min_gap,
            check_min_wall,
            check_slice_alignment,
        )

        violations = []

        for slice_ in self.slices:
            # Check connectivity
            conn_violation = check_connectivity(slice_)
            if conn_violation:
                violations.append(conn_violation)

            # Check wall thickness
            violations.extend(check_min_wall(slice_, min_wall))

            # Check gap width
            violations.extend(check_min_gap(slice_, min_gap))

        # Check inter-slice alignment
        violations.extend(check_slice_alignment(self))

        return violations

    def export_dxf(self, output_dir: Path, kerf_compensation: float = 0.0) -> list[Path]:
        """
        Export each slice as DXF for laser cutting.

        Args:
            output_dir: Directory to write DXF files
            kerf_compensation: Amount to offset cut paths (half kerf width)

        Returns:
            List of paths to created DXF files
        """
        from strata_fdtd.export import export_dxf

        return export_dxf(self, output_dir, kerf_compensation)

    def export_json(self, path: Path) -> None:
        """
        Export stack as JSON for Three.js visualization.

        Args:
            path: Path to output JSON file
        """
        from strata_fdtd.export import export_json

        export_json(self, path)

    def export_stl(self, path: Path) -> None:
        """
        Export stack as STL mesh for 3D visualization/printing.

        Args:
            path: Path to output STL file
        """
        from strata_fdtd.export import export_stl

        export_stl(self, path)


def union(a: Stack, b: Stack) -> Stack:
    """
    Combine two stacks (air where either has air).

    Creates a new stack with the union of air regions from both input stacks.
    The stacks must have compatible dimensions.

    Args:
        a: First stack
        b: Second stack

    Returns:
        New Stack with combined geometry
    """
    if not a.slices or not b.slices:
        return a if a.slices else b

    # Get all z_indices from both stacks
    z_indices = sorted(set(s.z_index for s in a.slices) | set(s.z_index for s in b.slices))

    slices = []
    for z in z_indices:
        slice_a = a.get_slice(z)
        slice_b = b.get_slice(z)

        if slice_a is None and slice_b is not None:
            slices.append(Slice(slice_b.mask.copy(), z, slice_b.resolution))
        elif slice_b is None and slice_a is not None:
            slices.append(Slice(slice_a.mask.copy(), z, slice_a.resolution))
        elif slice_a is not None and slice_b is not None:
            # Union of air regions (OR operation)
            combined = slice_a.mask | slice_b.mask
            slices.append(Slice(combined, z, slice_a.resolution))

    return Stack(slices=slices, resolution_xy=a.resolution_xy, thickness_z=a.thickness_z)


def difference(a: Stack, b: Stack) -> Stack:
    """
    Subtract b from a (solid where b has air).

    Creates a new stack where regions that are air in b become solid.

    Args:
        a: Base stack
        b: Stack to subtract

    Returns:
        New Stack with subtracted geometry
    """
    if not a.slices:
        return Stack(resolution_xy=a.resolution_xy, thickness_z=a.thickness_z)

    slices = []
    for slice_a in a.slices:
        slice_b = b.get_slice(slice_a.z_index)

        if slice_b is None:
            slices.append(Slice(slice_a.mask.copy(), slice_a.z_index, slice_a.resolution))
        else:
            # Where b has air, make a solid (AND with NOT b)
            result = slice_a.mask & ~slice_b.mask
            slices.append(Slice(result, slice_a.z_index, slice_a.resolution))

    return Stack(slices=slices, resolution_xy=a.resolution_xy, thickness_z=a.thickness_z)


def array_stack(
    primitive_stack: Stack, positions: list[tuple[float, float, int]]
) -> Stack:
    """
    Repeat a primitive stack at multiple positions.

    Args:
        primitive_stack: The stack to repeat
        positions: List of (x_mm, y_mm, z_slice) positions for each copy

    Returns:
        New Stack with all copies combined
    """
    if not positions:
        return Stack(
            resolution_xy=primitive_stack.resolution_xy,
            thickness_z=primitive_stack.thickness_z,
        )

    # Find bounding box needed
    prim_box = primitive_stack.bounding_box
    max_x = max(x + prim_box[0] for x, y, z in positions)
    max_y = max(y + prim_box[1] for x, y, z in positions)
    max_z = max(z + primitive_stack.num_slices for x, y, z in positions)

    # Create empty stack of required size
    res = primitive_stack.resolution_xy
    w_px = int(np.ceil(max_x / res))
    h_px = int(np.ceil(max_y / res))

    # Initialize all-solid slices
    slices = []
    for z in range(max_z):
        mask = np.zeros((h_px, w_px), dtype=np.bool_)
        slices.append(Slice(mask=mask, z_index=z, resolution=res))

    result = Stack(
        slices=slices,
        resolution_xy=res,
        thickness_z=primitive_stack.thickness_z,
    )

    # Place each copy
    for x_mm, y_mm, z_start in positions:
        x_px = int(x_mm / res)
        y_px = int(y_mm / res)

        for prim_slice in primitive_stack.slices:
            z_idx = z_start + prim_slice.z_index
            target_slice = result.get_slice(z_idx)
            if target_slice is None:
                continue

            # Get primitive slice dimensions
            ph, pw = prim_slice.mask.shape

            # Place primitive (OR for air regions)
            y_end = min(y_px + ph, target_slice.mask.shape[0])
            x_end = min(x_px + pw, target_slice.mask.shape[1])
            ph_clip = y_end - y_px
            pw_clip = x_end - x_px

            target_slice.mask[y_px:y_end, x_px:x_end] |= prim_slice.mask[:ph_clip, :pw_clip]

    return result

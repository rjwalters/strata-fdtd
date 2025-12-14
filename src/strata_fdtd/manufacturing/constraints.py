"""
Manufacturing constraint checking for laminated strata_fdtd structures.

Provides functions to verify that geometry meets laser cutting and
structural requirements for MDF fabrication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from strata_fdtd.manufacturing.lamination import Slice, Stack, Violation


# Default constraints
DEFAULT_MIN_WALL = 3.0  # mm minimum solid wall thickness
DEFAULT_MIN_GAP = 2.0  # mm minimum air gap/slot width
DEFAULT_MAX_ALIGNMENT_SHIFT = 1.0  # mm maximum feature shift between slices


def check_connectivity(slice_: Slice) -> Violation | None:
    """
    Verify the solid region is a single connected component.

    Uses 4-connectivity (no diagonal connections) which matches
    the physical reality of cut material. If the solid region
    has multiple disconnected pieces, some pieces will fall out
    when cut.

    Args:
        slice_: The slice to check

    Returns:
        Violation if disconnected, None if connected
    """
    from strata_fdtd.manufacturing.lamination import Violation

    solid = slice_.solid_mask

    if not solid.any():
        # All air - this is a problem (nothing left after cutting)
        return Violation(
            constraint="connectivity",
            slice_index=slice_.z_index,
            location=(0.0, 0.0),
            measured=0.0,
            required=1.0,
        )

    # Label connected components using 4-connectivity
    structure = ndimage.generate_binary_structure(2, 1)
    labeled, num_features = ndimage.label(solid, structure=structure)

    if num_features > 1:
        # Find location of second component (the "island")
        # Report the centroid of the smallest component
        component_sizes = ndimage.sum(solid, labeled, range(1, num_features + 1))
        smallest_idx = int(np.argmin(component_sizes)) + 1

        # Find centroid of smallest component
        component_mask = labeled == smallest_idx
        y_coords, x_coords = np.where(component_mask)
        centroid_x = float(np.mean(x_coords)) * slice_.resolution
        centroid_y = float(np.mean(y_coords)) * slice_.resolution

        return Violation(
            constraint="connectivity",
            slice_index=slice_.z_index,
            location=(centroid_x, centroid_y),
            measured=float(num_features),
            required=1.0,
        )

    return None


def check_min_wall(
    slice_: Slice,
    min_thickness: float = DEFAULT_MIN_WALL,
) -> list[Violation]:
    """
    Find walls thinner than the minimum allowed thickness.

    Uses distance transform to find regions where walls are too thin.
    A wall is "too thin" at a point if the distance to the nearest
    air pixel (times 2, for the diameter) is less than min_thickness.

    Args:
        slice_: The slice to check
        min_thickness: Minimum wall thickness in mm

    Returns:
        List of violations (empty if all walls are thick enough)
    """
    from strata_fdtd.manufacturing.lamination import Violation

    violations = []
    solid = slice_.solid_mask

    if not solid.any():
        return violations

    # Use distance transform to find thin regions
    # Distance gives distance to nearest background (air) pixel
    dist = ndimage.distance_transform_edt(solid)

    # Convert threshold to pixels
    half_min_px = min_thickness / (2 * slice_.resolution)

    # Find thin regions: solid pixels where distance < half min thickness
    # This means the wall is thinner than required at this point
    thin_mask = solid & (dist < half_min_px)

    if not thin_mask.any():
        return violations

    # Label connected thin regions
    labeled, num_features = ndimage.label(thin_mask)

    for i in range(1, num_features + 1):
        component = labeled == i

        # Skip very small regions (noise)
        if component.sum() < 4:
            continue

        # Find centroid
        y_coords, x_coords = np.where(component)
        if len(x_coords) == 0:
            continue

        centroid_x = float(np.mean(x_coords)) * slice_.resolution
        centroid_y = float(np.mean(y_coords)) * slice_.resolution

        # Estimate actual wall thickness at this location
        local_dist = dist[component]
        if local_dist.size > 0:
            # Use minimum distance in the region (narrowest point)
            measured = float(np.min(local_dist) * 2 * slice_.resolution)
        else:
            measured = 0.0

        violations.append(
            Violation(
                constraint="min_wall_thickness",
                slice_index=slice_.z_index,
                location=(centroid_x, centroid_y),
                measured=measured,
                required=min_thickness,
            )
        )

    return violations


def check_min_gap(
    slice_: Slice,
    min_width: float = DEFAULT_MIN_GAP,
) -> list[Violation]:
    """
    Find gaps/slots narrower than the minimum allowed width.

    Uses distance transform on air regions to find narrow slots
    that may be difficult to cut cleanly or may leave debris.

    Args:
        slice_: The slice to check
        min_width: Minimum gap/slot width in mm

    Returns:
        List of violations (empty if all gaps are wide enough)
    """
    from strata_fdtd.manufacturing.lamination import Violation

    violations = []
    air = slice_.mask

    if not air.any():
        return violations

    # Use distance transform to find narrow regions
    # Distance gives distance to nearest solid pixel
    dist = ndimage.distance_transform_edt(air)

    # Convert threshold to pixels
    half_min_px = min_width / (2 * slice_.resolution)

    # Find narrow regions: air pixels where distance < half min width
    # This means the gap is narrower than required at this point
    narrow_mask = air & (dist < half_min_px)

    if not narrow_mask.any():
        return violations

    # Label connected narrow regions
    labeled, num_features = ndimage.label(narrow_mask)

    for i in range(1, num_features + 1):
        component = labeled == i

        # Skip very small regions (noise)
        if component.sum() < 4:
            continue

        # Find centroid
        y_coords, x_coords = np.where(component)
        if len(x_coords) == 0:
            continue

        centroid_x = float(np.mean(x_coords)) * slice_.resolution
        centroid_y = float(np.mean(y_coords)) * slice_.resolution

        # Estimate actual gap width at this location
        local_dist = dist[component]
        if local_dist.size > 0:
            # Use minimum distance in the region (narrowest point)
            measured = float(np.min(local_dist) * 2 * slice_.resolution)
        else:
            measured = 0.0

        violations.append(
            Violation(
                constraint="min_gap_width",
                slice_index=slice_.z_index,
                location=(centroid_x, centroid_y),
                measured=measured,
                required=min_width,
            )
        )

    return violations


def check_slice_alignment(
    stack: Stack,
    max_shift: float = DEFAULT_MAX_ALIGNMENT_SHIFT,
) -> list[Violation]:
    """
    Verify features align properly between adjacent slices.

    Large shifts in feature positions between slices can cause:
    - Structural weakness at layer boundaries
    - Acoustic discontinuities
    - Assembly difficulties

    This checks that the centroid of solid regions doesn't shift
    too much between adjacent layers.

    Args:
        stack: The stack to check
        max_shift: Maximum allowed shift in mm between adjacent slices

    Returns:
        List of violations (empty if alignment is good)
    """
    from strata_fdtd.manufacturing.lamination import Violation

    violations = []

    if len(stack.slices) < 2:
        return violations

    prev_centroids: dict[int, tuple[float, float]] = {}

    for i, slice_ in enumerate(stack.slices):
        solid = slice_.solid_mask

        if not solid.any():
            continue

        # Label components and get their centroids
        labeled, num_features = ndimage.label(solid)
        current_centroids: dict[int, tuple[float, float]] = {}

        for j in range(1, num_features + 1):
            component = labeled == j
            y_coords, x_coords = np.where(component)
            if len(x_coords) == 0:
                continue

            centroid_x = float(np.mean(x_coords)) * slice_.resolution
            centroid_y = float(np.mean(y_coords)) * slice_.resolution
            current_centroids[j] = (centroid_x, centroid_y)

        # Compare with previous slice
        if i > 0 and prev_centroids:
            # Simple matching: find closest centroid pairs
            for _curr_idx, (cx, cy) in current_centroids.items():
                min_dist = float("inf")
                closest_prev = None

                for _prev_idx, (px, py) in prev_centroids.items():
                    dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_prev = (px, py)

                if closest_prev is not None and min_dist > max_shift:
                    violations.append(
                        Violation(
                            constraint="slice_alignment",
                            slice_index=slice_.z_index,
                            location=(cx, cy),
                            measured=min_dist,
                            required=max_shift,
                        )
                    )

        prev_centroids = current_centroids

    return violations


def find_thin_regions(
    slice_: Slice,
    thickness_threshold: float,
) -> np.ndarray:
    """
    Find all regions thinner than a given threshold.

    Useful for visualization and debugging of constraint violations.

    Args:
        slice_: The slice to analyze
        thickness_threshold: Threshold in mm

    Returns:
        Boolean mask where True indicates regions thinner than threshold
    """
    solid = slice_.solid_mask

    if not solid.any():
        return np.zeros_like(solid)

    # Use distance transform
    dist = ndimage.distance_transform_edt(solid)

    # Regions where 2*distance < threshold are thin
    thin = solid & (dist * 2 * slice_.resolution < thickness_threshold)

    return thin


def find_narrow_gaps(
    slice_: Slice,
    width_threshold: float,
) -> np.ndarray:
    """
    Find all gaps narrower than a given threshold.

    Useful for visualization and debugging of constraint violations.

    Args:
        slice_: The slice to analyze
        width_threshold: Threshold in mm

    Returns:
        Boolean mask where True indicates gaps narrower than threshold
    """
    air = slice_.mask

    if not air.any():
        return np.zeros_like(air)

    # Use distance transform on air regions
    dist = ndimage.distance_transform_edt(air)

    # Regions where 2*distance < threshold are narrow
    narrow = air & (dist * 2 * slice_.resolution < width_threshold)

    return narrow

"""
Signed Distance Function (SDF) primitives for geometry representation.

This module provides SDF-based geometric primitives that enable:
- Accurate surface positioning during voxelization
- Easy CSG operations (union = min, intersection = max)
- Smooth blending between shapes
- Efficient bounding box queries

Classes:
    SDFPrimitive: Abstract base class for all SDF shapes
    Box: Axis-aligned box with optional rounding
    Sphere: Centered sphere
    Cylinder: Arbitrary orientation cylinder
    Cone: Truncated cone (frustum)
    Horn: Acoustic horn with various flare profiles

Example:
    >>> from strata_fdtd import sdf, UniformGrid
    >>>
    >>> # Create a box
    >>> box = sdf.Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
    >>>
    >>> # Voxelize to grid
    >>> grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)
    >>> mask = box.voxelize(grid)
    >>>
    >>> # Check if points are inside
    >>> points = np.array([[0.05, 0.05, 0.05], [0.0, 0.0, 0.0]])
    >>> inside = box.contains(points)

References:
    Inigo Quilez SDF Functions: https://iquilezles.org/articles/distfunctions/
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from strata_fdtd.grid import NonuniformGrid, UniformGrid


class SDFPrimitive(ABC):
    """Base class for Signed Distance Function primitives.

    All SDF primitives must implement:
    - sdf(): Evaluate signed distance at points
    - bounding_box: Return axis-aligned bounding box

    The signed distance convention is:
    - Negative values = inside the shape
    - Positive values = outside the shape
    - Zero = on the surface

    Note: For voxelization, inside (negative distance) maps to True (air),
    outside (positive distance) maps to False (solid).
    """

    @abstractmethod
    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate SDF at Nx3 array of points.

        Args:
            points: (N, 3) array of (x, y, z) coordinates

        Returns:
            (N,) array of signed distances (negative = inside)
        """
        pass

    @property
    @abstractmethod
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return (min_corner, max_corner) axis-aligned bounding box.

        Returns:
            Tuple of two (3,) arrays representing the minimum and maximum
            corners of the bounding box.
        """
        pass

    def contains(self, points: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Check if points are inside the shape.

        Args:
            points: (N, 3) array of (x, y, z) coordinates

        Returns:
            (N,) boolean array where True = inside
        """
        return self.sdf(points) <= 0

    def voxelize(self, grid: UniformGrid | NonuniformGrid) -> NDArray[np.bool_]:
        """Voxelize to boolean mask matching grid shape.

        Args:
            grid: UniformGrid or NonuniformGrid defining the voxel structure

        Returns:
            Boolean array where True = inside (air), False = outside (solid)
        """
        # Generate meshgrid of cell centers
        X, Y, Z = np.meshgrid(
            grid.x_coords,
            grid.y_coords,
            grid.z_coords,
            indexing="ij",
        )

        # Reshape to Nx3 array of points
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # Evaluate SDF
        distances = self.sdf(points)

        # Inside = negative distance = True (air)
        mask = distances.reshape(grid.shape) <= 0

        return mask

    # === Fluent API for transformations ===

    def translate(self, offset: tuple[float, float, float] | NDArray[np.floating]) -> SDFPrimitive:
        """Translate this primitive by an offset vector.

        Args:
            offset: (x, y, z) translation vector

        Returns:
            Translated primitive

        Example:
            >>> box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))
            >>> translated = box.translate((0.05, 0.05, 0.05))
        """
        # Note: Translate class is defined later in this module
        # We use forward reference to avoid issues with type checking
        return Translate(self, offset)  # type: ignore[name-defined]

    def scale(
        self, factor: float | tuple[float, float, float] | NDArray[np.floating]
    ) -> SDFPrimitive:
        """Scale this primitive uniformly or non-uniformly.

        Args:
            factor: Uniform scale (float) or per-axis scale (3-element)

        Returns:
            Scaled primitive

        Example:
            >>> box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))
            >>> scaled = box.scale(2.0)  # 2x larger
            >>> scaled = box.scale((2, 1, 0.5))  # Non-uniform
        """
        # Note: Scale class is defined later in this module
        return Scale(self, factor)  # type: ignore[name-defined]

    def rotate(
        self,
        axis: tuple[float, float, float] | NDArray[np.floating],
        angle: float,
    ) -> SDFPrimitive:
        """Rotate this primitive around an axis by an angle.

        Args:
            axis: (x, y, z) rotation axis (will be normalized)
            angle: Rotation angle in radians

        Returns:
            Rotated primitive

        Example:
            >>> box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.04))
            >>> rotated = box.rotate((0, 0, 1), np.pi/4)  # 45° around z
        """
        # Note: Rotate class is defined later in this module
        return Rotate(self, axis=axis, angle=angle)  # type: ignore[name-defined]

    def rotate_x(self, angle: float) -> SDFPrimitive:
        """Rotate this primitive around the x-axis.

        Args:
            angle: Rotation angle in radians

        Returns:
            Rotated primitive

        Example:
            >>> box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.04))
            >>> rotated = box.rotate_x(np.pi/2)  # 90° around x-axis
        """
        return self.rotate((1, 0, 0), angle)

    def rotate_y(self, angle: float) -> SDFPrimitive:
        """Rotate this primitive around the y-axis.

        Args:
            angle: Rotation angle in radians

        Returns:
            Rotated primitive

        Example:
            >>> box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.04))
            >>> rotated = box.rotate_y(np.pi/2)  # 90° around y-axis
        """
        return self.rotate((0, 1, 0), angle)

    def rotate_z(self, angle: float) -> SDFPrimitive:
        """Rotate this primitive around the z-axis.

        Args:
            angle: Rotation angle in radians

        Returns:
            Rotated primitive

        Example:
            >>> box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.04))
            >>> rotated = box.rotate_z(np.pi/2)  # 90° around z-axis
        """
        return self.rotate((0, 0, 1), angle)


class Box(SDFPrimitive):
    """Axis-aligned box SDF primitive.

    Can be constructed using either:
    - center and size, or
    - min_corner and max_corner

    Args:
        center: (x, y, z) center position
        size: (width, height, depth) dimensions
        min_corner: (x_min, y_min, z_min) corner
        max_corner: (x_max, y_max, z_max) corner

    Example:
        >>> # Using center and size
        >>> box1 = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        >>>
        >>> # Using corners
        >>> box2 = Box(min_corner=(0.04, 0.04, 0.04), max_corner=(0.06, 0.06, 0.06))
    """

    def __init__(
        self,
        center: tuple[float, float, float] | None = None,
        size: tuple[float, float, float] | None = None,
        min_corner: tuple[float, float, float] | None = None,
        max_corner: tuple[float, float, float] | None = None,
    ):
        if center is not None and size is not None:
            self.center = np.array(center, dtype=np.float64)
            self.size = np.array(size, dtype=np.float64)
        elif min_corner is not None and max_corner is not None:
            min_c = np.array(min_corner, dtype=np.float64)
            max_c = np.array(max_corner, dtype=np.float64)
            self.center = (min_c + max_c) / 2
            self.size = max_c - min_c
        else:
            raise ValueError("Must provide either (center, size) or (min_corner, max_corner)")

        if np.any(self.size <= 0):
            raise ValueError(f"Box size must be positive, got {self.size}")

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate box SDF.

        Uses the standard box SDF: distance is the maximum of distances
        to each face plane.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be Nx3 array, got shape {points.shape}")

        # Distance from center
        q = np.abs(points - self.center) - self.size / 2

        # Outside distance: length of the positive components
        outside = np.linalg.norm(np.maximum(q, 0), axis=1)

        # Inside distance: maximum component (most negative = deepest inside)
        inside = np.minimum(np.max(q, axis=1), 0)

        return outside + inside

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return axis-aligned bounding box."""
        half_size = self.size / 2
        return self.center - half_size, self.center + half_size


class Sphere(SDFPrimitive):
    """Sphere SDF primitive.

    Args:
        center: (x, y, z) center position
        radius: Sphere radius

    Example:
        >>> sphere = Sphere(center=(0.05, 0.05, 0.05), radius=0.01)
    """

    def __init__(self, center: tuple[float, float, float], radius: float):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)

        if self.radius <= 0:
            raise ValueError(f"Sphere radius must be positive, got {self.radius}")

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate sphere SDF.

        Simple distance to center minus radius.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be Nx3 array, got shape {points.shape}")

        # Distance from center to each point
        dist_to_center = np.linalg.norm(points - self.center, axis=1)

        return dist_to_center - self.radius

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return axis-aligned bounding box."""
        r = np.array([self.radius, self.radius, self.radius])
        return self.center - r, self.center + r


class Cylinder(SDFPrimitive):
    """Cylinder SDF primitive with arbitrary orientation.

    The cylinder is defined by two axis endpoints and a radius.
    It is capped at both ends.

    Args:
        p1: (x, y, z) first endpoint
        p2: (x, y, z) second endpoint
        radius: Cylinder radius

    Example:
        >>> # Vertical cylinder along z-axis
        >>> cyl = Cylinder(p1=(0.05, 0.05, 0.0), p2=(0.05, 0.05, 0.1), radius=0.01)
        >>>
        >>> # Horizontal cylinder along x-axis
        >>> cyl2 = Cylinder(p1=(0.0, 0.05, 0.05), p2=(0.1, 0.05, 0.05), radius=0.005)
    """

    def __init__(
        self,
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
        radius: float,
    ):
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.radius = float(radius)

        if self.radius <= 0:
            raise ValueError(f"Cylinder radius must be positive, got {self.radius}")

        # Precompute axis vector and length
        self.axis = self.p2 - self.p1
        self.length = np.linalg.norm(self.axis)

        if self.length == 0:
            raise ValueError("Cylinder endpoints must be distinct")

        self.axis_normalized = self.axis / self.length

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate cylinder SDF.

        Algorithm:
        1. Project points onto cylinder axis
        2. Compute distance to axis line (radial distance)
        3. Clamp projection to cylinder length (capping)
        4. Combine radial and axial distances
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be Nx3 array, got shape {points.shape}")

        # Vector from p1 to each point
        v = points - self.p1

        # Project onto axis (scalar projection)
        t = np.dot(v, self.axis_normalized)

        # Clamp t to [0, length] for capping
        t_clamped = np.clip(t, 0, self.length)

        # Closest point on axis to each point
        closest_on_axis = self.p1 + t_clamped[:, np.newaxis] * self.axis_normalized

        # Distance from each point to its closest point on axis
        radial_dist = np.linalg.norm(points - closest_on_axis, axis=1)

        # Distance from axis (considering capping)
        # For points between caps: distance is radial_dist - radius
        # For points beyond caps: add axial component
        axial_dist = np.maximum(0, np.abs(t - self.length / 2) - self.length / 2)

        # Combine: inside cylinder body = negative radial
        # Outside cylinder (radial or axial) = positive distance
        inside_body = (t >= 0) & (t <= self.length)
        dist = np.where(
            inside_body,
            radial_dist - self.radius,  # Pure radial distance
            np.sqrt((radial_dist - self.radius) ** 2 + axial_dist**2),  # Combined
        )

        return dist

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return axis-aligned bounding box."""
        # Box contains both endpoints plus radius in all directions
        r = np.array([self.radius, self.radius, self.radius])
        min_corner = np.minimum(self.p1, self.p2) - r
        max_corner = np.maximum(self.p1, self.p2) + r
        return min_corner, max_corner


class Cone(SDFPrimitive):
    """Truncated cone (frustum) SDF primitive.

    A cone with two different radii at the endpoints. This generalizes:
    - Cylinder: r1 == r2
    - Full cone: r2 == 0 (or r1 == 0)
    - Frustum: r1 != r2

    Args:
        p1: (x, y, z) first endpoint
        p2: (x, y, z) second endpoint
        r1: Radius at p1
        r2: Radius at p2

    Example:
        >>> # Tapered port: wider at one end
        >>> cone = Cone(
        ...     p1=(0.05, 0.05, 0.0),
        ...     p2=(0.05, 0.05, 0.02),
        ...     r1=0.01,  # 10mm radius at bottom
        ...     r2=0.005,  # 5mm radius at top
        ... )
        >>>
        >>> # Full cone (r2 = 0)
        >>> cone2 = Cone(p1=(0.0, 0.0, 0.0), p2=(0.0, 0.0, 0.05), r1=0.02, r2=0.0)
    """

    def __init__(
        self,
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
        r1: float,
        r2: float,
    ):
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.r1 = float(r1)
        self.r2 = float(r2)

        if self.r1 < 0 or self.r2 < 0:
            raise ValueError(f"Cone radii must be non-negative, got r1={r1}, r2={r2}")

        if self.r1 == 0 and self.r2 == 0:
            raise ValueError("Cone cannot have both radii zero")

        # Precompute axis vector and length
        self.axis = self.p2 - self.p1
        self.length = np.linalg.norm(self.axis)

        if self.length == 0:
            raise ValueError("Cone endpoints must be distinct")

        self.axis_normalized = self.axis / self.length

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate cone SDF.

        This is a simplified cone SDF that interpolates radius linearly
        along the axis and computes distance accordingly.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be Nx3 array, got shape {points.shape}")

        # Vector from p1 to each point
        v = points - self.p1

        # Project onto axis (normalized to [0, 1])
        t = np.dot(v, self.axis_normalized) / self.length
        t_clamped = np.clip(t, 0, 1)

        # Interpolate radius at projection point
        radius_at_t = self.r1 + t_clamped * (self.r2 - self.r1)

        # Closest point on axis
        closest_on_axis = self.p1 + (t_clamped * self.length)[:, np.newaxis] * self.axis_normalized

        # Radial distance
        radial_dist = np.linalg.norm(points - closest_on_axis, axis=1)

        # Axial distance (for capping)
        axial_dist = np.maximum(0, np.abs(t - 0.5) - 0.5) * self.length

        # Combined distance
        # Inside cone body (0 <= t <= 1): use radial distance
        inside_body = (t >= 0) & (t <= 1)
        dist = np.where(
            inside_body,
            radial_dist - radius_at_t,  # Pure radial
            # Outside caps: combine radial and axial
            np.sqrt((radial_dist - radius_at_t) ** 2 + axial_dist**2),
        )

        return dist

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return axis-aligned bounding box."""
        # Box contains both endpoints plus max radius
        max_r = max(self.r1, self.r2)
        r = np.array([max_r, max_r, max_r])
        min_corner = np.minimum(self.p1, self.p2) - r
        max_corner = np.maximum(self.p1, self.p2) + r
        return min_corner, max_corner

# === CSG Operations ===


class CSGNode(SDFPrimitive):
    """Base class for CSG operations on SDF primitives."""
    pass


class Union(CSGNode):
    """
    Union of multiple shapes (logical OR).

    The union is the set of points inside any of the child shapes.
    SDF implementation: min(d1, d2, ..., dn)
    """

    def __init__(self, *children: SDFPrimitive):
        """
        Create union of multiple primitives.

        Args:
            *children: Variable number of SDFPrimitive objects to union
        """
        self.children = children

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute union SDF as minimum of all child SDFs.

        Args:
            points: Array of shape (N, 3) with (x, y, z) coordinates

        Returns:
            Array of shape (N,) with signed distances
        """
        if not self.children:
            # Empty union - all points infinitely far from any surface
            return np.full(len(points), np.inf, dtype=np.float64)

        # Compute SDF for each child
        distances = [child.sdf(points) for child in self.children]

        # Union is the minimum distance (closest surface from any child)
        return np.minimum.reduce(distances)

    @cached_property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Bounding box is the union of all child bounding boxes.

        Returns:
            Tuple of (min_point, max_point)
        """
        if not self.children:
            # Empty union - degenerate bounding box
            return (
                np.array([np.inf, np.inf, np.inf]),
                np.array([-np.inf, -np.inf, -np.inf]),
            )

        # Get all child bounding boxes
        mins = [child.bounding_box[0] for child in self.children]
        maxs = [child.bounding_box[1] for child in self.children]

        # Union bounding box is min of mins, max of maxs
        return (
            np.minimum.reduce(mins),
            np.maximum.reduce(maxs),
        )


class Intersection(CSGNode):
    """
    Intersection of multiple shapes (logical AND).

    The intersection is the set of points inside all child shapes.
    SDF implementation: max(d1, d2, ..., dn)
    """

    def __init__(self, *children: SDFPrimitive):
        """
        Create intersection of multiple primitives.

        Args:
            *children: Variable number of SDFPrimitive objects to intersect
        """
        self.children = children

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute intersection SDF as maximum of all child SDFs.

        Args:
            points: Array of shape (N, 3) with (x, y, z) coordinates

        Returns:
            Array of shape (N,) with signed distances
        """
        if not self.children:
            # Empty intersection - all points inside (distance = -inf)
            return np.full(len(points), -np.inf, dtype=np.float64)

        # Compute SDF for each child
        distances = [child.sdf(points) for child in self.children]

        # Intersection is the maximum distance (must be inside all shapes)
        return np.maximum.reduce(distances)

    @cached_property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Bounding box is approximately the intersection of all child bounding boxes.

        Note: This is conservative - the actual intersection may be smaller or empty.

        Returns:
            Tuple of (min_point, max_point)
        """
        if not self.children:
            # Empty intersection - infinite bounding box
            return (
                np.array([-np.inf, -np.inf, -np.inf]),
                np.array([np.inf, np.inf, np.inf]),
            )

        # Get all child bounding boxes
        mins = [child.bounding_box[0] for child in self.children]
        maxs = [child.bounding_box[1] for child in self.children]

        # Intersection bounding box is max of mins, min of maxs
        bb_min = np.maximum.reduce(mins)
        bb_max = np.minimum.reduce(maxs)

        # Check if intersection is empty (min > max on any axis)
        if np.any(bb_min > bb_max):
            # Empty intersection - degenerate bounding box
            return (
                np.array([np.inf, np.inf, np.inf]),
                np.array([-np.inf, -np.inf, -np.inf]),
            )

        return (bb_min, bb_max)


class Difference(CSGNode):
    """
    Subtract shapes from a base shape.

    The difference is the set of points inside the base but outside all subtracted shapes.
    SDF implementation: max(base_sdf, -subtracted1_sdf, -subtracted2_sdf, ...)
    """

    def __init__(self, base: SDFPrimitive, *subtracted: SDFPrimitive):
        """
        Create difference of base minus subtracted shapes.

        Args:
            base: Base shape to subtract from
            *subtracted: Variable number of shapes to subtract from base
        """
        self.base = base
        self.subtracted = subtracted

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute difference SDF.

        Args:
            points: Array of shape (N, 3) with (x, y, z) coordinates

        Returns:
            Array of shape (N,) with signed distances
        """
        # Start with base SDF
        result = self.base.sdf(points)

        # Subtract each shape by taking max with negative of its SDF
        for shape in self.subtracted:
            result = np.maximum(result, -shape.sdf(points))

        return result

    @cached_property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Bounding box is the same as the base shape.

        Subtracting only removes material, never adds beyond base bounds.

        Returns:
            Tuple of (min_point, max_point)
        """
        return self.base.bounding_box


class SmoothUnion(CSGNode):
    """
    Smooth union with blend radius.

    Creates a smooth transition between shapes rather than a sharp edge.
    Uses polynomial smooth minimum function.
    """

    def __init__(self, a: SDFPrimitive, b: SDFPrimitive, radius: float):
        """
        Create smooth union of two primitives.

        Args:
            a: First primitive
            b: Second primitive
            radius: Blend radius (larger = smoother transition)
        """
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")

        self.a = a
        self.b = b
        self.radius = radius

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute smooth union SDF.

        Uses polynomial smooth minimum:
        h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
        result = lerp(d2, d1, h) - k * h * (1 - h)

        Args:
            points: Array of shape (N, 3) with (x, y, z) coordinates

        Returns:
            Array of shape (N,) with signed distances
        """
        d1 = self.a.sdf(points)
        d2 = self.b.sdf(points)

        # Smooth blend factor
        h = np.clip(0.5 + 0.5 * (d2 - d1) / self.radius, 0, 1)

        # Polynomial interpolation with subtraction for smooth minimum
        return d2 * (1 - h) + d1 * h - self.radius * h * (1 - h)

    @cached_property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Bounding box is the union of both primitives, expanded by blend radius.

        Returns:
            Tuple of (min_point, max_point)
        """
        min_a, max_a = self.a.bounding_box
        min_b, max_b = self.b.bounding_box

        # Union of bounding boxes
        bb_min = np.minimum(min_a, min_b)
        bb_max = np.maximum(max_a, max_b)

        # Expand by blend radius (smooth transition extends slightly beyond)
        expansion = np.array([self.radius, self.radius, self.radius])
        return (bb_min - expansion, bb_max + expansion)


class SmoothIntersection(CSGNode):
    """
    Smooth intersection with blend radius.

    Creates a smooth transition between shapes rather than a sharp edge.
    Uses polynomial smooth maximum function.
    """

    def __init__(self, a: SDFPrimitive, b: SDFPrimitive, radius: float):
        """
        Create smooth intersection of two primitives.

        Args:
            a: First primitive
            b: Second primitive
            radius: Blend radius (larger = smoother transition)
        """
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")

        self.a = a
        self.b = b
        self.radius = radius

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute smooth intersection SDF.

        Uses polynomial smooth maximum (same formula as smooth union,
        but with addition instead of subtraction).

        Args:
            points: Array of shape (N, 3) with (x, y, z) coordinates

        Returns:
            Array of shape (N,) with signed distances
        """
        d1 = self.a.sdf(points)
        d2 = self.b.sdf(points)

        # Smooth blend factor
        h = np.clip(0.5 + 0.5 * (d2 - d1) / self.radius, 0, 1)

        # Polynomial interpolation with addition for smooth maximum
        return d2 * (1 - h) + d1 * h + self.radius * h * (1 - h)

    @cached_property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Bounding box is approximately the intersection, contracted by blend radius.

        Returns:
            Tuple of (min_point, max_point)
        """
        min_a, max_a = self.a.bounding_box
        min_b, max_b = self.b.bounding_box

        # Intersection of bounding boxes
        bb_min = np.maximum(min_a, min_b)
        bb_max = np.minimum(max_a, max_b)

        # Contract by blend radius (smooth transition may shrink slightly)
        contraction = np.array([self.radius, self.radius, self.radius])
        bb_min = bb_min + contraction
        bb_max = bb_max - contraction

        # Check if still valid
        if np.any(bb_min > bb_max):
            # Empty intersection
            return (
                np.array([np.inf, np.inf, np.inf]),
                np.array([-np.inf, -np.inf, -np.inf]),
            )

        return (bb_min, bb_max)


class SmoothDifference(CSGNode):
    """
    Smooth difference with blend radius.

    Creates a smooth transition when subtracting rather than a sharp edge.
    """

    def __init__(self, base: SDFPrimitive, subtracted: SDFPrimitive, radius: float):
        """
        Create smooth difference.

        Args:
            base: Base shape to subtract from
            subtracted: Shape to subtract
            radius: Blend radius (larger = smoother transition)
        """
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")

        self.base = base
        self.subtracted = subtracted
        self.radius = radius

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute smooth difference SDF.

        Args:
            points: Array of shape (N, 3) with (x, y, z) coordinates

        Returns:
            Array of shape (N,) with signed distances
        """
        d1 = self.base.sdf(points)
        d2 = -self.subtracted.sdf(points)  # Negate for subtraction

        # Smooth blend factor (same as smooth intersection)
        h = np.clip(0.5 + 0.5 * (d2 - d1) / self.radius, 0, 1)

        # Polynomial interpolation
        return d2 * (1 - h) + d1 * h + self.radius * h * (1 - h)

    @cached_property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Bounding box is the base shape, potentially expanded by blend radius.

        Returns:
            Tuple of (min_point, max_point)
        """
        bb_min, bb_max = self.base.bounding_box

        # Smooth difference can slightly expand beyond base
        expansion = np.array([self.radius, self.radius, self.radius])
        return (bb_min - expansion, bb_max + expansion)


class Horn(SDFPrimitive):
    """Horn with specified flare profile.

    The horn is defined by throat and mouth positions/radii with various
    acoustic flare profiles. Different profiles have different acoustic
    loading characteristics and cutoff frequencies.

    Args:
        throat_position: (x, y, z) position of throat (narrow end)
        mouth_position: (x, y, z) position of mouth (wide end)
        throat_radius: Radius at throat
        mouth_radius: Radius at mouth
        profile: Flare profile type ("conical", "exponential", "hyperbolic", "tractrix")

    Profile Characteristics:
        - Conical: Linear taper, simple geometry
        - Exponential: Smooth loading, classic horn design
        - Hyperbolic: Good low-frequency loading
        - Tractrix: Constant directivity, wide bandwidth

    Example:
        >>> # Exponential horn for midrange driver
        >>> horn = Horn(
        ...     throat_position=(0.1, 0.1, 0.2),
        ...     mouth_position=(0.1, 0.1, 0.0),
        ...     throat_radius=0.025,  # 50mm throat
        ...     mouth_radius=0.075,   # 150mm mouth
        ...     profile="exponential"
        ... )
        >>> print(f"Cutoff: {horn.cutoff_frequency:.1f} Hz")
        >>>
        >>> # Conical port flare
        >>> flare = Horn(
        ...     throat_position=(0.05, 0.15, 0.12),
        ...     mouth_position=(0.05, 0.15, 0.15),
        ...     throat_radius=0.02,
        ...     mouth_radius=0.03,
        ...     profile="conical"
        ... )
    """

    def __init__(
        self,
        throat_position: tuple[float, float, float],
        mouth_position: tuple[float, float, float],
        throat_radius: float,
        mouth_radius: float,
        profile: str = "exponential",
    ):
        valid_profiles = {"conical", "exponential", "hyperbolic", "tractrix"}
        if profile not in valid_profiles:
            raise ValueError(f"profile must be one of {valid_profiles}, got '{profile}'")

        self.throat_position = np.array(throat_position, dtype=np.float64)
        self.mouth_position = np.array(mouth_position, dtype=np.float64)
        self.throat_radius = float(throat_radius)
        self.mouth_radius = float(mouth_radius)
        self.profile = profile

        if self.throat_radius <= 0 or self.mouth_radius <= 0:
            raise ValueError(
                f"Horn radii must be positive, got throat={throat_radius}, mouth={mouth_radius}"
            )

        # Compute axis and length
        self._axis = self.mouth_position - self.throat_position
        self._length = np.linalg.norm(self._axis)

        if self._length == 0:
            raise ValueError("Horn throat and mouth positions must be distinct")

        self._axis = self._axis / self._length  # Normalize

        # Compute profile-specific parameters
        self._compute_profile_params()

    def _compute_profile_params(self) -> None:
        """Compute profile-specific parameters for radius calculations."""
        A0 = np.pi * self.throat_radius**2
        Am = np.pi * self.mouth_radius**2
        L = self._length

        if self.profile == "conical":
            # A(x) = A0 * (1 + m*t)^2, where t = x/L
            # At t=1: Am = A0 * (1 + m)^2, so m = sqrt(Am/A0) - 1
            self._m = np.sqrt(Am / A0) - 1

        elif self.profile == "exponential":
            # A(x) = A0 * exp(2*alpha*x)
            # At x=L: Am = A0 * exp(2*alpha*L), so alpha = ln(Am/A0) / (2*L)
            self._alpha = np.log(Am / A0) / (2 * L)

        elif self.profile == "hyperbolic":
            # A(x) = A0 * cosh^2(alpha*x)
            # At x=L: Am = A0 * cosh^2(alpha*L), so alpha = arccosh(sqrt(Am/A0)) / L
            self._alpha = np.arccosh(np.sqrt(Am / A0)) / L

        elif self.profile == "tractrix":
            # A(x) = A0 / cos^4(x/R)
            # At x=L: Am = A0 / cos^4(L/R), so R = L / arccos((A0/Am)^(1/4))
            self._R = L / np.arccos(np.power(A0 / Am, 0.25))

    def radius_at_position(self, t: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
        """Get horn radius at normalized position t (0=throat, 1=mouth).

        Args:
            t: Normalized position along horn axis (0 to 1) or array of positions

        Returns:
            Radius at position(s) t
        """
        t = np.asarray(t)
        x = t * self._length

        if self.profile == "conical":
            return self.throat_radius * (1 + self._m * t)
        elif self.profile == "exponential":
            return self.throat_radius * np.exp(self._alpha * x)
        elif self.profile == "hyperbolic":
            return self.throat_radius * np.cosh(self._alpha * x)
        elif self.profile == "tractrix":
            # r(x) = r0 / cos^2(x/R)
            return self.throat_radius / np.cos(x / self._R) ** 2

    @property
    def cutoff_frequency(self) -> float:
        """Theoretical low-frequency cutoff for this horn.

        Returns:
            Cutoff frequency in Hz (speed of sound assumed 343 m/s)
        """
        c = 343.0  # Speed of sound in air at 20°C

        if self.profile == "conical":
            # fc ≈ c / (2π * L) for conical
            return c / (2 * np.pi * self._length)
        elif self.profile == "exponential":
            # fc = c * alpha / (2π)
            return c * self._alpha / (2 * np.pi)
        elif self.profile == "hyperbolic":
            # Similar to exponential
            return c * self._alpha / (2 * np.pi)
        elif self.profile == "tractrix":
            # fc = c / (4 * R)
            return c / (4 * self._R)

        return 0.0  # Fallback (should never reach here)

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate SDF for horn geometry.

        The horn is treated as a surface of revolution around the axis,
        with radius varying according to the profile function.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be Nx3 array, got shape {points.shape}")

        # Vector from throat to each point
        v = points - self.throat_position

        # Project onto horn axis
        t = np.dot(v, self._axis)  # Axial position
        t_norm = t / self._length  # Normalized to [0, 1]

        # Clamp to horn extent for radius calculation
        t_clamped = np.clip(t_norm, 0, 1)

        # Radial distance from axis
        axial_component = np.outer(t, self._axis)
        radial_vec = v - axial_component
        r = np.linalg.norm(radial_vec, axis=1)

        # Horn radius at each point's axial position
        horn_r = self.radius_at_position(t_clamped)

        # Radial component of distance
        radial_dist = r - horn_r

        # Axial distance for end caps
        throat_dist = -t  # Negative if beyond throat
        mouth_dist = t - self._length  # Positive if beyond mouth

        # Combined distance: inside if within radial envelope AND between caps
        # For points between caps (0 <= t <= L): use radial distance
        # For points beyond caps: combine radial and axial distances
        inside_body = (t >= 0) & (t <= self._length)

        dist = np.where(
            inside_body,
            radial_dist,  # Pure radial distance when between caps
            # Beyond caps: combine with axial distance
            np.sqrt(radial_dist**2 + np.maximum(throat_dist, mouth_dist) ** 2),
        )

        return dist

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return axis-aligned bounding box.

        The bounding box contains both throat and mouth positions plus
        the maximum radius (at mouth) in all directions.
        """
        max_r = self.mouth_radius  # Mouth is always wider
        r = np.array([max_r, max_r, max_r])
        min_corner = np.minimum(self.throat_position, self.mouth_position) - r
        max_corner = np.maximum(self.throat_position, self.mouth_position) + r
        return min_corner, max_corner


# === Transformations ===


def rotation_matrix(axis: NDArray[np.floating], angle: float) -> NDArray[np.floating]:
    """Create 3x3 rotation matrix for rotation around axis by angle (radians).

    Uses Rodrigues' rotation formula to construct the rotation matrix.

    Args:
        axis: (3,) array defining rotation axis (will be normalized)
        angle: Rotation angle in radians

    Returns:
        (3, 3) rotation matrix

    Example:
        >>> # Rotate 90 degrees around z-axis
        >>> R = rotation_matrix(np.array([0, 0, 1]), np.pi/2)
        >>> # Rotate around arbitrary axis
        >>> R = rotation_matrix(np.array([1, 1, 1]), np.pi/4)
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)  # Normalize

    # Rodrigues' formula: R = I + sin(θ)K + (1 - cos(θ))K²
    # where K is the cross-product matrix of axis
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )

    identity = np.eye(3)
    R = identity + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R


class Transform(SDFPrimitive):
    """Base class for transformed SDF primitives.

    Transformations wrap an existing primitive and modify the coordinate
    space of SDF evaluation.
    """

    def __init__(self, child: SDFPrimitive):
        self.child = child


class Translate(Transform):
    """Translate a primitive by offset vector.

    Translation works by subtracting the offset from query points
    (inverse transform) before evaluating the child SDF.

    Args:
        child: The primitive to translate
        offset: (x, y, z) translation vector

    Example:
        >>> box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))
        >>> translated = Translate(box, offset=(0.05, 0.05, 0.05))
        >>>
        >>> # Fluent API
        >>> translated = box.translate((0.05, 0.05, 0.05))
    """

    def __init__(
        self, child: SDFPrimitive, offset: tuple[float, float, float] | NDArray[np.floating]
    ):
        super().__init__(child)
        self.offset = np.asarray(offset, dtype=np.float64)

        if self.offset.shape != (3,):
            raise ValueError(f"offset must be a 3-element vector, got shape {self.offset.shape}")

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate SDF with inverse translation applied to points."""
        # Inverse transform: subtract offset from points
        transformed = points - self.offset
        return self.child.sdf(transformed)

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return bounding box transformed by translation."""
        bb_min, bb_max = self.child.bounding_box
        return bb_min + self.offset, bb_max + self.offset


class Scale(Transform):
    """Scale a primitive uniformly or non-uniformly.

    Scaling works by dividing query points by the scale factor
    (inverse transform). For uniform scaling, the SDF property
    is preserved by multiplying the result by the scale factor.

    For non-uniform scaling, the result is approximate (not a true SDF).

    Args:
        child: The primitive to scale
        scale: Uniform scale factor (float) or per-axis scale (3-element array)

    Example:
        >>> box = Box(center=(0, 0, 0), size=(0.01, 0.01, 0.01))
        >>> scaled = Scale(box, scale=2.0)  # Uniform 2x scaling
        >>>
        >>> # Non-uniform scaling
        >>> scaled = Scale(box, scale=(2.0, 1.0, 0.5))
        >>>
        >>> # Fluent API
        >>> scaled = box.scale(2.0)
    """

    def __init__(
        self, child: SDFPrimitive, scale: float | tuple[float, float, float] | NDArray[np.floating]
    ):
        super().__init__(child)
        if np.isscalar(scale):
            self.scale = np.array([scale, scale, scale], dtype=np.float64)
            self.uniform = True
        else:
            self.scale = np.asarray(scale, dtype=np.float64)
            self.uniform = np.allclose(self.scale, self.scale[0])

        if self.scale.shape != (3,):
            raise ValueError(
                f"scale must be scalar or 3-element vector, got shape {self.scale.shape}"
            )

        if np.any(self.scale <= 0):
            raise ValueError(f"scale factors must be positive, got {self.scale}")

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate SDF with inverse scaling applied to points."""
        # Inverse transform: divide points by scale
        transformed = points / self.scale
        distances = self.child.sdf(transformed)

        # For uniform scale, multiply distance by scale factor to preserve SDF
        if self.uniform:
            return distances * self.scale[0]

        # For non-uniform scale: approximate using minimum scale factor
        # This is not a perfect SDF but maintains reasonable distance estimates
        return distances * np.min(self.scale)

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return bounding box transformed by scaling."""
        bb_min, bb_max = self.child.bounding_box
        return bb_min * self.scale, bb_max * self.scale


class Rotate(Transform):
    """Rotate a primitive around an axis or by rotation matrix.

    Rotation works by applying the inverse rotation matrix to query
    points before evaluating the child SDF.

    Args:
        child: The primitive to rotate
        axis: (3,) array defining rotation axis (used with angle)
        angle: Rotation angle in radians (used with axis)
        matrix: Explicit 3x3 rotation matrix (alternative to axis/angle)

    Example:
        >>> box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.04))
        >>>
        >>> # Rotate around z-axis
        >>> rotated = Rotate(box, axis=(0, 0, 1), angle=np.pi/4)
        >>>
        >>> # Rotate using explicit matrix
        >>> R = rotation_matrix([1, 0, 0], np.pi/2)
        >>> rotated = Rotate(box, matrix=R)
        >>>
        >>> # Fluent API
        >>> rotated = box.rotate((0, 0, 1), np.pi/4)
        >>> rotated = box.rotate_z(np.pi/4)  # Convenience method
    """

    def __init__(
        self,
        child: SDFPrimitive,
        axis: tuple[float, float, float] | NDArray[np.floating] | None = None,
        angle: float | None = None,
        matrix: NDArray[np.floating] | None = None,
    ):
        super().__init__(child)

        if matrix is not None:
            self.matrix = np.asarray(matrix, dtype=np.float64)
            if self.matrix.shape != (3, 3):
                raise ValueError(f"matrix must be 3x3, got shape {self.matrix.shape}")
        elif axis is not None and angle is not None:
            self.matrix = rotation_matrix(np.asarray(axis), angle)
        else:
            raise ValueError("Must provide either (axis, angle) or matrix")

        # Inverse rotation matrix = transpose (for rotation matrices)
        self.inv_matrix = self.matrix.T

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate SDF with inverse rotation applied to points."""
        # Apply inverse rotation: points @ R^T
        transformed = points @ self.inv_matrix.T
        return self.child.sdf(transformed)

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return axis-aligned bounding box of rotated primitive.

        Computes a conservative AABB by rotating all 8 corners of the
        child's bounding box and finding min/max extents.
        """
        bb_min, bb_max = self.child.bounding_box

        # Generate all 8 corners of child bounding box
        corners = np.array(
            [
                [bb_min[0], bb_min[1], bb_min[2]],
                [bb_min[0], bb_min[1], bb_max[2]],
                [bb_min[0], bb_max[1], bb_min[2]],
                [bb_min[0], bb_max[1], bb_max[2]],
                [bb_max[0], bb_min[1], bb_min[2]],
                [bb_max[0], bb_min[1], bb_max[2]],
                [bb_max[0], bb_max[1], bb_min[2]],
                [bb_max[0], bb_max[1], bb_max[2]],
            ]
        )

        # Rotate all corners: corners @ R.T
        rotated_corners = corners @ self.matrix.T

        # Find min/max of rotated corners
        new_min = rotated_corners.min(axis=0)
        new_max = rotated_corners.max(axis=0)

        return new_min, new_max

# === Swept-volume primitives ===


class SpiralHorn(SDFPrimitive):
    """Horn following a spiral path with expanding cross-section.

    Combines a 2D spiral path (logarithmic or Archimedean) with a horn-like
    area expansion. The horn is extruded in the Z direction to form a 3D shape.

    This is useful for:
    - Compact bass horn loading
    - Folded horns in limited cabinet space
    - Impedance transformation in serpentine channels

    Args:
        center: (x, y) center of spiral in XY plane
        throat_radius: Cross-section radius at spiral start
        mouth_radius: Cross-section radius at spiral end
        inner_spiral_radius: Starting distance from center
        outer_spiral_radius: Ending distance from center
        turns: Number of spiral turns
        height: Extrusion height in Z direction
        profile: Horn expansion profile ("exponential", "linear", "hyperbolic")
        spiral_type: "logarithmic" or "archimedean"

    Note:
        The SDF is approximated using distance to a discretized path with
        varying radius. For high accuracy, the internal sampling density
        is automatically increased.

    Example:
        >>> # Compact spiral horn for bass loading
        >>> horn = SpiralHorn(
        ...     center=(0.15, 0.15),
        ...     throat_radius=0.025,
        ...     mouth_radius=0.075,
        ...     inner_spiral_radius=0.04,
        ...     outer_spiral_radius=0.12,
        ...     turns=2.5,
        ...     height=0.15,
        ...     profile="exponential"
        ... )
        >>> grid = UniformGrid(shape=(200, 200, 150), resolution=1e-3)
        >>> mask = horn.voxelize(grid)
    """

    def __init__(
        self,
        center: tuple[float, float],
        throat_radius: float,
        mouth_radius: float,
        inner_spiral_radius: float,
        outer_spiral_radius: float,
        turns: float,
        height: float,
        profile: str = "exponential",
        spiral_type: str = "logarithmic",
    ):
        from strata_fdtd.geometry.paths import SpiralPath

        self.center_2d = np.array(center, dtype=np.float64)
        self.throat_radius = float(throat_radius)
        self.mouth_radius = float(mouth_radius)
        self.inner_spiral_radius = float(inner_spiral_radius)
        self.outer_spiral_radius = float(outer_spiral_radius)
        self.turns = float(turns)
        self.height = float(height)
        self.profile = profile
        self.spiral_type = spiral_type

        if self.throat_radius <= 0 or self.mouth_radius <= 0:
            raise ValueError("Throat and mouth radii must be positive")
        if self.height <= 0:
            raise ValueError("Height must be positive")
        if self.profile not in ("exponential", "linear", "hyperbolic"):
            raise ValueError("profile must be 'exponential', 'linear', or 'hyperbolic'")

        # Create spiral path
        self.spiral = SpiralPath(
            center=self.center_2d,
            inner_radius=self.inner_spiral_radius,
            outer_radius=self.outer_spiral_radius,
            turns=self.turns,
            spiral_type=self.spiral_type,  # type: ignore
        )

        # Sample path with high density for accurate SDF
        # Use ~100 points per turn
        n_samples = max(200, int(100 * self.turns))
        self.path_2d = self.spiral.sample_points(n_samples)

        # Compute radius at each path point
        self.radii = self._compute_radii(n_samples)

    def _compute_radii(self, n_samples: int) -> NDArray[np.floating]:
        """Compute cross-section radius at each path point.

        Uses the specified profile (exponential, linear, hyperbolic) to
        interpolate between throat and mouth radii.
        """
        t = np.linspace(0, 1, n_samples)

        if self.profile == "exponential":
            # r(t) = r0 * (r1/r0)^t
            radii = self.throat_radius * (self.mouth_radius / self.throat_radius) ** t
        elif self.profile == "linear":
            # r(t) = r0 + (r1 - r0) * t
            radii = self.throat_radius + (self.mouth_radius - self.throat_radius) * t
        else:  # hyperbolic
            # r(t) = r0 / (1 - t * (1 - r0/r1))
            # This gives a hyperbolic expansion
            ratio = self.throat_radius / self.mouth_radius
            radii = self.throat_radius / (1 - t * (1 - ratio))

        return radii

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate SDF for spiral horn.

        For each point:
        1. Find distance to spiral path in XY plane
        2. Get radius at nearest path point
        3. Combine with Z distance for extrusion
        """
        from strata_fdtd.geometry.paths import distance_to_polyline

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be Nx3 array, got shape {points.shape}")

        # Extract XY and Z components
        points_xy = points[:, :2]
        z = points[:, 2]

        # Distance to spiral path in XY plane
        dist_to_path = distance_to_polyline(points_xy, self.path_2d)

        # For each point, find nearest path segment to determine radius
        # This is a simplified approach - find nearest path point
        n_points = points.shape[0]
        radii_at_points = np.zeros(n_points)

        for i in range(n_points):
            # Find nearest path point
            dists_to_path_points = np.linalg.norm(self.path_2d - points_xy[i], axis=1)
            nearest_idx = np.argmin(dists_to_path_points)
            radii_at_points[i] = self.radii[nearest_idx]

        # Distance from path considering varying radius
        radial_dist = dist_to_path - radii_at_points

        # Z distance (extrusion bounds)
        z_dist = np.maximum(0, np.maximum(-z, z - self.height))

        # Combine radial and axial distances
        # Inside extrusion bounds: use radial distance only
        # Outside bounds: combine with z distance
        inside_z_bounds = (z >= 0) & (z <= self.height)
        dist = np.where(
            inside_z_bounds,
            radial_dist,  # Pure radial distance
            np.sqrt(radial_dist**2 + z_dist**2),  # Combined distance
        )

        return dist

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return axis-aligned bounding box."""
        # Box contains spiral extents plus maximum radius
        max_radius = max(self.throat_radius, self.mouth_radius)

        min_corner = np.array(
            [
                self.center_2d[0] - self.outer_spiral_radius - max_radius,
                self.center_2d[1] - self.outer_spiral_radius - max_radius,
                0.0,
            ]
        )
        max_corner = np.array(
            [
                self.center_2d[0] + self.outer_spiral_radius + max_radius,
                self.center_2d[1] + self.outer_spiral_radius + max_radius,
                self.height,
            ]
        )

        return min_corner, max_corner


class HelicalTube(SDFPrimitive):
    """Tube following a helical path with optional taper.

    Creates a 3D tube that follows a helix. The tube cross-section can vary
    from start to end for tapered transmission lines or impedance transformers.

    This is useful for:
    - Helical transmission lines
    - 3D serpentine channels
    - Tapered impedance transformers
    - Coil-shaped resonators

    Args:
        center: (x, y, z) center at bottom of helix
        helix_radius: Radius of helix path (distance from center to tube axis)
        tube_radius_start: Tube cross-section radius at start
        tube_radius_end: Tube cross-section radius at end
        pitch: Height per complete turn
        turns: Number of complete turns

    Example:
        >>> # Constant-diameter helical tube
        >>> tube = HelicalTube(
        ...     center=(0.1, 0.1, 0.0),
        ...     helix_radius=0.06,
        ...     tube_radius_start=0.02,
        ...     tube_radius_end=0.02,
        ...     pitch=0.04,
        ...     turns=3.0
        ... )
        >>>
        >>> # Tapered helical tube
        >>> tapered = HelicalTube(
        ...     center=(0.1, 0.1, 0.0),
        ...     helix_radius=0.06,
        ...     tube_radius_start=0.01,
        ...     tube_radius_end=0.03,
        ...     pitch=0.04,
        ...     turns=3.0
        ... )
    """

    def __init__(
        self,
        center: tuple[float, float, float],
        helix_radius: float,
        tube_radius_start: float,
        tube_radius_end: float,
        pitch: float,
        turns: float,
    ):
        from strata_fdtd.geometry.paths import HelixPath

        self.center = np.array(center, dtype=np.float64)
        self.helix_radius = float(helix_radius)
        self.tube_radius_start = float(tube_radius_start)
        self.tube_radius_end = float(tube_radius_end)
        self.pitch = float(pitch)
        self.turns = float(turns)

        if self.helix_radius <= 0:
            raise ValueError("helix_radius must be positive")
        if self.tube_radius_start <= 0 or self.tube_radius_end <= 0:
            raise ValueError("tube radii must be positive")
        if self.pitch <= 0:
            raise ValueError("pitch must be positive")
        if self.turns <= 0:
            raise ValueError("turns must be positive")

        # Create helix path
        self.helix = HelixPath(
            center=self.center,
            radius=self.helix_radius,
            pitch=self.pitch,
            turns=self.turns,
        )

        # Sample path with high density
        n_samples = max(200, int(100 * self.turns))
        self.path_3d = self.helix.sample_points(n_samples)

        # Compute tube radius at each path point
        t = np.linspace(0, 1, n_samples)
        self.tube_radii = (
            self.tube_radius_start + (self.tube_radius_end - self.tube_radius_start) * t
        )

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate SDF for helical tube.

        For each point:
        1. Find distance to helix path
        2. Get tube radius at nearest path point
        3. Distance is path_distance - tube_radius
        """
        from strata_fdtd.geometry.paths import distance_to_polyline

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be Nx3 array, got shape {points.shape}")

        # Distance to helix path
        dist_to_path = distance_to_polyline(points, self.path_3d)

        # For each point, find nearest path segment to determine tube radius
        n_points = points.shape[0]
        tube_radii_at_points = np.zeros(n_points)

        for i in range(n_points):
            # Find nearest path point
            dists_to_path_points = np.linalg.norm(self.path_3d - points[i], axis=1)
            nearest_idx = np.argmin(dists_to_path_points)
            tube_radii_at_points[i] = self.tube_radii[nearest_idx]

        # Distance is: distance to path minus tube radius
        return dist_to_path - tube_radii_at_points

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return axis-aligned bounding box."""
        # Box contains helix extents plus maximum tube radius
        max_tube_radius = max(self.tube_radius_start, self.tube_radius_end)
        total_radius = self.helix_radius + max_tube_radius

        min_corner = np.array(
            [
                self.center[0] - total_radius,
                self.center[1] - total_radius,
                self.center[2],
            ]
        )
        max_corner = np.array(
            [
                self.center[0] + total_radius,
                self.center[1] + total_radius,
                self.center[2] + self.helix.total_height,
            ]
        )

        return min_corner, max_corner


# === Factory Functions ===


def conical_horn(
    throat_pos: tuple[float, float, float],
    mouth_pos: tuple[float, float, float],
    throat_r: float,
    mouth_r: float,
) -> Horn:
    """Create a conical horn with linear taper.

    Args:
        throat_pos: (x, y, z) throat position
        mouth_pos: (x, y, z) mouth position
        throat_r: Throat radius
        mouth_r: Mouth radius

    Returns:
        Horn with conical profile

    Example:
        >>> horn = conical_horn((0, 0, 0.1), (0, 0, 0), 0.01, 0.02)
    """
    return Horn(throat_pos, mouth_pos, throat_r, mouth_r, "conical")


def exponential_horn(
    throat_pos: tuple[float, float, float],
    mouth_pos: tuple[float, float, float],
    throat_r: float,
    mouth_r: float,
) -> Horn:
    """Create an exponential horn with smooth loading.

    Args:
        throat_pos: (x, y, z) throat position
        mouth_pos: (x, y, z) mouth position
        throat_r: Throat radius
        mouth_r: Mouth radius

    Returns:
        Horn with exponential profile

    Example:
        >>> horn = exponential_horn((0, 0, 0.1), (0, 0, 0), 0.025, 0.075)
    """
    return Horn(throat_pos, mouth_pos, throat_r, mouth_r, "exponential")


def hyperbolic_horn(
    throat_pos: tuple[float, float, float],
    mouth_pos: tuple[float, float, float],
    throat_r: float,
    mouth_r: float,
) -> Horn:
    """Create a hyperbolic horn with good low-frequency loading.

    Args:
        throat_pos: (x, y, z) throat position
        mouth_pos: (x, y, z) mouth position
        throat_r: Throat radius
        mouth_r: Mouth radius

    Returns:
        Horn with hyperbolic profile

    Example:
        >>> horn = hyperbolic_horn((0, 0, 0.1), (0, 0, 0), 0.02, 0.06)
    """
    return Horn(throat_pos, mouth_pos, throat_r, mouth_r, "hyperbolic")


def tractrix_horn(
    throat_pos: tuple[float, float, float],
    mouth_pos: tuple[float, float, float],
    throat_r: float,
    mouth_r: float,
) -> Horn:
    """Create a tractrix horn with constant directivity.

    Args:
        throat_pos: (x, y, z) throat position
        mouth_pos: (x, y, z) mouth position
        throat_r: Throat radius
        mouth_r: Mouth radius

    Returns:
        Horn with tractrix profile

    Example:
        >>> horn = tractrix_horn((0, 0, 0.1), (0, 0, 0), 0.025, 0.08)
    """
    return Horn(throat_pos, mouth_pos, throat_r, mouth_r, "tractrix")

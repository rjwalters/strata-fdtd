"""
Spiral and helical path generators for folded horns and serpentine channels.

This module provides parametric path generators for creating compact acoustic
paths within limited volumes. These are especially useful for:
- Folded horn loudspeakers
- Serpentine transmission lines
- Tapered impedance transformers
- Labyrinth absorbers

Classes:
    SpiralPath: 2D logarithmic or Archimedean spiral
    HelixPath: 3D helical path with constant radius and pitch

Functions:
    distance_to_polyline: Distance from points to a polyline
    distance_to_helix: Distance from points to a helix curve
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpiralPath:
    """2D logarithmic or Archimedean spiral path.

    Generates smooth spiral curves for compact folded paths. Two types:
    - Logarithmic: r = a * exp(b * theta) - constant curvature ratio
    - Archimedean: r = a + b * theta - constant spacing between turns

    Logarithmic spirals maintain constant angle between radius and tangent,
    making them ideal for smooth horn expansions. Archimedean spirals have
    uniform spacing, useful for uniform-width channels.

    Attributes:
        center: (x, y) center point of spiral
        inner_radius: Starting radius
        outer_radius: Ending radius
        turns: Number of complete 360° rotations
        spiral_type: "logarithmic" or "archimedean"

    Example:
        >>> # Logarithmic spiral for horn loading
        >>> spiral = SpiralPath(
        ...     center=(0.15, 0.15),
        ...     inner_radius=0.04,
        ...     outer_radius=0.12,
        ...     turns=2.5,
        ...     spiral_type="logarithmic"
        ... )
        >>> points = spiral.sample_points(100)
        >>> print(f"Path length: {spiral.total_length:.3f} m")
    """

    center: NDArray[np.floating] | tuple[float, float]
    inner_radius: float
    outer_radius: float
    turns: float
    spiral_type: Literal["logarithmic", "archimedean"] = "logarithmic"

    def __post_init__(self) -> None:
        """Validate and normalize parameters."""
        self.center = np.asarray(self.center, dtype=np.float64)
        if self.center.shape != (2,):
            raise ValueError(f"center must be 2D point, got shape {self.center.shape}")

        if self.inner_radius <= 0:
            raise ValueError(f"inner_radius must be positive, got {self.inner_radius}")
        if self.outer_radius <= self.inner_radius:
            raise ValueError(
                f"outer_radius ({self.outer_radius}) must be > inner_radius ({self.inner_radius})"
            )
        if self.turns <= 0:
            raise ValueError(f"turns must be positive, got {self.turns}")

        if self.spiral_type not in ("logarithmic", "archimedean"):
            raise ValueError(
                f"spiral_type must be 'logarithmic' or 'archimedean', got '{self.spiral_type}'"
            )

    def sample_points(self, n_points: int = 100) -> NDArray[np.floating]:
        """Generate (n, 2) array of points along spiral.

        Args:
            n_points: Number of points to sample

        Returns:
            (n_points, 2) array of (x, y) coordinates
        """
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")

        theta = np.linspace(0, 2 * np.pi * self.turns, n_points)

        if self.spiral_type == "logarithmic":
            # r = r0 * exp(k * theta)
            # Solve for k: r1 = r0 * exp(k * theta_max)
            # k = ln(r1/r0) / theta_max
            k = np.log(self.outer_radius / self.inner_radius) / (2 * np.pi * self.turns)
            r = self.inner_radius * np.exp(k * theta)
        else:  # archimedean
            # r = r0 + k * theta
            # Solve for k: r1 = r0 + k * theta_max
            # k = (r1 - r0) / theta_max
            k = (self.outer_radius - self.inner_radius) / (2 * np.pi * self.turns)
            r = self.inner_radius + k * theta

        x = self.center[0] + r * np.cos(theta)
        y = self.center[1] + r * np.sin(theta)

        return np.stack([x, y], axis=1)

    @property
    def total_length(self) -> float:
        """Approximate path length via dense sampling.

        For logarithmic spirals, there's an analytical formula, but
        numerical integration is more general and works for both types.

        Returns:
            Total arc length in meters (or same units as radii)
        """
        # Use dense sampling for accurate length estimate
        points = self.sample_points(1000)
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return float(np.sum(segment_lengths))


@dataclass
class HelixPath:
    """3D helical path with constant radius and pitch.

    Generates a helix centered on the Z-axis (or arbitrary axis via rotation).
    Useful for vertical stacking of spiral paths or 3D serpentine channels.

    Attributes:
        center: (x, y, z) center point at bottom of helix
        radius: Radial distance from helix axis
        pitch: Height per complete turn (vertical spacing)
        turns: Number of complete 360° rotations

    Example:
        >>> # Helical transmission line
        >>> helix = HelixPath(
        ...     center=(0.1, 0.1, 0.0),
        ...     radius=0.06,
        ...     pitch=0.04,
        ...     turns=3.0
        ... )
        >>> points = helix.sample_points(100)
        >>> print(f"Total height: {helix.total_height:.3f} m")
        >>> print(f"Arc length: {helix.total_length:.3f} m")
    """

    center: NDArray[np.floating] | tuple[float, float, float]
    radius: float
    pitch: float
    turns: float

    def __post_init__(self) -> None:
        """Validate and normalize parameters."""
        self.center = np.asarray(self.center, dtype=np.float64)
        if self.center.shape != (3,):
            raise ValueError(f"center must be 3D point, got shape {self.center.shape}")

        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.pitch <= 0:
            raise ValueError(f"pitch must be positive, got {self.pitch}")
        if self.turns <= 0:
            raise ValueError(f"turns must be positive, got {self.turns}")

    def sample_points(self, n_points: int = 100) -> NDArray[np.floating]:
        """Generate (n, 3) array of points along helix.

        Args:
            n_points: Number of points to sample

        Returns:
            (n_points, 3) array of (x, y, z) coordinates
        """
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")

        theta = np.linspace(0, 2 * np.pi * self.turns, n_points)

        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        z = self.center[2] + (theta / (2 * np.pi)) * self.pitch

        return np.stack([x, y, z], axis=1)

    @property
    def total_length(self) -> float:
        """Exact helix arc length via analytical formula.

        For a helix: L = turns * sqrt((2*pi*r)^2 + pitch^2)

        Returns:
            Total arc length in meters (or same units as radius/pitch)
        """
        circumference = 2 * np.pi * self.radius
        length_per_turn = np.sqrt(circumference**2 + self.pitch**2)
        return float(self.turns * length_per_turn)

    @property
    def total_height(self) -> float:
        """Total vertical extent of helix.

        Returns:
            Height in meters (or same units as pitch)
        """
        return float(self.turns * self.pitch)


# === Distance-to-curve utilities ===


def distance_to_polyline(
    points: NDArray[np.floating], polyline: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Compute distance from points to a polyline (sequence of segments).

    For each query point, finds the minimum distance to any segment of the
    polyline. This is useful for swept-volume SDFs where the path is
    discretized into segments.

    Args:
        points: (N, d) query points (d = 2 or 3)
        polyline: (M, d) polyline vertices

    Returns:
        (N,) distances to nearest point on polyline

    Example:
        >>> # Distance to 2D path
        >>> path = np.array([[0, 0], [1, 0], [1, 1]])
        >>> points = np.array([[0.5, 0.5], [2, 2]])
        >>> dists = distance_to_polyline(points, path)
    """
    points = np.asarray(points, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)

    if points.shape[1] != polyline.shape[1]:
        raise ValueError(
            f"Dimension mismatch: points are {points.shape[1]}D, polyline is {polyline.shape[1]}D"
        )

    if polyline.shape[0] < 2:
        raise ValueError(f"Polyline must have >= 2 vertices, got {polyline.shape[0]}")

    n_points = points.shape[0]
    n_segments = polyline.shape[0] - 1

    # For each segment, compute distance from all points
    min_dists = np.full(n_points, np.inf, dtype=np.float64)

    for i in range(n_segments):
        p1 = polyline[i]
        p2 = polyline[i + 1]

        # Vector from p1 to p2
        segment = p2 - p1
        segment_length_sq = np.dot(segment, segment)

        if segment_length_sq < 1e-14:
            # Degenerate segment (p1 == p2), treat as point
            dists = np.linalg.norm(points - p1, axis=1)
        else:
            # Project points onto segment line
            # t = dot(point - p1, segment) / ||segment||^2
            t = np.dot(points - p1, segment) / segment_length_sq

            # Clamp t to [0, 1] for segment endpoints
            t_clamped = np.clip(t, 0, 1)

            # Closest point on segment
            closest = p1 + t_clamped[:, np.newaxis] * segment

            # Distance to closest point
            dists = np.linalg.norm(points - closest, axis=1)

        # Update minimum distances
        min_dists = np.minimum(min_dists, dists)

    return min_dists


def distance_to_helix(
    points: NDArray[np.floating],
    center: NDArray[np.floating] | tuple[float, float, float],
    radius: float,
    pitch: float,
    turns: float,
) -> NDArray[np.floating]:
    """Compute distance from points to helix curve.

    Uses parametric projection onto the helix. For each point, finds the
    closest point on the helix via optimization (approximated by discretization).

    This is useful for HelicalTube SDF evaluation.

    Args:
        points: (N, 3) query points
        center: (x, y, z) helix center at bottom
        radius: Helix radius
        pitch: Height per turn
        turns: Number of turns

    Returns:
        (N,) distances to helix curve

    Note:
        Uses discretized helix sampling for efficiency. For very high accuracy,
        increase the sampling density (currently uses ~50 samples per turn).

    Example:
        >>> points = np.array([[0.1, 0.1, 0.05], [0.2, 0.2, 0.1]])
        >>> dists = distance_to_helix(points, center=(0.1, 0.1, 0.0),
        ...                           radius=0.05, pitch=0.04, turns=3.0)
    """
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3 array, got shape {points.shape}")
    if center.shape != (3,):
        raise ValueError(f"center must be 3D point, got shape {center.shape}")

    # Sample helix with ~50 points per turn for accuracy
    n_samples = max(100, int(50 * turns))
    helix_points = HelixPath(center=center, radius=radius, pitch=pitch, turns=turns).sample_points(
        n_samples
    )

    # Use distance to polyline approximation
    return distance_to_polyline(points, helix_points)

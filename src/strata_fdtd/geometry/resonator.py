"""
Helmholtz resonator primitives for acoustic strata_fdtd design.

This module provides SDF-based Helmholtz resonators combining cavity and neck
geometries for use in broadband acoustic absorbers and strata_fdtd structures.

Classes:
    HelmholtzResonator: Composite SDF primitive with cavity and neck

Functions:
    helmholtz_array: Generate arrays of detuned resonators

Example:
    >>> from strata_fdtd import HelmholtzResonator, Box
    >>>
    >>> # Single resonator tuned to 500 Hz
    >>> resonator = HelmholtzResonator(
    ...     position=(0.1, 0.1, 0.1),
    ...     cavity_shape="box",
    ...     cavity_size=(0.04, 0.04, 0.04),  # 40mm cube
    ...     neck_length=0.015,                # 15mm neck
    ...     neck_radius=0.004,                # 8mm diameter
    ...     neck_direction=(0, 0, 1),         # Opens upward
    ... )
    >>> print(f"f₀ = {resonator.resonant_frequency:.1f} Hz")
    >>>
    >>> # Array for 300-1200 Hz absorption
    >>> region = Box(center=(0.1, 0.1, 0.15), size=(0.08, 0.08, 0.1))
    >>> absorbers = helmholtz_array(
    ...     region=region,
    ...     frequency_range=(300, 1200),
    ...     n_resonators=8,
    ...     spacing="log",
    ... )

References:
    - Helmholtz resonator theory: Fletcher & Rossing, Physics of Musical Instruments
    - Metamaterial absorbers: Ma et al., Nature Materials (2014)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from strata_fdtd.geometry.sdf import Box, Cylinder, SDFPrimitive, Sphere, Union

if TYPE_CHECKING:
    pass

# Physical constants
SPEED_OF_SOUND = 343.0  # m/s at 20°C


@dataclass
class HelmholtzResonator(SDFPrimitive):
    """
    Helmholtz resonator as a composite SDF primitive.

    Combines a cavity (box, sphere, or cylinder) with a cylindrical neck
    to create an acoustic resonator. The resonant frequency is determined
    by the cavity volume and neck dimensions.

    Attributes:
        position: (x, y, z) center of cavity in meters
        cavity_shape: Shape of cavity ("box", "sphere", "cylinder")
        cavity_size: Size parameters in meters:
            - box: (width, height, depth)
            - sphere: (radius,)
            - cylinder: (radius, length)
        neck_length: Physical neck length in meters
        neck_radius: Neck radius in meters (circular cross-section)
        neck_direction: (x, y, z) unit vector for neck direction
        opening_type: "flanged" or "unflanged" (affects end correction)

    Example:
        >>> # Box cavity resonator
        >>> res = HelmholtzResonator(
        ...     position=(0.05, 0.05, 0.05),
        ...     cavity_shape="box",
        ...     cavity_size=(0.03, 0.03, 0.03),
        ...     neck_length=0.01,
        ...     neck_radius=0.005,
        ...     neck_direction=(0, 0, 1),
        ... )
        >>> print(f"Resonates at {res.resonant_frequency:.1f} Hz")
        >>>
        >>> # Spherical cavity
        >>> res2 = HelmholtzResonator(
        ...     position=(0.05, 0.05, 0.05),
        ...     cavity_shape="sphere",
        ...     cavity_size=(0.015,),  # 15mm radius
        ...     neck_length=0.01,
        ...     neck_radius=0.003,
        ... )
    """

    position: NDArray[np.floating] | tuple[float, float, float]
    cavity_shape: Literal["box", "sphere", "cylinder"] = "box"
    cavity_size: tuple[float, ...] = (0.03, 0.03, 0.03)  # meters
    neck_length: float = 0.01  # meters
    neck_radius: float = 0.005  # meters
    neck_direction: NDArray[np.floating] | tuple[float, float, float] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    opening_type: Literal["flanged", "unflanged"] = "unflanged"

    # Internal state
    _cavity: SDFPrimitive = field(init=False, repr=False)
    _neck: Cylinder = field(init=False, repr=False)
    _combined: Union = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize and validate the resonator geometry."""
        # Convert to numpy arrays
        self.position = np.asarray(self.position, dtype=np.float64)
        self.neck_direction = np.asarray(self.neck_direction, dtype=np.float64)

        # Normalize neck direction
        norm = np.linalg.norm(self.neck_direction)
        if norm < 1e-10:
            raise ValueError("neck_direction cannot be zero vector")
        self.neck_direction = self.neck_direction / norm

        # Validate dimensions
        if self.neck_length <= 0:
            raise ValueError(f"neck_length must be positive, got {self.neck_length}")
        if self.neck_radius <= 0:
            raise ValueError(f"neck_radius must be positive, got {self.neck_radius}")

        # Build internal CSG representation
        self._build_geometry()

    def _build_geometry(self) -> None:
        """Construct cavity + neck as CSG tree."""
        # Create cavity based on shape
        if self.cavity_shape == "box":
            if len(self.cavity_size) != 3:
                raise ValueError(f"Box cavity requires 3 size parameters, got {len(self.cavity_size)}")
            if any(s <= 0 for s in self.cavity_size):
                raise ValueError(f"Cavity dimensions must be positive, got {self.cavity_size}")
            self._cavity = Box(center=tuple(self.position), size=self.cavity_size)

        elif self.cavity_shape == "sphere":
            if len(self.cavity_size) != 1:
                raise ValueError(f"Sphere cavity requires 1 size parameter (radius), got {len(self.cavity_size)}")
            if self.cavity_size[0] <= 0:
                raise ValueError(f"Cavity radius must be positive, got {self.cavity_size[0]}")
            self._cavity = Sphere(center=tuple(self.position), radius=self.cavity_size[0])

        elif self.cavity_shape == "cylinder":
            if len(self.cavity_size) != 2:
                raise ValueError(f"Cylinder cavity requires 2 size parameters (radius, length), got {len(self.cavity_size)}")
            if any(s <= 0 for s in self.cavity_size):
                raise ValueError(f"Cavity dimensions must be positive, got {self.cavity_size}")
            # Cylinder aligned with neck direction
            cavity_radius, cavity_length = self.cavity_size
            axis = self.neck_direction * cavity_length
            p1 = self.position - axis / 2
            p2 = self.position + axis / 2
            self._cavity = Cylinder(p1=tuple(p1), p2=tuple(p2), radius=cavity_radius)

        else:
            raise ValueError(f"Unknown cavity_shape: {self.cavity_shape}")

        # Create neck
        neck_start = self._cavity_surface_point()
        neck_end = neck_start + self.neck_direction * self.neck_length
        self._neck = Cylinder(p1=tuple(neck_start), p2=tuple(neck_end), radius=self.neck_radius)

        # Combined geometry
        self._combined = Union(self._cavity, self._neck)

    def _cavity_surface_point(self) -> NDArray[np.floating]:
        """Find point on cavity surface in neck direction."""
        if self.cavity_shape == "box":
            # Project to box surface
            half_size = np.array(self.cavity_size, dtype=np.float64) / 2
            # Find which face the neck direction points to
            # Scale factor to reach each face
            t_values = half_size / (np.abs(self.neck_direction) + 1e-10)
            t = np.min(t_values)
            return self.position + self.neck_direction * t

        elif self.cavity_shape == "sphere":
            # Sphere surface is simply center + radius * direction
            return self.position + self.neck_direction * self.cavity_size[0]

        elif self.cavity_shape == "cylinder":
            # Neck connects to one end of cylinder
            cavity_length = self.cavity_size[1]
            return self.position + self.neck_direction * (cavity_length / 2)

        raise ValueError(f"Unknown cavity_shape: {self.cavity_shape}")

    def sdf(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate SDF at points (delegates to combined cavity+neck)."""
        return self._combined.sdf(points)

    @property
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return bounding box of combined cavity+neck."""
        return self._combined.bounding_box

    @property
    def cavity_volume(self) -> float:
        """Cavity volume in m³."""
        if self.cavity_shape == "box":
            return np.prod(self.cavity_size)
        elif self.cavity_shape == "sphere":
            radius = self.cavity_size[0]
            return (4 / 3) * np.pi * radius**3
        elif self.cavity_shape == "cylinder":
            radius, length = self.cavity_size
            return np.pi * radius**2 * length
        raise ValueError(f"Unknown cavity_shape: {self.cavity_shape}")

    @property
    def neck_area(self) -> float:
        """Neck cross-sectional area in m²."""
        return np.pi * self.neck_radius**2

    @property
    def effective_neck_length(self) -> float:
        """
        Effective neck length with end corrections.

        The end correction accounts for the air mass oscillating just
        outside the neck opening. Different corrections apply for
        flanged vs unflanged openings.

        Returns:
            Effective length in meters
        """
        # End correction depends on opening type
        # See Fletcher & Rossing, "The Physics of Musical Instruments"
        if self.opening_type == "flanged":
            # Flanged: 0.85 * radius correction at opening
            # Interior end (connects to cavity): ~0.85 * radius
            # Exterior end (flanged opening): ~0.85 * radius
            end_correction = 2 * 0.85 * self.neck_radius
        else:
            # Unflanged: 0.6 * radius at open end
            # Interior end (connects to cavity): ~0.6 * radius
            # Exterior end (unflanged): ~0.6 * radius
            end_correction = 2 * 0.6 * self.neck_radius

        return self.neck_length + end_correction

    @property
    def resonant_frequency(self) -> float:
        """
        Analytical resonant frequency in Hz.

        Uses the Helmholtz resonator formula:
            f₀ = (c / 2π) * sqrt(S / (V * L_eff))

        Where:
            c = speed of sound
            S = neck cross-sectional area
            V = cavity volume
            L_eff = effective neck length (with end corrections)

        Returns:
            Resonant frequency in Hz
        """
        c = SPEED_OF_SOUND
        S = self.neck_area
        V = self.cavity_volume
        L = self.effective_neck_length

        if V <= 0 or S <= 0 or L <= 0:
            raise ValueError(f"Invalid geometry: V={V}, S={S}, L={L}")

        f0 = (c / (2 * np.pi)) * np.sqrt(S / (V * L))
        return float(f0)

    @property
    def quality_factor(self) -> float:
        """
        Estimated Q factor (higher = narrower bandwidth).

        This is a simplified estimate based on geometry.
        Real Q depends on viscous and thermal losses in the neck,
        which require more detailed modeling.

        Returns:
            Dimensionless Q factor
        """
        # Simplified estimate: Q ∝ sqrt(V / S*L)
        # Typical range: 5-50 for physical Helmholtz resonators
        # This is a placeholder - proper Q calculation requires
        # viscothermal boundary layer analysis
        return 10.0


def helmholtz_array(
    region: SDFPrimitive,
    frequency_range: tuple[float, float],
    n_resonators: int,
    spacing: Literal["log", "linear"] = "log",
    cavity_shape: Literal["box", "sphere", "cylinder"] = "box",
    neck_length: float = 0.01,
    neck_radius: float = 0.003,
) -> list[HelmholtzResonator]:
    """
    Generate array of detuned Helmholtz resonators for broadband absorption.

    Creates resonators with frequencies distributed across the specified range,
    positioned to fit within the given region. Uses a fixed neck geometry and
    varies cavity volume to achieve target frequencies.

    Args:
        region: Bounding region (SDFPrimitive) for resonator placement
        frequency_range: (f_min, f_max) in Hz
        n_resonators: Number of resonators to create
        spacing: Frequency distribution ("log" or "linear")
        cavity_shape: Shape for all cavities
        neck_length: Neck length in meters (constant for all resonators)
        neck_radius: Neck radius in meters (constant for all resonators)

    Returns:
        List of HelmholtzResonator objects with frequencies spanning the range

    Example:
        >>> from strata_fdtd import Box, helmholtz_array
        >>> region = Box(center=(0.1, 0.1, 0.1), size=(0.08, 0.08, 0.1))
        >>> absorbers = helmholtz_array(
        ...     region=region,
        ...     frequency_range=(300, 1200),
        ...     n_resonators=8,
        ...     spacing="log",
        ... )
        >>> for i, res in enumerate(absorbers):
        ...     print(f"Resonator {i}: {res.resonant_frequency:.1f} Hz")
    """
    f_min, f_max = frequency_range

    if f_min <= 0 or f_max < f_min:
        raise ValueError(f"Invalid frequency range: ({f_min}, {f_max})")
    if n_resonators < 1:
        raise ValueError(f"n_resonators must be >= 1, got {n_resonators}")
    if neck_length <= 0 or neck_radius <= 0:
        raise ValueError("neck_length and neck_radius must be positive")

    # Generate target frequencies
    if spacing == "log":
        frequencies = np.geomspace(f_min, f_max, n_resonators)
    elif spacing == "linear":
        frequencies = np.linspace(f_min, f_max, n_resonators)
    else:
        raise ValueError(f"spacing must be 'log' or 'linear', got {spacing}")

    # Get region bounding box for positioning
    bb_min, bb_max = region.bounding_box
    region_size = bb_max - bb_min

    resonators = []
    for i, f in enumerate(frequencies):
        # Design resonator for target frequency
        # Solve Helmholtz equation for required cavity volume
        # f = (c / 2π) * sqrt(S / (V * L_eff))
        # V = S / (L_eff * (2π*f / c)²)

        c = SPEED_OF_SOUND
        S = np.pi * neck_radius**2

        # Estimate effective length (we'll refine after creating resonator)
        # For unflanged: L_eff ≈ L + 1.2*r
        L_eff = neck_length + 1.2 * neck_radius

        # Required volume
        V = S / (L_eff * (2 * np.pi * f / c) ** 2)

        # Convert volume to cavity size based on shape
        cavity_size = _volume_to_size(V, cavity_shape)

        # Position within region (simple grid layout)
        position = _compute_position(region, i, n_resonators, bb_min, region_size)

        # Create resonator
        resonators.append(
            HelmholtzResonator(
                position=position,
                cavity_shape=cavity_shape,
                cavity_size=cavity_size,
                neck_length=neck_length,
                neck_radius=neck_radius,
                neck_direction=(0, 0, 1),  # Default: upward opening
            )
        )

    return resonators


def _volume_to_size(volume: float, shape: str) -> tuple[float, ...]:
    """
    Convert volume to shape-specific size parameters.

    Args:
        volume: Volume in m³
        shape: "box", "sphere", or "cylinder"

    Returns:
        Size tuple appropriate for the shape
    """
    if shape == "box":
        # Cube with given volume
        side = np.cbrt(volume)
        return (side, side, side)

    elif shape == "sphere":
        # V = (4/3) π r³ → r = ∛(3V / 4π)
        radius = np.cbrt(3 * volume / (4 * np.pi))
        return (radius,)

    elif shape == "cylinder":
        # Arbitrary choice: length = 2 * radius
        # V = π r² L = π r² (2r) = 2π r³
        # r = ∛(V / 2π)
        radius = np.cbrt(volume / (2 * np.pi))
        length = 2 * radius
        return (radius, length)

    raise ValueError(f"Unknown shape: {shape}")


def _compute_position(
    region: SDFPrimitive,
    index: int,
    total: int,
    bb_min: NDArray[np.floating],
    region_size: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute position for resonator within region.

    Uses a simple grid layout. For more sophisticated packing,
    this could be extended to use sphere packing algorithms.

    Args:
        region: Bounding region
        index: Index of this resonator (0 to total-1)
        total: Total number of resonators
        bb_min: Minimum corner of bounding box
        region_size: Size of bounding box

    Returns:
        (x, y, z) position in meters
    """
    # Simple grid layout: arrange in approximate cube
    n = int(np.ceil(np.cbrt(total)))

    ix = index % n
    iy = (index // n) % n
    iz = index // (n * n)

    # Position within grid (with small margin from edges)
    margin = 0.05  # 5% margin
    effective_size = region_size * (1 - 2 * margin)
    cell_size = effective_size / n

    offset = bb_min + region_size * margin + cell_size * np.array([ix + 0.5, iy + 0.5, iz + 0.5])

    return offset

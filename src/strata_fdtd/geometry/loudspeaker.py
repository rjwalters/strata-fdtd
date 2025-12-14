"""
Loudspeaker enclosure builder API.

This module provides a high-level builder API for constructing loudspeaker
cabinet geometries. It uses CSG operations internally to build complex
enclosures from simple components.

Classes:
    LoudspeakerEnclosure: High-level builder for speaker cabinets

Factory Functions:
    tower_speaker: 3-way tower speaker enclosure
    bookshelf_speaker: 2-way bookshelf speaker with bass reflex port

Example:
    >>> from strata_fdtd import LoudspeakerEnclosure, UniformGrid
    >>>
    >>> # Build a bookshelf speaker
    >>> enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.35))
    >>> enc.add_driver((0.1, 0.25), 0.025, baffle_face="front", name="tweeter")
    >>> enc.add_driver((0.1, 0.12), 0.065, baffle_face="front", name="woofer")
    >>> enc.add_port((0.1, 0.05), 0.05, 0.15, baffle_face="front")
    >>>
    >>> # Build and voxelize
    >>> geometry = enc.build()
    >>> grid = UniformGrid(shape=(100, 125, 175), resolution=2e-3)
    >>> mask = geometry.voxelize(grid)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from strata_fdtd.resonator import HelmholtzResonator, helmholtz_array
from strata_fdtd.sdf import (
    Box,
    Cone,
    Cylinder,
    Difference,
    SDFPrimitive,
    Union,
)


@dataclass
class LoudspeakerEnclosure:
    """High-level builder for loudspeaker cabinet simulation geometry.

    Uses a CSG approach internally: start with solid box, subtract
    internal cavities and driver/port cutouts.

    Args:
        external_size: (width, depth, height) in meters
        wall_thickness: Wall thickness in meters (default 19mm MDF)

    Example:
        >>> enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        >>> enc.add_driver((0.1, 0.35), 0.13, baffle_face="front")
        >>> enc.add_port((0.1, 0.1), 0.05, 0.15, baffle_face="front")
        >>> geometry = enc.build()
    """

    external_size: tuple[float, float, float]  # (width, depth, height) in meters
    wall_thickness: float = 0.019  # 19mm MDF default

    # Internal lists for components (initialized in __post_init__)
    _drivers: list[dict] = field(default_factory=list, init=False, repr=False)
    _ports: list[dict] = field(default_factory=list, init=False, repr=False)
    _chambers: list[dict] = field(default_factory=list, init=False, repr=False)
    _bracing: list[dict] = field(default_factory=list, init=False, repr=False)
    _absorbers: list[dict] = field(default_factory=list, init=False, repr=False)
    _resonators: list[dict] = field(default_factory=list, init=False, repr=False)

    # Geometry components (initialized in __post_init__)
    _shell: SDFPrimitive = field(init=False, repr=False)
    _cavity: SDFPrimitive = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize geometry components."""
        # Initialize component lists
        self._drivers = []
        self._ports = []
        self._chambers = []
        self._bracing = []
        self._absorbers = []
        self._resonators = []

        # Build external shell
        outer = Box(
            center=tuple(np.array(self.external_size) / 2),
            size=self.external_size
        )
        inner_size = tuple(s - 2 * self.wall_thickness for s in self.external_size)
        inner = Box(
            center=tuple(np.array(self.external_size) / 2),
            size=inner_size
        )
        self._shell = Difference(outer, inner)
        self._cavity = inner  # Internal air volume

    def add_driver(
        self,
        position: tuple[float, float],  # (x, y) on baffle
        diameter: float,
        mounting_depth: float = 0.0,
        baffle_face: Literal["front", "back", "top", "bottom", "left", "right"] = "front",
        name: str | None = None,
    ) -> LoudspeakerEnclosure:
        """Add driver cutout on specified baffle face.

        Args:
            position: (x, y) position on baffle face
            diameter: Driver frame diameter
            mounting_depth: How deep the driver sits into baffle
            baffle_face: Which face to mount driver on
            name: Optional name for reference

        Returns:
            self for method chaining

        Example:
            >>> enc = LoudspeakerEnclosure((0.2, 0.25, 0.4))
            >>> enc.add_driver((0.1, 0.3), 0.025, name="tweeter")
            >>> enc.add_driver((0.1, 0.15), 0.065, name="woofer")
        """
        self._drivers.append({
            "position": position,
            "diameter": diameter,
            "depth": mounting_depth,
            "face": baffle_face,
            "name": name or f"driver_{len(self._drivers)}",
        })
        return self

    def add_port(
        self,
        position: tuple[float, float],  # (x, y) on baffle
        diameter: float,
        length: float,
        baffle_face: str = "front",
        flare_ratio: float = 1.0,  # mouth/throat ratio
        flare_length: float = 0.0,  # Length of flared section
    ) -> LoudspeakerEnclosure:
        """Add bass reflex port.

        Args:
            position: (x, y) position on baffle face
            diameter: Port internal diameter (at throat if flared)
            length: Port tube length
            baffle_face: Which face port opens on
            flare_ratio: Ratio of mouth to throat diameter
            flare_length: Length of flared section at mouth

        Returns:
            self for method chaining

        Example:
            >>> enc = LoudspeakerEnclosure((0.2, 0.25, 0.4))
            >>> enc.add_port((0.1, 0.1), 0.05, 0.15, baffle_face="front")
        """
        self._ports.append({
            "position": position,
            "diameter": diameter,
            "length": length,
            "face": baffle_face,
            "flare_ratio": flare_ratio,
            "flare_length": flare_length,
        })
        return self

    def add_chamber_divider(
        self,
        z_position: float,
        thickness: float = 0.019,
        hole_positions: list[tuple[float, float, float]] | None = None,  # (x, y, diameter)
    ) -> LoudspeakerEnclosure:
        """Add horizontal divider creating separate chambers.

        Args:
            z_position: Height of divider
            thickness: Divider thickness
            hole_positions: Optional holes through divider (for ports, wiring)

        Returns:
            self for method chaining

        Example:
            >>> enc = LoudspeakerEnclosure((0.2, 0.25, 0.4))
            >>> enc.add_chamber_divider(0.2)  # Divider at 20cm height
        """
        self._chambers.append({
            "z": z_position,
            "thickness": thickness,
            "holes": hole_positions or [],
        })
        return self

    def add_brace(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        width: float,
        height: float,
    ) -> LoudspeakerEnclosure:
        """Add internal brace beam.

        Args:
            start: Starting corner (x, y, z)
            end: Ending corner (x, y, z)
            width: Brace width
            height: Brace height

        Returns:
            self for method chaining

        Example:
            >>> enc = LoudspeakerEnclosure((0.2, 0.25, 0.4))
            >>> enc.add_brace((0.02, 0.125, 0.2), (0.18, 0.125, 0.2), 0.02, 0.015)
        """
        self._bracing.append({
            "start": start,
            "end": end,
            "width": width,
            "height": height,
        })
        return self

    def add_absorber_region(
        self,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        material_id: int = 1,
    ) -> LoudspeakerEnclosure:
        """Define region filled with absorbing material.

        Args:
            bounds: ((x0, y0, z0), (x1, y1, z1)) region bounds
            material_id: ID for material assignment

        Returns:
            self for method chaining

        Example:
            >>> enc = LoudspeakerEnclosure((0.2, 0.25, 0.4))
            >>> enc.add_absorber_region(((0.02, 0.02, 0.02), (0.18, 0.23, 0.1)), material_id=1)
        """
        self._absorbers.append({
            "bounds": bounds,
            "material_id": material_id,
        })
        return self

    def add_helmholtz_array(
        self,
        region_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        frequency_range: tuple[float, float],
        n_resonators: int,
    ) -> LoudspeakerEnclosure:
        """Add array of Helmholtz resonators.

        Args:
            region_bounds: Bounding box for resonator placement
            frequency_range: (f_min, f_max) target frequencies
            n_resonators: Number of resonators

        Returns:
            self for method chaining

        Example:
            >>> enc = LoudspeakerEnclosure((0.2, 0.25, 0.4))
            >>> enc.add_helmholtz_array(
            ...     ((0.02, 0.02, 0.02), (0.18, 0.23, 0.2)),
            ...     (300, 1200),
            ...     8
            ... )
        """
        self._resonators.append({
            "bounds": region_bounds,
            "freq_range": frequency_range,
            "n": n_resonators,
        })
        return self

    def build(self) -> SDFPrimitive:
        """Construct final CSG geometry.

        Returns:
            Combined SDF primitive representing complete enclosure

        Example:
            >>> enc = LoudspeakerEnclosure((0.2, 0.25, 0.4))
            >>> enc.add_driver((0.1, 0.3), 0.025)
            >>> geometry = enc.build()
        """
        # Start with shell
        parts = [self._shell]
        subtractions = []

        # Add chamber dividers
        for chamber in self._chambers:
            divider = self._make_divider(chamber)
            parts.append(divider)
            # Add holes through divider
            for x, y, d in chamber["holes"]:
                hole = self._make_divider_hole(x, y, chamber["z"], chamber["thickness"], d)
                subtractions.append(hole)

        # Add bracing
        for brace in self._bracing:
            parts.append(self._make_brace(brace))

        # Build solid structure
        solid = Union(*parts) if len(parts) > 1 else parts[0]

        # Subtract driver cutouts
        for driver in self._drivers:
            cutout = self._make_driver_cutout(driver)
            subtractions.append(cutout)

        # Subtract port tubes
        for port in self._ports:
            port_geo = self._make_port(port)
            subtractions.append(port_geo)

        # Apply all subtractions
        if subtractions:
            solid = Difference(solid, *subtractions)

        # Add Helmholtz resonator arrays
        for hr_spec in self._resonators:
            resonators = self._build_helmholtz_array(hr_spec)
            for resonator in resonators:
                # Resonators are air cavities - subtract from solid
                solid = Difference(solid, resonator)

        return solid

    def get_material_regions(self) -> list[tuple[SDFPrimitive, int]]:
        """Get absorber regions with material IDs.

        Returns:
            List of (region_primitive, material_id) pairs

        Example:
            >>> enc = LoudspeakerEnclosure((0.2, 0.25, 0.4))
            >>> enc.add_absorber_region(((0.02, 0.02, 0.02), (0.18, 0.23, 0.1)))
            >>> regions = enc.get_material_regions()
        """
        regions = []
        for absorber in self._absorbers:
            min_corner, max_corner = absorber["bounds"]
            center = tuple(np.mean([min_corner, max_corner], axis=0))
            size = tuple(np.abs(np.array(max_corner) - np.array(min_corner)))
            box = Box(center=center, size=size)
            regions.append((box, absorber["material_id"]))
        return regions

    # === Helper methods for building geometry ===

    def _build_helmholtz_array(self, spec: dict) -> list[HelmholtzResonator]:
        """Build array of detuned Helmholtz resonators.

        Uses helmholtz_array() to compute frequencies and positions, then
        creates HelmholtzResonator instances.

        Args:
            spec: Dictionary with keys:
                - bounds: ((x0, y0, z0), (x1, y1, z1)) region bounds
                - freq_range: (f_min, f_max) frequency range
                - n: number of resonators

        Returns:
            List of positioned HelmholtzResonator SDFs
        """
        # Extract parameters from spec
        bounds = spec["bounds"]
        freq_range = spec["freq_range"]
        n_resonators = spec["n"]

        # Create bounding box for the region
        min_corner, max_corner = bounds
        center = tuple(np.mean([min_corner, max_corner], axis=0))
        size = tuple(np.abs(np.array(max_corner) - np.array(min_corner)))
        region = Box(center=center, size=size)

        # Use helmholtz_array to generate detuned resonators
        # Default parameters: box cavity, 10mm neck, 3mm radius
        resonators = helmholtz_array(
            region=region,
            frequency_range=freq_range,
            n_resonators=n_resonators,
            spacing="log",  # Logarithmic spacing for broadband absorption
            cavity_shape="box",
            neck_length=0.01,  # 10mm neck
            neck_radius=0.003,  # 3mm radius (6mm diameter)
        )

        return resonators

    def _make_divider(self, chamber: dict) -> SDFPrimitive:
        """Create chamber divider box."""
        w, d, h = self.external_size
        z = chamber["z"]
        t = chamber["thickness"]

        # Divider spans full internal width/depth, positioned at z
        inner_w = w - 2 * self.wall_thickness
        inner_d = d - 2 * self.wall_thickness

        return Box(
            center=(w / 2, d / 2, z),
            size=(inner_w, inner_d, t)
        )

    def _make_divider_hole(self, x: float, y: float, z: float, thickness: float, diameter: float) -> SDFPrimitive:
        """Create hole through chamber divider."""
        # Cylinder through the divider at (x, y, z)
        p1 = (x, y, z - thickness / 2 - 0.001)  # Slightly extend
        p2 = (x, y, z + thickness / 2 + 0.001)
        return Cylinder(p1=p1, p2=p2, radius=diameter / 2)

    def _make_brace(self, brace: dict) -> SDFPrimitive:
        """Create internal bracing beam."""
        start = np.array(brace["start"])
        end = np.array(brace["end"])
        width = brace["width"]
        height = brace["height"]

        # Brace is a box oriented along start-end direction
        center = (start + end) / 2
        length = np.linalg.norm(end - start)

        # Create box aligned with brace direction
        # For simplicity, assume brace is axis-aligned
        # (Full implementation would require rotation)
        return Box(
            center=tuple(center),
            size=(width, height, length)
        )

    def _make_driver_cutout(self, driver: dict) -> SDFPrimitive:
        """Create driver mounting cutout."""
        pos_2d = driver["position"]
        diameter = driver["diameter"]
        depth = driver["depth"]
        face = driver["face"]

        # Convert 2D position on face to 3D cylinder endpoints
        w, d, h = self.external_size

        if face == "front":
            # Front face is at y=0
            x, z = pos_2d
            p1 = (x, -0.001, z)  # Slightly outside
            p2 = (x, depth, z)
        elif face == "back":
            # Back face is at y=d
            x, z = pos_2d
            p1 = (x, 0, z)
            p2 = (x, d + 0.001, z)
        elif face == "top":
            # Top face is at z=h
            x, y = pos_2d
            p1 = (x, y, 0)
            p2 = (x, y, h + 0.001)
        elif face == "bottom":
            # Bottom face is at z=0
            x, y = pos_2d
            p1 = (x, y, -0.001)
            p2 = (x, y, h)
        elif face == "left":
            # Left face is at x=0
            y, z = pos_2d
            p1 = (-0.001, y, z)
            p2 = (w, y, z)
        elif face == "right":
            # Right face is at x=w
            y, z = pos_2d
            p1 = (0, y, z)
            p2 = (w + 0.001, y, z)
        else:
            raise ValueError(f"Unknown baffle face: {face}")

        return Cylinder(p1=p1, p2=p2, radius=diameter / 2)

    def _make_port(self, port: dict) -> SDFPrimitive:
        """Create bass reflex port tube."""
        pos_2d = port["position"]
        diameter = port["diameter"]
        length = port["length"]
        face = port["face"]
        flare_ratio = port["flare_ratio"]
        flare_length = port["flare_length"]

        # Convert 2D position on face to 3D endpoints
        w, d, h = self.external_size

        if face == "front":
            x, z = pos_2d
            # Port extends inward from front face
            p1 = (x, -0.001, z)  # Mouth (outside)
            p2 = (x, length, z)  # Throat (inside)
        elif face == "back":
            x, z = pos_2d
            p1 = (x, d + 0.001, z)
            p2 = (x, d - length, z)
        elif face == "top":
            x, y = pos_2d
            p1 = (x, y, h + 0.001)
            p2 = (x, y, h - length)
        elif face == "bottom":
            x, y = pos_2d
            p1 = (x, y, -0.001)
            p2 = (x, y, length)
        elif face == "left":
            y, z = pos_2d
            p1 = (-0.001, y, z)
            p2 = (length, y, z)
        elif face == "right":
            y, z = pos_2d
            p1 = (w + 0.001, y, z)
            p2 = (w - length, y, z)
        else:
            raise ValueError(f"Unknown baffle face: {face}")

        # If flared, use conical port
        if flare_ratio > 1.0 and flare_length > 0:
            # Split into straight section + flared section
            # For simplicity, use a single cone
            mouth_r = diameter / 2 * flare_ratio
            throat_r = diameter / 2
            return Cone(p1=p1, p2=p2, r1=mouth_r, r2=throat_r)
        else:
            # Simple cylindrical port
            return Cylinder(p1=p1, p2=p2, radius=diameter / 2)


def tower_speaker(
    width: float,
    depth: float,
    height: float,
    woofer_diameter: float,
    midrange_diameter: float,
    tweeter_diameter: float,
) -> LoudspeakerEnclosure:
    """Create a basic 3-way tower speaker enclosure.

    Standard layout: tweeter at top, midrange below, woofer at bottom.
    Includes chamber dividers between driver sections.

    Args:
        width: Cabinet width in meters
        depth: Cabinet depth in meters
        height: Cabinet height in meters
        woofer_diameter: Woofer frame diameter in meters
        midrange_diameter: Midrange frame diameter in meters
        tweeter_diameter: Tweeter frame diameter in meters

    Returns:
        LoudspeakerEnclosure configured as 3-way tower

    Example:
        >>> enc = tower_speaker(0.2, 0.3, 1.0, 0.2, 0.13, 0.025)
        >>> geometry = enc.build()
    """
    enc = LoudspeakerEnclosure(
        external_size=(width, depth, height),
    )

    # Driver positions (centered on front baffle)
    cx = width / 2

    enc.add_driver((cx, height * 0.85), tweeter_diameter, baffle_face="front", name="tweeter")
    enc.add_driver((cx, height * 0.6), midrange_diameter, baffle_face="front", name="midrange")
    enc.add_driver((cx, height * 0.25), woofer_diameter, baffle_face="front", name="woofer")

    # Chamber dividers
    enc.add_chamber_divider(z_position=height * 0.45)  # Above woofer
    enc.add_chamber_divider(z_position=height * 0.75)  # Above midrange

    return enc


def bookshelf_speaker(
    width: float,
    depth: float,
    height: float,
    woofer_diameter: float,
    tweeter_diameter: float,
    port_diameter: float = 0.05,
    port_length: float = 0.15,
) -> LoudspeakerEnclosure:
    """Create a basic 2-way bookshelf speaker enclosure.

    Includes rear-firing port for bass reflex alignment.

    Args:
        width: Cabinet width in meters
        depth: Cabinet depth in meters
        height: Cabinet height in meters
        woofer_diameter: Woofer frame diameter in meters
        tweeter_diameter: Tweeter frame diameter in meters
        port_diameter: Port diameter in meters
        port_length: Port tube length in meters

    Returns:
        LoudspeakerEnclosure configured as 2-way bookshelf

    Example:
        >>> enc = bookshelf_speaker(0.2, 0.25, 0.35, 0.13, 0.025)
        >>> geometry = enc.build()
    """
    enc = LoudspeakerEnclosure(
        external_size=(width, depth, height),
    )

    cx = width / 2

    enc.add_driver((cx, height * 0.75), tweeter_diameter, baffle_face="front", name="tweeter")
    enc.add_driver((cx, height * 0.35), woofer_diameter, baffle_face="front", name="woofer")
    enc.add_port((cx, height * 0.15), port_diameter, port_length, baffle_face="back")

    return enc

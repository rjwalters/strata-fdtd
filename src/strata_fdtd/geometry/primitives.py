"""
Primitive geometry building blocks for acoustic metamaterials.

Provides HelmholtzCell, Channel, and related primitives for constructing
acoustic absorber and resonator structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from strata_fdtd.manufacturing.lamination import Stack, Violation

# Physical constants
SPEED_OF_SOUND = 343.0  # m/s at 20C

# Default manufacturing constraints
DEFAULT_SLICE_THICKNESS = 6.0  # mm per slice
DEFAULT_MIN_WALL = 3.0  # mm
DEFAULT_MIN_GAP = 2.0  # mm


@dataclass
class HelmholtzCell:
    """
    Helmholtz resonator: a cavity with a neck, the primary acoustic building block.

    The resonator consists of:
    - A rectangular cavity (the "body")
    - A narrow neck connecting the cavity to the outside

    The resonant frequency is determined by the cavity volume and neck dimensions
    according to the Helmholtz equation.

    Attributes:
        cavity_x: Cavity width in mm
        cavity_y: Cavity depth in mm
        cavity_z: Cavity height in slices (each slice = 6mm)
        neck_x: Neck width in mm
        neck_y: Neck depth in mm
        neck_z: Neck length in slices
        position: (x, y, z_slice) position of neck opening
        neck_direction: Direction neck opens ('+x', '-x', '+y', '-y')
    """

    cavity_x: float
    cavity_y: float
    cavity_z: int
    neck_x: float
    neck_y: float
    neck_z: int
    position: tuple[float, float, int]
    neck_direction: str = "+y"

    def __post_init__(self) -> None:
        """Validate dimensions."""
        if self.neck_direction not in ("+x", "-x", "+y", "-y"):
            raise ValueError(
                f"neck_direction must be '+x', '-x', '+y', or '-y', "
                f"got '{self.neck_direction}'"
            )

    @property
    def cavity_volume_mm3(self) -> float:
        """Cavity volume in cubic millimeters."""
        return self.cavity_x * self.cavity_y * (self.cavity_z * DEFAULT_SLICE_THICKNESS)

    @property
    def cavity_volume_m3(self) -> float:
        """Cavity volume in cubic meters."""
        return self.cavity_volume_mm3 * 1e-9

    @property
    def neck_area_mm2(self) -> float:
        """Neck cross-sectional area in square millimeters."""
        return self.neck_x * self.neck_y

    @property
    def neck_area_m2(self) -> float:
        """Neck cross-sectional area in square meters."""
        return self.neck_area_mm2 * 1e-6

    @property
    def neck_length_mm(self) -> float:
        """Neck length in millimeters."""
        return self.neck_z * DEFAULT_SLICE_THICKNESS

    @property
    def neck_length_m(self) -> float:
        """Neck length in meters."""
        return self.neck_length_mm * 1e-3

    def resonant_frequency(self) -> float:
        """
        Calculate analytical resonant frequency with end corrections.

        Uses the Helmholtz resonator formula:
            f = (c / 2π) * sqrt(S / (V * L_eff))

        Where:
            c = speed of sound
            S = neck cross-sectional area
            V = cavity volume
            L_eff = effective neck length (with end corrections)

        The end correction accounts for the air mass that oscillates
        just outside the neck opening.

        Returns:
            Resonant frequency in Hz
        """
        c = SPEED_OF_SOUND
        V = self.cavity_volume_m3
        S = self.neck_area_m2
        L = self.neck_length_m

        # End correction: add 1.7 * sqrt(S/π) for unflanged opening
        # This accounts for the radiating mass of air at the neck mouth
        L_eff = L + 1.7 * np.sqrt(S / np.pi)

        # Helmholtz formula
        f0 = (c / (2 * np.pi)) * np.sqrt(S / (V * L_eff))

        return float(f0)

    def total_height_slices(self) -> int:
        """Total height of the cell in slices."""
        return self.cavity_z + self.neck_z

    def total_height_mm(self) -> float:
        """Total height of the cell in mm."""
        return self.total_height_slices() * DEFAULT_SLICE_THICKNESS

    def to_stack(self, resolution: float = 0.5) -> Stack:
        """
        Generate a Stack representing this Helmholtz cell.

        The cell is oriented with:
        - Cavity at the bottom (lower z)
        - Neck at the top (higher z), opening in neck_direction

        Args:
            resolution: mm per pixel

        Returns:
            Stack with the cell geometry
        """
        from strata_fdtd.manufacturing.lamination import Slice, Stack

        # Calculate total dimensions
        # The outer dimensions include walls around the cavity
        # We assume the cell is centered at position
        pos_x, pos_y, z_start = self.position

        # Determine layout based on neck direction
        if self.neck_direction in ("+x", "-x"):
            # Neck extends in X direction
            outer_x = self.cavity_x + self.neck_z * DEFAULT_SLICE_THICKNESS
            outer_y = self.cavity_y
        else:  # +y or -y
            # Neck extends in Y direction
            outer_x = self.cavity_x
            outer_y = self.cavity_y + self.neck_z * DEFAULT_SLICE_THICKNESS

        # Create slices
        slices = []

        # Cavity slices (air region inside cavity)
        for z in range(self.cavity_z):
            # Create slice with cavity as air
            slice_mask = self._create_cavity_slice(
                outer_x, outer_y, resolution
            )
            slices.append(
                Slice(
                    mask=slice_mask,
                    z_index=z_start + z,
                    resolution=resolution,
                )
            )

        # Neck slices (narrow channel connecting to outside)
        for z in range(self.neck_z):
            slice_mask = self._create_neck_slice(
                outer_x, outer_y, resolution
            )
            slices.append(
                Slice(
                    mask=slice_mask,
                    z_index=z_start + self.cavity_z + z,
                    resolution=resolution,
                )
            )

        return Stack(
            slices=slices,
            resolution_xy=resolution,
            thickness_z=DEFAULT_SLICE_THICKNESS,
        )

    def _create_cavity_slice(
        self,
        outer_x: float,
        outer_y: float,
        resolution: float,
    ) -> np.ndarray:
        """Create a slice mask for the cavity portion."""
        w_px = int(np.ceil(outer_x / resolution))
        h_px = int(np.ceil(outer_y / resolution))
        mask = np.zeros((h_px, w_px), dtype=np.bool_)

        # Cavity position depends on neck direction
        if self.neck_direction == "+y":
            # Cavity at bottom of Y extent
            x0 = int((outer_x - self.cavity_x) / 2 / resolution)
            y0 = 0
        elif self.neck_direction == "-y":
            # Cavity at top of Y extent
            x0 = int((outer_x - self.cavity_x) / 2 / resolution)
            y0 = int((outer_y - self.cavity_y) / resolution)
        elif self.neck_direction == "+x":
            # Cavity at left of X extent
            x0 = 0
            y0 = int((outer_y - self.cavity_y) / 2 / resolution)
        else:  # -x
            # Cavity at right of X extent
            x0 = int((outer_x - self.cavity_x) / resolution)
            y0 = int((outer_y - self.cavity_y) / 2 / resolution)

        x1 = x0 + int(self.cavity_x / resolution)
        y1 = y0 + int(self.cavity_y / resolution)

        mask[y0:y1, x0:x1] = True
        return mask

    def _create_neck_slice(
        self,
        outer_x: float,
        outer_y: float,
        resolution: float,
    ) -> np.ndarray:
        """Create a slice mask for the neck portion."""
        w_px = int(np.ceil(outer_x / resolution))
        h_px = int(np.ceil(outer_y / resolution))
        mask = np.zeros((h_px, w_px), dtype=np.bool_)

        # Neck is centered on the cavity
        if self.neck_direction in ("+y", "-y"):
            # Neck centered in X, at Y edge
            x0 = int((outer_x - self.neck_x) / 2 / resolution)
            x1 = x0 + int(self.neck_x / resolution)

            if self.neck_direction == "+y":
                y0 = int(self.cavity_y / resolution)
                y1 = h_px
            else:
                y0 = 0
                y1 = int((outer_y - self.cavity_y) / resolution)

            mask[y0:y1, x0:x1] = True

        else:  # +x or -x
            # Neck centered in Y, at X edge
            y0 = int((outer_y - self.neck_y) / 2 / resolution)
            y1 = y0 + int(self.neck_y / resolution)

            if self.neck_direction == "+x":
                x0 = int(self.cavity_x / resolution)
                x1 = w_px
            else:
                x0 = 0
                x1 = int((outer_x - self.cavity_x) / resolution)

            mask[y0:y1, x0:x1] = True

        return mask

    def check_manufacturable(
        self,
        min_wall: float = DEFAULT_MIN_WALL,
        min_gap: float = DEFAULT_MIN_GAP,
    ) -> list[Violation]:
        """
        Verify dimensions meet manufacturing constraints.

        Args:
            min_wall: Minimum wall thickness in mm
            min_gap: Minimum gap/slot width in mm

        Returns:
            List of violations (empty if all constraints pass)
        """
        from strata_fdtd.manufacturing.lamination import Violation

        violations = []

        # Check neck dimensions
        if self.neck_x < min_gap:
            violations.append(
                Violation(
                    constraint="min_gap_width",
                    slice_index=self.position[2] + self.cavity_z,
                    location=(self.position[0], self.position[1]),
                    measured=self.neck_x,
                    required=min_gap,
                )
            )

        if self.neck_y < min_gap:
            violations.append(
                Violation(
                    constraint="min_gap_width",
                    slice_index=self.position[2] + self.cavity_z,
                    location=(self.position[0], self.position[1]),
                    measured=self.neck_y,
                    required=min_gap,
                )
            )

        # Check that cavity is big enough to cut
        if self.cavity_x < min_gap:
            violations.append(
                Violation(
                    constraint="min_gap_width",
                    slice_index=self.position[2],
                    location=(self.position[0], self.position[1]),
                    measured=self.cavity_x,
                    required=min_gap,
                )
            )

        if self.cavity_y < min_gap:
            violations.append(
                Violation(
                    constraint="min_gap_width",
                    slice_index=self.position[2],
                    location=(self.position[0], self.position[1]),
                    measured=self.cavity_y,
                    required=min_gap,
                )
            )

        return violations


@dataclass
class Channel:
    """
    Rectangular channel that can span multiple slices.

    Used for connecting cavities, creating waveguides, or providing
    ventilation paths.

    Attributes:
        width: Channel width in mm
        height: Channel height in slices
        path: List of (x, y) waypoints in mm defining the channel path
        z_start: Starting slice index
    """

    width: float
    height: int
    path: list[tuple[float, float]]
    z_start: int

    def __post_init__(self) -> None:
        """Validate path."""
        if len(self.path) < 2:
            raise ValueError("Channel path must have at least 2 waypoints")

    def total_length(self) -> float:
        """Calculate total path length in mm."""
        length = 0.0
        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i + 1]
            length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return length

    def to_stack(self, resolution: float = 0.5) -> Stack:
        """
        Generate a Stack representing this channel.

        Args:
            resolution: mm per pixel

        Returns:
            Stack with the channel geometry
        """
        from strata_fdtd.manufacturing.lamination import Slice, Stack

        # Calculate bounding box
        xs = [p[0] for p in self.path]
        ys = [p[1] for p in self.path]

        # Add margin for channel width
        margin = self.width / 2 + resolution
        min_x = min(xs) - margin
        min_y = min(ys) - margin
        max_x = max(xs) + margin
        max_y = max(ys) + margin

        # Create mask dimensions
        w_px = int(np.ceil((max_x - min_x) / resolution))
        h_px = int(np.ceil((max_y - min_y) / resolution))

        slices = []
        for z in range(self.height):
            mask = np.zeros((h_px, w_px), dtype=np.bool_)

            # Draw channel segments
            for i in range(len(self.path) - 1):
                x1, y1 = self.path[i]
                x2, y2 = self.path[i + 1]

                # Convert to pixel coordinates relative to bounding box
                px1 = (x1 - min_x) / resolution
                py1 = (y1 - min_y) / resolution
                px2 = (x2 - min_x) / resolution
                py2 = (y2 - min_y) / resolution

                # Draw thick line
                self._draw_thick_line(
                    mask, px1, py1, px2, py2, self.width / resolution
                )

            slices.append(
                Slice(
                    mask=mask,
                    z_index=self.z_start + z,
                    resolution=resolution,
                )
            )

        return Stack(
            slices=slices,
            resolution_xy=resolution,
            thickness_z=DEFAULT_SLICE_THICKNESS,
        )

    @staticmethod
    def _draw_thick_line(
        mask: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float,
    ) -> None:
        """Draw a thick line on the mask."""
        # Create coordinate arrays
        h, w = mask.shape
        yy, xx = np.ogrid[:h, :w]

        # Line vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length < 1e-6:
            # Degenerate line - just draw a circle
            mask |= ((xx - x1) ** 2 + (yy - y1) ** 2) <= (width / 2) ** 2
            return

        # Unit vectors
        ux, uy = dx / length, dy / length  # along line
        nx, ny = -uy, ux  # perpendicular to line

        # Point to line distance
        # Project (xx - x1, yy - y1) onto perpendicular direction
        perp_dist = np.abs((xx - x1) * nx + (yy - y1) * ny)

        # Project onto line direction and check if within segment
        along = (xx - x1) * ux + (yy - y1) * uy
        within_segment = (along >= 0) & (along <= length)

        # Within the thick line body
        mask |= (perp_dist <= width / 2) & within_segment

        # Add rounded end caps
        mask |= ((xx - x1) ** 2 + (yy - y1) ** 2) <= (width / 2) ** 2
        mask |= ((xx - x2) ** 2 + (yy - y2) ** 2) <= (width / 2) ** 2


@dataclass
class TaperedChannel:
    """
    Channel with linearly varying width.

    Useful for impedance matching or creating horn-like structures.

    Attributes:
        width_start: Width at start of channel in mm
        width_end: Width at end of channel in mm
        height: Channel height in slices
        path: List of (x, y) waypoints in mm
        z_start: Starting slice index
    """

    width_start: float
    width_end: float
    height: int
    path: list[tuple[float, float]]
    z_start: int

    def __post_init__(self) -> None:
        """Validate path."""
        if len(self.path) < 2:
            raise ValueError("Channel path must have at least 2 waypoints")

    def total_length(self) -> float:
        """Calculate total path length in mm."""
        length = 0.0
        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i + 1]
            length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return length

    def width_at_position(self, t: float) -> float:
        """
        Get channel width at normalized position (0=start, 1=end).

        Args:
            t: Normalized position along path (0 to 1)

        Returns:
            Width in mm at that position
        """
        t = np.clip(t, 0, 1)
        return self.width_start + t * (self.width_end - self.width_start)

    def to_stack(self, resolution: float = 0.5) -> Stack:
        """
        Generate a Stack representing this tapered channel.

        Args:
            resolution: mm per pixel

        Returns:
            Stack with the channel geometry
        """
        from strata_fdtd.manufacturing.lamination import Slice, Stack

        # Calculate bounding box with max width
        xs = [p[0] for p in self.path]
        ys = [p[1] for p in self.path]
        max_width = max(self.width_start, self.width_end)

        margin = max_width / 2 + resolution
        min_x = min(xs) - margin
        min_y = min(ys) - margin
        max_x = max(xs) + margin
        max_y = max(ys) + margin

        w_px = int(np.ceil((max_x - min_x) / resolution))
        h_px = int(np.ceil((max_y - min_y) / resolution))

        # Calculate total length for normalization
        total_len = self.total_length()

        slices = []
        for z in range(self.height):
            mask = np.zeros((h_px, w_px), dtype=np.bool_)

            # Draw tapered channel segments
            cum_len = 0.0
            for i in range(len(self.path) - 1):
                x1, y1 = self.path[i]
                x2, y2 = self.path[i + 1]

                seg_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                t_start = cum_len / total_len if total_len > 0 else 0
                t_end = (cum_len + seg_len) / total_len if total_len > 0 else 1

                w_start = self.width_at_position(t_start)
                w_end = self.width_at_position(t_end)

                # Convert to pixel coordinates
                px1 = (x1 - min_x) / resolution
                py1 = (y1 - min_y) / resolution
                px2 = (x2 - min_x) / resolution
                py2 = (y2 - min_y) / resolution

                self._draw_tapered_line(
                    mask,
                    px1, py1, px2, py2,
                    w_start / resolution,
                    w_end / resolution,
                )

                cum_len += seg_len

            slices.append(
                Slice(
                    mask=mask,
                    z_index=self.z_start + z,
                    resolution=resolution,
                )
            )

        return Stack(
            slices=slices,
            resolution_xy=resolution,
            thickness_z=DEFAULT_SLICE_THICKNESS,
        )

    @staticmethod
    def _draw_tapered_line(
        mask: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width1: float,
        width2: float,
    ) -> None:
        """Draw a tapered line on the mask."""
        h, w = mask.shape
        yy, xx = np.ogrid[:h, :w]

        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length < 1e-6:
            # Degenerate - draw circle with average width
            avg_width = (width1 + width2) / 2
            mask |= ((xx - x1) ** 2 + (yy - y1) ** 2) <= (avg_width / 2) ** 2
            return

        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux

        perp_dist = np.abs((xx - x1) * nx + (yy - y1) * ny)
        along = (xx - x1) * ux + (yy - y1) * uy

        # Normalized position
        t = np.clip(along / length, 0, 1)

        # Width varies linearly with position
        half_width = (width1 + t * (width2 - width1)) / 2

        within_segment = (along >= 0) & (along <= length)
        mask |= (perp_dist <= half_width) & within_segment

        # End caps
        mask |= ((xx - x1) ** 2 + (yy - y1) ** 2) <= (width1 / 2) ** 2
        mask |= ((xx - x2) ** 2 + (yy - y2) ** 2) <= (width2 / 2) ** 2


@dataclass
class SerpentineChannel:
    """
    Folded channel to increase path length in a compact space.

    Creates a serpentine/meandering pattern useful for quarter-wave
    resonators or delay lines.

    Attributes:
        width: Channel width in mm
        height: Channel height in slices
        fold_count: Number of folds (turns)
        fold_spacing: Distance between parallel sections in mm
        total_length: Desired unfolded total length in mm
        z_start: Starting slice index
        start_position: (x, y) start position in mm
        direction: Initial direction ('+x', '-x', '+y', '-y')
    """

    width: float
    height: int
    fold_count: int
    fold_spacing: float
    total_length: float
    z_start: int
    start_position: tuple[float, float] = (0.0, 0.0)
    direction: str = "+x"

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.fold_count < 1:
            raise ValueError("fold_count must be at least 1")
        if self.direction not in ("+x", "-x", "+y", "-y"):
            raise ValueError(f"Invalid direction: {self.direction}")

    def _generate_path(self) -> list[tuple[float, float]]:
        """Generate the serpentine path waypoints."""
        path = [self.start_position]

        # Calculate segment lengths
        # Total turns = fold_count
        # Straight segments = fold_count + 1
        num_straights = self.fold_count + 1
        turn_length = self.fold_spacing  # Each turn traverses fold_spacing
        total_turn_length = self.fold_count * turn_length
        straight_length = (self.total_length - total_turn_length) / num_straights

        if straight_length <= 0:
            raise ValueError(
                f"Total length {self.total_length} mm is too short for "
                f"{self.fold_count} folds with {self.fold_spacing} mm spacing"
            )

        # Direction vectors
        if self.direction == "+x":
            main_dir = (1, 0)
            turn_dir = (0, 1)
        elif self.direction == "-x":
            main_dir = (-1, 0)
            turn_dir = (0, 1)
        elif self.direction == "+y":
            main_dir = (0, 1)
            turn_dir = (1, 0)
        else:  # -y
            main_dir = (0, -1)
            turn_dir = (1, 0)

        current_x, current_y = self.start_position
        turn_sign = 1

        for i in range(num_straights):
            # Straight segment
            current_x += main_dir[0] * straight_length
            current_y += main_dir[1] * straight_length
            path.append((current_x, current_y))

            # Turn (except after last straight)
            if i < self.fold_count:
                current_x += turn_dir[0] * turn_length * turn_sign
                current_y += turn_dir[1] * turn_length * turn_sign
                path.append((current_x, current_y))
                turn_sign *= -1
                main_dir = (-main_dir[0], -main_dir[1])

        return path

    def actual_length(self) -> float:
        """Calculate actual path length (may differ slightly from total_length)."""
        path = self._generate_path()
        length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return length

    def to_stack(self, resolution: float = 0.5) -> Stack:
        """
        Generate a Stack representing this serpentine channel.

        Args:
            resolution: mm per pixel

        Returns:
            Stack with the channel geometry
        """
        path = self._generate_path()
        channel = Channel(
            width=self.width,
            height=self.height,
            path=path,
            z_start=self.z_start,
        )
        return channel.to_stack(resolution)


@dataclass
class SphereCell:
    """
    Spherical cavity approximated as stacked circular cuts.

    Each slice through the sphere at height z from center produces a circular
    cut with radius r(z) = sqrt((D/2)² - z²). This provides a good approximation
    of a spherical cavity using laminated manufacturing.

    Spheres act as acoustic cavities in Helmholtz resonator configurations
    when connected by cylindrical tubes.

    Attributes:
        diameter: Sphere diameter in mm
        position: (x, y, z_slice) center position of sphere
    """

    diameter: float
    position: tuple[float, float, int]

    def __post_init__(self) -> None:
        """Validate dimensions."""
        if self.diameter <= 0:
            raise ValueError(f"diameter must be positive, got {self.diameter}")

    @property
    def radius(self) -> float:
        """Sphere radius in mm."""
        return self.diameter / 2

    @property
    def volume_mm3(self) -> float:
        """Exact sphere volume in cubic millimeters."""
        return (4 / 3) * np.pi * self.radius**3

    @property
    def volume_m3(self) -> float:
        """Exact sphere volume in cubic meters."""
        return self.volume_mm3 * 1e-9

    def num_slices(self) -> int:
        """
        Number of slices needed to approximate this sphere.

        Returns the number of DEFAULT_SLICE_THICKNESS layers that span
        the sphere diameter.
        """
        return max(1, int(np.ceil(self.diameter / DEFAULT_SLICE_THICKNESS)))

    def radius_at_slice(self, slice_offset: int) -> float:
        """
        Calculate the circular cut radius at a given slice offset from center.

        At height z from sphere center: r(z) = sqrt((D/2)² - z²)

        Args:
            slice_offset: Number of slices from center (can be negative)

        Returns:
            Radius of circular cut in mm, or 0 if outside sphere
        """
        # Height from center in mm
        z = slice_offset * DEFAULT_SLICE_THICKNESS
        r_squared = self.radius**2 - z**2
        if r_squared <= 0:
            return 0.0
        return np.sqrt(r_squared)

    def to_stack(self, resolution: float = 0.5) -> Stack:
        """
        Generate a Stack representing this sphere.

        Creates circular cuts at each slice level, sized according to
        the sphere equation.

        Args:
            resolution: mm per pixel

        Returns:
            Stack with the sphere geometry
        """
        from strata_fdtd.manufacturing.lamination import Slice, Stack

        pos_x, pos_y, z_center = self.position
        num_slices = self.num_slices()

        # Calculate z range (slices above and below center)
        half_slices = num_slices // 2
        z_start = z_center - half_slices
        z_end = z_center + half_slices + (1 if num_slices % 2 == 1 else 0)

        # Bounding box size (sphere diameter + margin)
        margin = resolution * 2
        size = self.diameter + margin * 2
        w_px = int(np.ceil(size / resolution))
        h_px = w_px  # Square

        # Center in pixel coordinates
        cx_px = w_px / 2
        cy_px = h_px / 2

        slices = []
        for z_idx in range(z_start, z_end):
            slice_offset = z_idx - z_center
            r = self.radius_at_slice(slice_offset)

            if r <= 0:
                continue

            # Create circular mask
            mask = np.zeros((h_px, w_px), dtype=np.bool_)
            yy, xx = np.ogrid[:h_px, :w_px]

            # Distance from center in pixels
            dist_sq = (xx - cx_px) ** 2 + (yy - cy_px) ** 2
            r_px = r / resolution

            # Circle: air where distance <= radius
            mask[dist_sq <= r_px**2] = True

            slices.append(
                Slice(
                    mask=mask,
                    z_index=z_idx,
                    resolution=resolution,
                )
            )

        return Stack(
            slices=slices,
            resolution_xy=resolution,
            thickness_z=DEFAULT_SLICE_THICKNESS,
        )

    def check_manufacturable(
        self,
        min_gap: float = DEFAULT_MIN_GAP,
    ) -> list[Violation]:
        """
        Verify sphere dimensions meet manufacturing constraints.

        Args:
            min_gap: Minimum gap/slot width in mm

        Returns:
            List of violations (empty if all constraints pass)
        """
        from strata_fdtd.manufacturing.lamination import Violation

        violations = []

        # Need at least 2 slices to approximate a curve
        if self.num_slices() < 2:
            violations.append(
                Violation(
                    constraint="min_sphere_slices",
                    slice_index=self.position[2],
                    location=(self.position[0], self.position[1]),
                    measured=self.num_slices(),
                    required=2,
                )
            )

        # Minimum diameter for smallest cut to be manufacturable
        # The smallest slice radius should be >= min_gap/2
        min_diameter = min_gap * 2
        if self.diameter < min_diameter:
            violations.append(
                Violation(
                    constraint="min_sphere_diameter",
                    slice_index=self.position[2],
                    location=(self.position[0], self.position[1]),
                    measured=self.diameter,
                    required=min_diameter,
                )
            )

        return violations


@dataclass
class SphereLattice:
    """
    Lattice of spherical cavities connected by cylindrical tubes.

    Supports cubic, BCC (body-centered cubic), and FCC (face-centered cubic)
    lattice arrangements. Each sphere acts as a Helmholtz resonator cavity,
    with tubes between adjacent spheres acting as necks.

    Lattice geometry reference:
    | Lattice | Packing | Neighbors | Pattern                       |
    |---------|---------|-----------|-------------------------------|
    | Cubic   | 52%     | 6         | Square grid, aligned layers   |
    | BCC     | 68%     | 8         | Square grid, alternating offset|
    | FCC     | 74%     | 12        | Hex grid, ABC stacking        |

    Attributes:
        sphere_diameter: Cavity diameter in mm
        tube_diameter: Neck/tube diameter in mm
        lattice_type: 'cubic', 'bcc', or 'fcc'
        extent: (nx, ny, nz) number of unit cells in each direction
        lattice_constant: Spacing between sphere centers in mm
                         (auto-calculated if not specified)
    """

    sphere_diameter: float
    tube_diameter: float
    lattice_type: str
    extent: tuple[int, int, int]
    lattice_constant: float | None = None

    def __post_init__(self) -> None:
        """Validate parameters and compute lattice constant if needed."""
        if self.lattice_type not in ("cubic", "bcc", "fcc"):
            raise ValueError(
                f"lattice_type must be 'cubic', 'bcc', or 'fcc', "
                f"got '{self.lattice_type}'"
            )

        if self.sphere_diameter <= 0:
            raise ValueError(f"sphere_diameter must be positive, got {self.sphere_diameter}")

        if self.tube_diameter <= 0:
            raise ValueError(f"tube_diameter must be positive, got {self.tube_diameter}")

        if any(e < 1 for e in self.extent):
            raise ValueError(f"extent values must be >= 1, got {self.extent}")

        # Auto-calculate lattice constant if not provided
        if self.lattice_constant is None:
            # Minimum spacing: spheres must not overlap, tubes need length > 0
            # Default: sphere diameter + tube diameter (so tubes are 1 tube_diameter long)
            self.lattice_constant = self.sphere_diameter + self.tube_diameter

        # Validate lattice constant
        min_lattice = self.sphere_diameter + 0.1  # Tiny margin for tubes
        if self.lattice_constant < min_lattice:
            raise ValueError(
                f"lattice_constant {self.lattice_constant} too small for "
                f"sphere_diameter {self.sphere_diameter}. Minimum: {min_lattice}"
            )

    @property
    def tube_length(self) -> float:
        """Length of tubes connecting adjacent spheres in mm."""
        return self.lattice_constant - self.sphere_diameter

    @property
    def packing_efficiency(self) -> float:
        """Theoretical packing efficiency for this lattice type."""
        efficiencies = {
            "cubic": 0.52,
            "bcc": 0.68,
            "fcc": 0.74,
        }
        return efficiencies[self.lattice_type]

    @property
    def neighbors_per_sphere(self) -> int:
        """Number of nearest neighbors in this lattice type."""
        neighbors = {
            "cubic": 6,
            "bcc": 8,
            "fcc": 12,
        }
        return neighbors[self.lattice_type]

    def sphere_positions(self) -> list[tuple[float, float, float]]:
        """
        Calculate center positions of all spheres in the lattice.

        Returns:
            List of (x, y, z) positions in mm for each sphere center
        """
        nx, ny, nz = self.extent
        a = self.lattice_constant
        positions = []

        if self.lattice_type == "cubic":
            # Simple cubic: spheres at integer multiples of lattice constant
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        x = ix * a + a / 2  # Center in unit cell
                        y = iy * a + a / 2
                        z = iz * a + a / 2
                        positions.append((x, y, z))

        elif self.lattice_type == "bcc":
            # Body-centered cubic: corner + center positions
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        # Corner position
                        x = ix * a + a / 2
                        y = iy * a + a / 2
                        z = iz * a + a / 2
                        positions.append((x, y, z))

                        # Body center (offset by half lattice constant)
                        # Only add if within bounds
                        if ix < nx - 1 and iy < ny - 1 and iz < nz - 1:
                            x_c = (ix + 1) * a
                            y_c = (iy + 1) * a
                            z_c = (iz + 1) * a
                            positions.append((x_c, y_c, z_c))

        elif self.lattice_type == "fcc":
            # Face-centered cubic: corner + face center positions
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        # Corner position
                        x = ix * a + a / 2
                        y = iy * a + a / 2
                        z = iz * a + a / 2
                        positions.append((x, y, z))

                        # Face centers (3 faces per unit cell)
                        # XY face center
                        if iz < nz - 1:
                            positions.append((x, y, z + a / 2))
                        # XZ face center
                        if iy < ny - 1:
                            positions.append((x, y + a / 2, z))
                        # YZ face center
                        if ix < nx - 1:
                            positions.append((x + a / 2, y, z))

        return positions

    def tube_connections(self) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """
        Calculate tube connections between adjacent spheres.

        Returns:
            List of ((x1, y1, z1), (x2, y2, z2)) pairs for tube endpoints
        """
        positions = self.sphere_positions()
        connections = []

        # Distance threshold for adjacency (slightly more than sphere diameter)
        # For different lattice types:
        # - Cubic: neighbors at distance = a
        # - BCC: neighbors at distance = a*sqrt(3)/2 (body diagonal / 2)
        # - FCC: neighbors at distance = a*sqrt(2)/2 (face diagonal / 2)
        a = self.lattice_constant
        if self.lattice_type == "cubic":
            max_dist = a * 1.1  # Small tolerance
        elif self.lattice_type == "bcc":
            max_dist = a * np.sqrt(3) / 2 * 1.1
        else:  # fcc
            max_dist = a * np.sqrt(2) / 2 * 1.1

        # Find all pairs within max_dist
        seen = set()
        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                if i >= j:
                    continue

                dist = np.sqrt(
                    (p1[0] - p2[0]) ** 2 +
                    (p1[1] - p2[1]) ** 2 +
                    (p1[2] - p2[2]) ** 2
                )

                if dist < max_dist:
                    # Use sorted tuple for deduplication
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        connections.append((p1, p2))

        return connections

    def resonant_frequency(self) -> float:
        """
        Calculate analytical resonant frequency for a single sphere-tube resonator.

        Uses Helmholtz formula: f = (c / 2π) * sqrt(S / (V * L_eff))

        Where:
            c = speed of sound
            S = tube cross-sectional area
            V = sphere volume
            L_eff = effective tube length (with end corrections)

        Note: This gives the frequency for a single sphere with one tube.
        Coupled resonators in a lattice will have more complex behavior.

        Returns:
            Resonant frequency in Hz
        """
        c = SPEED_OF_SOUND
        V = SphereCell(self.sphere_diameter, (0, 0, 0)).volume_m3
        S = np.pi * (self.tube_diameter / 2 / 1000) ** 2  # Convert to m²
        L = self.tube_length / 1000  # Convert to m

        # End correction for circular tube opening
        # 0.85 * radius for each end (unflanged)
        r_tube = self.tube_diameter / 2 / 1000
        L_eff = L + 2 * 0.85 * r_tube  # Both ends

        # Helmholtz formula
        f0 = (c / (2 * np.pi)) * np.sqrt(S / (V * L_eff))

        return float(f0)

    def to_stack(self, resolution: float = 0.5) -> Stack:
        """
        Generate a Stack representing the complete lattice structure.

        Creates spheres at all lattice positions and tubes connecting
        adjacent spheres.

        Args:
            resolution: mm per pixel

        Returns:
            Stack with the complete lattice geometry
        """
        from strata_fdtd.manufacturing.lamination import Slice, Stack

        # Calculate bounding box
        positions = self.sphere_positions()
        if not positions:
            return Stack(resolution_xy=resolution)

        # Find extents
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]

        margin = self.sphere_diameter / 2 + resolution * 2
        min_x = min(xs) - margin
        min_y = min(ys) - margin
        max_x = max(xs) + margin
        max_y = max(ys) + margin
        min_z = min(zs) - self.sphere_diameter / 2
        max_z = max(zs) + self.sphere_diameter / 2

        # Convert z to slice indices
        z_start_slice = int(np.floor(min_z / DEFAULT_SLICE_THICKNESS))
        z_end_slice = int(np.ceil(max_z / DEFAULT_SLICE_THICKNESS))

        # Create empty stack
        w_px = int(np.ceil((max_x - min_x) / resolution))
        h_px = int(np.ceil((max_y - min_y) / resolution))

        slices = []
        for z_idx in range(z_start_slice, z_end_slice + 1):
            mask = np.zeros((h_px, w_px), dtype=np.bool_)
            slices.append(Slice(mask=mask, z_index=z_idx, resolution=resolution))

        result = Stack(
            slices=slices,
            resolution_xy=resolution,
            thickness_z=DEFAULT_SLICE_THICKNESS,
        )

        # Add spheres
        for x, y, z in positions:
            # Convert z from mm to slice index for sphere center
            z_slice = int(round(z / DEFAULT_SLICE_THICKNESS))
            sphere = SphereCell(
                diameter=self.sphere_diameter,
                position=(x - min_x, y - min_y, z_slice - z_start_slice),
            )
            sphere_stack = sphere.to_stack(resolution)

            # Manual merge (more efficient than union for large structures)
            for s_slice in sphere_stack.slices:
                target = result.get_slice(s_slice.z_index + z_start_slice)
                if target is not None:
                    # Offset position in pixels
                    x_off = int((x - min_x - self.sphere_diameter / 2) / resolution)
                    y_off = int((y - min_y - self.sphere_diameter / 2) / resolution)

                    # Clamp to bounds
                    src_h, src_w = s_slice.mask.shape
                    dst_h, dst_w = target.mask.shape

                    x0 = max(0, x_off)
                    y0 = max(0, y_off)
                    x1 = min(dst_w, x_off + src_w)
                    y1 = min(dst_h, y_off + src_h)

                    sx0 = max(0, -x_off)
                    sy0 = max(0, -y_off)
                    sx1 = sx0 + (x1 - x0)
                    sy1 = sy0 + (y1 - y0)

                    if x1 > x0 and y1 > y0:
                        target.mask[y0:y1, x0:x1] |= s_slice.mask[sy0:sy1, sx0:sx1]

        # Add tubes between adjacent spheres
        connections = self.tube_connections()
        for (x1, y1, z1), (x2, y2, z2) in connections:
            self._add_tube(
                result,
                (x1 - min_x, y1 - min_y, z1),
                (x2 - min_x, y2 - min_y, z2),
                z_start_slice,
                resolution,
            )

        return result

    def _add_tube(
        self,
        stack: Stack,
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
        z_offset: int,
        resolution: float,
    ) -> None:
        """
        Add a cylindrical tube between two sphere centers.

        The tube connects from surface to surface of the two spheres,
        not center to center.

        Args:
            stack: Stack to modify in place
            p1: (x, y, z_mm) first endpoint
            p2: (x, y, z_mm) second endpoint
            z_offset: Slice index offset for the stack
            resolution: mm per pixel
        """
        x1, y1, z1_mm = p1
        x2, y2, z2_mm = p2

        # Direction vector
        dx = x2 - x1
        dy = y2 - y1
        dz = z2_mm - z1_mm
        length = np.sqrt(dx**2 + dy**2 + dz**2)

        if length < 1e-6:
            return

        # Unit direction
        ux, uy, uz = dx / length, dy / length, dz / length

        # Tube endpoints (offset from sphere centers by radius)
        r = self.sphere_diameter / 2
        t1_x = x1 + ux * r
        t1_y = y1 + uy * r
        t1_z = z1_mm + uz * r

        t2_x = x2 - ux * r
        t2_y = y2 - uy * r
        t2_z = z2_mm - uz * r

        # Tube length
        tube_len = length - self.sphere_diameter
        if tube_len <= 0:
            return

        # Draw tube as series of circles along its length
        tube_r = self.tube_diameter / 2
        num_steps = max(2, int(tube_len / (resolution * 2)))

        for i in range(num_steps + 1):
            t = i / num_steps
            # Position along tube
            px = t1_x + t * (t2_x - t1_x)
            py = t1_y + t * (t2_y - t1_y)
            pz = t1_z + t * (t2_z - t1_z)

            # Which slice?
            z_slice = int(round(pz / DEFAULT_SLICE_THICKNESS)) - z_offset
            slice_ = stack.get_slice(z_slice + z_offset)
            if slice_ is None:
                continue

            # Draw circle at this position
            h, w = slice_.mask.shape
            yy, xx = np.ogrid[:h, :w]

            cx_px = px / resolution
            cy_px = py / resolution
            r_px = tube_r / resolution

            circle_mask = ((xx - cx_px) ** 2 + (yy - cy_px) ** 2) <= r_px**2
            slice_.mask |= circle_mask

    def check_manufacturable(
        self,
        min_wall: float = DEFAULT_MIN_WALL,
        min_gap: float = DEFAULT_MIN_GAP,
    ) -> list[Violation]:
        """
        Verify lattice dimensions meet manufacturing constraints.

        Args:
            min_wall: Minimum wall thickness in mm
            min_gap: Minimum gap/slot width in mm

        Returns:
            List of violations (empty if all constraints pass)
        """
        from strata_fdtd.manufacturing.lamination import Violation

        violations = []

        # Check sphere diameter (needs ≥2 slices)
        sphere = SphereCell(self.sphere_diameter, (0, 0, 0))
        violations.extend(sphere.check_manufacturable(min_gap))

        # Check tube diameter
        if self.tube_diameter < min_gap:
            violations.append(
                Violation(
                    constraint="min_tube_diameter",
                    slice_index=0,
                    location=(0.0, 0.0),
                    measured=self.tube_diameter,
                    required=min_gap,
                )
            )

        # Check tube length (must be positive)
        if self.tube_length <= 0:
            violations.append(
                Violation(
                    constraint="tube_length_positive",
                    slice_index=0,
                    location=(0.0, 0.0),
                    measured=self.tube_length,
                    required=0.1,
                )
            )

        return violations


def connect_points(
    points: list[tuple[float, float, int]],
    width: float,
    resolution: float = 0.5,
) -> Stack:
    """
    Create a Stack with channels connecting a list of 3D points.

    Points at the same z-level are connected with straight channels.
    Points at different z-levels are connected with vertical channels.

    Args:
        points: List of (x_mm, y_mm, z_slice) points to connect
        width: Channel width in mm
        resolution: mm per pixel

    Returns:
        Stack with connecting channels
    """
    from strata_fdtd.manufacturing.lamination import Stack, union

    if len(points) < 2:
        return Stack(resolution_xy=resolution)

    result = Stack(resolution_xy=resolution)

    for i in range(len(points) - 1):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[i + 1]

        if z1 == z2:
            # Same level - horizontal channel
            channel = Channel(
                width=width,
                height=1,
                path=[(x1, y1), (x2, y2)],
                z_start=z1,
            )
            result = union(result, channel.to_stack(resolution))
        else:
            # Different levels - create vertical connection
            # First horizontal to align, then vertical
            z_min, z_max = min(z1, z2), max(z1, z2)
            for z in range(z_min, z_max + 1):
                # Interpolate x, y position (unused but kept for future offset support)
                t = (z - z_min) / (z_max - z_min) if z_max > z_min else 0
                _x = x1 + t * (x2 - x1)  # noqa: F841
                _y = y1 + t * (y2 - y1)  # noqa: F841

                # Small square at each level
                from strata_fdtd.manufacturing.lamination import Slice

                size = int(np.ceil(width / resolution))
                mask = np.ones((size, size), dtype=np.bool_)
                s = Slice(mask=mask, z_index=z, resolution=resolution)
                temp = Stack(slices=[s], resolution_xy=resolution)
                result = union(result, temp)

    return result

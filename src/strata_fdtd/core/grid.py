"""
Grid specifications for FDTD simulation.

This module provides grid classes for defining computational domains with
uniform or nonuniform cell spacing. Nonuniform grids enable efficient
simulations where high resolution is needed in specific regions while using
coarser resolution elsewhere.

Classes:
    UniformGrid: Traditional uniform cell spacing (same as legacy behavior)
    NonuniformGrid: Variable cell spacing along any axis

Example:
    >>> from strata_fdtd import NonuniformGrid, FDTDSolver
    >>>
    >>> # Option 1: Geometric stretch ratio per axis
    >>> grid = NonuniformGrid.from_stretch(
    ...     shape=(100, 100, 200),
    ...     base_resolution=1e-3,
    ...     stretch_z=1.05,  # 5% geometric stretch in z
    ... )
    >>>
    >>> # Option 2: Explicit coordinate arrays
    >>> import numpy as np
    >>> grid = NonuniformGrid(
    ...     x_coords=np.linspace(0, 0.1, 100),
    ...     y_coords=np.linspace(0, 0.1, 100),
    ...     z_coords=np.geomspace(0.001, 0.2, 200),
    ... )
    >>>
    >>> # Option 3: Piecewise regions with different resolutions
    >>> grid = NonuniformGrid.from_regions(
    ...     x_regions=[(0, 0.05, 1e-3), (0.05, 0.15, 2e-3), (0.15, 0.2, 1e-3)],
    ...     y_regions=[(0, 0.1, 1e-3)],  # uniform
    ...     z_regions=[(0, 0.1, 1e-3)],  # uniform
    ... )
    >>>
    >>> solver = FDTDSolver(grid=grid)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class UniformGrid:
    """Uniform grid specification with equal cell spacing.

    This is equivalent to the legacy FDTDSolver behavior where a single
    resolution value is used for all cells in all directions.

    Args:
        shape: Grid dimensions (nx, ny, nz) in cells
        resolution: Cell spacing in meters (same for all axes)

    Attributes:
        shape: Grid dimensions tuple
        dx, dy, dz: Cell spacing (all equal to resolution)
        x_coords: 1D array of cell center x-coordinates
        y_coords: 1D array of cell center y-coordinates
        z_coords: 1D array of cell center z-coordinates
        min_spacing: Minimum cell spacing (equals resolution)
        is_uniform: Always True for UniformGrid

    Example:
        >>> grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)
        >>> grid.min_spacing
        0.001
    """

    shape: tuple[int, int, int]
    resolution: float

    def __post_init__(self):
        nx, ny, nz = self.shape
        self._dx = np.full(nx, self.resolution, dtype=np.float64)
        self._dy = np.full(ny, self.resolution, dtype=np.float64)
        self._dz = np.full(nz, self.resolution, dtype=np.float64)

        # Cell center coordinates
        self._x_coords = np.arange(nx) * self.resolution + self.resolution / 2
        self._y_coords = np.arange(ny) * self.resolution + self.resolution / 2
        self._z_coords = np.arange(nz) * self.resolution + self.resolution / 2

    @property
    def dx(self) -> NDArray[np.float64]:
        """Cell spacing in x-direction for each cell."""
        return self._dx

    @property
    def dy(self) -> NDArray[np.float64]:
        """Cell spacing in y-direction for each cell."""
        return self._dy

    @property
    def dz(self) -> NDArray[np.float64]:
        """Cell spacing in z-direction for each cell."""
        return self._dz

    @property
    def x_coords(self) -> NDArray[np.float64]:
        """Cell center x-coordinates."""
        return self._x_coords

    @property
    def y_coords(self) -> NDArray[np.float64]:
        """Cell center y-coordinates."""
        return self._y_coords

    @property
    def z_coords(self) -> NDArray[np.float64]:
        """Cell center z-coordinates."""
        return self._z_coords

    @property
    def min_spacing(self) -> float:
        """Minimum cell spacing across all axes."""
        return self.resolution

    @property
    def max_spacing(self) -> float:
        """Maximum cell spacing across all axes."""
        return self.resolution

    @property
    def is_uniform(self) -> bool:
        """Whether the grid has uniform spacing."""
        return True

    def physical_extent(self) -> tuple[float, float, float]:
        """Get physical domain size in meters.

        Returns:
            Tuple (Lx, Ly, Lz) of domain dimensions
        """
        return (
            self.shape[0] * self.resolution,
            self.shape[1] * self.resolution,
            self.shape[2] * self.resolution,
        )


class NonuniformGrid:
    """Nonuniform grid specification with variable cell spacing.

    Enables efficient simulations where high resolution is needed in specific
    regions (near boundaries, material interfaces, sources) while using
    coarser resolution in bulk regions.

    The grid stores 1D coordinate arrays defining cell centers along each axis.
    Cell spacings are derived as differences between consecutive coordinates.

    Args:
        x_coords: 1D array of cell center x-coordinates in meters
        y_coords: 1D array of cell center y-coordinates in meters
        z_coords: 1D array of cell center z-coordinates in meters

    Attributes:
        shape: Grid dimensions (nx, ny, nz)
        dx, dy, dz: Cell spacing arrays for each axis
        x_coords, y_coords, z_coords: Cell center coordinate arrays
        min_spacing: Minimum cell spacing (used for CFL condition)
        is_uniform: False for NonuniformGrid

    Note:
        For proper FDTD stability, the CFL condition uses the minimum cell
        spacing: dt <= min(dx, dy, dz) / (c * sqrt(3))

    Example:
        >>> import numpy as np
        >>> grid = NonuniformGrid(
        ...     x_coords=np.linspace(0, 0.1, 100),
        ...     y_coords=np.linspace(0, 0.1, 100),
        ...     z_coords=np.geomspace(0.001, 0.2, 200),
        ... )
        >>> print(f"Min spacing: {grid.min_spacing:.4f} m")
    """

    def __init__(
        self,
        x_coords: NDArray[np.floating],
        y_coords: NDArray[np.floating],
        z_coords: NDArray[np.floating],
    ):
        # Store coordinates (ensure 1D and sorted)
        self._x_coords = np.asarray(x_coords, dtype=np.float64).ravel()
        self._y_coords = np.asarray(y_coords, dtype=np.float64).ravel()
        self._z_coords = np.asarray(z_coords, dtype=np.float64).ravel()

        if len(self._x_coords) < 2:
            raise ValueError("x_coords must have at least 2 points")
        if len(self._y_coords) < 2:
            raise ValueError("y_coords must have at least 2 points")
        if len(self._z_coords) < 2:
            raise ValueError("z_coords must have at least 2 points")

        # Validate monotonically increasing
        if not np.all(np.diff(self._x_coords) > 0):
            raise ValueError("x_coords must be monotonically increasing")
        if not np.all(np.diff(self._y_coords) > 0):
            raise ValueError("y_coords must be monotonically increasing")
        if not np.all(np.diff(self._z_coords) > 0):
            raise ValueError("z_coords must be monotonically increasing")

        # Compute cell spacings from coordinate differences
        # dx[i] is the spacing for cell i (distance to next cell center)
        # For the last cell, we extrapolate using the previous spacing
        self._dx = self._compute_spacing(self._x_coords)
        self._dy = self._compute_spacing(self._y_coords)
        self._dz = self._compute_spacing(self._z_coords)

        self._shape = (len(self._x_coords), len(self._y_coords), len(self._z_coords))

    @staticmethod
    def _compute_spacing(coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute cell spacings from coordinate array.

        Returns array of same length as coords, where spacing[i] is the
        effective cell size at index i.
        """
        n = len(coords)
        spacing = np.zeros(n, dtype=np.float64)

        # Interior spacings: average of left and right distances
        for i in range(1, n - 1):
            spacing[i] = (coords[i + 1] - coords[i - 1]) / 2

        # Boundary spacings: extrapolate from neighbors
        spacing[0] = coords[1] - coords[0]
        spacing[-1] = coords[-1] - coords[-2]

        return spacing

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid dimensions (nx, ny, nz)."""
        return self._shape

    @property
    def dx(self) -> NDArray[np.float64]:
        """Cell spacing in x-direction for each cell."""
        return self._dx

    @property
    def dy(self) -> NDArray[np.float64]:
        """Cell spacing in y-direction for each cell."""
        return self._dy

    @property
    def dz(self) -> NDArray[np.float64]:
        """Cell spacing in z-direction for each cell."""
        return self._dz

    @property
    def x_coords(self) -> NDArray[np.float64]:
        """Cell center x-coordinates."""
        return self._x_coords

    @property
    def y_coords(self) -> NDArray[np.float64]:
        """Cell center y-coordinates."""
        return self._y_coords

    @property
    def z_coords(self) -> NDArray[np.float64]:
        """Cell center z-coordinates."""
        return self._z_coords

    @property
    def min_spacing(self) -> float:
        """Minimum cell spacing across all axes.

        This value is used for the CFL stability condition.
        """
        return float(min(
            np.min(self._dx),
            np.min(self._dy),
            np.min(self._dz),
        ))

    @property
    def max_spacing(self) -> float:
        """Maximum cell spacing across all axes."""
        return float(max(
            np.max(self._dx),
            np.max(self._dy),
            np.max(self._dz),
        ))

    @property
    def is_uniform(self) -> bool:
        """Whether the grid has uniform spacing."""
        return False

    @property
    def stretch_ratio(self) -> tuple[float, float, float]:
        """Approximate stretch ratio for each axis.

        Returns the ratio of max/min spacing for each axis.
        """
        return (
            float(np.max(self._dx) / np.min(self._dx)),
            float(np.max(self._dy) / np.min(self._dy)),
            float(np.max(self._dz) / np.min(self._dz)),
        )

    def physical_extent(self) -> tuple[float, float, float]:
        """Get physical domain size in meters.

        Returns:
            Tuple (Lx, Ly, Lz) of domain dimensions
        """
        return (
            float(self._x_coords[-1] - self._x_coords[0] + self._dx[-1] / 2 + self._dx[0] / 2),
            float(self._y_coords[-1] - self._y_coords[0] + self._dy[-1] / 2 + self._dy[0] / 2),
            float(self._z_coords[-1] - self._z_coords[0] + self._dz[-1] / 2 + self._dz[0] / 2),
        )

    @classmethod
    def from_stretch(
        cls,
        shape: tuple[int, int, int],
        base_resolution: float,
        stretch_x: float = 1.0,
        stretch_y: float = 1.0,
        stretch_z: float = 1.0,
        center_fine: bool = True,
    ) -> NonuniformGrid:
        """Create grid with geometric stretch ratio per axis.

        Cells grow geometrically from the center (if center_fine=True) or
        from one end of the domain.

        Args:
            shape: Grid dimensions (nx, ny, nz)
            base_resolution: Finest cell spacing in meters
            stretch_x: Geometric stretch ratio in x (1.0 = uniform)
            stretch_y: Geometric stretch ratio in y (1.0 = uniform)
            stretch_z: Geometric stretch ratio in z (1.0 = uniform)
            center_fine: If True, finest cells are at center; if False,
                         finest cells are at x=0, y=0, z=0 boundaries

        Returns:
            NonuniformGrid with specified stretch pattern

        Example:
            >>> # Fine resolution at center, coarser at edges
            >>> grid = NonuniformGrid.from_stretch(
            ...     shape=(100, 100, 200),
            ...     base_resolution=1e-3,
            ...     stretch_z=1.05,  # 5% growth per cell in z
            ... )
        """
        nx, ny, nz = shape

        x_coords = cls._generate_stretched_coords(nx, base_resolution, stretch_x, center_fine)
        y_coords = cls._generate_stretched_coords(ny, base_resolution, stretch_y, center_fine)
        z_coords = cls._generate_stretched_coords(nz, base_resolution, stretch_z, center_fine)

        return cls(x_coords=x_coords, y_coords=y_coords, z_coords=z_coords)

    @staticmethod
    def _generate_stretched_coords(
        n: int,
        base_dx: float,
        stretch: float,
        center_fine: bool,
    ) -> NDArray[np.float64]:
        """Generate stretched coordinates along one axis."""
        if stretch == 1.0:
            # Uniform spacing
            return np.arange(n, dtype=np.float64) * base_dx + base_dx / 2

        if center_fine:
            # Fine resolution at center, coarser at edges
            # Generate half the grid with increasing spacing, then mirror
            n_half = n // 2

            # Cell sizes: base_dx, base_dx * stretch, base_dx * stretch^2, ...
            sizes = base_dx * (stretch ** np.arange(n_half, dtype=np.float64))

            # Cumulative positions from center outward
            positions = np.cumsum(sizes) - sizes[0] / 2

            if n % 2 == 0:
                # Even: symmetric around center
                left = -positions[::-1] - sizes[::-1] / 2
                right = positions + sizes / 2
                coords = np.concatenate([left, right])
            else:
                # Odd: center cell at origin
                left = -positions[::-1] - sizes[::-1] / 2
                center = np.array([0.0])
                right = positions[1:] + sizes[1:] / 2 if n_half > 0 else np.array([])
                coords = np.concatenate([left, center, right])

            # Shift to positive coordinates
            coords = coords - coords[0] + base_dx / 2

        else:
            # Fine resolution at start, coarser toward end
            sizes = base_dx * (stretch ** np.arange(n, dtype=np.float64))
            coords = np.cumsum(sizes) - sizes / 2

        return coords

    @classmethod
    def from_regions(
        cls,
        x_regions: Sequence[tuple[float, float, float]],
        y_regions: Sequence[tuple[float, float, float]] | None = None,
        z_regions: Sequence[tuple[float, float, float]] | None = None,
    ) -> NonuniformGrid:
        """Create grid from piecewise regions with different resolutions.

        Each region is specified as (start, end, resolution) tuples.
        Regions must be contiguous (end of one equals start of next).

        Args:
            x_regions: List of (start, end, resolution) for x-axis
            y_regions: List of (start, end, resolution) for y-axis.
                       If None, uses same as x_regions.
            z_regions: List of (start, end, resolution) for z-axis.
                       If None, uses same as x_regions.

        Returns:
            NonuniformGrid with piecewise-constant resolution

        Example:
            >>> # Fine resolution in middle, coarse at ends
            >>> grid = NonuniformGrid.from_regions(
            ...     x_regions=[
            ...         (0, 0.05, 2e-3),     # Coarse: 2mm
            ...         (0.05, 0.15, 1e-3),  # Fine: 1mm
            ...         (0.15, 0.2, 2e-3),   # Coarse: 2mm
            ...     ],
            ...     y_regions=[(0, 0.1, 1e-3)],  # Uniform 1mm
            ...     z_regions=[(0, 0.1, 1e-3)],  # Uniform 1mm
            ... )
        """
        if y_regions is None:
            y_regions = x_regions
        if z_regions is None:
            z_regions = x_regions

        x_coords = cls._generate_region_coords(x_regions)
        y_coords = cls._generate_region_coords(y_regions)
        z_coords = cls._generate_region_coords(z_regions)

        return cls(x_coords=x_coords, y_coords=y_coords, z_coords=z_coords)

    @staticmethod
    def _generate_region_coords(
        regions: Sequence[tuple[float, float, float]],
    ) -> NDArray[np.float64]:
        """Generate coordinates from piecewise regions."""
        if not regions:
            raise ValueError("At least one region must be specified")

        coords_list = []
        for i, (start, end, resolution) in enumerate(regions):
            if end <= start:
                raise ValueError(f"Region {i}: end ({end}) must be > start ({start})")
            if resolution <= 0:
                raise ValueError(f"Region {i}: resolution must be positive")

            # Check contiguity
            if i > 0:
                prev_end = regions[i - 1][1]
                if abs(start - prev_end) > 1e-10:
                    raise ValueError(
                        f"Regions must be contiguous: region {i-1} ends at {prev_end}, "
                        f"region {i} starts at {start}"
                    )

            # Generate cell centers for this region
            n_cells = max(1, int(round((end - start) / resolution)))
            actual_resolution = (end - start) / n_cells

            # Cell centers
            region_coords = start + (np.arange(n_cells) + 0.5) * actual_resolution

            # Avoid duplicating boundary points
            if coords_list and len(region_coords) > 0:
                # Skip if too close to previous point
                if abs(region_coords[0] - coords_list[-1]) < actual_resolution / 2:
                    region_coords = region_coords[1:]

            coords_list.extend(region_coords.tolist())

        return np.array(coords_list, dtype=np.float64)

    def get_spacing_arrays_for_stencil(self) -> dict[str, NDArray[np.float32]]:
        """Get spacing arrays formatted for FDTD stencil operations.

        Returns arrays needed for the nonuniform FDTD update equations:
        - inv_dx_x[i]: 1/dx for velocity x-update at face i
        - inv_dx_p[i]: 1/dx for pressure update at cell i
        - etc.

        Returns:
            Dict with keys: inv_dx_x, inv_dx_p, inv_dy_y, inv_dy_p,
                           inv_dz_z, inv_dz_p
        """
        nx, ny, nz = self.shape

        # For velocity updates: vx[i] += coeff * (p[i+1] - p[i]) / dx_face[i]
        # dx_face[i] is distance between cell centers i and i+1
        dx_face = np.diff(self._x_coords)  # shape (nx-1,)
        dy_face = np.diff(self._y_coords)  # shape (ny-1,)
        dz_face = np.diff(self._z_coords)  # shape (nz-1,)

        # For pressure updates: p[i] += coeff * (vx[i] - vx[i-1]) / dx_cell[i]
        # dx_cell[i] is the cell size at index i
        # We use the stored spacing arrays
        dx_cell = self._dx  # shape (nx,)
        dy_cell = self._dy  # shape (ny,)
        dz_cell = self._dz  # shape (nz,)

        return {
            "inv_dx_face": (1.0 / dx_face).astype(np.float32),
            "inv_dx_cell": (1.0 / dx_cell).astype(np.float32),
            "inv_dy_face": (1.0 / dy_face).astype(np.float32),
            "inv_dy_cell": (1.0 / dy_cell).astype(np.float32),
            "inv_dz_face": (1.0 / dz_face).astype(np.float32),
            "inv_dz_cell": (1.0 / dz_cell).astype(np.float32),
        }

    def __repr__(self) -> str:
        return (
            f"NonuniformGrid(shape={self.shape}, "
            f"min_spacing={self.min_spacing:.4g}, "
            f"max_spacing={self.max_spacing:.4g}, "
            f"stretch_ratio={tuple(f'{r:.2f}' for r in self.stretch_ratio)})"
        )

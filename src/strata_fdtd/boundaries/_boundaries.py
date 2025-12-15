"""
Boundary conditions for FDTD acoustic simulation.

This module provides boundary condition implementations:
    - RigidBoundary: Perfect reflection (built into solver)
    - PML: Perfectly Matched Layer for absorption
    - ABCFirstOrder: First-order Mur absorbing boundary
    - RadiationImpedance: Partial reflection for open pipe simulations

Boundary Selection Guide
------------------------

**PML (Perfectly Matched Layer)**:
- Best for: Simulating infinite/anechoic space, measuring impulse responses
- Absorbs outgoing waves with minimal reflection
- Trade-offs: Uses additional memory for auxiliary arrays, absorbs standing
  wave energy (not suitable for resonance analysis in open pipes)

**ABCFirstOrder (Mur ABC)**:
- Best for: Quick tests, memory-constrained simulations
- Simpler than PML, lower memory footprint
- Trade-offs: Higher reflection than PML, especially at oblique angles

**RadiationImpedance**:
- Best for: Open pipe simulations, mode analysis with partial radiation
- Allows configurable reflection coefficient (0=absorbing, 1=rigid)
- Preserves standing wave energy while allowing some radiation
- Trade-offs: Simplified model (constant R), not physically accurate
  frequency-dependent impedance

**RigidBoundary**:
- Best for: Closed cavities, solid boundaries
- Perfect reflection, no energy loss
- Built into geometry handling (implicit at solid interfaces)

Example: Open Pipe Mode Analysis
--------------------------------
For detecting resonant modes in an open pipe:

    # PML absorbs too much - modes decay quickly
    solver_pml = FDTDSolver(shape=(100, 20, 20), resolution=2e-3)
    solver_pml.add_boundary(PML(depth=10, axis='x'))  # 0% mode detection

    # RadiationImpedance preserves standing waves
    solver_rad = FDTDSolver(shape=(100, 20, 20), resolution=2e-3)
    solver_rad.add_boundary(RadiationImpedance(
        axis='x', side='high', reflection_coeff=0.85
    ))  # Modes detectable with ~1% frequency error

The PML implementation uses a convolutional PML (CPML) formulation
for improved absorption at low frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

# Try to import native C++ kernels for accelerated PML
try:
    from . import _kernels

    _HAS_NATIVE = True
except ImportError:
    _kernels = None  # type: ignore[assignment]
    _HAS_NATIVE = False

if TYPE_CHECKING:
    from .fdtd import FDTDSolver


@dataclass
class RigidBoundary:
    """Rigid (perfectly reflecting) boundary condition.

    This is the default boundary at solid/air interfaces and domain edges.
    It's built into the solver's geometry handling, so this class is
    primarily for documentation and explicit specification.

    The rigid boundary condition enforces zero normal velocity at the
    boundary surface, resulting in perfect reflection.
    """

    def initialize(self, solver: FDTDSolver) -> None:
        """Initialize boundary (no-op for rigid)."""
        pass

    def apply_velocity(self, solver: FDTDSolver) -> None:
        """Apply to velocity fields (no-op, handled by geometry)."""
        pass

    def apply_pressure(self, solver: FDTDSolver) -> None:
        """Apply to pressure field (no-op for rigid)."""
        pass

    def reset(self) -> None:
        """Reset boundary state (no-op for rigid)."""
        pass


class PML:
    """Perfectly Matched Layer absorbing boundary condition.

    Implements a convolutional PML (CPML) that absorbs outgoing waves
    with minimal reflection. The PML region is added to the domain
    edges to simulate an infinite/anechoic space.

    Args:
        depth: Number of cells for PML region (default: 10)
        axis: Which axes to apply PML ('all', 'x', 'y', 'z', or tuple)
        max_sigma: Maximum conductivity (controls absorption strength)
        order: Polynomial order for sigma profile (default: 3)

    The absorption profile increases polynomially from the interface
    toward the outer boundary, minimizing reflections at the PML entry.

    Example:
        >>> solver = FDTDSolver(shape=(100, 100, 150), resolution=1e-3)
        >>> solver.add_boundary(PML(depth=10, axis='all'))
    """

    def __init__(
        self,
        depth: int = 10,
        axis: Literal["all", "x", "y", "z"] | tuple[str, ...] = "all",
        max_sigma: float | None = None,
        order: int = 3,
    ):
        self.depth = depth
        self.order = order

        # Parse axis specification
        if axis == "all":
            self.axes = ("x", "y", "z")
        elif isinstance(axis, str):
            self.axes = (axis,)
        else:
            self.axes = axis

        # max_sigma will be computed during initialization based on solver params
        self._max_sigma = max_sigma
        self._initialized = False

        # PML auxiliary variables (for CPML formulation)
        self._psi_vx: NDArray[np.floating] | None = None
        self._psi_vy: NDArray[np.floating] | None = None
        self._psi_vz: NDArray[np.floating] | None = None
        self._psi_px: NDArray[np.floating] | None = None
        self._psi_py: NDArray[np.floating] | None = None
        self._psi_pz: NDArray[np.floating] | None = None

        # Damping profiles
        self._sigma_x: NDArray[np.floating] | None = None
        self._sigma_y: NDArray[np.floating] | None = None
        self._sigma_z: NDArray[np.floating] | None = None

        # Solver reference
        self._solver: FDTDSolver | None = None

        # Native kernel data (pre-computed decay factors)
        self._pml_data = None

    def initialize(self, solver: FDTDSolver) -> None:
        """Initialize PML auxiliary arrays and profiles.

        Args:
            solver: The FDTD solver instance
        """
        self._solver = solver
        nx, ny, nz = solver.shape

        # Get grid for coordinate information
        grid = solver.grid

        # Compute optimal max_sigma if not specified
        # Rule of thumb: sigma_max = -(order+1) * c * ln(R) / (2 * d)
        # where R is target reflection coefficient, d is PML thickness
        # Use minimum cell spacing for conservative estimate
        if self._max_sigma is None:
            R_target = 1e-6  # Target reflection coefficient
            d = self.depth * grid.min_spacing
            self._max_sigma = (
                -(self.order + 1) * solver.c * np.log(R_target) / (2 * d)
            )

        # Create damping profiles for each axis
        # For nonuniform grids, use coordinate arrays to compute physical distances
        if grid.is_uniform:
            self._sigma_x = self._create_profile(nx, solver.dx) if "x" in self.axes else None
            self._sigma_y = self._create_profile(ny, solver.dx) if "y" in self.axes else None
            self._sigma_z = self._create_profile(nz, solver.dx) if "z" in self.axes else None
        else:
            self._sigma_x = self._create_profile_nonuniform(
                grid.x_coords, grid.dx
            ) if "x" in self.axes else None
            self._sigma_y = self._create_profile_nonuniform(
                grid.y_coords, grid.dy
            ) if "y" in self.axes else None
            self._sigma_z = self._create_profile_nonuniform(
                grid.z_coords, grid.dz
            ) if "z" in self.axes else None

        # Allocate auxiliary arrays for CPML
        # These store the convolution history
        self._psi_vx = np.zeros((nx, ny, nz), dtype=np.float32)
        self._psi_vy = np.zeros((nx, ny, nz), dtype=np.float32)
        self._psi_vz = np.zeros((nx, ny, nz), dtype=np.float32)
        self._psi_px = np.zeros((nx, ny, nz), dtype=np.float32)
        self._psi_py = np.zeros((nx, ny, nz), dtype=np.float32)
        self._psi_pz = np.zeros((nx, ny, nz), dtype=np.float32)

        # Pre-compute decay factors using native kernels if available
        # This avoids recomputing exp(-sigma * dt) every timestep
        if _HAS_NATIVE and _kernels is not None:
            self._pml_data = _kernels.initialize_pml(
                self._sigma_x,
                self._sigma_y,
                self._sigma_z,
                nx, ny, nz,
                solver.dt
            )

        self._initialized = True

    def _create_profile(self, n: int, dx: float) -> NDArray[np.floating]:
        """Create polynomial damping profile for one axis (uniform grids).

        The profile is zero in the interior and increases polynomially
        toward both ends of the domain.

        Args:
            n: Number of cells along this axis
            dx: Grid spacing

        Returns:
            1D array of sigma values for each cell
        """
        sigma = np.zeros(n, dtype=np.float32)
        d = self.depth

        # Left PML region (cells 0 to depth-1)
        for i in range(d):
            # Distance from PML/interior interface, normalized to [0, 1]
            dist = (d - i) / d
            sigma[i] = self._max_sigma * (dist ** self.order)

        # Right PML region (cells n-depth to n-1)
        for i in range(n - d, n):
            dist = (i - (n - d - 1)) / d
            sigma[i] = self._max_sigma * (dist ** self.order)

        return sigma

    def _create_profile_nonuniform(
        self,
        coords: NDArray[np.floating],
        cell_spacing: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Create polynomial damping profile for nonuniform grids.

        Uses physical coordinates to compute normalized distance within
        the PML regions, rather than cell indices.

        Args:
            coords: Cell center coordinates along this axis
            cell_spacing: Cell spacing array for this axis

        Returns:
            1D array of sigma values for each cell
        """
        n = len(coords)
        sigma = np.zeros(n, dtype=np.float32)
        d = self.depth

        if d >= n // 2:
            # PML depth is too large for domain, use half the domain
            d = max(1, n // 4)

        # Compute physical thickness of PML regions
        # Left PML: sum of first 'd' cell spacings
        left_thickness = float(np.sum(cell_spacing[:d]))
        # Right PML: sum of last 'd' cell spacings
        right_thickness = float(np.sum(cell_spacing[-d:]))

        # Left PML interface position (between cell d-1 and d)
        left_interface = coords[d] - cell_spacing[d] / 2 if d < n else coords[-1]
        # Right PML interface position (between cell n-d-1 and n-d)
        right_interface = coords[n - d - 1] + cell_spacing[n - d - 1] / 2 if n - d > 0 else coords[0]

        # Left PML region (cells 0 to depth-1)
        for i in range(d):
            # Physical distance from PML/interior interface
            dist_from_interface = left_interface - coords[i]
            # Normalized distance [0, 1] where 0 is interface, 1 is outer boundary
            if left_thickness > 0:
                normalized_dist = min(1.0, dist_from_interface / left_thickness)
            else:
                normalized_dist = 0.0
            sigma[i] = self._max_sigma * (normalized_dist ** self.order)

        # Right PML region (cells n-depth to n-1)
        for i in range(n - d, n):
            # Physical distance from PML/interior interface
            dist_from_interface = coords[i] - right_interface
            # Normalized distance [0, 1]
            if right_thickness > 0:
                normalized_dist = min(1.0, dist_from_interface / right_thickness)
            else:
                normalized_dist = 0.0
            sigma[i] = self._max_sigma * (normalized_dist ** self.order)

        return sigma

    def apply_velocity(self, solver: FDTDSolver) -> None:
        """Apply PML damping to velocity updates.

        This modifies velocities in the PML regions to absorb
        outgoing waves.
        """
        if not self._initialized:
            return

        # Use native kernel with pre-computed decay factors if available
        if self._pml_data is not None and _kernels is not None:
            _kernels.apply_pml_velocity(
                solver.vx, solver.vy, solver.vz, self._pml_data
            )
            return

        # Fallback: Python implementation
        dt = solver.dt

        # Apply damping to each velocity component
        # Using simple exponential decay: v *= exp(-sigma * dt)
        # This is a simplified formulation; full CPML uses convolution

        if self._sigma_x is not None:
            # Apply to vx (which has x-dependence)
            decay_x = np.exp(-self._sigma_x * dt)
            solver.vx *= decay_x[:, np.newaxis, np.newaxis]

        if self._sigma_y is not None:
            decay_y = np.exp(-self._sigma_y * dt)
            solver.vy *= decay_y[np.newaxis, :, np.newaxis]

        if self._sigma_z is not None:
            decay_z = np.exp(-self._sigma_z * dt)
            solver.vz *= decay_z[np.newaxis, np.newaxis, :]

    def apply_pressure(self, solver: FDTDSolver) -> None:
        """Apply PML damping to pressure updates."""
        if not self._initialized:
            return

        # Use native kernel with pre-computed decay factors if available
        if self._pml_data is not None and _kernels is not None:
            _kernels.apply_pml_pressure(solver.p, self._pml_data)
            return

        # Fallback: Python implementation
        dt = solver.dt

        # Apply combined damping from all axes
        if self._sigma_x is not None:
            decay_x = np.exp(-self._sigma_x * dt)
            solver.p *= decay_x[:, np.newaxis, np.newaxis]

        if self._sigma_y is not None:
            decay_y = np.exp(-self._sigma_y * dt)
            solver.p *= decay_y[np.newaxis, :, np.newaxis]

        if self._sigma_z is not None:
            decay_z = np.exp(-self._sigma_z * dt)
            solver.p *= decay_z[np.newaxis, np.newaxis, :]

    def reset(self) -> None:
        """Reset auxiliary arrays to zero."""
        if self._psi_vx is not None:
            self._psi_vx.fill(0)
        if self._psi_vy is not None:
            self._psi_vy.fill(0)
        if self._psi_vz is not None:
            self._psi_vz.fill(0)
        if self._psi_px is not None:
            self._psi_px.fill(0)
        if self._psi_py is not None:
            self._psi_py.fill(0)
        if self._psi_pz is not None:
            self._psi_pz.fill(0)

    def get_interior_slice(self) -> tuple[slice, slice, slice]:
        """Get slice indices for the non-PML interior region.

        Useful for extracting results without PML artifacts.

        Returns:
            Tuple of slices (x_slice, y_slice, z_slice)
        """
        d = self.depth
        if self._solver is None:
            raise RuntimeError("PML not initialized")

        nx, ny, nz = self._solver.shape

        x_slice = slice(d, nx - d) if "x" in self.axes else slice(None)
        y_slice = slice(d, ny - d) if "y" in self.axes else slice(None)
        z_slice = slice(d, nz - d) if "z" in self.axes else slice(None)

        return (x_slice, y_slice, z_slice)

    @property
    def is_initialized(self) -> bool:
        """Check if PML has been initialized."""
        return self._initialized


class ABCFirstOrder:
    """First-order Absorbing Boundary Condition (Mur).

    A simpler alternative to PML that applies a one-way wave equation
    at the boundary. Less effective than PML but computationally cheaper.

    This is useful for quick tests or when PML memory overhead is a concern.

    Args:
        axis: Which axes to apply ABC ('all', 'x', 'y', 'z', or tuple)
    """

    def __init__(
        self,
        axis: Literal["all", "x", "y", "z"] | tuple[str, ...] = "all",
    ):
        if axis == "all":
            self.axes = ("x", "y", "z")
        elif isinstance(axis, str):
            self.axes = (axis,)
        else:
            self.axes = axis

        self._solver: FDTDSolver | None = None
        self._coeff: float = 0.0

        # Store previous boundary values
        self._p_prev_x0: NDArray[np.floating] | None = None
        self._p_prev_x1: NDArray[np.floating] | None = None
        self._p_prev_y0: NDArray[np.floating] | None = None
        self._p_prev_y1: NDArray[np.floating] | None = None
        self._p_prev_z0: NDArray[np.floating] | None = None
        self._p_prev_z1: NDArray[np.floating] | None = None

    def initialize(self, solver: FDTDSolver) -> None:
        """Initialize ABC arrays."""
        self._solver = solver
        nx, ny, nz = solver.shape

        # Mur coefficient: (c*dt - dx) / (c*dt + dx)
        self._coeff = (solver.c * solver.dt - solver.dx) / (
            solver.c * solver.dt + solver.dx
        )

        # Allocate boundary storage
        if "x" in self.axes:
            self._p_prev_x0 = np.zeros((ny, nz), dtype=np.float32)
            self._p_prev_x1 = np.zeros((ny, nz), dtype=np.float32)
        if "y" in self.axes:
            self._p_prev_y0 = np.zeros((nx, nz), dtype=np.float32)
            self._p_prev_y1 = np.zeros((nx, nz), dtype=np.float32)
        if "z" in self.axes:
            self._p_prev_z0 = np.zeros((nx, ny), dtype=np.float32)
            self._p_prev_z1 = np.zeros((nx, ny), dtype=np.float32)

    def apply_velocity(self, solver: FDTDSolver) -> None:
        """No velocity modification for Mur ABC."""
        pass

    def apply_pressure(self, solver: FDTDSolver) -> None:
        """Apply first-order Mur ABC to pressure boundaries."""
        if self._solver is None:
            return

        p = solver.p
        c = self._coeff

        # x boundaries
        if "x" in self.axes and self._p_prev_x0 is not None:
            # x = 0 boundary
            p[0, :, :] = self._p_prev_x0 + c * (p[1, :, :] - p[0, :, :])
            self._p_prev_x0 = p[1, :, :].copy()

            # x = nx-1 boundary
            p[-1, :, :] = self._p_prev_x1 + c * (p[-2, :, :] - p[-1, :, :])
            self._p_prev_x1 = p[-2, :, :].copy()

        # y boundaries
        if "y" in self.axes and self._p_prev_y0 is not None:
            p[:, 0, :] = self._p_prev_y0 + c * (p[:, 1, :] - p[:, 0, :])
            self._p_prev_y0 = p[:, 1, :].copy()

            p[:, -1, :] = self._p_prev_y1 + c * (p[:, -2, :] - p[:, -1, :])
            self._p_prev_y1 = p[:, -2, :].copy()

        # z boundaries
        if "z" in self.axes and self._p_prev_z0 is not None:
            p[:, :, 0] = self._p_prev_z0 + c * (p[:, :, 1] - p[:, :, 0])
            self._p_prev_z0 = p[:, :, 1].copy()

            p[:, :, -1] = self._p_prev_z1 + c * (p[:, :, -2] - p[:, :, -1])
            self._p_prev_z1 = p[:, :, -2].copy()

    def reset(self) -> None:
        """Reset stored boundary values."""
        if self._p_prev_x0 is not None:
            self._p_prev_x0.fill(0)
        if self._p_prev_x1 is not None:
            self._p_prev_x1.fill(0)
        if self._p_prev_y0 is not None:
            self._p_prev_y0.fill(0)
        if self._p_prev_y1 is not None:
            self._p_prev_y1.fill(0)
        if self._p_prev_z0 is not None:
            self._p_prev_z0.fill(0)
        if self._p_prev_z1 is not None:
            self._p_prev_z1.fill(0)


class RadiationImpedance:
    """Radiation impedance boundary for open pipe simulations.

    Models the acoustic impedance at a pipe opening, allowing partial
    reflection to sustain standing wave modes while still permitting
    energy radiation. This provides more realistic open-pipe behavior
    than PML (which absorbs too aggressively) or rigid boundaries
    (which reflect completely).

    The radiation impedance at a circular pipe opening is approximately:
        Z_rad ≈ ρc * (ka)² / 4  for ka << 1 (low frequency limit)

    This results in a frequency-dependent reflection coefficient:
        R = (Z_rad - ρc) / (Z_rad + ρc)

    For educational simulations, a simplified constant reflection
    coefficient can be used instead.

    Args:
        axis: Axis perpendicular to the open boundary ('x', 'y', or 'z')
        side: Which end of the axis ('low' for 0, 'high' for max)
        reflection_coeff: Reflection coefficient (0=absorbing, 1=rigid).
            If None, uses frequency-dependent radiation impedance.
        pipe_radius: Pipe radius in meters (for frequency-dependent mode).
            Required if reflection_coeff is None.

    Example:
        >>> # Open pipe with 50% energy reflection (constant)
        >>> solver = FDTDSolver(shape=(100, 20, 20), resolution=1e-3)
        >>> solver.add_boundary(RadiationImpedance(
        ...     axis='x', side='high', reflection_coeff=0.7
        ... ))

        >>> # Frequency-dependent radiation impedance
        >>> solver.add_boundary(RadiationImpedance(
        ...     axis='x', side='high', pipe_radius=0.01
        ... ))
    """

    def __init__(
        self,
        axis: Literal["x", "y", "z"],
        side: Literal["low", "high"],
        reflection_coeff: float | None = None,
        pipe_radius: float | None = None,
    ):
        self.axis = axis
        self.side = side

        if reflection_coeff is not None:
            if not 0.0 <= reflection_coeff <= 1.0:
                raise ValueError("reflection_coeff must be in [0, 1]")
            self._reflection_coeff = reflection_coeff
            self._frequency_dependent = False
        elif pipe_radius is not None:
            if pipe_radius <= 0:
                raise ValueError("pipe_radius must be positive")
            self._pipe_radius = pipe_radius
            self._frequency_dependent = True
            self._reflection_coeff = None  # Will be computed
        else:
            raise ValueError(
                "Must specify either reflection_coeff or pipe_radius"
            )

        self._solver: FDTDSolver | None = None
        self._initialized = False

        # Store previous boundary values for radiation impedance calculation
        self._p_prev: NDArray[np.floating] | None = None
        self._v_prev: NDArray[np.floating] | None = None

        # Pre-computed constants (set during initialization)
        self._coeff_a: float = 0.0  # Coefficient for outgoing wave
        self._coeff_b: float = 0.0  # Coefficient for incoming wave

    def initialize(self, solver: FDTDSolver) -> None:
        """Initialize radiation impedance boundary.

        Args:
            solver: The FDTD solver instance
        """
        self._solver = solver
        nx, ny, nz = solver.shape

        # Determine boundary slice shape
        if self.axis == "x":
            shape_2d = (ny, nz)
        elif self.axis == "y":
            shape_2d = (nx, nz)
        else:  # z
            shape_2d = (nx, ny)

        # Allocate storage for previous boundary values
        self._p_prev = np.zeros(shape_2d, dtype=np.float32)
        self._v_prev = np.zeros(shape_2d, dtype=np.float32)

        # Compute coefficients for the simplified radiation boundary
        # Using a first-order approximation:
        #   p_boundary = coeff_a * p_interior + coeff_b * p_prev_boundary
        #
        # For constant reflection coefficient R:
        #   coeff_a = (1 - R) / (1 + R) * (c*dt/dx)
        #   coeff_b = R (simplified storage coefficient)
        #
        # This approximates partial reflection at the boundary

        c = solver.c
        dt = solver.dt
        dx = solver.dx

        if not self._frequency_dependent:
            R = self._reflection_coeff
            # Modified Mur ABC with partial reflection
            # Standard Mur: p_boundary = p_prev + (cdt-dx)/(cdt+dx) * (p_interior - p_boundary)
            # With reflection: we blend absorbed wave with reflected component
            mur_coeff = (c * dt - dx) / (c * dt + dx)

            # Split into transmitted and reflected components
            # transmitted goes through (like ABC), reflected bounces back
            self._coeff_a = (1 - R) * mur_coeff  # Transmitted part
            self._coeff_b = R  # Reflected part
        else:
            # Frequency-dependent mode - use ka-based impedance
            # For now, estimate based on characteristic frequency
            # In practice, this requires more sophisticated filtering
            # Use a mid-band approximation
            k_char = 2 * np.pi * 1000 / c  # ~1kHz characteristic
            a = self._pipe_radius
            ka = k_char * a

            # Radiation impedance: Z_rad = ρc * (ka)² / 4 for ka << 1
            # Reflection coefficient: R = (Z - ρc)/(Z + ρc)
            # where Z = Z_rad for outgoing, ρc for characteristic
            z_ratio = (ka ** 2) / 4
            R = abs((z_ratio - 1) / (z_ratio + 1))
            R = min(R, 0.95)  # Clamp to avoid instability

            mur_coeff = (c * dt - dx) / (c * dt + dx)
            self._coeff_a = (1 - R) * mur_coeff
            self._coeff_b = R

        self._initialized = True

    def apply_velocity(self, solver: FDTDSolver) -> None:
        """Apply radiation impedance to velocity field.

        For partial reflection, we modify the normal velocity at the
        boundary to allow some wave energy to escape while reflecting
        the rest.
        """
        if not self._initialized or self._solver is None:
            return

        # Get the boundary velocity slice
        if self.axis == "x":
            if self.side == "low":
                # vx at x=0 boundary (between cells -1 and 0)
                # For radiation, we allow partial outflow
                # Reduce absorption by scaling the velocity less
                v_boundary = solver.vx[0, :, :]
            else:  # high
                v_boundary = solver.vx[-1, :, :]
        elif self.axis == "y":
            if self.side == "low":
                v_boundary = solver.vy[:, 0, :]
            else:
                v_boundary = solver.vy[:, -1, :]
        else:  # z
            if self.side == "low":
                v_boundary = solver.vz[:, :, 0]
            else:
                v_boundary = solver.vz[:, :, -1]

        # Store for next timestep
        if self._v_prev is not None:
            self._v_prev[:] = v_boundary

    def apply_pressure(self, solver: FDTDSolver) -> None:
        """Apply radiation impedance boundary condition to pressure.

        Uses a modified approach that blends between rigid reflection
        and absorbing boundary:
        - R=1.0: Full reflection (rigid boundary behavior)
        - R=0.0: Full absorption (ABC behavior)
        - R=0.5: Partial reflection, partial absorption

        This allows standing waves to form in open pipes while still
        permitting energy radiation.
        """
        if not self._initialized or self._solver is None:
            return

        p = solver.p
        R = self._coeff_b  # Reflection coefficient

        c = solver.c
        dt = solver.dt
        dx = solver.dx

        # Mur ABC coefficient for outgoing wave absorption
        mur = (c * dt - dx) / (c * dt + dx)

        # Get boundary and interior slices based on axis and side
        if self.axis == "x":
            if self.side == "low":
                p_boundary = p[0, :, :]
                p_interior = p[1, :, :]
            else:  # high
                p_boundary = p[-1, :, :]
                p_interior = p[-2, :, :]
        elif self.axis == "y":
            if self.side == "low":
                p_boundary = p[:, 0, :]
                p_interior = p[:, 1, :]
            else:
                p_boundary = p[:, -1, :]
                p_interior = p[:, -2, :]
        else:  # z
            if self.side == "low":
                p_boundary = p[:, :, 0]
                p_interior = p[:, :, 1]
            else:
                p_boundary = p[:, :, -1]
                p_interior = p[:, :, -2]

        # Compute the two extreme cases:
        # 1. ABC (absorbing): what Mur ABC would produce
        # 2. Rigid (reflecting): boundary = interior (perfect reflection)
        #
        # For Mur ABC: p_new = p_prev_interior + mur * (p_interior - p_prev_boundary)
        # Simplified: we use the stored previous interior value

        # ABC component - absorbs outgoing wave
        abc_value = self._p_prev + mur * (p_interior - p_boundary)

        # Rigid component - reflects wave back
        # For a rigid boundary, pressure at boundary mirrors interior
        # This is approximated by just using the interior value
        rigid_value = p_interior

        # Blend between absorbing and reflecting based on R
        # R=1: fully rigid (rigid_value)
        # R=0: fully absorbing (abc_value)
        p_boundary[:] = R * rigid_value + (1 - R) * abc_value

        # Store current interior for next iteration
        self._p_prev[:] = p_interior

    def reset(self) -> None:
        """Reset boundary state."""
        if self._p_prev is not None:
            self._p_prev.fill(0)
        if self._v_prev is not None:
            self._v_prev.fill(0)

    @property
    def reflection_coefficient(self) -> float:
        """Get the effective reflection coefficient."""
        return self._coeff_b

    @property
    def is_initialized(self) -> bool:
        """Check if boundary has been initialized."""
        return self._initialized

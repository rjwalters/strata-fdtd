"""3D Finite-Difference Time-Domain (FDTD) acoustic simulator.

This module implements a pressure-velocity formulation FDTD solver
on a staggered Yee grid for simulating acoustic wave propagation
through arbitrary 3D geometries.

Physics:
    ∂p/∂t = -ρc²(∂vx/∂x + ∂vy/∂y + ∂vz/∂z)
    ∂vx/∂t = -(1/ρ)∂p/∂x
    ∂vy/∂t = -(1/ρ)∂p/∂y
    ∂vz/∂t = -(1/ρ)∂p/∂z

Stability: Δt ≤ min(Δx)/(c√3) (CFL condition for 3D)

Example:
    >>> from strata_fdtd import FDTDSolver, GaussianPulse, PML
    >>> solver = FDTDSolver(shape=(100, 100, 150), resolution=1e-3)
    >>> solver.add_source(GaussianPulse(position=(10, 50, 75), frequency=1000))
    >>> solver.add_probe('center', position=(50, 50, 75))
    >>> solver.run(duration=0.05)
    >>> traces = solver.get_probe_data()

Nonuniform Grids:
    For simulations requiring variable resolution, use a NonuniformGrid:

    >>> from strata_fdtd import NonuniformGrid
    >>> grid = NonuniformGrid.from_stretch(
    ...     shape=(100, 100, 200),
    ...     base_resolution=1e-3,
    ...     stretch_z=1.05,  # 5% geometric stretch in z
    ... )
    >>> solver = FDTDSolver(grid=grid)

Backend Selection:
    The solver supports multiple backends:
    - "python": Pure NumPy implementation (always available)
    - "native": C++ implementation with OpenMP/SIMD (optional, faster)
    - "gpu": GPU acceleration via PyTorch MPS (Apple Silicon)
    - "auto": Selects best available (native > gpu > python)

    >>> solver = FDTDSolver(shape=(100,)*3, resolution=1e-3, backend="auto")
    >>> print(f"Using native: {solver.using_native}, GPU: {solver.using_gpu}")

    Note: GPU backend currently has limited feature support (no PML, geometry,
    or ADE materials). For full feature support, use "native" or "python".

Frequency Weighting:
    Microphones support A-weighting and C-weighting per IEC 61672-1:

    >>> mic = solver.add_microphone(position=(0.05, 0.05, 0.05), name="mic")
    >>> solver.run(duration=0.1)
    >>> waveform_a = mic.get_waveform(weighting="A")  # A-weighted
    >>> spl_dba = mic.get_spl(weighting="A")  # SPL in dBA
    >>> leq = mic.get_leq(weighting="A", duration=1.0)  # Leq,1s in dBA
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from .grid import NonuniformGrid, UniformGrid

if TYPE_CHECKING:
    from strata_fdtd.analysis.weighting import WeightingType

    from .grid import NonuniformGrid, UniformGrid
    from .material_geometry import MaterializedGeometry
    from .sdf import SDFPrimitive

# =============================================================================
# Native C++ Extension Support
# =============================================================================
# Try to import the native C++ kernels. If unavailable, fall back to pure Python.

try:
    from . import _kernels

    _HAS_NATIVE = True
    _NATIVE_VERSION = _kernels.__version__
    _NATIVE_HAS_OPENMP = _kernels.has_openmp
except ImportError:
    _kernels = None  # type: ignore[assignment]
    _HAS_NATIVE = False
    _NATIVE_VERSION = None
    _NATIVE_HAS_OPENMP = False


def has_native_kernels() -> bool:
    """Check if native C++ FDTD kernels are available.

    Returns:
        True if native kernels are available, False otherwise.
    """
    return _HAS_NATIVE


def get_native_info() -> dict:
    """Get information about the native kernel implementation.

    Returns:
        Dict with keys: available, version, has_openmp, num_threads
    """
    if not _HAS_NATIVE:
        return {
            "available": False,
            "version": None,
            "has_openmp": False,
            "num_threads": 1,
        }
    return {
        "available": True,
        "version": _NATIVE_VERSION,
        "has_openmp": _NATIVE_HAS_OPENMP,
        "num_threads": _kernels.get_num_threads() if _NATIVE_HAS_OPENMP else 1,
    }


# =============================================================================
# GPU Backend Support (MPS via PyTorch)
# =============================================================================
# Import GPU availability check from fdtd_gpu module

try:
    from .fdtd_gpu import get_gpu_info
    from .fdtd_gpu import has_gpu_support as _has_gpu_support

    _HAS_GPU = _has_gpu_support()
except ImportError:
    _HAS_GPU = False

    def get_gpu_info() -> dict:
        """Get GPU support information (stub when fdtd_gpu unavailable)."""
        return {"available": False, "backend": None, "pytorch_version": None}


def has_gpu_backend() -> bool:
    """Check if GPU (MPS) backend is available.

    Returns:
        True if PyTorch MPS backend is available, False otherwise.
    """
    return _HAS_GPU


@dataclass
class GaussianPulse:
    """Gaussian-modulated sinusoidal pulse source.

    Creates a broadband acoustic source with energy concentrated
    around the specified center frequency.

    Args:
        position: Grid coordinates (i, j, k) for point source,
                  or dict with 'axis' and 'index' for planar source
        frequency: Center frequency in Hz
        bandwidth: Frequency bandwidth in Hz (default: 2 * frequency)
        amplitude: Peak pressure amplitude (default: 1.0)
        source_type: 'point' or 'plane'

    Example:
        >>> # Point source at grid location (10, 50, 75)
        >>> source = GaussianPulse(position=(10, 50, 75), frequency=1000)
        >>> # Planar source at x=10
        >>> source = GaussianPulse(
        ...     position={'axis': 0, 'index': 10},
        ...     frequency=1000,
        ...     source_type='plane'
        ... )
    """

    position: tuple[int, int, int] | dict
    frequency: float
    bandwidth: float | None = None
    amplitude: float = 1.0
    source_type: Literal["point", "plane"] = "point"

    def __post_init__(self):
        if self.bandwidth is None:
            self.bandwidth = 2.0 * self.frequency

    def waveform(self, t: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        """Generate the source waveform at given times.

        Args:
            t: Array of time values in seconds
            dt: Timestep in seconds (used for pulse timing)

        Returns:
            Array of pressure values at each time
        """
        # Gaussian envelope parameters
        # Bandwidth determines the Gaussian width
        sigma = 1.0 / (np.pi * self.bandwidth)
        t0 = 4.0 * sigma  # Delay to avoid initial discontinuity

        # Gaussian-modulated sinusoid
        envelope = np.exp(-((t - t0) ** 2) / (2 * sigma**2))
        carrier = np.sin(2 * np.pi * self.frequency * (t - t0))

        return self.amplitude * envelope * carrier


@dataclass
class Probe:
    """Pressure recording probe at a specific location.

    Args:
        name: Identifier for this probe
        position: Grid coordinates (i, j, k)
    """

    name: str
    position: tuple[int, int, int]
    data: list[float] = field(default_factory=list)

    def record(self, pressure: float) -> None:
        """Record a pressure sample."""
        self.data.append(pressure)

    def get_data(self) -> NDArray[np.floating]:
        """Get recorded data as numpy array."""
        return np.array(self.data, dtype=np.float32)

    def clear(self) -> None:
        """Clear recorded data."""
        self.data.clear()


# Standard polar pattern coefficients
# Pattern equation: output = (1-k)*pressure + k*(ρc*velocity·direction)
# Or equivalently: S(θ) = (1-k) + k*cos(θ)
POLAR_PATTERNS: dict[str, float] = {
    "omni": 0.0,           # S(θ) = 1
    "subcardioid": 0.3,    # S(θ) = 0.7 + 0.3*cos(θ)
    "cardioid": 0.5,       # S(θ) = 0.5 + 0.5*cos(θ)
    "supercardioid": 0.63, # S(θ) = 0.37 + 0.63*cos(θ)
    "hypercardioid": 0.75, # S(θ) = 0.25 + 0.75*cos(θ)
    "figure8": 1.0,        # S(θ) = cos(θ)
}


class Microphone:
    """Virtual microphone for recording pressure waveforms at arbitrary positions.

    Unlike Probe (which uses grid indices), Microphone accepts physical positions
    in meters and supports trilinear interpolation for sub-grid accuracy.

    Supports directional pickup patterns including cardioid, figure-8, and custom
    patterns. Directional microphones combine pressure and velocity (pressure
    gradient) components to achieve their polar response.

    Args:
        position: Physical coordinates (x, y, z) in meters
        name: Optional identifier for this microphone
        pattern: Polar pattern type. Options:
            - "omni": Omnidirectional (default, equal sensitivity all directions)
            - "cardioid": Heart-shaped, rejects rear sound
            - "supercardioid": Narrower than cardioid with small rear lobe
            - "hypercardioid": Even narrower with larger rear lobe
            - "figure8": Bidirectional, rejects sound from sides
            - callable: Custom pattern function f(theta) -> gain, where theta
                       is angle from microphone axis in radians
        direction: Direction the microphone points (unit vector). Required for
                  directional patterns. Default: (1, 0, 0) for +x direction.
        up: Up vector for figure-8 orientation (determines null plane).
            Only used with figure-8 pattern. Default: (0, 0, 1) for z-up.

    Attributes:
        position: Physical position in meters (tuple of floats)
        name: Microphone identifier
        pattern: Current polar pattern (string or callable)
        direction: Normalized direction vector
        data: Recorded pressure samples

    Example:
        >>> solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)
        >>> # Omnidirectional (default)
        >>> mic_omni = Microphone(position=(0.05, 0.05, 0.05), name="omni")
        >>> # Cardioid pointing in +x direction
        >>> mic_cardioid = Microphone(
        ...     position=(0.05, 0.05, 0.05),
        ...     name="cardioid",
        ...     pattern="cardioid",
        ...     direction=(1, 0, 0)
        ... )
        >>> # Custom subcardioid pattern
        >>> mic_custom = Microphone(
        ...     position=(0.05, 0.05, 0.05),
        ...     name="custom",
        ...     pattern=lambda theta: 0.7 + 0.3 * np.cos(theta)
        ... )
        >>> solver.add_microphone(mic_omni)
        >>> solver.add_microphone(mic_cardioid)
        >>> solver.run(duration=0.01)
        >>> waveform = mic_cardioid.get_waveform()
    """

    def __init__(
        self,
        position: tuple[float, float, float],
        name: str | None = None,
        pattern: str | Callable[[float], float] = "omni",
        direction: tuple[float, float, float] | None = None,
        up: tuple[float, float, float] | None = None,
    ):
        self.position = position
        self.name = name

        # Validate and store pattern
        if isinstance(pattern, str):
            if pattern not in POLAR_PATTERNS:
                raise ValueError(
                    f"Unknown pattern '{pattern}'. "
                    f"Valid patterns: {list(POLAR_PATTERNS.keys())}"
                )
            self._pattern_name = pattern
            self._pattern_k = POLAR_PATTERNS[pattern]
            self._custom_pattern = None
        elif callable(pattern):
            self._pattern_name = "custom"
            self._pattern_k = None  # Will use custom function
            self._custom_pattern = pattern
        else:
            raise TypeError(
                f"pattern must be str or callable, got {type(pattern).__name__}"
            )

        # Store original pattern for repr
        self.pattern = pattern

        # Normalize direction vector
        if direction is None:
            direction = (1.0, 0.0, 0.0)  # Default: +x direction
        dir_arr = np.array(direction, dtype=np.float64)
        dir_norm = np.linalg.norm(dir_arr)
        if dir_norm < 1e-10:
            raise ValueError("direction vector cannot be zero")
        self._direction = dir_arr / dir_norm

        # Store up vector for figure-8 (determines null plane)
        if up is None:
            up = (0.0, 0.0, 1.0)  # Default: z-up
        up_arr = np.array(up, dtype=np.float64)
        up_norm = np.linalg.norm(up_arr)
        if up_norm < 1e-10:
            raise ValueError("up vector cannot be zero")
        self._up = up_arr / up_norm

        # Check direction and up are not parallel (for figure-8)
        dot = abs(np.dot(self._direction, self._up))
        if dot > 0.999:
            raise ValueError("direction and up vectors cannot be parallel")

        self._data: list[float] = []
        self._times: list[float] = []
        self._solver_dt: float | None = None
        self._solver_rho: float | None = None
        self._solver_c: float | None = None
        self._grid_position: tuple[float, float, float] | None = None
        self._interpolation_weights: dict | None = None

    @property
    def direction(self) -> tuple[float, float, float]:
        """Get the microphone direction as a tuple."""
        return tuple(self._direction)

    def is_directional(self) -> bool:
        """Check if this microphone has a directional pattern.

        Returns:
            True if pattern is anything other than omnidirectional.
        """
        return self._pattern_name != "omni"

    def _initialize(self, solver: FDTDSolver) -> None:
        """Initialize microphone with solver parameters.

        Called automatically when microphone is added to solver.
        Pre-computes grid position and interpolation weights.
        """
        self._solver_dt = solver.dt
        self._solver_rho = solver.rho
        self._solver_c = solver.c

        # Convert physical position to grid coordinates
        gx = self.position[0] / solver.dx
        gy = self.position[1] / solver.dx
        gz = self.position[2] / solver.dx

        self._grid_position = (gx, gy, gz)

        # Validate position is within domain
        if not (0 <= gx < solver.shape[0] - 1 and
                0 <= gy < solver.shape[1] - 1 and
                0 <= gz < solver.shape[2] - 1):
            raise ValueError(
                f"Microphone position {self.position} is outside simulation domain. "
                f"Valid range: (0, 0, 0) to "
                f"({(solver.shape[0]-1)*solver.dx:.4f}, "
                f"{(solver.shape[1]-1)*solver.dx:.4f}, "
                f"{(solver.shape[2]-1)*solver.dx:.4f})"
            )

        # Pre-compute trilinear interpolation weights
        self._interpolation_weights = self._compute_trilinear_weights(gx, gy, gz)

    def _compute_trilinear_weights(
        self, gx: float, gy: float, gz: float
    ) -> dict:
        """Compute trilinear interpolation weights for the grid position.

        Returns dict with 'indices' (8 corner indices) and 'weights' (8 weights).
        """
        # Integer grid indices (lower corner of cell)
        i0, j0, k0 = int(gx), int(gy), int(gz)
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1

        # Fractional position within cell [0, 1)
        fx = gx - i0
        fy = gy - j0
        fz = gz - k0

        # Trilinear interpolation weights for 8 corners
        # w_ijk = (1-fx or fx) * (1-fy or fy) * (1-fz or fz)
        weights = [
            (1 - fx) * (1 - fy) * (1 - fz),  # (i0, j0, k0)
            fx * (1 - fy) * (1 - fz),         # (i1, j0, k0)
            (1 - fx) * fy * (1 - fz),         # (i0, j1, k0)
            fx * fy * (1 - fz),               # (i1, j1, k0)
            (1 - fx) * (1 - fy) * fz,         # (i0, j0, k1)
            fx * (1 - fy) * fz,               # (i1, j0, k1)
            (1 - fx) * fy * fz,               # (i0, j1, k1)
            fx * fy * fz,                     # (i1, j1, k1)
        ]

        indices = [
            (i0, j0, k0),
            (i1, j0, k0),
            (i0, j1, k0),
            (i1, j1, k0),
            (i0, j0, k1),
            (i1, j0, k1),
            (i0, j1, k1),
            (i1, j1, k1),
        ]

        return {"indices": indices, "weights": weights}

    def record(
        self,
        pressure_field: NDArray[np.floating],
        time: float,
        vx: NDArray[np.floating] | None = None,
        vy: NDArray[np.floating] | None = None,
        vz: NDArray[np.floating] | None = None,
    ) -> None:
        """Record interpolated pressure from the field.

        For omnidirectional microphones, only pressure is used.
        For directional microphones, velocity fields are combined with
        pressure to compute the polar pattern response.

        Args:
            pressure_field: 3D pressure array from solver
            time: Current simulation time in seconds
            vx: X-velocity field (required for directional patterns)
            vy: Y-velocity field (required for directional patterns)
            vz: Z-velocity field (required for directional patterns)
        """
        if self._interpolation_weights is None:
            raise RuntimeError("Microphone not initialized. Add to solver first.")

        # Trilinear interpolation of pressure
        pressure = 0.0
        for (i, j, k), w in zip(
            self._interpolation_weights["indices"],
            self._interpolation_weights["weights"],
            strict=True,
        ):
            pressure += w * pressure_field[i, j, k]

        # For omnidirectional, just record pressure
        if not self.is_directional():
            self._data.append(pressure)
            self._times.append(time)
            return

        # Directional pattern: combine pressure and velocity components
        if vx is None or vy is None or vz is None:
            raise ValueError(
                "Velocity fields (vx, vy, vz) are required for directional microphones"
            )

        # Interpolate velocity components at microphone position
        vel_x = self._interpolate_velocity(vx, axis=0)
        vel_y = self._interpolate_velocity(vy, axis=1)
        vel_z = self._interpolate_velocity(vz, axis=2)

        # Compute directional output
        output = self._compute_directional_output(pressure, vel_x, vel_y, vel_z)

        self._data.append(output)
        self._times.append(time)

    def _interpolate_velocity(
        self,
        velocity_field: NDArray[np.floating],
        axis: int,
    ) -> float:
        """Interpolate velocity component at microphone position.

        Velocity components are on a staggered grid:
        - vx[i,j,k] is at position (i+0.5, j, k)
        - vy[i,j,k] is at position (i, j+0.5, k)
        - vz[i,j,k] is at position (i, j, k+0.5)

        Args:
            velocity_field: 3D velocity component array
            axis: 0 for vx, 1 for vy, 2 for vz

        Returns:
            Interpolated velocity value
        """
        gx, gy, gz = self._grid_position

        # Adjust grid position for staggered grid
        # vx is shifted by -0.5 in x, etc.
        if axis == 0:
            gx -= 0.5
        elif axis == 1:
            gy -= 0.5
        else:
            gz -= 0.5

        # Clamp to valid range
        nx, ny, nz = velocity_field.shape
        gx = max(0.0, min(gx, nx - 1.001))
        gy = max(0.0, min(gy, ny - 1.001))
        gz = max(0.0, min(gz, nz - 1.001))

        # Compute interpolation weights for this position
        weights = self._compute_trilinear_weights(gx, gy, gz)

        # Interpolate
        velocity = 0.0
        for (i, j, k), w in zip(
            weights["indices"],
            weights["weights"],
            strict=True,
        ):
            # Clamp indices to valid range
            i = min(i, nx - 1)
            j = min(j, ny - 1)
            k = min(k, nz - 1)
            velocity += w * velocity_field[i, j, k]

        return velocity

    def _compute_directional_output(
        self,
        pressure: float,
        vel_x: float,
        vel_y: float,
        vel_z: float,
    ) -> float:
        """Compute directional microphone output.

        For standard patterns:
            output = (1-k)*P + k*Z*V·d

        Where:
            P = pressure at microphone
            V = velocity vector
            d = microphone direction unit vector
            Z = ρc (acoustic impedance)
            k = pattern coefficient (0=omni, 0.5=cardioid, 1=figure-8)

        For custom patterns:
            The custom function receives the angle θ between the wave
            direction and microphone axis, returning a gain multiplier.

        Args:
            pressure: Interpolated pressure
            vel_x: Interpolated x-velocity
            vel_y: Interpolated y-velocity
            vel_z: Interpolated z-velocity

        Returns:
            Directional microphone output
        """
        # Acoustic impedance
        Z = self._solver_rho * self._solver_c

        # Velocity dot direction (cosine of angle for plane wave)
        v_dot_d = (
            vel_x * self._direction[0] +
            vel_y * self._direction[1] +
            vel_z * self._direction[2]
        )

        if self._custom_pattern is not None:
            # Custom pattern: compute angle from velocity vector
            v_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
            if v_mag > 1e-20:
                cos_theta = v_dot_d / v_mag
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
            else:
                theta = 0.0  # No wave, assume on-axis

            # Custom function returns gain vs angle
            gain = self._custom_pattern(theta)
            # Apply gain to pressure (assuming unit gain on-axis)
            return pressure * gain
        else:
            # Standard pattern: combine pressure and velocity
            k = self._pattern_k
            return (1 - k) * pressure + k * Z * v_dot_d

    def get_waveform(
        self,
        weighting: WeightingType = None,
    ) -> NDArray[np.floating]:
        """Get recorded pressure waveform, optionally with frequency weighting.

        Args:
            weighting: Frequency weighting to apply:
                - "A": A-weighting (approximates human hearing at moderate levels)
                - "C": C-weighting (flatter response, used for peak measurements)
                - "Z" or None: No weighting (default, raw pressure)

        Returns:
            1D numpy array of pressure values (weighted if specified)

        Example:
            >>> waveform = mic.get_waveform()  # Raw pressure
            >>> waveform_a = mic.get_waveform(weighting="A")  # A-weighted
        """
        waveform = np.array(self._data, dtype=np.float32)

        if weighting in ("Z", None):
            return waveform

        if len(waveform) == 0:
            return waveform

        # Import here to avoid circular import
        from strata_fdtd.analysis.weighting import apply_weighting

        fs = self.get_sample_rate()
        return apply_weighting(waveform, fs, weighting).astype(np.float32)

    def get_time_axis(self) -> NDArray[np.floating]:
        """Get time axis for recorded waveform.

        Returns:
            1D numpy array of time values in seconds
        """
        return np.array(self._times, dtype=np.float64)

    def get_sample_rate(self) -> float:
        """Get effective sample rate of recording.

        Returns:
            Sample rate in Hz (based on solver timestep)
        """
        if self._solver_dt is None:
            raise RuntimeError("Microphone not initialized. Add to solver first.")
        return 1.0 / self._solver_dt

    def get_spl(
        self,
        weighting: WeightingType = None,
        p_ref: float = 2e-5,
    ) -> float:
        """Calculate sound pressure level (SPL) from recorded waveform.

        SPL = 20 * log10(p_rms / p_ref)

        Args:
            weighting: Frequency weighting ("A", "C", "Z", or None)
            p_ref: Reference pressure in Pa (default: 20 µPa for air)

        Returns:
            SPL in dB (dBA if weighting="A", dBC if weighting="C")

        Raises:
            RuntimeError: If no data has been recorded

        Example:
            >>> mic.get_spl()  # Unweighted SPL (dB)
            >>> mic.get_spl(weighting="A")  # A-weighted SPL (dBA)
            >>> mic.get_spl(weighting="C")  # C-weighted SPL (dBC)
        """
        if len(self._data) == 0:
            raise RuntimeError("No data recorded. Run simulation first.")

        waveform = self.get_waveform(weighting=weighting)

        # Import here to avoid circular import
        from strata_fdtd.analysis.weighting import calculate_spl

        return calculate_spl(waveform, p_ref)

    def get_leq(
        self,
        weighting: WeightingType = None,
        duration: float | None = None,
        p_ref: float = 2e-5,
    ) -> float:
        """Calculate equivalent continuous sound level (Leq).

        Leq is the constant sound level that would deliver the same
        acoustic energy over the measurement period.

        Args:
            weighting: Frequency weighting ("A", "C", "Z", or None)
            duration: Integration time in seconds (default: use full recording)
                      If specified, uses the most recent `duration` seconds
            p_ref: Reference pressure in Pa (default: 20 µPa for air)

        Returns:
            Leq in dB (LAeq if weighting="A", LCeq if weighting="C")

        Raises:
            RuntimeError: If no data has been recorded

        Example:
            >>> mic.get_leq(weighting="A")  # LAeq over full recording
            >>> mic.get_leq(weighting="A", duration=1.0)  # LAeq,1s
        """
        if len(self._data) == 0:
            raise RuntimeError("No data recorded. Run simulation first.")

        waveform = self.get_waveform(weighting=weighting)
        fs = self.get_sample_rate()

        # Import here to avoid circular import
        from strata_fdtd.analysis.weighting import calculate_leq

        return calculate_leq(waveform, fs, duration, p_ref)

    def get_time_weighted_level(
        self,
        weighting: WeightingType = None,
        time_constant: Literal["fast", "slow", "impulse"] = "fast",
        p_ref: float = 2e-5,
    ) -> NDArray[np.floating]:
        """Calculate time-weighted instantaneous sound levels.

        Applies exponential averaging with standard time constants:
        - Fast: 125 ms (default, for general measurements)
        - Slow: 1000 ms (for fluctuating noise)
        - Impulse: 35 ms attack (for impulsive sounds)

        Args:
            weighting: Frequency weighting ("A", "C", "Z", or None)
            time_constant: "fast", "slow", or "impulse"
            p_ref: Reference pressure in Pa (default: 20 µPa for air)

        Returns:
            Array of instantaneous SPL values in dB, same length as recording

        Raises:
            RuntimeError: If no data has been recorded

        Example:
            >>> levels = mic.get_time_weighted_level(weighting="A", time_constant="fast")
            >>> max_level = np.max(levels)  # LAFmax
        """
        if len(self._data) == 0:
            raise RuntimeError("No data recorded. Run simulation first.")

        waveform = self.get_waveform(weighting=weighting)
        fs = self.get_sample_rate()

        # Import here to avoid circular import
        from strata_fdtd.analysis.weighting import calculate_time_weighted_level

        return calculate_time_weighted_level(waveform, fs, time_constant, p_ref)

    def to_wav(
        self,
        filepath: str,
        sample_rate: int = 44100,
        normalize: bool = True,
        bit_depth: int = 16,
    ) -> None:
        """Export recorded waveform to WAV file.

        The waveform is resampled to the target sample rate if needed.

        Args:
            filepath: Output file path (.wav extension recommended)
            sample_rate: Target sample rate in Hz (default: 44100)
            normalize: If True, normalize to [-1, 1] before export (default: True)
            bit_depth: Bit depth for output (16 or 32, default: 16)

        Raises:
            RuntimeError: If no data has been recorded
            ValueError: If bit_depth is not 16 or 32
        """
        import wave

        if len(self._data) == 0:
            raise RuntimeError("No data recorded. Run simulation first.")

        if bit_depth not in (16, 32):
            raise ValueError("bit_depth must be 16 or 32")

        waveform = self.get_waveform()

        # Resample if necessary
        if self._solver_dt is not None:
            native_sr = 1.0 / self._solver_dt
            if abs(native_sr - sample_rate) > 1.0:  # More than 1 Hz difference
                waveform = self._resample(waveform, native_sr, sample_rate)

        # Normalize
        if normalize:
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val * 0.95  # Leave headroom

        # Convert to integer format
        if bit_depth == 16:
            dtype = np.int16
            max_int = 32767
        else:
            dtype = np.int32
            max_int = 2147483647

        waveform_int = (waveform * max_int).astype(dtype)

        # Write WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(bit_depth // 8)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(waveform_int.tobytes())

    def _resample(
        self,
        waveform: NDArray[np.floating],
        src_rate: float,
        target_rate: int,
    ) -> NDArray[np.floating]:
        """Resample waveform to target sample rate.

        Uses linear interpolation for simplicity. For production use,
        consider scipy.signal.resample for better quality.
        """
        src_len = len(waveform)
        target_len = int(src_len * target_rate / src_rate)

        if target_len == src_len:
            return waveform

        # Linear interpolation
        src_times = np.arange(src_len)
        target_times = np.linspace(0, src_len - 1, target_len)

        return np.interp(target_times, src_times, waveform).astype(np.float32)

    def clear(self) -> None:
        """Clear recorded data."""
        self._data.clear()
        self._times.clear()

    def __len__(self) -> int:
        """Return number of recorded samples."""
        return len(self._data)

    def __repr__(self) -> str:
        name_str = f"'{self.name}'" if self.name else "unnamed"
        pattern_str = self._pattern_name
        if self.is_directional():
            dir_str = f", direction={self.direction}"
        else:
            dir_str = ""
        return (
            f"Microphone({name_str}, position={self.position}, "
            f"pattern='{pattern_str}'{dir_str}, samples={len(self._data)})"
        )


class FDTDSolver:
    """3D acoustic FDTD solver with staggered Yee grid.

    Implements pressure-velocity formulation for acoustic wave
    propagation with support for arbitrary geometry, sources,
    probes, and absorbing boundaries. Supports both uniform and
    nonuniform grids for efficient multi-scale simulations.

    Args:
        shape: Grid dimensions (nx, ny, nz) in cells. Required if grid
            is not provided.
        resolution: Grid spacing in meters (same for all axes). Required
            if grid is not provided.
        grid: Grid specification (UniformGrid or NonuniformGrid). If
            provided, shape and resolution are ignored.
        c: Speed of sound in m/s (default: 343.0 for air)
        rho: Medium density in kg/m³ (default: 1.2 for air)
        courant: Courant number < 1 for stability (default: 0.95)
        backend: Computation backend - "auto", "native", "gpu", or "python"
            - "auto": Use best available (native > gpu > python) (default)
            - "native": Use C++ kernels (raises ImportError if unavailable)
            - "gpu": Use GPU/MPS via PyTorch (raises RuntimeError if unavailable)
            - "python": Force pure Python/NumPy implementation
        warn_energy_drift: If True, emit warning when energy changes
            significantly during simulation (default: False)
        energy_drift_threshold: Fractional threshold for energy drift
            warning (default: 0.01 = 1%)

    Attributes:
        p: Pressure field array
        vx, vy, vz: Velocity component arrays
        geometry: Boolean mask (True=air, False=solid)
        dt: Timestep in seconds
        dx: Grid spacing in meters (for uniform grids) or minimum spacing
        grid: The grid specification (UniformGrid or NonuniformGrid)
        using_native: True if using C++ backend
        using_gpu: True if using GPU/MPS backend

    Example:
        >>> # Uniform grid (traditional usage)
        >>> solver = FDTDSolver(shape=(100, 100, 150), resolution=1e-3)
        >>> solver.set_geometry(mask_3d)  # True=air, False=solid
        >>> solver.add_source(GaussianPulse(position=(10,50,75), frequency=1000))
        >>> solver.add_probe('outlet', position=(85, 50, 75))
        >>> solver.run(duration=0.05)

    Nonuniform Grid Example:
        >>> from strata_fdtd import NonuniformGrid
        >>> grid = NonuniformGrid.from_stretch(
        ...     shape=(100, 100, 200),
        ...     base_resolution=1e-3,
        ...     stretch_z=1.05,
        ... )
        >>> solver = FDTDSolver(grid=grid)
        >>> print(f"Min spacing: {solver.grid.min_spacing:.4f} m")

    Energy Tracking Example:
        >>> solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)
        >>> solver.p[50, 50, 50] = 1.0
        >>> solver.run(duration=0.01, track_energy=True)
        >>> history = solver.get_energy_history()
        >>> report = solver.energy_report()
        >>> print(f"Energy changed by {report['energy_change_percent']:.2f}%")
    """

    def __init__(
        self,
        shape: tuple[int, int, int] | None = None,
        resolution: float | None = None,
        grid: UniformGrid | NonuniformGrid | None = None,
        c: float = 343.0,
        rho: float = 1.2,
        courant: float = 0.95,
        backend: Literal["auto", "native", "gpu", "python"] = "auto",
        warn_energy_drift: bool = False,
        energy_drift_threshold: float = 0.01,
    ):
        # Grid setup: either from grid parameter or shape/resolution
        if grid is not None:
            self._grid = grid
            self.shape = grid.shape
        elif shape is not None and resolution is not None:
            self._grid = UniformGrid(shape=shape, resolution=resolution)
            self.shape = shape
        else:
            raise ValueError(
                "Must provide either 'grid' or both 'shape' and 'resolution'"
            )

        # For backward compatibility, dx is the minimum spacing
        self.dx = self._grid.min_spacing
        self.c = c
        self.rho = rho

        # Backend selection
        # Priority for "auto": native (C++) > gpu (MPS) > python
        self._use_gpu = False
        if backend == "native":
            if not _HAS_NATIVE:
                raise ImportError(
                    "Native FDTD kernels not available. "
                    "Install with: pip install -e '.[native]' "
                    "(requires CMake and C++ compiler)"
                )
            self._use_native = True
        elif backend == "gpu":
            if not _HAS_GPU:
                raise RuntimeError(
                    "GPU (MPS) backend not available. "
                    "Requires PyTorch with MPS support on Apple Silicon. "
                    "Install PyTorch with: pip install torch"
                )
            self._use_native = False
            self._use_gpu = True
            # Warn about feature limitations
            warnings.warn(
                "GPU backend has limited feature support. "
                "PML boundaries, geometry masks, and ADE materials are not yet "
                "implemented for GPU. Consider using 'native' or 'python' backend "
                "if these features are needed.",
                UserWarning,
                stacklevel=2,
            )
        elif backend == "python":
            self._use_native = False
        else:  # auto - prefer native > gpu > python
            if _HAS_NATIVE:
                self._use_native = True
            elif _HAS_GPU:
                self._use_native = False
                self._use_gpu = True
            else:
                self._use_native = False

        # Compute CFL-stable timestep based on minimum cell spacing
        # For 3D: dt <= min_dx / (c * sqrt(3))
        min_spacing = self._grid.min_spacing
        dt_max = min_spacing / (self.c * np.sqrt(3))
        self.dt = courant * dt_max

        # For uniform grids, precompute scalar update coefficients
        # For nonuniform grids, these are computed per-cell in _step_python
        if self._grid.is_uniform:
            self._coeff_p = -self.rho * self.c**2 * self.dt / self.dx
            self._coeff_v = -self.dt / (self.rho * self.dx)
        else:
            # Base coefficients (without the 1/dx factor)
            self._coeff_p_base = -self.rho * self.c**2 * self.dt
            self._coeff_v_base = -self.dt / self.rho
            # Precompute spacing arrays for stencil operations
            self._spacing_arrays = self._grid.get_spacing_arrays_for_stencil()

            # Pre-compute native nonuniform grid data if using native backend
            if self._use_native and _kernels is not None:
                sa = self._spacing_arrays
                self._native_grid_data = _kernels.create_nonuniform_grid_data(
                    sa["inv_dx_face"], sa["inv_dy_face"], sa["inv_dz_face"],
                    sa["inv_dx_cell"], sa["inv_dy_cell"], sa["inv_dz_cell"],
                )
            else:
                self._native_grid_data = None

        # Allocate field arrays (contiguous, float32 for memory efficiency)
        self.p = np.zeros(self.shape, dtype=np.float32)
        self.vx = np.zeros(self.shape, dtype=np.float32)
        self.vy = np.zeros(self.shape, dtype=np.float32)
        self.vz = np.zeros(self.shape, dtype=np.float32)

        # GPU backend: allocate PyTorch tensors for accelerated computation
        self._gpu_tensors = None
        if self._use_gpu:
            self._init_gpu_tensors()

        # Geometry mask: True = air (propagates), False = solid (blocks)
        self.geometry = np.ones(self.shape, dtype=bool)

        # Sources, probes, boundaries, microphones
        self._sources: list[GaussianPulse] = []
        self._probes: dict[str, Probe] = {}
        self._microphones: dict[str, Microphone] = {}
        self._boundaries: list = []

        # Simulation state
        self._step_count = 0
        self._time = 0.0

        # Snapshot storage
        self._snapshots: list[tuple[float, NDArray[np.floating]]] = []
        self._velocity_snapshots: list[
            tuple[float, NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
        ] = []
        self._snapshot_interval: int | None = None
        self._snapshot_velocity: bool = False

        # Native backend state (pre-computed boundary cells, etc.)
        self._boundary_cells = None

        # Native microphone recording state
        self._microphone_data = None  # Pre-computed MicrophoneData for native kernel
        self._microphone_output = None  # Reusable output buffer for native recording
        self._microphone_names_order: list[str] = []  # Ordered list of mic names

        # Energy tracking state
        self._warn_energy_drift = warn_energy_drift
        self._energy_drift_threshold = energy_drift_threshold
        self._energy_history: list[tuple[int, float, float]] = []
        self._track_energy = False
        self._energy_sample_interval = 1

        # ADE Material system
        self._materials: dict = {}  # material_id -> AcousticMaterial
        self._material_id = np.zeros(self.shape, dtype=np.uint8)  # Per-cell material ID
        self._ade_fields: dict = {}  # material_id -> {pole_idx -> auxiliary field(s)}
        self._ade_initialized = False

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self._time

    @property
    def step_count(self) -> int:
        """Number of timesteps completed."""
        return self._step_count

    @property
    def using_native(self) -> bool:
        """True if using native C++ backend."""
        return self._use_native

    @property
    def using_gpu(self) -> bool:
        """True if using GPU/MPS backend."""
        return self._use_gpu

    @property
    def grid(self) -> UniformGrid | NonuniformGrid:
        """The grid specification used by this solver."""
        return self._grid

    def set_geometry(
        self,
        geometry: NDArray[np.bool_] | SDFPrimitive | MaterializedGeometry,
    ) -> None:
        """Set simulation geometry with optional material assignments.

        Accepts three input types:
        1. Boolean array: Direct geometry mask (legacy)
        2. SDFPrimitive: Voxelized with air as only material
        3. MaterializedGeometry: Voxelized with material assignments

        Args:
            geometry: Either:
                - np.ndarray: Boolean mask (True=air, False=solid)
                - SDFPrimitive: Will be voxelized with air as only material
                - MaterializedGeometry: Voxelized with material assignments

        Raises:
            ValueError: If geometry shape doesn't match solver shape

        Example:
            >>> # Direct mask (legacy)
            >>> solver.set_geometry(geometry_mask)
            >>>
            >>> # SDF primitive
            >>> box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
            >>> solver.set_geometry(box)
            >>>
            >>> # With materials
            >>> mat_geo = MaterializedGeometry(box)
            >>> mat_geo.set_material(absorber_region, FIBERGLASS_48)
            >>> solver.set_geometry(mat_geo)
        """
        from strata_fdtd.geometry.material_assignment import MaterializedGeometry
        from strata_fdtd.geometry.sdf import SDFPrimitive

        if isinstance(geometry, MaterializedGeometry):
            # Voxelize with materials
            geometry_mask, material_ids = geometry.voxelize_with_materials(self.grid)
            material_table = geometry.get_material_table()

            # Validate shape
            if geometry_mask.shape != self.shape:
                raise ValueError(
                    f"Geometry shape {geometry_mask.shape} doesn't match solver shape {self.shape}"
                )

            # Set geometry mask
            self.geometry = geometry_mask

            # Register all materials and set material regions
            for mat_id, material in material_table.items():
                if mat_id == 0:
                    # Default material - assign to all air cells without explicit material
                    default_mask = geometry_mask & (material_ids == 0)
                    if np.any(default_mask):
                        # Register default material with ID 0 (special case)
                        if 0 not in self._materials:
                            # We need to handle default material specially
                            # For now, we'll skip it and let air be handled by solver defaults
                            pass
                else:
                    # Non-default materials
                    if mat_id not in self._materials:
                        self.register_material(material, material_id=mat_id)
                    # Set material region
                    material_mask = material_ids == mat_id
                    if np.any(material_mask):
                        self.set_material_region(material_mask, material_id=mat_id)

            # Mark ADE as needing initialization
            self._ade_initialized = False

        elif isinstance(geometry, SDFPrimitive):
            # Voxelize SDF primitive (air only, no special materials)
            geometry_mask = geometry.voxelize(self.grid)

            # Validate shape
            if geometry_mask.shape != self.shape:
                raise ValueError(
                    f"Geometry shape {geometry_mask.shape} doesn't match solver shape {self.shape}"
                )

            # Set geometry mask
            self.geometry = geometry_mask

            # Reset materials (air only)
            self._material_id.fill(0)

        else:
            # Legacy: direct boolean array
            mask = geometry
            if mask.shape != self.shape:
                raise ValueError(f"Geometry shape {mask.shape} doesn't match solver shape {self.shape}")
            self.geometry = mask.astype(bool)

        # Pre-compute boundary cells for native backend
        if self._use_native and _kernels is not None:
            self._boundary_cells = _kernels.precompute_boundary_cells(self.geometry)

    def add_source(self, source: GaussianPulse) -> None:
        """Add an acoustic source to the simulation."""
        self._sources.append(source)

    def add_probe(self, name: str, position: tuple[int, int, int]) -> None:
        """Add a pressure recording probe.

        Args:
            name: Unique identifier for this probe
            position: Grid coordinates (i, j, k)
        """
        if name in self._probes:
            raise ValueError(f"Probe '{name}' already exists")
        self._probes[name] = Probe(name=name, position=position)

    def add_microphone(
        self,
        position: tuple[float, float, float] | Microphone,
        name: str | None = None,
        pattern: str | Callable[[float], float] = "omni",
        direction: tuple[float, float, float] | None = None,
        up: tuple[float, float, float] | None = None,
    ) -> Microphone:
        """Add a virtual microphone for recording pressure waveforms.

        Microphones accept physical positions in meters (not grid indices)
        and use trilinear interpolation for sub-grid accuracy.

        Supports directional patterns (cardioid, figure-8, etc.) that combine
        pressure and velocity measurements to achieve realistic polar responses.

        Args:
            position: Either a Microphone object, or physical coordinates
                      (x, y, z) in meters
            name: Optional identifier (required if position is a tuple)
            pattern: Polar pattern type (ignored if position is a Microphone).
                - "omni": Omnidirectional (default)
                - "cardioid": Heart-shaped, rejects rear
                - "supercardioid": Narrower than cardioid
                - "hypercardioid": Even narrower
                - "figure8": Bidirectional
                - callable: Custom f(theta) -> gain function
            direction: Direction the microphone points (unit vector).
                      Default: (1, 0, 0) for +x direction.
            up: Up vector for figure-8 orientation. Default: (0, 0, 1).

        Returns:
            The Microphone object (for method chaining or direct access)

        Raises:
            ValueError: If microphone name already exists or position is invalid

        Example:
            >>> solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)
            >>> # Omnidirectional (default)
            >>> solver.add_microphone(position=(0.05, 0.05, 0.05), name="omni")
            >>> # Cardioid pointing in +x direction
            >>> solver.add_microphone(
            ...     position=(0.05, 0.05, 0.05),
            ...     name="cardioid",
            ...     pattern="cardioid",
            ...     direction=(1, 0, 0)
            ... )
            >>> # Pass Microphone object directly
            >>> mic = Microphone(
            ...     position=(0.08, 0.05, 0.05),
            ...     name="figure8",
            ...     pattern="figure8",
            ...     direction=(0, 1, 0)
            ... )
            >>> solver.add_microphone(mic)
        """
        if isinstance(position, Microphone):
            mic = position
        else:
            mic = Microphone(
                position=position,
                name=name,
                pattern=pattern,
                direction=direction,
                up=up,
            )

        # Use provided name or generate one
        mic_name = mic.name or f"mic_{len(self._microphones)}"
        mic.name = mic_name

        if mic_name in self._microphones:
            raise ValueError(f"Microphone '{mic_name}' already exists")

        # Initialize microphone with solver parameters
        mic._initialize(self)

        self._microphones[mic_name] = mic

        # Invalidate native microphone cache (will be recomputed on next recording)
        self._microphone_data = None

        return mic

    @property
    def microphones(self) -> dict[str, Microphone]:
        """Dict-like access to microphones by name.

        Example:
            >>> solver.microphones["center"].get_waveform()
            >>> for name, mic in solver.microphones.items():
            ...     print(f"{name}: {len(mic)} samples")
        """
        return self._microphones

    def add_boundary(self, boundary) -> None:
        """Add a boundary condition handler.

        Args:
            boundary: Boundary condition object (e.g., PML)
        """
        boundary.initialize(self)
        self._boundaries.append(boundary)

    def enable_snapshots(self, interval: int, capture_velocity: bool = False) -> None:
        """Enable field snapshots at regular intervals.

        Args:
            interval: Save snapshot every N timesteps
            capture_velocity: If True, also capture velocity field snapshots.
                Velocity is interpolated from staggered Yee grid to cell centers.
        """
        self._snapshot_interval = interval
        self._snapshot_velocity = capture_velocity

    def step(self) -> None:
        """Advance simulation by one timestep.

        This is the core FDTD update loop using leapfrog scheme.
        Velocities are at half-integer time steps, pressure at integer steps.

        Grid layout (staggered Yee grid):
        - p[i,j,k] at cell center (i, j, k)
        - vx[i,j,k] at face (i+1/2, j, k) - right face of cell i
        - vy[i,j,k] at face (i, j+1/2, k) - top face of cell j
        - vz[i,j,k] at face (i, j, k+1/2) - front face of cell k

        ADE Material Update Sequence:
        When materials are registered, the step sequence becomes:
        1. Update density auxiliary fields
        2. Update velocities (standard + ADE density correction)
        3. Apply rigid boundaries
        4. Update modulus auxiliary fields
        5. Update pressure (standard + ADE modulus correction)
        6. Zero pressure in solids
        """
        # Initialize ADE fields on first step with materials
        if self.has_materials and not self._ade_initialized:
            self._initialize_ade_fields()

        # Backend dispatch
        if self._use_gpu:
            # GPU backend - sync to GPU first if this is after source injection
            if self._step_count > 0:
                self._sync_to_gpu()
            self._step_gpu()
        elif self._use_native and _kernels is not None:
            if self._grid.is_uniform:
                self._step_native()
            else:
                self._step_native_nonuniform()
        else:
            self._step_python()

        # Apply absorbing boundary conditions (PML, etc.)
        # Note: These are applied after the core step for both backends
        for boundary in self._boundaries:
            boundary.apply_velocity(self)
        for boundary in self._boundaries:
            boundary.apply_pressure(self)

        # Inject sources
        self._inject_sources()

        # Record probes and microphones
        self._record_probes()
        self._record_microphones()

        # Save snapshot if enabled
        if self._snapshot_interval and self._step_count % self._snapshot_interval == 0:
            self._snapshots.append((self._time, self.p.copy()))

            # Capture velocity snapshots if enabled
            if self._snapshot_velocity:
                # Interpolate staggered velocities to cell centers
                vx_centered = self._interpolate_vx_to_centers()
                vy_centered = self._interpolate_vy_to_centers()
                vz_centered = self._interpolate_vz_to_centers()
                self._velocity_snapshots.append(
                    (self._time, vx_centered, vy_centered, vz_centered)
                )

        # Update simulation state
        self._step_count += 1
        self._time += self.dt

        # Record energy if tracking enabled
        if self._track_energy and self._step_count % self._energy_sample_interval == 0:
            energy = self.compute_energy()
            self._energy_history.append((self._step_count, self._time, energy))

    def _step_native(self) -> None:
        """Native C++ implementation of FDTD step.

        Uses separate velocity and pressure updates to allow rigid boundary
        application between phases, matching the Python implementation's order
        of operations for proper energy conservation.
        """
        # Phase 1: Update velocities from pressure gradient
        _kernels.update_velocity(
            self.p, self.vx, self.vy, self.vz, self._coeff_v
        )

        # Phase 2: Apply rigid boundaries (must come AFTER velocity update,
        # BEFORE pressure update to prevent boundary velocities from
        # contributing to pressure divergence)
        if self._boundary_cells is not None:
            _kernels.apply_rigid_boundaries(
                self.vx, self.vy, self.vz, self._boundary_cells
            )

        # Phase 3: Update pressure from velocity divergence
        _kernels.update_pressure(
            self.p, self.vx, self.vy, self.vz, self.geometry, self._coeff_p
        )

    def _step_native_nonuniform(self) -> None:
        """Native C++ implementation of FDTD step for nonuniform grids.

        Uses separate velocity and pressure updates with per-cell spacing
        arrays for variable cell sizes.
        """
        # Phase 1: Update velocities from pressure gradient
        _kernels.update_velocity_nonuniform(
            self.p, self.vx, self.vy, self.vz,
            self._native_grid_data, self._coeff_v_base
        )

        # Phase 2: Apply rigid boundaries
        if self._boundary_cells is not None:
            _kernels.apply_rigid_boundaries(
                self.vx, self.vy, self.vz, self._boundary_cells
            )

        # Phase 3: Update pressure from velocity divergence
        _kernels.update_pressure_nonuniform(
            self.p, self.vx, self.vy, self.vz, self.geometry,
            self._native_grid_data, self._coeff_p_base
        )

    def _step_python(self) -> None:
        """Pure Python/NumPy implementation of FDTD step."""
        if self._grid.is_uniform:
            self._step_python_uniform()
        else:
            self._step_python_nonuniform()

    def _step_python_uniform(self) -> None:
        """FDTD step for uniform grids (original efficient implementation)."""
        # === ADE Phase 1: Update density auxiliary fields ===
        if self.has_materials:
            self._update_ade_density_poles()

        # Update velocity components from pressure gradient
        # vx[i,j,k] at right face of cell i, driven by p[i+1] - p[i]
        self.vx[:-1, :, :] += self._coeff_v * (self.p[1:, :, :] - self.p[:-1, :, :])
        self.vy[:, :-1, :] += self._coeff_v * (self.p[:, 1:, :] - self.p[:, :-1, :])
        self.vz[:, :, :-1] += self._coeff_v * (self.p[:, :, 1:] - self.p[:, :, :-1])

        # === ADE Phase 2: Apply density ADE correction to velocities ===
        if self.has_materials:
            self._apply_ade_velocity_correction()

        # Apply rigid boundary conditions (zero normal velocity at solids)
        self._apply_rigid_boundaries()

        # === ADE Phase 3: Update modulus auxiliary fields ===
        if self.has_materials:
            self._update_ade_modulus_poles()

        # Update pressure from velocity divergence
        # With vx[i] at face (i+1/2), the divergence for cell i is:
        # (vx[i] - vx[i-1])/dx where vx[-1] = 0 (boundary)
        #
        # For nx cells (0 to nx-1), we need:
        # Cell 0: vx[0] - 0 = vx[0]
        # Cell i: vx[i] - vx[i-1]
        # Cell nx-1: vx[nx-1] - vx[nx-2]

        nx, ny, nz = self.shape

        # Divergence computation with proper boundary handling
        div = np.zeros(self.shape, dtype=np.float32)

        # x-divergence: for cell i, we need vx[i] - vx[i-1]
        # Cell 0: use vx[0] (vx[-1] = 0 at left boundary)
        div[0, :, :] = self.vx[0, :, :]
        # Cells 1 to nx-1: vx[i] - vx[i-1]
        div[1:, :, :] = self.vx[1:nx, :, :] - self.vx[:nx-1, :, :]

        # y-divergence
        div[:, 0, :] += self.vy[:, 0, :]
        div[:, 1:, :] += self.vy[:, 1:ny, :] - self.vy[:, :ny-1, :]

        # z-divergence
        div[:, :, 0] += self.vz[:, :, 0]
        div[:, :, 1:] += self.vz[:, :, 1:nz] - self.vz[:, :, :nz-1]

        self.p += self._coeff_p * div

        # === ADE Phase 4: Apply modulus ADE correction to pressure ===
        if self.has_materials:
            self._apply_ade_pressure_correction()

        # Zero pressure in solid regions
        self.p[~self.geometry] = 0.0

    def _step_python_nonuniform(self) -> None:
        """FDTD step for nonuniform grids with variable cell spacing.

        Uses precomputed inverse spacing arrays for efficiency.
        The update equations are:
        - Velocity: v[i] += coeff_v_base * inv_dx_face[i] * (p[i+1] - p[i])
        - Pressure: p[i] += coeff_p_base * inv_dx_cell[i] * (v[i] - v[i-1])
        """
        # === ADE Phase 1: Update density auxiliary fields ===
        if self.has_materials:
            self._update_ade_density_poles()

        nx, ny, nz = self.shape
        sa = self._spacing_arrays  # Precomputed spacing arrays

        # Update velocity components from pressure gradient
        # Each component uses the appropriate face spacing

        # vx update: vx[i] += coeff * inv_dx_face[i] * (p[i+1] - p[i])
        # inv_dx_face has shape (nx-1,), broadcast to (nx-1, ny, nz)
        dp_x = self.p[1:, :, :] - self.p[:-1, :, :]
        self.vx[:-1, :, :] += (
            self._coeff_v_base
            * sa["inv_dx_face"][:, np.newaxis, np.newaxis]
            * dp_x
        )

        # vy update
        dp_y = self.p[:, 1:, :] - self.p[:, :-1, :]
        self.vy[:, :-1, :] += (
            self._coeff_v_base
            * sa["inv_dy_face"][np.newaxis, :, np.newaxis]
            * dp_y
        )

        # vz update
        dp_z = self.p[:, :, 1:] - self.p[:, :, :-1]
        self.vz[:, :, :-1] += (
            self._coeff_v_base
            * sa["inv_dz_face"][np.newaxis, np.newaxis, :]
            * dp_z
        )

        # === ADE Phase 2: Apply density ADE correction to velocities ===
        if self.has_materials:
            self._apply_ade_velocity_correction()

        # Apply rigid boundary conditions (zero normal velocity at solids)
        self._apply_rigid_boundaries()

        # === ADE Phase 3: Update modulus auxiliary fields ===
        if self.has_materials:
            self._update_ade_modulus_poles()

        # Update pressure from velocity divergence with variable cell sizes
        # For nonuniform grids, each cell has its own size for the divergence

        # Compute divergence with per-cell spacing
        # div[i] = (vx[i] - vx[i-1]) / dx_cell[i] + ...

        # x-divergence with per-cell spacing
        div_x = np.zeros(self.shape, dtype=np.float32)
        # Cell 0: vx[0] / dx_cell[0] (vx[-1] = 0 at boundary)
        div_x[0, :, :] = self.vx[0, :, :] * sa["inv_dx_cell"][0]
        # Interior cells
        for i in range(1, nx):
            div_x[i, :, :] = (self.vx[i, :, :] - self.vx[i - 1, :, :]) * sa["inv_dx_cell"][i]

        # y-divergence with per-cell spacing
        div_y = np.zeros(self.shape, dtype=np.float32)
        div_y[:, 0, :] = self.vy[:, 0, :] * sa["inv_dy_cell"][0]
        for j in range(1, ny):
            div_y[:, j, :] = (self.vy[:, j, :] - self.vy[:, j - 1, :]) * sa["inv_dy_cell"][j]

        # z-divergence with per-cell spacing
        div_z = np.zeros(self.shape, dtype=np.float32)
        div_z[:, :, 0] = self.vz[:, :, 0] * sa["inv_dz_cell"][0]
        for k in range(1, nz):
            div_z[:, :, k] = (self.vz[:, :, k] - self.vz[:, :, k - 1]) * sa["inv_dz_cell"][k]

        # Total divergence
        div = div_x + div_y + div_z

        # Pressure update (coeff_p_base already includes -ρc²dt)
        self.p += self._coeff_p_base * div

        # === ADE Phase 4: Apply modulus ADE correction to pressure ===
        if self.has_materials:
            self._apply_ade_pressure_correction()

        # Zero pressure in solid regions
        self.p[~self.geometry] = 0.0

    def _init_gpu_tensors(self) -> None:
        """Initialize PyTorch tensors for GPU computation."""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for GPU backend. Install with: pip install torch"
            ) from None

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "cpu":
            warnings.warn(
                "MPS not available, GPU backend running on CPU. "
                "This provides no acceleration benefit.",
                UserWarning,
                stacklevel=3,
            )

        # Create GPU tensors
        self._gpu_tensors = {
            "p": torch.zeros(*self.shape, device=device, dtype=torch.float32),
            "vx": torch.zeros(*self.shape, device=device, dtype=torch.float32),
            "vy": torch.zeros(*self.shape, device=device, dtype=torch.float32),
            "vz": torch.zeros(*self.shape, device=device, dtype=torch.float32),
            "device": device,
            "torch": torch,
        }

        # FDTD coefficients for GPU
        self._gpu_tensors["coeff_v"] = -self.dt / (self.rho * self.dx)
        self._gpu_tensors["coeff_p"] = -self.rho * self.c**2 * self.dt / self.dx

    def _step_gpu(self) -> None:
        """GPU-accelerated FDTD step using PyTorch MPS backend.

        This is a simplified step that handles basic FDTD without
        geometry masks, PML, or ADE materials.
        """
        if self._gpu_tensors is None:
            raise RuntimeError("GPU tensors not initialized")

        p = self._gpu_tensors["p"]
        vx = self._gpu_tensors["vx"]
        vy = self._gpu_tensors["vy"]
        vz = self._gpu_tensors["vz"]
        coeff_v = self._gpu_tensors["coeff_v"]
        coeff_p = self._gpu_tensors["coeff_p"]

        # Velocity update: v += coeff_v * grad(p)
        vx[:-1, :, :] += coeff_v * (p[1:, :, :] - p[:-1, :, :])
        vy[:, :-1, :] += coeff_v * (p[:, 1:, :] - p[:, :-1, :])
        vz[:, :, :-1] += coeff_v * (p[:, :, 1:] - p[:, :, :-1])

        # Pressure update: p += coeff_p * div(v)
        p[1:, :, :] += coeff_p * (vx[1:, :, :] - vx[:-1, :, :])
        p[:, 1:, :] += coeff_p * (vy[:, 1:, :] - vy[:, :-1, :])
        p[:, :, 1:] += coeff_p * (vz[:, :, 1:] - vz[:, :, :-1])

        # Sync back to numpy arrays for compatibility with probes/sources
        self.p[:] = p.cpu().numpy()
        self.vx[:] = vx.cpu().numpy()
        self.vy[:] = vy.cpu().numpy()
        self.vz[:] = vz.cpu().numpy()

    def _sync_to_gpu(self) -> None:
        """Sync numpy arrays to GPU tensors (after source injection)."""
        if self._gpu_tensors is None:
            return
        torch = self._gpu_tensors["torch"]
        device = self._gpu_tensors["device"]
        self._gpu_tensors["p"] = torch.from_numpy(self.p).to(device)
        self._gpu_tensors["vx"] = torch.from_numpy(self.vx).to(device)
        self._gpu_tensors["vy"] = torch.from_numpy(self.vy).to(device)
        self._gpu_tensors["vz"] = torch.from_numpy(self.vz).to(device)

    def _apply_rigid_boundaries(self) -> None:
        """Zero velocity at solid boundaries.

        For each velocity component, zero it if either adjacent
        cell is solid (the face is blocked).

        With the convention vx[i] at face (i+1/2):
        - vx[i] is between cells i and i+1
        - Zero if either cell i or i+1 is solid
        """
        # vx[i,j,k] is between cells (i,j,k) and (i+1,j,k)
        # Zero if either cell is solid
        solid_vx = ~self.geometry[:-1, :, :] | ~self.geometry[1:, :, :]
        self.vx[:-1, :, :][solid_vx] = 0.0

        # vy[i,j,k] is between cells (i,j,k) and (i,j+1,k)
        solid_vy = ~self.geometry[:, :-1, :] | ~self.geometry[:, 1:, :]
        self.vy[:, :-1, :][solid_vy] = 0.0

        # vz[i,j,k] is between cells (i,j,k) and (i,j,k+1)
        solid_vz = ~self.geometry[:, :, :-1] | ~self.geometry[:, :, 1:]
        self.vz[:, :, :-1][solid_vz] = 0.0

    def _inject_sources(self) -> None:
        """Inject source waveforms into the pressure field."""
        for source in self._sources:
            # Compute waveform value at current time
            waveform_val = source.waveform(np.array([self._time]), self.dt)[0]

            if source.source_type == "point":
                i, j, k = source.position
                if self.geometry[i, j, k]:  # Only inject in air
                    self.p[i, j, k] += waveform_val
            else:  # plane source
                axis = source.position["axis"]
                idx = source.position["index"]
                if axis == 0:
                    mask = self.geometry[idx, :, :]
                    self.p[idx, :, :][mask] += waveform_val
                elif axis == 1:
                    mask = self.geometry[:, idx, :]
                    self.p[:, idx, :][mask] += waveform_val
                else:  # axis == 2
                    mask = self.geometry[:, :, idx]
                    self.p[:, :, idx][mask] += waveform_val

    def _record_probes(self) -> None:
        """Record pressure at all probe locations."""
        for probe in self._probes.values():
            i, j, k = probe.position
            probe.record(float(self.p[i, j, k]))

    def _record_microphones(self) -> None:
        """Record pressure at all microphone locations with interpolation.

        Uses native C++ kernel with OpenMP parallelization when available
        and there are enough microphones to benefit from parallelization.

        Note: Native kernel currently only supports omnidirectional microphones.
        Directional microphones always use Python implementation.
        """
        if not self._microphones:
            return

        # Check if any microphones are directional
        has_directional = any(mic.is_directional() for mic in self._microphones.values())

        # Use native kernel when available AND no directional microphones
        # (Native kernel doesn't support velocity interpolation yet)
        if self._use_native and _kernels is not None and not has_directional:
            self._record_microphones_native()
        else:
            self._record_microphones_python()

    def _record_microphones_python(self) -> None:
        """Python implementation of microphone recording.

        For directional microphones, velocity fields are passed to enable
        polar pattern response calculation.
        """
        for mic in self._microphones.values():
            if mic.is_directional():
                mic.record(self.p, self._time, self.vx, self.vy, self.vz)
            else:
                mic.record(self.p, self._time)

    def _record_microphones_native(self) -> None:
        """Native C++ implementation of microphone recording.

        Pre-computes interpolation data on first call, then uses batch
        trilinear interpolation with OpenMP parallelization.
        """
        # Lazy initialization of native microphone data
        if self._microphone_data is None:
            self._prepare_native_microphones()

        # Record all microphones in batch
        _kernels.record_microphones_batch(
            self.p, self._microphone_data, self._microphone_output
        )

        # Distribute results to individual Microphone objects
        for i, name in enumerate(self._microphone_names_order):
            mic = self._microphones[name]
            mic._data.append(float(self._microphone_output[i]))
            mic._times.append(self._time)

    def _prepare_native_microphones(self) -> None:
        """Pre-compute native microphone interpolation data."""
        n_mics = len(self._microphones)

        # Build ordered list of microphone names and grid positions
        self._microphone_names_order = list(self._microphones.keys())
        grid_positions = np.zeros(n_mics * 3, dtype=np.float32)

        for i, name in enumerate(self._microphone_names_order):
            mic = self._microphones[name]
            gx, gy, gz = mic._grid_position
            grid_positions[i * 3 + 0] = gx
            grid_positions[i * 3 + 1] = gy
            grid_positions[i * 3 + 2] = gz

        # Pre-compute interpolation data
        nx, ny, nz = self.shape
        self._microphone_data = _kernels.precompute_microphone_data(
            grid_positions, nx, ny, nz
        )

        # Allocate reusable output buffer
        self._microphone_output = np.zeros(n_mics, dtype=np.float32)

    def run(
        self,
        duration: float,
        progress: bool = False,
        track_energy: bool = False,
        energy_sample_interval: int = 1,
        output_file: str | None = None,
        script_content: str | None = None,
        callback: Callable[[int], None] | None = None,
        snapshot_interval: int | None = None,
    ) -> None:
        """Run simulation for specified duration.

        Args:
            duration: Simulation time in seconds
            progress: If True, print progress updates
            track_energy: If True, track energy at each sample interval
            energy_sample_interval: Record energy every N steps (default: 1)
            output_file: Path to HDF5 output file (optional)
            script_content: Source script content for reproducibility (stored in HDF5)
            callback: Function called after each timestep with signature callback(step)
            snapshot_interval: Save pressure field every N steps to HDF5 (default: no snapshots)
        """
        import time as time_module

        n_steps = int(np.ceil(duration / self.dt))
        start_time = time_module.time()

        # Set up energy tracking for this run
        self._track_energy = track_energy
        self._energy_sample_interval = energy_sample_interval

        # Record initial energy if tracking
        if track_energy and len(self._energy_history) == 0:
            initial_energy = self.compute_energy()
            self._energy_history.append((self._step_count, self._time, initial_energy))

        # Set up HDF5 output if requested
        hdf5_writer = None
        if output_file:
            from strata_fdtd.io import HDF5ResultWriter

            hdf5_writer = HDF5ResultWriter(output_file, self, script_content)

        try:
            if progress:
                from tqdm import tqdm

                iterator = tqdm(range(n_steps), desc="FDTD simulation")
            else:
                iterator = range(n_steps)

            for step in iterator:
                self.step()

                # Write to HDF5 if enabled
                if hdf5_writer:
                    save_snapshot = (
                        snapshot_interval is not None and step % snapshot_interval == 0
                    )
                    hdf5_writer.write_timestep(self._step_count - 1, save_snapshot)

                # Call user callback if provided
                if callback:
                    callback(self._step_count - 1)

            # Check for energy drift warning after run completes
            if self._warn_energy_drift and len(self._energy_history) >= 2:
                report = self.energy_report()
                if abs(report["energy_change_percent"]) > self._energy_drift_threshold * 100:
                    warnings.warn(
                        f"Energy drift detected: {report['energy_change_percent']:.2f}% change "
                        f"(threshold: {self._energy_drift_threshold * 100:.1f}%). "
                        f"Status: {report['conservation_status']}",
                        UserWarning,
                        stacklevel=2,
                    )

        finally:
            # Finalize HDF5 file if it was opened
            if hdf5_writer:
                runtime = time_module.time() - start_time
                backend = "native" if self.using_native else "python"
                num_threads = get_native_info()["num_threads"]
                hdf5_writer.finalize(
                    runtime=runtime, backend=backend, num_threads=num_threads
                )

    def get_probe_data(self, name: str | None = None) -> dict[str, NDArray[np.floating]]:
        """Get recorded probe data.

        Args:
            name: Specific probe name, or None for all probes

        Returns:
            Dict mapping probe names to pressure time series
        """
        if name is not None:
            if name not in self._probes:
                raise KeyError(f"Probe '{name}' not found")
            return {name: self._probes[name].get_data()}
        return {name: probe.get_data() for name, probe in self._probes.items()}

    def get_snapshots(self) -> list[tuple[float, NDArray[np.floating]]]:
        """Get saved field snapshots.

        Returns:
            List of (time, pressure_field) tuples
        """
        return self._snapshots

    def get_velocity_snapshots(
        self,
    ) -> list[
        tuple[float, NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
    ]:
        """Get saved velocity field snapshots.

        Velocity components are interpolated from staggered Yee grid positions
        to cell centers for visualization.

        Returns:
            List of (time, vx, vy, vz) tuples where each velocity component
            has shape matching the solver grid.
        """
        return self._velocity_snapshots

    def _interpolate_vx_to_centers(self) -> NDArray[np.floating]:
        """Interpolate vx from staggered face positions to cell centers.

        On the Yee grid, vx[i,j,k] is at face (i+1/2, j, k).
        Cell center (i, j, k) averages vx[i] and vx[i-1].
        """
        nx, ny, nz = self.shape
        vx_centered = np.zeros((nx, ny, nz), dtype=np.float32)
        # Average adjacent face values
        vx_centered[1:, :, :] = 0.5 * (self.vx[:-1, :, :] + self.vx[1:, :, :])
        # Boundary: use single adjacent value
        vx_centered[0, :, :] = 0.5 * self.vx[0, :, :]
        return vx_centered

    def _interpolate_vy_to_centers(self) -> NDArray[np.floating]:
        """Interpolate vy from staggered face positions to cell centers.

        On the Yee grid, vy[i,j,k] is at face (i, j+1/2, k).
        Cell center (i, j, k) averages vy[j] and vy[j-1].
        """
        nx, ny, nz = self.shape
        vy_centered = np.zeros((nx, ny, nz), dtype=np.float32)
        # Average adjacent face values
        vy_centered[:, 1:, :] = 0.5 * (self.vy[:, :-1, :] + self.vy[:, 1:, :])
        # Boundary: use single adjacent value
        vy_centered[:, 0, :] = 0.5 * self.vy[:, 0, :]
        return vy_centered

    def _interpolate_vz_to_centers(self) -> NDArray[np.floating]:
        """Interpolate vz from staggered face positions to cell centers.

        On the Yee grid, vz[i,j,k] is at face (i, j, k+1/2).
        Cell center (i, j, k) averages vz[k] and vz[k-1].
        """
        nx, ny, nz = self.shape
        vz_centered = np.zeros((nx, ny, nz), dtype=np.float32)
        # Average adjacent face values
        vz_centered[:, :, 1:] = 0.5 * (self.vz[:, :, :-1] + self.vz[:, :, 1:])
        # Boundary: use single adjacent value
        vz_centered[:, :, 0] = 0.5 * self.vz[:, :, 0]
        return vz_centered

    def compute_energy(self) -> float:
        """Compute total acoustic energy in the domain.

        Energy = (1/2) * integral of (p²/ρc² + ρv²) dV

        Returns:
            Total acoustic energy in Joules
        """
        dV = self.dx**3

        # Pressure energy (only in air)
        p_energy = 0.5 * np.sum(self.p[self.geometry] ** 2) / (self.rho * self.c**2) * dV

        # Kinetic energy (velocity components)
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        k_energy = 0.5 * self.rho * np.sum(v2[self.geometry]) * dV

        return p_energy + k_energy

    def get_energy_history(self) -> list[tuple[int, float, float]]:
        """Get energy history recorded during simulation.

        Returns:
            List of (step, time, total_energy) tuples. Empty if tracking
            was not enabled during run().

        Example:
            >>> solver.run(duration=0.01, track_energy=True)
            >>> history = solver.get_energy_history()
            >>> for step, time, energy in history:
            ...     print(f"Step {step}: t={time:.6f}s, E={energy:.6e}J")
        """
        return self._energy_history.copy()

    def energy_report(self) -> dict:
        """Generate energy conservation diagnostic report.

        Returns:
            Dict with keys:
            - initial_energy: Energy at first recorded step
            - final_energy: Energy at last recorded step
            - max_energy: Maximum energy observed
            - min_energy: Minimum energy observed
            - energy_change_percent: (final - initial) / initial * 100
            - conservation_status: "stable" | "growing" | "decaying"
            - n_samples: Number of energy samples recorded

        Raises:
            ValueError: If no energy history has been recorded

        Example:
            >>> solver.run(duration=0.01, track_energy=True)
            >>> report = solver.energy_report()
            >>> print(f"Energy changed by {report['energy_change_percent']:.2f}%")
            >>> print(f"Status: {report['conservation_status']}")
        """
        if not self._energy_history:
            raise ValueError(
                "No energy history recorded. Call run() with track_energy=True first."
            )

        energies = np.array([e for _, _, e in self._energy_history])
        initial_energy = energies[0]
        final_energy = energies[-1]
        max_energy = float(np.max(energies))
        min_energy = float(np.min(energies))

        # Handle case where initial energy is zero (no excitation yet)
        if initial_energy == 0:
            energy_change_percent = 0.0 if final_energy == 0 else float('inf')
        else:
            energy_change_percent = (final_energy - initial_energy) / initial_energy * 100

        # Determine conservation status
        # Use 1% threshold for stable classification
        if abs(energy_change_percent) <= 1.0:
            status = "stable"
        elif energy_change_percent > 0:
            status = "growing"
        else:
            status = "decaying"

        return {
            "initial_energy": float(initial_energy),
            "final_energy": float(final_energy),
            "max_energy": max_energy,
            "min_energy": min_energy,
            "energy_change_percent": energy_change_percent,
            "conservation_status": status,
            "n_samples": len(self._energy_history),
        }

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.p.fill(0)
        self.vx.fill(0)
        self.vy.fill(0)
        self.vz.fill(0)
        self._step_count = 0
        self._time = 0.0
        self._snapshots.clear()
        self._energy_history.clear()
        self._track_energy = False
        for probe in self._probes.values():
            probe.clear()
        for mic in self._microphones.values():
            mic.clear()
        for boundary in self._boundaries:
            boundary.reset()
        # Reset ADE auxiliary fields
        if self._ade_initialized:
            self._reset_ade_fields()

    def get_sample_rate(self) -> float:
        """Get effective sample rate for probe recordings.

        Returns:
            Sample rate in Hz (1/dt)
        """
        return 1.0 / self.dt

    def get_frequency_response(
        self, probe_name: str, n_fft: int | None = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute frequency response from probe data.

        Args:
            probe_name: Name of probe to analyze
            n_fft: FFT size (default: next power of 2 >= data length)

        Returns:
            Tuple of (frequencies, magnitude) arrays
        """
        data = self._probes[probe_name].get_data()
        if n_fft is None:
            n_fft = int(2 ** np.ceil(np.log2(len(data))))

        spectrum = np.fft.rfft(data, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, self.dt)
        magnitude = np.abs(spectrum)

        return freqs, magnitude

    # =========================================================================
    # ADE Material System
    # =========================================================================

    def register_material(self, material, material_id: int | None = None) -> int:
        """Register an acoustic material for use in the simulation.

        Materials must be registered before they can be assigned to grid
        regions. Each material gets a unique ID (1-255).

        Args:
            material: AcousticMaterial instance
            material_id: Optional ID (1-255). If None, auto-assigns next available.

        Returns:
            The assigned material ID

        Raises:
            ValueError: If material_id is invalid or already in use

        Example:
            >>> from strata_fdtd.materials import FIBERGLASS_48
            >>> mat_id = solver.register_material(FIBERGLASS_48)
            >>> solver.set_material_region(absorber_mask, material_id=mat_id)
        """
        if material_id is None:
            # Auto-assign next available ID
            used_ids = set(self._materials.keys())
            for candidate in range(1, 256):
                if candidate not in used_ids:
                    material_id = candidate
                    break
            else:
                raise ValueError("Maximum of 255 materials reached")

        if not 1 <= material_id <= 255:
            raise ValueError("material_id must be in range 1-255 (0 is reserved for air)")

        if material_id in self._materials:
            raise ValueError(f"Material ID {material_id} is already registered")

        self._materials[material_id] = material
        self._ade_initialized = False  # Need to reinitialize ADE fields

        return material_id

    def set_material_region(
        self,
        mask: NDArray[np.bool_],
        material_id: int,
    ) -> None:
        """Assign a material to a region of the grid.

        Args:
            mask: Boolean array matching solver shape. True cells get the material.
            material_id: ID from register_material()

        Raises:
            ValueError: If mask shape doesn't match or material_id is invalid
        """
        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match solver shape {self.shape}")

        if material_id != 0 and material_id not in self._materials:
            raise ValueError(f"Material ID {material_id} not registered. Use register_material() first.")

        self._material_id[mask] = material_id
        self._ade_initialized = False

    def set_material_box(
        self,
        material_id: int,
        x_range: tuple[int, int],
        y_range: tuple[int, int],
        z_range: tuple[int, int],
    ) -> None:
        """Assign material to a rectangular box region.

        Args:
            material_id: ID from register_material()
            x_range: (x_start, x_end) cell indices
            y_range: (y_start, y_end) cell indices
            z_range: (z_start, z_end) cell indices
        """
        mask = np.zeros(self.shape, dtype=bool)
        mask[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] = True
        self.set_material_region(mask, material_id)

    def get_material_at(self, position: tuple[int, int, int]):
        """Get the material at a grid position.

        Args:
            position: Grid coordinates (i, j, k)

        Returns:
            AcousticMaterial instance, or None if air (material_id=0)
        """
        mat_id = self._material_id[position]
        if mat_id == 0:
            return None
        return self._materials.get(mat_id)

    @property
    def has_materials(self) -> bool:
        """Check if any materials are registered."""
        return len(self._materials) > 0

    @property
    def material_count(self) -> int:
        """Number of registered materials."""
        return len(self._materials)

    def _initialize_ade_fields(self) -> None:
        """Initialize auxiliary fields for all registered materials.

        Called automatically before first step if materials are registered.
        Creates storage for Debye and Lorentz auxiliary variables.
        """
        if self._ade_initialized:
            return

        self._ade_fields.clear()

        for mat_id, material in self._materials.items():
            self._ade_fields[mat_id] = {}

            for pole_idx, pole in enumerate(material.poles):
                # Get mask for cells with this material
                mask = self._material_id == mat_id
                n_cells = np.sum(mask)

                if n_cells == 0:
                    continue

                if pole.is_debye:
                    # Debye: single auxiliary field J
                    self._ade_fields[mat_id][pole_idx] = {
                        "J": np.zeros(self.shape, dtype=np.float32),
                        "type": "debye",
                        "target": pole.target,
                        "coeffs": pole.fdtd_coefficients(self.dt),
                    }
                else:
                    # Lorentz: two auxiliary fields (J and J_prev)
                    self._ade_fields[mat_id][pole_idx] = {
                        "J": np.zeros(self.shape, dtype=np.float32),
                        "J_prev": np.zeros(self.shape, dtype=np.float32),
                        "type": "lorentz",
                        "target": pole.target,
                        "coeffs": pole.fdtd_coefficients(self.dt),
                    }

        # Initialize native ADE data if native kernels are available
        self._ade_native_data = None
        self._ade_J_debye = None
        self._ade_J_lorentz = None
        self._ade_J_lorentz_prev = None
        self._ade_divergence = None

        if self._use_native and _kernels is not None:
            self._initialize_ade_native()

        self._ade_initialized = True

    def _initialize_ade_native(self) -> None:
        """Initialize native ADE data structures.

        Creates ADEMaterialData structure and flat auxiliary arrays
        for efficient native kernel execution.
        """
        if _kernels is None:
            return

        # Create ADEMaterialData structure
        ade_data = _kernels.ADEMaterialData()

        # Get max material ID
        max_mat_id = max(self._materials.keys()) if self._materials else 0

        # Initialize material property arrays (index 0 is unused/air)
        ade_data.rho_inf = [0.0] * (max_mat_id + 1)
        ade_data.K_inf = [0.0] * (max_mat_id + 1)

        # Populate material properties
        for mat_id, material in self._materials.items():
            ade_data.rho_inf[mat_id] = float(material.rho_inf)
            ade_data.K_inf[mat_id] = float(material.K_inf)

        # Count poles and assign field indices
        debye_field_idx = 0
        lorentz_field_idx = 0

        for mat_id, material in self._materials.items():
            for pole in material.poles:
                if pole.is_debye:
                    debye_pole = _kernels.ADEDebyePole()
                    debye_pole.material_id = mat_id
                    debye_pole.target = 0 if pole.target == "density" else 1
                    debye_pole.field_index = debye_field_idx

                    coeffs = _kernels.DebyePoleCoeffs()
                    alpha, beta = pole.fdtd_coefficients(self.dt)
                    coeffs.alpha = float(alpha)
                    coeffs.beta = float(beta)
                    debye_pole.coeffs = coeffs

                    ade_data.debye_poles.append(debye_pole)
                    debye_field_idx += 1
                else:  # Lorentz
                    lorentz_pole = _kernels.ADELorentzPole()
                    lorentz_pole.material_id = mat_id
                    lorentz_pole.target = 0 if pole.target == "density" else 1
                    lorentz_pole.field_index = lorentz_field_idx

                    coeffs = _kernels.LorentzPoleCoeffs()
                    a, b, d = pole.fdtd_coefficients(self.dt)
                    coeffs.a = float(a)
                    coeffs.b = float(b)
                    coeffs.d = float(d)
                    lorentz_pole.coeffs = coeffs

                    ade_data.lorentz_poles.append(lorentz_pole)
                    lorentz_field_idx += 1

        ade_data.n_debye_fields = debye_field_idx
        ade_data.n_lorentz_fields = lorentz_field_idx

        self._ade_native_data = ade_data

        # Allocate flat auxiliary field arrays
        grid_size = self.shape[0] * self.shape[1] * self.shape[2]

        if debye_field_idx > 0:
            self._ade_J_debye = np.zeros(
                debye_field_idx * grid_size, dtype=np.float32
            )
        else:
            self._ade_J_debye = np.zeros(1, dtype=np.float32)  # Dummy

        if lorentz_field_idx > 0:
            self._ade_J_lorentz = np.zeros(
                lorentz_field_idx * grid_size, dtype=np.float32
            )
            self._ade_J_lorentz_prev = np.zeros(
                lorentz_field_idx * grid_size, dtype=np.float32
            )
        else:
            self._ade_J_lorentz = np.zeros(1, dtype=np.float32)  # Dummy
            self._ade_J_lorentz_prev = np.zeros(1, dtype=np.float32)

        # Pre-allocate divergence array
        self._ade_divergence = np.zeros(self.shape, dtype=np.float32)

    def _update_ade_density_poles(self) -> None:
        """Update auxiliary fields for density poles.

        Called before velocity update. Density poles add dispersive
        correction to the velocity update equation.
        """
        # Use native kernels if available
        if (self._ade_native_data is not None and _kernels is not None):
            # Update Debye density poles
            if self._ade_native_data.n_density_debye() > 0:
                _kernels.update_ade_density_debye(
                    self._ade_J_debye,
                    self.p,
                    self._material_id,
                    self._ade_native_data,
                )
            # Update Lorentz density poles
            if self._ade_native_data.n_density_lorentz() > 0:
                _kernels.update_ade_density_lorentz(
                    self._ade_J_lorentz,
                    self._ade_J_lorentz_prev,
                    self.p,
                    self._material_id,
                    self._ade_native_data,
                )
            return

        # Fallback: Python implementation
        for mat_id, poles_data in self._ade_fields.items():
            mask = self._material_id == mat_id

            for _pole_idx, pole_data in poles_data.items():
                if pole_data["target"] != "density":
                    continue

                # Source term is pressure gradient (computed during velocity update)
                # For now, use pressure as proxy
                source = self.p

                if pole_data["type"] == "debye":
                    alpha, beta = pole_data["coeffs"]
                    J = pole_data["J"]

                    # J^{n+1} = alpha * J^n + beta * source
                    J[mask] = alpha * J[mask] + beta * source[mask]

                else:  # lorentz
                    a, b, d = pole_data["coeffs"]
                    J = pole_data["J"]
                    J_prev = pole_data["J_prev"]

                    # J^{n+1} = a*J^n + b*J^{n-1} + d*source
                    J_new = a * J[mask] + b * J_prev[mask] + d * source[mask]

                    # Shift history
                    J_prev[mask] = J[mask]
                    J[mask] = J_new

    def _update_ade_modulus_poles(self) -> None:
        """Update auxiliary fields for modulus poles.

        Called before pressure update. Modulus poles add dispersive
        correction to the pressure update equation.
        """
        # Use native kernels if available
        if (self._ade_native_data is not None and _kernels is not None):
            # Compute divergence using native kernel
            if self._grid.is_uniform:
                _kernels.compute_divergence(
                    self._ade_divergence,
                    self.vx,
                    self.vy,
                    self.vz,
                    1.0 / self.dx,
                )
            else:
                _kernels.compute_divergence_nonuniform(
                    self._ade_divergence,
                    self.vx,
                    self.vy,
                    self.vz,
                    self._native_grid_data,
                )
            # Update Debye modulus poles
            if self._ade_native_data.n_modulus_debye() > 0:
                _kernels.update_ade_modulus_debye(
                    self._ade_J_debye,
                    self._ade_divergence,
                    self._material_id,
                    self._ade_native_data,
                )
            # Update Lorentz modulus poles
            if self._ade_native_data.n_modulus_lorentz() > 0:
                _kernels.update_ade_modulus_lorentz(
                    self._ade_J_lorentz,
                    self._ade_J_lorentz_prev,
                    self._ade_divergence,
                    self._material_id,
                    self._ade_native_data,
                )
            return

        # Fallback: Python implementation
        for mat_id, poles_data in self._ade_fields.items():
            mask = self._material_id == mat_id

            for _pole_idx, pole_data in poles_data.items():
                if pole_data["target"] != "modulus":
                    continue

                # Source term is velocity divergence (computed during pressure update)
                # Use divergence computed in pressure update
                source = self._compute_divergence()

                if pole_data["type"] == "debye":
                    alpha, beta = pole_data["coeffs"]
                    J = pole_data["J"]

                    J[mask] = alpha * J[mask] + beta * source[mask]

                else:  # lorentz
                    a, b, d = pole_data["coeffs"]
                    J = pole_data["J"]
                    J_prev = pole_data["J_prev"]

                    J_new = a * J[mask] + b * J_prev[mask] + d * source[mask]
                    J_prev[mask] = J[mask]
                    J[mask] = J_new

    def _compute_divergence(self) -> NDArray[np.floating]:
        """Compute velocity divergence for ADE pressure update.

        For uniform grids, uses scalar 1/dx for all cells.
        For nonuniform grids, uses per-cell inverse spacing arrays.
        """
        nx, ny, nz = self.shape

        if self._grid.is_uniform:
            # Uniform grid: use efficient scalar division
            div = np.zeros(self.shape, dtype=np.float32)

            # x-divergence
            div[0, :, :] = self.vx[0, :, :]
            div[1:, :, :] = self.vx[1:nx, :, :] - self.vx[:nx-1, :, :]

            # y-divergence
            div[:, 0, :] += self.vy[:, 0, :]
            div[:, 1:, :] += self.vy[:, 1:ny, :] - self.vy[:, :ny-1, :]

            # z-divergence
            div[:, :, 0] += self.vz[:, :, 0]
            div[:, :, 1:] += self.vz[:, :, 1:nz] - self.vz[:, :, :nz-1]

            return div / self.dx
        else:
            # Nonuniform grid: use per-cell spacing
            sa = self._spacing_arrays

            # x-divergence with per-cell spacing
            div_x = np.zeros(self.shape, dtype=np.float32)
            div_x[0, :, :] = self.vx[0, :, :] * sa["inv_dx_cell"][0]
            for i in range(1, nx):
                div_x[i, :, :] = (
                    self.vx[i, :, :] - self.vx[i - 1, :, :]
                ) * sa["inv_dx_cell"][i]

            # y-divergence with per-cell spacing
            div_y = np.zeros(self.shape, dtype=np.float32)
            div_y[:, 0, :] = self.vy[:, 0, :] * sa["inv_dy_cell"][0]
            for j in range(1, ny):
                div_y[:, j, :] = (
                    self.vy[:, j, :] - self.vy[:, j - 1, :]
                ) * sa["inv_dy_cell"][j]

            # z-divergence with per-cell spacing
            div_z = np.zeros(self.shape, dtype=np.float32)
            div_z[:, :, 0] = self.vz[:, :, 0] * sa["inv_dz_cell"][0]
            for k in range(1, nz):
                div_z[:, :, k] = (
                    self.vz[:, :, k] - self.vz[:, :, k - 1]
                ) * sa["inv_dz_cell"][k]

            return div_x + div_y + div_z

    def _apply_ade_velocity_correction(self) -> None:
        """Apply ADE corrections to velocity fields using gradient of J.

        For density dispersion in acoustic media, the auxiliary field J
        represents a polarization-like term that modifies the effective
        density. The velocity correction requires computing the gradient
        of J at each velocity component location.

        The correction is:
            vx[i] += -dt/ρ_inf * (J[i+1] - J[i]) / dx  # ∂J/∂x at face i
            vy[j] += -dt/ρ_inf * (J[j+1] - J[j]) / dy  # ∂J/∂y at face j
            vz[k] += -dt/ρ_inf * (J[k+1] - J[k]) / dz  # ∂J/∂z at face k

        This is analogous to how pressure gradients drive velocity updates
        in the standard FDTD update, using the same staggered grid stencil.

        For nonuniform grids, the spacing arrays (inv_dx_face, etc.) provide
        per-face metric scaling.
        """
        # Use native kernels if available
        if (self._ade_native_data is not None and _kernels is not None):
            has_density_poles = (
                self._ade_native_data.n_density_debye() > 0 or
                self._ade_native_data.n_density_lorentz() > 0
            )
            if has_density_poles:
                _kernels.apply_ade_velocity_correction(
                    self.vx,
                    self.vy,
                    self.vz,
                    self._ade_J_debye,
                    self._ade_J_lorentz,
                    self._material_id,
                    self._ade_native_data,
                    self.dt,
                    1.0 / self.dx,  # inv_dx for gradient computation
                )
            return

        # Fallback: Python implementation
        for mat_id, poles_data in self._ade_fields.items():
            material = self._materials[mat_id]
            mask = self._material_id == mat_id

            for pole_data in poles_data.values():
                if pole_data["target"] != "density":
                    continue

                J = pole_data["J"]

                # Base correction coefficient (without spatial derivative)
                # For density dispersion: adds to -∂p/∂x term
                coeff_base = -self.dt / material.rho_inf

                if self._grid.is_uniform:
                    # Uniform grid: gradient is (J[i+1] - J[i]) / dx
                    # Combined coefficient includes 1/dx
                    coeff = coeff_base / self.dx

                    # Compute gradient of J at velocity face locations
                    # dJ_dx at face i (between cells i and i+1)
                    dJ_dx = J[1:, :, :] - J[:-1, :, :]
                    # dJ_dy at face j
                    dJ_dy = J[:, 1:, :] - J[:, :-1, :]
                    # dJ_dz at face k
                    dJ_dz = J[:, :, 1:] - J[:, :, :-1]

                    # Create masks for velocity face locations
                    # vx[i] is at face between cells (i, j, k) and (i+1, j, k)
                    # Apply correction only if both adjacent cells are in material
                    mask_vx = mask[:-1, :, :] & mask[1:, :, :]
                    mask_vy = mask[:, :-1, :] & mask[:, 1:, :]
                    mask_vz = mask[:, :, :-1] & mask[:, :, 1:]

                    # Apply gradient-based correction
                    self.vx[:-1, :, :][mask_vx] += coeff * dJ_dx[mask_vx]
                    self.vy[:, :-1, :][mask_vy] += coeff * dJ_dy[mask_vy]
                    self.vz[:, :, :-1][mask_vz] += coeff * dJ_dz[mask_vz]

                else:
                    # Nonuniform grid: use per-face spacing arrays
                    sa = self._spacing_arrays

                    # Compute gradient of J with per-face spacing
                    dJ_dx = J[1:, :, :] - J[:-1, :, :]
                    dJ_dy = J[:, 1:, :] - J[:, :-1, :]
                    dJ_dz = J[:, :, 1:] - J[:, :, :-1]

                    # Create masks for velocity face locations
                    mask_vx = mask[:-1, :, :] & mask[1:, :, :]
                    mask_vy = mask[:, :-1, :] & mask[:, 1:, :]
                    mask_vz = mask[:, :, :-1] & mask[:, :, 1:]

                    # Apply correction with per-face metric scaling
                    # vx: coeff_base * inv_dx_face[i] * dJ_dx
                    correction_vx = (
                        coeff_base
                        * sa["inv_dx_face"][:, np.newaxis, np.newaxis]
                        * dJ_dx
                    )
                    self.vx[:-1, :, :][mask_vx] += correction_vx[mask_vx]

                    # vy: coeff_base * inv_dy_face[j] * dJ_dy
                    correction_vy = (
                        coeff_base
                        * sa["inv_dy_face"][np.newaxis, :, np.newaxis]
                        * dJ_dy
                    )
                    self.vy[:, :-1, :][mask_vy] += correction_vy[mask_vy]

                    # vz: coeff_base * inv_dz_face[k] * dJ_dz
                    correction_vz = (
                        coeff_base
                        * sa["inv_dz_face"][np.newaxis, np.newaxis, :]
                        * dJ_dz
                    )
                    self.vz[:, :, :-1][mask_vz] += correction_vz[mask_vz]

    def _apply_ade_pressure_correction(self) -> None:
        """Apply ADE corrections to pressure field.

        Adds dispersive contribution from modulus poles:
        p += -K_inf * dt * Σ dJ/dt (for modulus poles)
        """
        # Use native kernels if available
        if (self._ade_native_data is not None and _kernels is not None):
            has_modulus_poles = (
                self._ade_native_data.n_modulus_debye() > 0 or
                self._ade_native_data.n_modulus_lorentz() > 0
            )
            if has_modulus_poles:
                _kernels.apply_ade_pressure_correction(
                    self.p,
                    self._ade_J_debye,
                    self._ade_J_lorentz,
                    self._material_id,
                    self._ade_native_data,
                    self.dt,
                )
            return

        # Fallback: Python implementation
        for mat_id, poles_data in self._ade_fields.items():
            material = self._materials[mat_id]
            mask = self._material_id == mat_id

            for pole_data in poles_data.values():
                if pole_data["target"] != "modulus":
                    continue

                J = pole_data["J"]

                # Correction coefficient
                coeff = -material.K_inf * self.dt

                # Apply to pressure at material cells
                self.p[mask] += coeff * J[mask]

    def _reset_ade_fields(self) -> None:
        """Reset all ADE auxiliary fields to zero."""
        # Reset native arrays
        if self._ade_J_debye is not None:
            self._ade_J_debye.fill(0)
        if self._ade_J_lorentz is not None:
            self._ade_J_lorentz.fill(0)
        if self._ade_J_lorentz_prev is not None:
            self._ade_J_lorentz_prev.fill(0)

        # Reset Python arrays
        for poles_data in self._ade_fields.values():
            for pole_data in poles_data.values():
                pole_data["J"].fill(0)
                if "J_prev" in pole_data:
                    pole_data["J_prev"].fill(0)

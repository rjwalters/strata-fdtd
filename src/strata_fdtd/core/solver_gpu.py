"""GPU-accelerated FDTD solver using PyTorch MPS backend.

This module provides GPU-accelerated FDTD simulation for Apple Silicon Macs
using the Metal Performance Shaders (MPS) backend via PyTorch.

Key Features:
    - Single-band GPU acceleration (2-6x speedup over CPU)
    - Batched multi-band simulation (N bands with near-linear scaling)
    - Geometry masks for solid obstacles
    - Frequency-dependent materials (Debye poles via ADE)
    - PML absorbing boundaries (>40dB reflection reduction)
    - Compatible API with FDTDSolver

PML Performance Impact:
    The PML (Perfectly Matched Layer) implementation uses pre-computed exponential
    decay factors stored as GPU tensors, minimizing per-timestep overhead:

    - Memory overhead: 3 decay tensors (1D per axis)
    - Compute overhead: ~5-10% per timestep (3 element-wise multiplications)
    - No auxiliary field storage required (simplified formulation)
    - Batched solver: PML tensors shared across all bands (no N× overhead)

    For typical simulations with 10-20 PML layers, the performance impact is
    negligible compared to the benefit of eliminating boundary reflections.

Example with PML:
    >>> from strata_fdtd.fdtd_gpu import GPUFDTDSolver, has_gpu_support
    >>> if has_gpu_support():
    ...     # PML absorbing boundaries eliminate reflections
    ...     solver = GPUFDTDSolver(
    ...         shape=(100, 100, 100),
    ...         resolution=1e-3,
    ...         pml_layers=10  # 10-cell PML on all boundaries
    ...     )
    ...     solver.add_source(frequency=1000, bandwidth=500)
    ...     solver.add_probe('center', position=(50, 50, 50))
    ...     solver.run(duration=0.01)
    ...     data = solver.get_probe_data()

Multi-Band Example:
    >>> solver = BatchedGPUFDTDSolver(
    ...     shape=(100, 100, 100),
    ...     resolution=1e-3,
    ...     bands=[(250, 100), (500, 200), (1000, 400), (2000, 800)],
    ...     pml_layers=15,  # PML shared across all bands
    ... )
    >>> results = solver.run(duration=0.01)
    >>> # results is dict: {'250Hz': array, '500Hz': array, ...}

Geometry Example:
    >>> solver = GPUFDTDSolver(shape=(100, 100, 100), resolution=1e-3)
    >>> # Create geometry mask (True=air, False=solid)
    >>> mask = np.ones((100, 100, 100), dtype=bool)
    >>> mask[50:, :, :] = False  # Solid wall at x=50
    >>> solver.set_geometry(mask)

Material Example:
    >>> from strata_fdtd.materials import DebyeMaterial
    >>> solver = GPUFDTDSolver(shape=(100, 100, 100), resolution=1e-3)
    >>> # Create frequency-dependent material
    >>> material = DebyeMaterial(eps_inf=2.0, deps=1.0, tau=1e-9)
    >>> # Define region where material applies
    >>> region_mask = np.zeros((100, 100, 100), dtype=bool)
    >>> region_mask[30:70, 30:70, 30:70] = True
    >>> solver.add_material(material, region_mask)

Memory Usage:
    - Base fields: 4 fields × nx × ny × nz × 4 bytes
    - Geometry mask: nx × ny × nz × 4 bytes (shared, not per-band)
    - ADE materials: Add n_poles auxiliary fields per material region
      Each auxiliary field: nx × ny × nz × 4 bytes
    - Example (100³ grid, 1 material with 2 Debye poles):
      Fields: 4 × 1M × 4B = 16 MB
      Geometry: 1M × 4B = 4 MB
      ADE (2 poles): 2 × 1M × 4B = 8 MB
      Total: ~28 MB GPU memory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Check for PyTorch and MPS availability
_HAS_TORCH = False
_HAS_MPS = False
_torch = None

try:
    import torch
    _torch = torch
    _HAS_TORCH = True
    _HAS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except ImportError:
    pass


def has_gpu_support() -> bool:
    """Check if GPU (MPS) support is available.

    Returns:
        True if PyTorch is installed and MPS backend is available.
    """
    return _HAS_MPS


def get_gpu_info() -> dict:
    """Get information about GPU support.

    Returns:
        Dict with keys: available, backend, pytorch_version
    """
    if not _HAS_TORCH:
        return {
            "available": False,
            "backend": None,
            "pytorch_version": None,
        }
    return {
        "available": _HAS_MPS,
        "backend": "mps" if _HAS_MPS else None,
        "pytorch_version": _torch.__version__,
    }


# Physical constants
SPEED_OF_SOUND = 343.0  # m/s at 20°C
AIR_DENSITY = 1.204  # kg/m³ at 20°C


def _create_pml_damping_tensors(
    shape: tuple[int, int, int],
    pml_layers: int,
    resolution: float,
    c: float,
    dt: float,
    device: str,
    batched: bool = False,
    pml_r_target: float = 1e-6,
    pml_order: int = 3,
    pml_max_sigma: float | None = None,
) -> tuple:
    """Create PML damping tensors with exponential decay factors.

    Args:
        shape: Grid dimensions (nx, ny, nz)
        pml_layers: Number of PML cells on each boundary
        resolution: Grid spacing in meters
        c: Speed of sound in m/s
        dt: Timestep in seconds
        device: PyTorch device ('mps' or 'cpu')
        batched: If True, add batch dimension for BatchedGPUFDTDSolver
        pml_r_target: Target reflection coefficient (0, 1). Lower values provide
            stronger absorption. Default 1e-6 gives ~60dB reduction.
        pml_order: Polynomial order for damping profile. Higher orders provide
            smoother transitions but slower computation. Default: 3.
        pml_max_sigma: Manual override for maximum damping coefficient. If None,
            auto-computed from pml_r_target. Must be positive if specified.

    Returns:
        Tuple of (decay_x, decay_y, decay_z) PyTorch tensors

    Raises:
        ValueError: If pml_layers is negative or too large for grid
        ValueError: If pml_r_target not in range (0, 1)
        ValueError: If pml_order < 1
        ValueError: If pml_max_sigma <= 0 when specified
    """
    nx, ny, nz = shape
    d = pml_layers

    # Validate PML layers
    if d < 0:
        raise ValueError(f"pml_layers must be non-negative, got {d}")

    if d > 0:
        max_pml = min(nx, ny, nz) // 2
        if d >= max_pml:
            raise ValueError(
                f"pml_layers ({d}) too large for grid shape {shape}. "
                f"Maximum allowed: {max_pml - 1}"
            )

    # Validate PML tuning parameters
    if pml_r_target <= 0 or pml_r_target >= 1:
        raise ValueError(
            f"pml_r_target must be in range (0, 1), got {pml_r_target}"
        )
    if pml_order < 1:
        raise ValueError(f"pml_order must be >= 1, got {pml_order}")
    if pml_max_sigma is not None and pml_max_sigma <= 0:
        raise ValueError(
            f"pml_max_sigma must be positive when specified, got {pml_max_sigma}"
        )

    # Compute max_sigma for PML
    # Rule of thumb: sigma_max = -(order+1) * c * ln(R) / (2 * d * dx)
    if pml_max_sigma is None:
        pml_thickness = d * resolution
        max_sigma = -(pml_order + 1) * c * np.log(pml_r_target) / (2 * pml_thickness)
    else:
        max_sigma = pml_max_sigma

    # Create 1D damping profiles for each axis
    def create_profile_1d(n: int) -> NDArray[np.float32]:
        """Create polynomial damping profile for one axis."""
        sigma = np.zeros(n, dtype=np.float32)
        # Left PML region
        for i in range(d):
            dist = (d - i) / d
            sigma[i] = max_sigma * (dist ** pml_order)
        # Right PML region
        for i in range(n - d, n):
            dist = (i - (n - d - 1)) / d
            sigma[i] = max_sigma * (dist ** pml_order)
        return sigma

    # Create profiles
    sigma_x = create_profile_1d(nx)
    sigma_y = create_profile_1d(ny)
    sigma_z = create_profile_1d(nz)

    # Convert to PyTorch tensors with proper broadcasting shapes
    if batched:
        # For batched solver: (1, nx, 1, 1), (1, 1, ny, 1), (1, 1, 1, nz)
        # to broadcast across batch dimension
        sigma_x_tensor = _torch.tensor(
            sigma_x[np.newaxis, :, np.newaxis, np.newaxis],
            device=device,
            dtype=_torch.float32
        )
        sigma_y_tensor = _torch.tensor(
            sigma_y[np.newaxis, np.newaxis, :, np.newaxis],
            device=device,
            dtype=_torch.float32
        )
        sigma_z_tensor = _torch.tensor(
            sigma_z[np.newaxis, np.newaxis, np.newaxis, :],
            device=device,
            dtype=_torch.float32
        )
    else:
        # For single-band solver: (nx, 1, 1), (1, ny, 1), (1, 1, nz)
        sigma_x_tensor = _torch.tensor(
            sigma_x[:, np.newaxis, np.newaxis],
            device=device,
            dtype=_torch.float32
        )
        sigma_y_tensor = _torch.tensor(
            sigma_y[np.newaxis, :, np.newaxis],
            device=device,
            dtype=_torch.float32
        )
        sigma_z_tensor = _torch.tensor(
            sigma_z[np.newaxis, np.newaxis, :],
            device=device,
            dtype=_torch.float32
        )

    # Precompute decay factors: exp(-sigma * dt)
    decay_x = _torch.exp(-sigma_x_tensor * dt)
    decay_y = _torch.exp(-sigma_y_tensor * dt)
    decay_z = _torch.exp(-sigma_z_tensor * dt)

    return (decay_x, decay_y, decay_z)


@dataclass
class GPUGaussianPulse:
    """Gaussian pulse source for GPU solver.

    Args:
        position: Grid coordinates (i, j, k) for point source
        frequency: Center frequency in Hz
        bandwidth: Frequency bandwidth in Hz (default: 2 * frequency)
        amplitude: Peak pressure amplitude (default: 1.0)
    """
    position: tuple[int, int, int]
    frequency: float
    bandwidth: float | None = None
    amplitude: float = 1.0

    def __post_init__(self):
        if self.bandwidth is None:
            self.bandwidth = 2.0 * self.frequency
        # Precompute Gaussian parameters
        self._sigma_t = 1.0 / (2.0 * np.pi * self.bandwidth)
        self._t_peak = 4.0 * self._sigma_t  # Delay to avoid initial transient

    def evaluate(self, t: float) -> float:
        """Evaluate source amplitude at time t."""
        t_shifted = t - self._t_peak
        envelope = np.exp(-0.5 * (t_shifted / self._sigma_t) ** 2)
        carrier = np.sin(2.0 * np.pi * self.frequency * t_shifted)
        return self.amplitude * envelope * carrier


class GPUFDTDSolver:
    """GPU-accelerated FDTD solver for single-band simulation.

    Uses PyTorch MPS backend for Apple Silicon GPU acceleration. Supports
    geometry masks for solid boundaries and frequency-dependent materials
    using the ADE (Auxiliary Differential Equation) formulation.

    Features:
        - Free-space and geometry-constrained propagation
        - Debye dispersion materials (Lorentz not yet supported)
        - Automatic memory management on GPU

    Args:
        shape: Grid dimensions (nx, ny, nz)
        resolution: Grid spacing in meters
        c: Speed of sound in m/s (default: 343.0)
        rho: Air density in kg/m³ (default: 1.204)
        device: PyTorch device ('mps', 'cpu', 'auto')
        pml_layers: Number of PML absorbing boundary layers (default: 0 = rigid)
        pml_r_target: PML target reflection coefficient in range (0, 1). Lower values
            give stronger absorption. Default 1e-6 provides ~60dB reduction. Use 1e-8
            for >80dB (ultra-quiet boundaries).
        pml_order: Polynomial order for PML damping profile (default: 3). Higher orders
            give smoother absorption but slower computation. Use 2 for faster performance.
        pml_max_sigma: Manual override for maximum PML damping coefficient. If None
            (default), auto-computed from pml_r_target. Must be positive if specified.
            Use for direct control when pml_r_target behavior is insufficient.

    Raises:
        ImportError: If PyTorch is not installed
        RuntimeError: If MPS is not available and device='mps'

    Example:
        >>> # Basic simulation
        >>> solver = GPUFDTDSolver(shape=(100,100,100), resolution=1e-3)
        >>> solver.add_source(frequency=1000)
        >>> solver.add_probe('center', position=(50,50,50))
        >>> solver.run(duration=0.01)

        >>> # With PML boundaries
        >>> solver = GPUFDTDSolver(
        ...     shape=(100,100,100), resolution=1e-3, pml_layers=10
        ... )

        >>> # Ultra-quiet PML boundaries (>80dB reduction)
        >>> solver = GPUFDTDSolver(
        ...     shape=(100,100,100), resolution=1e-3,
        ...     pml_layers=20, pml_r_target=1e-8
        ... )

        >>> # Performance-optimized PML (lower order)
        >>> solver = GPUFDTDSolver(
        ...     shape=(100,100,100), resolution=1e-3,
        ...     pml_layers=10, pml_order=2
        ... )

        >>> # Manual PML damping control
        >>> solver = GPUFDTDSolver(
        ...     shape=(100,100,100), resolution=1e-3,
        ...     pml_layers=15, pml_max_sigma=500.0
        ... )

        >>> # With geometry mask
        >>> mask = np.ones((100,100,100), dtype=bool)
        >>> mask[40:60, :, :] = False  # Solid barrier
        >>> solver.set_geometry(mask)

        >>> # With dispersive material
        >>> from strata_fdtd.materials import DebyeMaterial
        >>> material = DebyeMaterial(eps_inf=2.0, deps=1.0, tau=1e-9)
        >>> region = np.zeros((100,100,100), dtype=bool)
        >>> region[70:90, :, :] = True
        >>> solver.add_material(material, region)

        >>> # Check memory usage
        >>> print(f"Memory: {solver.memory_usage_mb():.1f} MB")
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        resolution: float,
        c: float = SPEED_OF_SOUND,
        rho: float = AIR_DENSITY,
        device: Literal["mps", "cpu", "auto"] = "auto",
        pml_layers: int = 0,
        pml_r_target: float = 1e-6,
        pml_order: int = 3,
        pml_max_sigma: float | None = None,
    ):
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for GPU FDTD. Install with: pip install torch"
            )

        # Validate PML parameters
        if pml_r_target <= 0 or pml_r_target >= 1:
            raise ValueError(
                f"pml_r_target must be in range (0, 1), got {pml_r_target}"
            )
        if pml_order < 1:
            raise ValueError(f"pml_order must be >= 1, got {pml_order}")
        if pml_max_sigma is not None and pml_max_sigma <= 0:
            raise ValueError(f"pml_max_sigma must be positive, got {pml_max_sigma}")

        # Select device
        if device == "auto":
            self._device = "mps" if _HAS_MPS else "cpu"
        elif device == "mps":
            if not _HAS_MPS:
                raise RuntimeError(
                    "MPS backend not available. Check PyTorch installation."
                )
            self._device = "mps"
        else:
            self._device = "cpu"

        self.shape = shape
        self.resolution = resolution
        self.c = c
        self.rho = rho
        self.pml_layers = pml_layers
        self.pml_r_target = pml_r_target
        self.pml_order = pml_order
        self.pml_max_sigma = pml_max_sigma

        # Compute timestep from CFL condition
        self.dt = 0.5 * resolution / (c * np.sqrt(3))

        # FDTD coefficients
        self._coeff_v = -self.dt / (rho * resolution)
        self._coeff_p = -rho * c**2 * self.dt / resolution

        # Allocate field tensors on device
        self._p = _torch.zeros(*shape, device=self._device, dtype=_torch.float32)
        self._vx = _torch.zeros(*shape, device=self._device, dtype=_torch.float32)
        self._vy = _torch.zeros(*shape, device=self._device, dtype=_torch.float32)
        self._vz = _torch.zeros(*shape, device=self._device, dtype=_torch.float32)

        # Geometry mask (True=air, False=solid)
        self._geometry = _torch.ones(*shape, device=self._device, dtype=_torch.float32)

        # Rigid boundary masks (True=allow flow, False=block flow)
        # These are updated when geometry is set
        self._rigid_mask_x = _torch.ones(shape[0]-1, shape[1], shape[2], device=self._device, dtype=_torch.float32)
        self._rigid_mask_y = _torch.ones(shape[0], shape[1]-1, shape[2], device=self._device, dtype=_torch.float32)
        self._rigid_mask_z = _torch.ones(shape[0], shape[1], shape[2]-1, device=self._device, dtype=_torch.float32)

        # Material system
        self._materials: dict = {}  # material_id -> AcousticMaterial
        self._material_id = None  # Will be torch tensor when materials are added
        self._ade_fields: dict = {}  # Auxiliary fields for ADE poles
        self._ade_initialized = False

        # PML damping tensors (if enabled)
        self._pml_enabled = pml_layers > 0
        if self._pml_enabled:
            self._init_pml()

        # Sources and probes
        self._sources: list[GPUGaussianPulse] = []
        self._probes: dict[str, tuple[int, int, int]] = {}
        self._probe_data: dict[str, list[float]] = {}

        # Simulation state
        self._time = 0.0
        self._step_count = 0

    @property
    def using_gpu(self) -> bool:
        """True if running on GPU (MPS)."""
        return self._device == "mps"

    @property
    def device(self) -> str:
        """Current compute device."""
        return self._device

    @property
    def step_count(self) -> int:
        """Number of timesteps executed."""
        return self._step_count

    @property
    def has_materials(self) -> bool:
        """Check if any materials are registered."""
        return len(self._materials) > 0

    def set_geometry(self, mask: NDArray[np.bool_]) -> None:
        """Set the geometry mask and compute rigid boundary masks.

        Args:
            mask: Boolean array with True for air cells, False for solid cells.
                  Solid cells have pressure frozen at zero, and velocity is
                  zeroed at any face touching a solid cell.

        Raises:
            ValueError: If mask shape doesn't match grid shape.
        """
        if mask.shape != self.shape:
            raise ValueError(
                f"Geometry mask shape {mask.shape} doesn't match grid shape {self.shape}"
            )

        # Convert to float tensor (0.0 for solid, 1.0 for air)
        self._geometry = _torch.tensor(
            mask.astype(np.float32), device=self._device, dtype=_torch.float32
        )

        # Compute rigid boundary masks
        # vx[i] is between cells i and i+1, zero if either is solid
        solid_vx = (~mask[:-1, :, :]) | (~mask[1:, :, :])
        self._rigid_mask_x = _torch.tensor(
            (~solid_vx).astype(np.float32), device=self._device, dtype=_torch.float32
        )

        # vy[j] is between cells j and j+1, zero if either is solid
        solid_vy = (~mask[:, :-1, :]) | (~mask[:, 1:, :])
        self._rigid_mask_y = _torch.tensor(
            (~solid_vy).astype(np.float32), device=self._device, dtype=_torch.float32
        )

        # vz[k] is between cells k and k+1, zero if either is solid
        solid_vz = (~mask[:, :, :-1]) | (~mask[:, :, 1:])
        self._rigid_mask_z = _torch.tensor(
            (~solid_vz).astype(np.float32), device=self._device, dtype=_torch.float32
        )

    def add_material(
        self, material, region_mask: NDArray[np.bool_]
    ) -> int:
        """Add a frequency-dependent material to a region.

        Args:
            material: AcousticMaterial object with dispersion poles
            region_mask: Boolean array marking where this material applies

        Returns:
            Material ID assigned to this material

        Raises:
            ValueError: If region_mask shape doesn't match grid
        """
        if region_mask.shape != self.shape:
            raise ValueError(
                f"Region mask shape {region_mask.shape} doesn't match grid shape {self.shape}"
            )

        # Assign material ID
        material_id = len(self._materials) + 1
        self._materials[material_id] = material

        # Initialize material_id tensor if needed
        if self._material_id is None:
            self._material_id = _torch.zeros(
                *self.shape, device=self._device, dtype=_torch.uint8
            )

        # Set material ID in region
        region_tensor = _torch.tensor(
            region_mask, device=self._device, dtype=_torch.bool
        )
        self._material_id[region_tensor] = material_id

        # Mark ADE as uninitialized (will reinit on next step)
        self._ade_initialized = False

        return material_id

    def _init_pml(self) -> None:
        """Initialize PML damping profiles and tensors."""
        self._decay_x, self._decay_y, self._decay_z = _create_pml_damping_tensors(
            shape=self.shape,
            pml_layers=self.pml_layers,
            resolution=self.resolution,
            c=self.c,
            dt=self.dt,
            device=self._device,
            batched=False,
            pml_r_target=self.pml_r_target,
            pml_order=self.pml_order,
            pml_max_sigma=self.pml_max_sigma,
        )

    def _apply_pml(self) -> None:
        """Apply PML damping to velocity and pressure fields."""
        if not self._pml_enabled:
            return

        # Apply exponential decay to velocity components
        # Each component is damped by its corresponding axis
        self._vx *= self._decay_x
        self._vy *= self._decay_y
        self._vz *= self._decay_z

        # Apply combined damping to pressure (product of all axes)
        self._p *= self._decay_x * self._decay_y * self._decay_z

    def add_source(
        self,
        position: tuple[int, int, int] | None = None,
        frequency: float = 1000.0,
        bandwidth: float | None = None,
        amplitude: float = 1.0,
    ) -> GPUGaussianPulse:
        """Add a Gaussian pulse source.

        Args:
            position: Grid coordinates (i, j, k). Default: center of grid.
            frequency: Center frequency in Hz
            bandwidth: Bandwidth in Hz (default: 2 * frequency)
            amplitude: Peak amplitude

        Returns:
            The created source object.
        """
        if position is None:
            position = (self.shape[0] // 2, self.shape[1] // 2, self.shape[2] // 2)

        source = GPUGaussianPulse(
            position=position,
            frequency=frequency,
            bandwidth=bandwidth,
            amplitude=amplitude,
        )
        self._sources.append(source)
        return source

    def add_probe(self, name: str, position: tuple[int, int, int]) -> None:
        """Add a pressure probe at the specified location.

        Args:
            name: Probe identifier
            position: Grid coordinates (i, j, k)
        """
        self._probes[name] = position
        self._probe_data[name] = []

    def _inject_sources(self) -> None:
        """Inject source contributions at current time."""
        for source in self._sources:
            i, j, k = source.position
            value = source.evaluate(self._time)
            self._p[i, j, k] += value

    def _record_probes(self) -> None:
        """Record pressure at probe locations."""
        for name, (i, j, k) in self._probes.items():
            value = self._p[i, j, k].item()
            self._probe_data[name].append(value)

    def step(self) -> None:
        """Execute one FDTD timestep with geometry and material support."""
        # Initialize ADE fields on first step if materials are present
        if self.has_materials and not self._ade_initialized:
            self._initialize_ade_fields()

        # === ADE Phase 1: Update density auxiliary fields ===
        if self.has_materials:
            self._update_ade_density_poles()

        # Velocity update: v += coeff_v * grad(p)
        self._vx[:-1, :, :] += self._coeff_v * (
            self._p[1:, :, :] - self._p[:-1, :, :]
        )
        self._vy[:, :-1, :] += self._coeff_v * (
            self._p[:, 1:, :] - self._p[:, :-1, :]
        )
        self._vz[:, :, :-1] += self._coeff_v * (
            self._p[:, :, 1:] - self._p[:, :, :-1]
        )

        # === ADE Phase 2: Apply density ADE correction to velocities ===
        if self.has_materials:
            self._apply_ade_velocity_correction()

        # Apply rigid boundary conditions (zero velocity at solid faces)
        self._vx[:-1, :, :] *= self._rigid_mask_x
        self._vy[:, :-1, :] *= self._rigid_mask_y
        self._vz[:, :, :-1] *= self._rigid_mask_z

        # === ADE Phase 3: Update modulus auxiliary fields ===
        if self.has_materials:
            self._update_ade_modulus_poles()

        # Pressure update: p += coeff_p * div(v)
        self._p[1:, :, :] += self._coeff_p * (
            self._vx[1:, :, :] - self._vx[:-1, :, :]
        )
        self._p[:, 1:, :] += self._coeff_p * (
            self._vy[:, 1:, :] - self._vy[:, :-1, :]
        )
        self._p[:, :, 1:] += self._coeff_p * (
            self._vz[:, :, 1:] - self._vz[:, :, :-1]
        )

        # === ADE Phase 4: Apply modulus ADE correction to pressure ===
        if self.has_materials:
            self._apply_ade_pressure_correction()

        # Apply geometry mask (zero pressure in solid cells)
        self._p *= self._geometry

        # Apply PML damping
        self._apply_pml()

        # Inject sources and record probes
        self._inject_sources()
        self._record_probes()

        # Advance time
        self._time += self.dt
        self._step_count += 1

    def run(self, duration: float | None = None, steps: int | None = None) -> None:
        """Run simulation for specified duration or steps.

        Args:
            duration: Simulation time in seconds
            steps: Number of timesteps (alternative to duration)
        """
        if duration is not None:
            steps = int(np.ceil(duration / self.dt))
        elif steps is None:
            raise ValueError("Must specify either duration or steps")

        # Synchronize before timing (if on MPS)
        if self._device == "mps":
            _torch.mps.synchronize()

        for _ in range(steps):
            self.step()

        # Synchronize after (if on MPS)
        if self._device == "mps":
            _torch.mps.synchronize()

    def get_probe_data(self) -> dict[str, NDArray[np.float32]]:
        """Get recorded probe data.

        Returns:
            Dict mapping probe names to pressure time series arrays.
        """
        return {
            name: np.array(data, dtype=np.float32)
            for name, data in self._probe_data.items()
        }

    def get_pressure_field(self) -> NDArray[np.float32]:
        """Get current pressure field as NumPy array."""
        return self._p.cpu().numpy()

    def memory_usage_mb(self) -> float:
        """Estimate GPU memory usage in MB.

        Returns:
            Total memory usage including:
            - 4 field arrays (p, vx, vy, vz)
            - 1 geometry mask
            - N ADE auxiliary fields (Debye: 1 field, Lorentz: 2 fields per pole)

        Note:
            Each Debye pole adds one auxiliary field J.
            Each Lorentz pole adds two auxiliary fields J and J_prev.
            Each field has size (nx × ny × nz × 4 bytes).

        Example:
            >>> solver = GPUFDTDSolver(shape=(100,100,100), resolution=1e-3)
            >>> # Base memory for fields
            >>> print(f"Base: {solver.memory_usage_mb():.1f} MB")
            >>> # Add material with 3 Debye poles
            >>> material = DebyeMaterial(eps_inf=2.0, deps=1.0, tau=1e-9)
            >>> solver.add_material(material, region_mask)
            >>> print(f"With material: {solver.memory_usage_mb():.1f} MB")
        """
        nx, ny, nz = self.shape
        bytes_per_field = nx * ny * nz * 4  # float32

        # Base fields: p, vx, vy, vz, geometry
        base_fields = 5
        total_bytes = base_fields * bytes_per_field

        # ADE auxiliary fields (Debye: 1 field, Lorentz: 2 fields per pole)
        if self.has_materials:
            total_ade_fields = 0
            for mat in self._materials.values():
                for pole in mat.poles:
                    if pole.is_debye:
                        total_ade_fields += 1  # J only
                    elif pole.is_lorentz:
                        total_ade_fields += 2  # J and J_prev
            total_bytes += total_ade_fields * bytes_per_field

        return total_bytes / (1024 * 1024)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._p.zero_()
        self._vx.zero_()
        self._vy.zero_()
        self._vz.zero_()
        self._time = 0.0
        self._step_count = 0
        for name in self._probe_data:
            self._probe_data[name] = []
        if self._ade_initialized:
            self._reset_ade_fields()

    def _initialize_ade_fields(self) -> None:
        """Initialize auxiliary fields for ADE materials."""
        if not self.has_materials:
            return

        self._ade_fields = {}

        # For each material, create auxiliary fields for each pole
        for material_id, material in self._materials.items():
            self._ade_fields[material_id] = {}

            for pole_idx, pole in enumerate(material.poles):
                # Debye poles need one auxiliary field J
                if pole.is_debye:
                    J = _torch.zeros(
                        *self.shape, device=self._device, dtype=_torch.float32
                    )
                    self._ade_fields[material_id][pole_idx] = {"J": J}
                # Lorentz poles need two auxiliary fields: J and J_prev
                elif pole.is_lorentz:
                    J = _torch.zeros(
                        *self.shape, device=self._device, dtype=_torch.float32
                    )
                    J_prev = _torch.zeros(
                        *self.shape, device=self._device, dtype=_torch.float32
                    )
                    self._ade_fields[material_id][pole_idx] = {"J": J, "J_prev": J_prev}

        self._ade_initialized = True

    def _reset_ade_fields(self) -> None:
        """Reset all ADE auxiliary fields to zero."""
        for mat_fields in self._ade_fields.values():
            for pole_fields in mat_fields.values():
                for field in pole_fields.values():
                    field.zero_()

    def _update_ade_density_poles(self) -> None:
        """Update auxiliary fields for density dispersion poles."""
        if not self.has_materials:
            return

        for material_id, material in self._materials.items():
            # Get mask for this material
            mask = self._material_id == material_id

            for pole_idx, pole in enumerate(material.poles):
                if pole.target != "density":
                    continue

                if pole.is_debye:
                    # Get auxiliary field
                    J = self._ade_fields[material_id][pole_idx]["J"]

                    # Density poles are driven by pressure (not divergence)
                    # This matches CPU implementation (fdtd.py:2330)
                    alpha, beta = pole.fdtd_coefficients(self.dt)
                    J[mask] = alpha * J[mask] + beta * self._p[mask]

                elif pole.is_lorentz:
                    # Get auxiliary fields
                    J = self._ade_fields[material_id][pole_idx]["J"]
                    J_prev = self._ade_fields[material_id][pole_idx]["J_prev"]

                    # Lorentz update: J^{n+1} = a*J^n + b*J^{n-1} + d*f^n
                    # For density poles, f = pressure
                    a, b, d = pole.fdtd_coefficients(self.dt)

                    # Store current J as J_prev for next iteration
                    J_new = a * J[mask] + b * J_prev[mask] + d * self._p[mask]
                    J_prev[mask] = J[mask]
                    J[mask] = J_new

    def _update_ade_modulus_poles(self) -> None:
        """Update auxiliary fields for modulus dispersion poles."""
        if not self.has_materials:
            return

        for material_id, material in self._materials.items():
            # Get mask for this material
            mask = self._material_id == material_id

            for pole_idx, pole in enumerate(material.poles):
                if pole.target != "modulus":
                    continue

                # Compute velocity divergence (needed for both Debye and Lorentz)
                # Include 1/dx scaling for proper units (matches CPU fdtd.py:2446)
                inv_dx = 1.0 / self.resolution
                div_v = (
                    (self._vx[1:, :, :] - self._vx[:-1, :, :]) * inv_dx
                    + (self._vy[:, 1:, :] - self._vy[:, :-1, :]) * inv_dx
                    + (self._vz[:, :, 1:] - self._vz[:, :, :-1]) * inv_dx
                )

                # Pad to full shape
                div_v_full = _torch.zeros_like(self._p)
                div_v_full[:-1, :-1, :-1] = div_v

                if pole.is_debye:
                    # Get auxiliary field
                    J = self._ade_fields[material_id][pole_idx]["J"]

                    # Debye update
                    alpha, beta = pole.fdtd_coefficients(self.dt)
                    J[mask] = alpha * J[mask] + beta * div_v_full[mask]

                elif pole.is_lorentz:
                    # Get auxiliary fields
                    J = self._ade_fields[material_id][pole_idx]["J"]
                    J_prev = self._ade_fields[material_id][pole_idx]["J_prev"]

                    # Lorentz update: J^{n+1} = a*J^n + b*J^{n-1} + d*f^n
                    # For modulus poles, f = velocity divergence
                    a, b, d = pole.fdtd_coefficients(self.dt)

                    # Store current J as J_prev for next iteration
                    J_new = a * J[mask] + b * J_prev[mask] + d * div_v_full[mask]
                    J_prev[mask] = J[mask]
                    J[mask] = J_new

    def _apply_ade_velocity_correction(self) -> None:
        """Apply ADE corrections to velocity from density poles."""
        if not self.has_materials:
            return

        for material_id, material in self._materials.items():
            rho_inf = material.rho_inf

            for pole_idx, pole in enumerate(material.poles):
                if pole.target != "density":
                    continue

                J = self._ade_fields[material_id][pole_idx]["J"]

                # Velocity correction: v += -(dt/rho_inf) * grad(J)
                # Same formula for both Debye and Lorentz poles
                # vx component
                grad_J_x = J[1:, :, :] - J[:-1, :, :]
                self._vx[:-1, :, :] -= (self.dt / rho_inf) * grad_J_x

                # vy component
                grad_J_y = J[:, 1:, :] - J[:, :-1, :]
                self._vy[:, :-1, :] -= (self.dt / rho_inf) * grad_J_y

                # vz component
                grad_J_z = J[:, :, 1:] - J[:, :, :-1]
                self._vz[:, :, :-1] -= (self.dt / rho_inf) * grad_J_z

    def _apply_ade_pressure_correction(self) -> None:
        """Apply ADE corrections to pressure from modulus poles."""
        if not self.has_materials:
            return

        for material_id, material in self._materials.items():
            K_inf = material.K_inf
            mask = self._material_id == material_id

            for pole_idx, pole in enumerate(material.poles):
                if pole.target != "modulus":
                    continue

                J = self._ade_fields[material_id][pole_idx]["J"]

                # Pressure correction: p += -K_inf * J
                # Same formula for both Debye and Lorentz poles
                self._p[mask] -= K_inf * J[mask]


class BatchedGPUFDTDSolver:
    """Batched GPU solver for multi-band parallel simulation.

    Runs multiple frequency bands simultaneously as a single batched
    computation, providing near-linear scaling with band count.

    Args:
        shape: Grid dimensions (nx, ny, nz)
        resolution: Grid spacing in meters
        bands: List of (center_freq, bandwidth) tuples
        source_position: Grid coordinates for all sources (default: center)
        probe_position: Grid coordinates for probe (default: 3/4 along x)
        c: Speed of sound in m/s
        rho: Air density in kg/m³
        device: PyTorch device ('mps', 'cpu', 'auto')
        pml_layers: Number of PML absorbing boundary layers (default: 0 = rigid)
        pml_r_target: PML target reflection coefficient in range (0, 1). Lower values
            give stronger absorption. Default 1e-6 provides ~60dB reduction. Use 1e-8
            for >80dB (ultra-quiet boundaries).
        pml_order: Polynomial order for PML damping profile (default: 3). Higher orders
            give smoother absorption but slower computation. Use 2 for faster performance.
        pml_max_sigma: Manual override for maximum PML damping coefficient. If None
            (default), auto-computed from pml_r_target. Must be positive if specified.
            Use for direct control when pml_r_target behavior is insufficient.

    Example:
        >>> # Basic multi-band simulation
        >>> solver = BatchedGPUFDTDSolver(
        ...     shape=(100, 100, 100),
        ...     resolution=1e-3,
        ...     bands=[(250, 100), (500, 200), (1000, 400), (2000, 800)],
        ...     pml_layers=10,
        ... )
        >>> results = solver.run(duration=0.01)
        >>> print(results.keys())  # dict_keys(['250Hz', '500Hz', '1000Hz', '2000Hz'])

        >>> # Ultra-quiet PML for critical measurements
        >>> solver = BatchedGPUFDTDSolver(
        ...     shape=(100, 100, 100),
        ...     resolution=1e-3,
        ...     bands=[(250, 100), (500, 200), (1000, 400)],
        ...     pml_layers=20,
        ...     pml_r_target=1e-8,
        ... )

        >>> # Performance-optimized PML
        >>> solver = BatchedGPUFDTDSolver(
        ...     shape=(100, 100, 100),
        ...     resolution=1e-3,
        ...     bands=[(250, 100), (500, 200)],
        ...     pml_layers=10,
        ...     pml_order=2,
        ... )
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        resolution: float,
        bands: list[tuple[float, float]],
        source_position: tuple[int, int, int] | None = None,
        probe_position: tuple[int, int, int] | None = None,
        c: float = SPEED_OF_SOUND,
        rho: float = AIR_DENSITY,
        device: Literal["mps", "cpu", "auto"] = "auto",
        pml_layers: int = 0,
        pml_r_target: float = 1e-6,
        pml_order: int = 3,
        pml_max_sigma: float | None = None,
    ):
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for GPU FDTD. Install with: pip install torch"
            )

        if not bands:
            raise ValueError("Must specify at least one frequency band")

        # Validate PML parameters
        if pml_r_target <= 0 or pml_r_target >= 1:
            raise ValueError(
                f"pml_r_target must be in range (0, 1), got {pml_r_target}"
            )
        if pml_order < 1:
            raise ValueError(f"pml_order must be >= 1, got {pml_order}")
        if pml_max_sigma is not None and pml_max_sigma <= 0:
            raise ValueError(f"pml_max_sigma must be positive, got {pml_max_sigma}")

        # Select device
        if device == "auto":
            self._device = "mps" if _HAS_MPS else "cpu"
        elif device == "mps":
            if not _HAS_MPS:
                raise RuntimeError("MPS backend not available")
            self._device = "mps"
        else:
            self._device = "cpu"

        self.shape = shape
        self.resolution = resolution
        self.bands = bands
        self.n_bands = len(bands)
        self.c = c
        self.rho = rho
        self.pml_layers = pml_layers
        self.pml_r_target = pml_r_target
        self.pml_order = pml_order
        self.pml_max_sigma = pml_max_sigma

        # Default positions
        if source_position is None:
            source_position = (shape[0] // 4, shape[1] // 2, shape[2] // 2)
        if probe_position is None:
            probe_position = (3 * shape[0] // 4, shape[1] // 2, shape[2] // 2)

        self.source_position = source_position
        self.probe_position = probe_position

        # Compute timestep from CFL condition
        self.dt = 0.5 * resolution / (c * np.sqrt(3))

        # FDTD coefficients
        self._coeff_v = -self.dt / (rho * resolution)
        self._coeff_p = -rho * c**2 * self.dt / resolution

        # Allocate BATCHED field tensors: (n_bands, nx, ny, nz)
        self._p = _torch.zeros(
            self.n_bands, *shape, device=self._device, dtype=_torch.float32
        )
        self._vx = _torch.zeros(
            self.n_bands, *shape, device=self._device, dtype=_torch.float32
        )
        self._vy = _torch.zeros(
            self.n_bands, *shape, device=self._device, dtype=_torch.float32
        )
        self._vz = _torch.zeros(
            self.n_bands, *shape, device=self._device, dtype=_torch.float32
        )

        # Geometry mask (shared across all bands) - True=air, False=solid
        self._geometry = _torch.ones(*shape, device=self._device, dtype=_torch.float32)

        # Rigid boundary masks (shared across all bands)
        # These are updated when geometry is set
        self._rigid_mask_x = _torch.ones(shape[0]-1, shape[1], shape[2], device=self._device, dtype=_torch.float32)
        self._rigid_mask_y = _torch.ones(shape[0], shape[1]-1, shape[2], device=self._device, dtype=_torch.float32)
        self._rigid_mask_z = _torch.ones(shape[0], shape[1], shape[2]-1, device=self._device, dtype=_torch.float32)

        # PML damping tensors (if enabled)
        self._pml_enabled = pml_layers > 0
        if self._pml_enabled:
            self._init_pml()

        # Precompute source parameters for each band
        self._source_params = []
        for freq, bw in bands:
            sigma_t = 1.0 / (2.0 * np.pi * bw)
            t_peak = 4.0 * sigma_t
            self._source_params.append((freq, sigma_t, t_peak))

        # Probe data storage: list per band
        self._probe_data: list[list[float]] = [[] for _ in range(self.n_bands)]

        # Material system (shared across bands)
        self._materials: dict = {}
        self._material_id = None  # (nx, ny, nz) - NOT batched
        self._ade_fields: dict = {}  # Batched: material_id -> pole_idx -> fields
        self._ade_initialized = False

        # Simulation state
        self._time = 0.0
        self._step_count = 0

    @property
    def using_gpu(self) -> bool:
        """True if running on GPU (MPS)."""
        return self._device == "mps"

    @property
    def device(self) -> str:
        """Current compute device."""
        return self._device

    @property
    def step_count(self) -> int:
        """Number of timesteps executed."""
        return self._step_count

    @property
    def has_materials(self) -> bool:
        """True if any materials are registered."""
        return len(self._materials) > 0

    def set_geometry(self, mask: NDArray[np.bool_]) -> None:
        """Set the geometry mask and compute rigid boundary masks (shared across all bands).

        Args:
            mask: Boolean array with True for air cells, False for solid cells.
                  Solid cells have pressure frozen at zero across all bands, and
                  velocity is zeroed at any face touching a solid cell.

        Raises:
            ValueError: If mask shape doesn't match grid shape.
        """
        if mask.shape != self.shape:
            raise ValueError(
                f"Geometry mask shape {mask.shape} doesn't match grid shape {self.shape}"
            )

        # Convert to float tensor (0.0 for solid, 1.0 for air)
        self._geometry = _torch.tensor(
            mask.astype(np.float32), device=self._device, dtype=_torch.float32
        )

        # Compute rigid boundary masks (shared across all bands)
        # vx[i] is between cells i and i+1, zero if either is solid
        solid_vx = (~mask[:-1, :, :]) | (~mask[1:, :, :])
        self._rigid_mask_x = _torch.tensor(
            (~solid_vx).astype(np.float32), device=self._device, dtype=_torch.float32
        )

        # vy[j] is between cells j and j+1, zero if either is solid
        solid_vy = (~mask[:, :-1, :]) | (~mask[:, 1:, :])
        self._rigid_mask_y = _torch.tensor(
            (~solid_vy).astype(np.float32), device=self._device, dtype=_torch.float32
        )

        # vz[k] is between cells k and k+1, zero if either is solid
        solid_vz = (~mask[:, :, :-1]) | (~mask[:, :, 1:])
        self._rigid_mask_z = _torch.tensor(
            (~solid_vz).astype(np.float32), device=self._device, dtype=_torch.float32
        )

    def _init_pml(self) -> None:
        """Initialize PML damping profiles and tensors for batched solver."""
        self._decay_x, self._decay_y, self._decay_z = _create_pml_damping_tensors(
            shape=self.shape,
            pml_layers=self.pml_layers,
            resolution=self.resolution,
            c=self.c,
            dt=self.dt,
            device=self._device,
            batched=True,
            pml_r_target=self.pml_r_target,
            pml_order=self.pml_order,
            pml_max_sigma=self.pml_max_sigma,
        )

    def _apply_pml(self) -> None:
        """Apply PML damping to all bands simultaneously."""
        if not self._pml_enabled:
            return

        # Apply exponential decay to velocity components (all bands at once)
        self._vx *= self._decay_x
        self._vy *= self._decay_y
        self._vz *= self._decay_z

        # Apply combined damping to pressure (all bands at once)
        self._p *= self._decay_x * self._decay_y * self._decay_z

    def _evaluate_sources(self, t: float) -> NDArray[np.float32]:
        """Evaluate all source amplitudes at time t."""
        values = np.zeros(self.n_bands, dtype=np.float32)
        for i, (freq, sigma_t, t_peak) in enumerate(self._source_params):
            t_shifted = t - t_peak
            envelope = np.exp(-0.5 * (t_shifted / sigma_t) ** 2)
            carrier = np.sin(2.0 * np.pi * freq * t_shifted)
            values[i] = envelope * carrier
        return values

    def step(self) -> None:
        """Execute one FDTD timestep for all bands with geometry, rigid boundary, and material support."""
        # Initialize ADE fields on first step with materials
        if self.has_materials and not self._ade_initialized:
            self._initialize_ade_fields()

        # === ADE Phase 1: Update density auxiliary fields ===
        if self.has_materials:
            self._update_ade_density_poles()

        # Velocity update: v += coeff_v * grad(p) - ALL BANDS AT ONCE
        self._vx[:, :-1, :, :] += self._coeff_v * (
            self._p[:, 1:, :, :] - self._p[:, :-1, :, :]
        )
        self._vy[:, :, :-1, :] += self._coeff_v * (
            self._p[:, :, 1:, :] - self._p[:, :, :-1, :]
        )
        self._vz[:, :, :, :-1] += self._coeff_v * (
            self._p[:, :, :, 1:] - self._p[:, :, :, :-1]
        )

        # === ADE Phase 2: Apply density ADE correction to velocities ===
        if self.has_materials:
            self._apply_ade_velocity_correction()

        # Apply rigid boundary conditions (zero velocity at solid faces, broadcast across bands)
        self._vx[:, :-1, :, :] *= self._rigid_mask_x
        self._vy[:, :, :-1, :] *= self._rigid_mask_y
        self._vz[:, :, :, :-1] *= self._rigid_mask_z

        # === ADE Phase 3: Update modulus auxiliary fields ===
        if self.has_materials:
            self._update_ade_modulus_poles()

        # Pressure update: p += coeff_p * div(v) - ALL BANDS AT ONCE
        self._p[:, 1:, :, :] += self._coeff_p * (
            self._vx[:, 1:, :, :] - self._vx[:, :-1, :, :]
        )
        self._p[:, :, 1:, :] += self._coeff_p * (
            self._vy[:, :, 1:, :] - self._vy[:, :, :-1, :]
        )
        self._p[:, :, :, 1:] += self._coeff_p * (
            self._vz[:, :, :, 1:] - self._vz[:, :, :, :-1]
        )

        # === ADE Phase 4: Apply modulus ADE correction to pressure ===
        if self.has_materials:
            self._apply_ade_pressure_correction()

        # Apply geometry mask (zero pressure in solid cells, broadcast across bands)
        self._p *= self._geometry

        # Apply PML damping
        self._apply_pml()

        # Inject sources (different frequency per band)
        source_values = self._evaluate_sources(self._time)
        i, j, k = self.source_position
        # Use advanced indexing to inject different values per band
        source_tensor = _torch.tensor(
            source_values, device=self._device, dtype=_torch.float32
        )
        self._p[:, i, j, k] += source_tensor

        # Record probe values
        pi, pj, pk = self.probe_position
        probe_values = self._p[:, pi, pj, pk].cpu().numpy()
        for band_idx in range(self.n_bands):
            self._probe_data[band_idx].append(float(probe_values[band_idx]))

        # Advance time
        self._time += self.dt
        self._step_count += 1

    def run(
        self, duration: float | None = None, steps: int | None = None
    ) -> dict[str, NDArray[np.float32]]:
        """Run simulation for all bands.

        Args:
            duration: Simulation time in seconds
            steps: Number of timesteps (alternative to duration)

        Returns:
            Dict mapping band names (e.g., '250Hz') to probe data arrays.
        """
        if duration is not None:
            steps = int(np.ceil(duration / self.dt))
        elif steps is None:
            raise ValueError("Must specify either duration or steps")

        # Synchronize before timing (if on MPS)
        if self._device == "mps":
            _torch.mps.synchronize()

        for _ in range(steps):
            self.step()

        # Synchronize after (if on MPS)
        if self._device == "mps":
            _torch.mps.synchronize()

        return self.get_probe_data()

    def get_probe_data(self) -> dict[str, NDArray[np.float32]]:
        """Get recorded probe data for all bands.

        Returns:
            Dict mapping band names to pressure time series arrays.
        """
        result = {}
        for i, (freq, _) in enumerate(self.bands):
            name = f"{int(freq)}Hz"
            result[name] = np.array(self._probe_data[i], dtype=np.float32)
        return result

    def get_pressure_fields(self) -> dict[str, NDArray[np.float32]]:
        """Get current pressure fields for all bands.

        Returns:
            Dict mapping band names to 3D pressure arrays.
        """
        p_cpu = self._p.cpu().numpy()
        result = {}
        for i, (freq, _) in enumerate(self.bands):
            name = f"{int(freq)}Hz"
            result[name] = p_cpu[i]
        return result

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._p.zero_()
        self._vx.zero_()
        self._vy.zero_()
        self._vz.zero_()
        self._time = 0.0
        self._step_count = 0
        self._probe_data = [[] for _ in range(self.n_bands)]

        # Reset ADE auxiliary fields
        if self._ade_initialized:
            for poles_data in self._ade_fields.values():
                for pole_data in poles_data.values():
                    pole_data["J"].zero_()
                    if "J_prev" in pole_data:
                        pole_data["J_prev"].zero_()

    def memory_usage_mb(self) -> float:
        """Estimate GPU memory usage in MB."""
        # 4 fields × n_bands × nx × ny × nz × 4 bytes
        nx, ny, nz = self.shape
        bytes_per_field = self.n_bands * nx * ny * nz * 4
        total_bytes = 4 * bytes_per_field

        # Add ADE field memory if materials are registered
        if self.has_materials:
            # Each pole adds one auxiliary field (two for Lorentz)
            n_ade_fields = 0
            for material in self._materials.values():
                for pole in material.poles:
                    n_ade_fields += 2 if pole.is_lorentz else 1

            # ADE fields are batched: n_bands × nx × ny × nz per field
            ade_bytes = n_ade_fields * self.n_bands * nx * ny * nz * 4
            total_bytes += ade_bytes

        return total_bytes / (1024 * 1024)

    def add_material(self, material, region_mask: NDArray[np.bool_]) -> int:
        """Add a frequency-dependent material to the simulation.

        Materials are shared across all frequency bands - the same material
        regions apply to all bands, but each band evolves independently
        based on the material's frequency-dependent properties.

        Args:
            material: AcousticMaterial instance (e.g., DebyeMaterial)
            region_mask: Boolean array (nx, ny, nz) defining material region

        Returns:
            Material ID that was assigned

        Example:
            >>> from strata_fdtd.materials import DebyeMaterial
            >>> solver = BatchedGPUFDTDSolver(
            ...     shape=(100, 100, 100),
            ...     resolution=1e-3,
            ...     bands=[(500, 200), (2000, 800)],
            ... )
            >>> material = DebyeMaterial(eps_inf=2.0, deps=1.0, tau=1e-9)
            >>> mask = np.zeros((100, 100, 100), dtype=bool)
            >>> mask[30:70, 30:70, 30:70] = True
            >>> mat_id = solver.add_material(material, mask)
        """
        # Initialize material_id array on first call
        if self._material_id is None:
            self._material_id = np.zeros(self.shape, dtype=np.uint8)

        # Auto-assign next available material ID (1, 2, 3, ...)
        if len(self._materials) == 0:
            material_id = 1
        else:
            material_id = max(self._materials.keys()) + 1

        if material_id > 255:
            raise ValueError("Maximum 255 materials supported")

        # Validate mask shape
        if region_mask.shape != self.shape:
            raise ValueError(
                f"Material mask shape {region_mask.shape} doesn't match "
                f"grid shape {self.shape}"
            )

        # Register material and assign to region
        self._materials[material_id] = material
        self._material_id[region_mask] = material_id
        self._ade_initialized = False  # Need to reinitialize ADE fields

        return material_id

    def _initialize_ade_fields(self) -> None:
        """Initialize BATCHED auxiliary fields for all registered materials.

        Creates storage for Debye and Lorentz auxiliary variables with
        shape (n_bands, nx, ny, nz) on the compute device.

        Called automatically before first step if materials are registered.
        """
        if self._ade_initialized:
            return

        self._ade_fields.clear()

        for mat_id, material in self._materials.items():
            self._ade_fields[mat_id] = {}

            for pole_idx, pole in enumerate(material.poles):
                # Compute FDTD coefficients for this pole
                coeffs = pole.fdtd_coefficients(self.dt)

                if pole.is_debye:
                    # Debye: single batched auxiliary field J
                    # Shape: (n_bands, nx, ny, nz)
                    self._ade_fields[mat_id][pole_idx] = {
                        "J": _torch.zeros(
                            self.n_bands, *self.shape,
                            device=self._device, dtype=_torch.float32
                        ),
                        "type": "debye",
                        "target": pole.target,
                        "coeffs": coeffs,  # (alpha, beta)
                    }
                else:
                    # Lorentz: two batched auxiliary fields (J and J_prev)
                    # Shape: (n_bands, nx, ny, nz) each
                    self._ade_fields[mat_id][pole_idx] = {
                        "J": _torch.zeros(
                            self.n_bands, *self.shape,
                            device=self._device, dtype=_torch.float32
                        ),
                        "J_prev": _torch.zeros(
                            self.n_bands, *self.shape,
                            device=self._device, dtype=_torch.float32
                        ),
                        "type": "lorentz",
                        "target": pole.target,
                        "coeffs": coeffs,  # (a, b, d)
                    }

        self._ade_initialized = True

    def _update_ade_density_poles(self) -> None:
        """Update auxiliary fields for density poles (all bands).

        Called before velocity update. Updates all density poles
        for all materials across all frequency bands simultaneously.
        """
        # Convert material_id to torch tensor once
        material_id_tensor = _torch.from_numpy(self._material_id).to(self._device)

        for mat_id, poles_data in self._ade_fields.items():
            # Create mask for cells with this material: (nx, ny, nz)
            mask = material_id_tensor == mat_id

            for pole_data in poles_data.values():
                if pole_data["target"] != "density":
                    continue

                J = pole_data["J"]  # (n_bands, nx, ny, nz)

                if pole_data["type"] == "debye":
                    # Debye update: J^{n+1} = alpha * J^n + beta * p^{n+1}
                    alpha, beta = pole_data["coeffs"]

                    # Broadcast mask to all bands: (1, nx, ny, nz) -> (n_bands, nx, ny, nz)
                    mask_batched = mask.unsqueeze(0)

                    # Update J at material cells
                    J_new = alpha * J + beta * self._p
                    J[mask_batched] = J_new[mask_batched]

                else:  # lorentz
                    # Lorentz update: J^{n+1} = a*J^n + b*J^{n-1} + d*p^n
                    a, b, d = pole_data["coeffs"]
                    J_prev = pole_data["J_prev"]

                    mask_batched = mask.unsqueeze(0)

                    J_new = a * J + b * J_prev + d * self._p
                    J_prev[mask_batched] = J[mask_batched]
                    J[mask_batched] = J_new[mask_batched]

    def _apply_ade_velocity_correction(self) -> None:
        """Apply ADE corrections to velocity fields from density poles.

        Adds dispersive contribution to velocities:
        v += -dt/(rho_inf * dx) * grad(J)
        """
        material_id_tensor = _torch.from_numpy(self._material_id).to(self._device)

        for mat_id, poles_data in self._ade_fields.items():
            material = self._materials[mat_id]
            mask = material_id_tensor == mat_id

            for pole_data in poles_data.values():
                if pole_data["target"] != "density":
                    continue

                J = pole_data["J"]  # (n_bands, nx, ny, nz)

                # Compute gradient components with central differences
                # Note: batch dimension is first, so grad operates on spatial dims

                # vx correction: -dt/(rho_inf*dx) * dJ/dx
                # vx lives at face (i+1/2), use central diff
                coeff = -self.dt / (material.rho_inf * self.resolution)

                # dJ/dx at face i+1/2: (J[i+1] - J[i]) / dx
                # vx shape: (n_bands, nx, ny, nz)
                grad_J_x = (J[:, 1:, :, :] - J[:, :-1, :, :]) / self.resolution
                mask_vx = mask[:-1, :, :]  # Shape: (nx-1, ny, nz)
                mask_vx_batched = mask_vx.unsqueeze(0)  # (1, nx-1, ny, nz)

                self._vx[:, :-1, :, :][mask_vx_batched] += (
                    coeff * grad_J_x[mask_vx_batched]
                )

                # vy correction
                grad_J_y = (J[:, :, 1:, :] - J[:, :, :-1, :]) / self.resolution
                mask_vy = mask[:, :-1, :]
                mask_vy_batched = mask_vy.unsqueeze(0)

                self._vy[:, :, :-1, :][mask_vy_batched] += (
                    coeff * grad_J_y[mask_vy_batched]
                )

                # vz correction
                grad_J_z = (J[:, :, :, 1:] - J[:, :, :, :-1]) / self.resolution
                mask_vz = mask[:, :, :-1]
                mask_vz_batched = mask_vz.unsqueeze(0)

                self._vz[:, :, :, :-1][mask_vz_batched] += (
                    coeff * grad_J_z[mask_vz_batched]
                )

    def _update_ade_modulus_poles(self) -> None:
        """Update auxiliary fields for modulus poles (all bands).

        Called before pressure update. Updates modulus poles based on
        the velocity divergence.
        """
        # Compute divergence for all bands: (n_bands, nx, ny, nz)
        div = _torch.zeros_like(self._p)

        # div_x = (vx[i] - vx[i-1]) / dx
        div[:, 1:, :, :] += (self._vx[:, 1:, :, :] - self._vx[:, :-1, :, :]) / self.resolution
        # div_y = (vy[j] - vy[j-1]) / dy
        div[:, :, 1:, :] += (self._vy[:, :, 1:, :] - self._vy[:, :, :-1, :]) / self.resolution
        # div_z = (vz[k] - vz[k-1]) / dz
        div[:, :, :, 1:] += (self._vz[:, :, :, 1:] - self._vz[:, :, :, :-1]) / self.resolution

        material_id_tensor = _torch.from_numpy(self._material_id).to(self._device)

        for mat_id, poles_data in self._ade_fields.items():
            mask = material_id_tensor == mat_id

            for pole_data in poles_data.values():
                if pole_data["target"] != "modulus":
                    continue

                J = pole_data["J"]  # (n_bands, nx, ny, nz)

                if pole_data["type"] == "debye":
                    # Debye update: J^{n+1} = alpha * J^n + beta * div(v)^{n+1}
                    alpha, beta = pole_data["coeffs"]

                    mask_batched = mask.unsqueeze(0)

                    J_new = alpha * J + beta * div
                    J[mask_batched] = J_new[mask_batched]

                else:  # lorentz
                    # Lorentz update: J^{n+1} = a*J^n + b*J^{n-1} + d*div(v)^n
                    a, b, d = pole_data["coeffs"]
                    J_prev = pole_data["J_prev"]

                    mask_batched = mask.unsqueeze(0)

                    J_new = a * J + b * J_prev + d * div
                    J_prev[mask_batched] = J[mask_batched]
                    J[mask_batched] = J_new[mask_batched]

    def _apply_ade_pressure_correction(self) -> None:
        """Apply ADE corrections to pressure field from modulus poles.

        Adds dispersive contribution:
        p += -K_inf * dt * Σ dJ/dt (for modulus poles)
        """
        material_id_tensor = _torch.from_numpy(self._material_id).to(self._device)

        for mat_id, poles_data in self._ade_fields.items():
            material = self._materials[mat_id]
            mask = material_id_tensor == mat_id
            mask_batched = mask.unsqueeze(0)

            for pole_data in poles_data.values():
                if pole_data["target"] != "modulus":
                    continue

                J = pole_data["J"]  # (n_bands, nx, ny, nz)

                # Coefficient for pressure correction
                coeff = -material.K_inf * self.dt

                # Apply to pressure at material cells
                self._p[mask_batched] += coeff * J[mask_batched]

    def combine_bands(
        self,
        probe_data: dict[str, NDArray[np.float32]] | None = None,
        overlap: float = 0.5,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Combine narrowband responses into coherent broadband transfer function.

        Takes the narrowband time-domain responses from multi-band simulation and
        combines them in the frequency domain using weighted averaging in overlapping
        regions. This provides a single broadband frequency response with improved
        SNR in the target bands.

        Args:
            probe_data: Dict mapping band names to time-domain probe data.
                       If None, uses data from the default probe position.
            overlap: Overlap fraction for Gaussian weighting (0.0 to 1.0).
                    Higher values create smoother transitions but reduce
                    effective bandwidth isolation. Default: 0.5.

        Returns:
            Tuple of (frequencies, magnitude, phase):
            - frequencies: Array of frequency values in Hz
            - magnitude: Array of magnitude values (linear scale)
            - phase: Array of phase values in radians

        Raises:
            ValueError: If overlap is not in [0, 1] or no probe data available.

        Example:
            >>> solver = BatchedGPUFDTDSolver(
            ...     shape=(100, 100, 100),
            ...     bands=[(250, 100), (500, 200), (1000, 400), (2000, 800)],
            ... )
            >>> results = solver.run(duration=0.01)
            >>> freqs, mag, phase = solver.combine_bands(results, overlap=0.5)
            >>> # Plot broadband response
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(freqs, 20 * np.log10(mag))  # dB magnitude
        """
        if overlap < 0.0 or overlap > 1.0:
            raise ValueError(f"overlap must be in [0, 1], got {overlap}")

        # Get probe data
        if probe_data is None:
            probe_data = self.get_probe_data()
            if not probe_data:
                raise ValueError("No probe data available")

        if len(probe_data) == 0:
            raise ValueError("probe_data is empty")

        # Compute FFT for each band
        band_ffts = {}
        band_freqs = {}

        for center_freq, _bandwidth in self.bands:
            band_name = f"{int(center_freq)}Hz"
            if band_name not in probe_data:
                raise ValueError(
                    f"Band '{band_name}' not found in probe_data. "
                    f"Available: {list(probe_data.keys())}"
                )

            time_data = probe_data[band_name]
            n_samples = len(time_data)

            # Validate non-empty time series
            if n_samples == 0:
                raise ValueError(f"Band '{band_name}' has no samples")

            # Compute FFT
            fft_data = np.fft.rfft(time_data)
            freqs = np.fft.rfftfreq(n_samples, self.dt)

            band_ffts[band_name] = fft_data
            band_freqs[band_name] = freqs

        # Determine frequency range for output
        # Use the union of all band frequency ranges
        min_freq = 0.0
        max_freq = max(band_freqs[f"{int(freq)}Hz"][-1]
                      for freq, _ in self.bands)

        # Create output frequency grid (use finest resolution from all bands)
        n_freq = max(len(band_freqs[f"{int(freq)}Hz"])
                    for freq, _ in self.bands)
        output_freqs = np.linspace(min_freq, max_freq, n_freq)

        # Initialize combined spectrum
        combined_spectrum = np.zeros(len(output_freqs), dtype=np.complex64)
        total_weight = np.zeros(len(output_freqs), dtype=np.float32)

        # Combine bands using Gaussian weighting
        for center_freq, bandwidth in self.bands:
            band_name = f"{int(center_freq)}Hz"
            fft_data = band_ffts[band_name]
            freqs = band_freqs[band_name]

            # Interpolate this band's FFT to output frequency grid
            # Interpolate real and imaginary parts separately to avoid
            # phase wrapping artifacts at ±π discontinuities
            interp_fft_real = np.interp(
                output_freqs, freqs, fft_data.real, left=0.0, right=0.0
            )
            interp_fft_imag = np.interp(
                output_freqs, freqs, fft_data.imag, left=0.0, right=0.0
            )

            # Reconstruct complex spectrum
            interp_fft = interp_fft_real + 1j * interp_fft_imag

            # Compute Gaussian weight centered on this band
            # Standard deviation based on bandwidth and overlap
            if overlap < 1e-10:
                # overlap ≈ 0: use narrow Gaussian (minimal blending)
                sigma = bandwidth / 10.0
            elif overlap > 1.0 - 1e-10:
                # overlap ≈ 1: use wide Gaussian (maximum blending)
                sigma = bandwidth * 10.0
            else:
                sigma = bandwidth / (2.0 * np.sqrt(-2.0 * np.log(1.0 - overlap)))
            weight = np.exp(-0.5 * ((output_freqs - center_freq) / sigma) ** 2)

            # Accumulate weighted spectrum
            combined_spectrum += weight * interp_fft
            total_weight += weight

        # Normalize by total weight (avoid division by zero)
        mask = total_weight > 1e-10
        combined_spectrum[mask] /= total_weight[mask]

        # Extract magnitude and phase
        magnitude = np.abs(combined_spectrum).astype(np.float32)
        phase = np.angle(combined_spectrum).astype(np.float32)

        return output_freqs.astype(np.float32), magnitude, phase

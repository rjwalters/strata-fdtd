"""Viscoelastic solid materials using standard linear solid (Zener) model.

This module implements the Zener model for viscoelastic solids commonly
used in speaker enclosure construction: MDF, plywood, and damping
materials like butyl rubber and sorbothane.

The Zener model (standard linear solid) captures:
- Instantaneous elastic response
- Relaxation over time (frequency-dependent damping)
- Loss factor η(ω) that varies with frequency

The complex modulus is:
    E*(ω) = E_∞ * [1 + Δ/(1 + iωτ)]

where Δ = (E_0 - E_∞)/E_∞ is the relaxation strength.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .base import AcousticMaterial, Pole, PoleType


@dataclass
class ViscoelasticSolid(AcousticMaterial):
    """Viscoelastic solid material using Zener (SLS) model.

    Models materials with frequency-dependent damping through the
    standard linear solid (SLS) or Zener model. Commonly used for
    speaker enclosure materials like MDF and plywood.

    The Zener model has three parameters:
    - E_0: Low-frequency (relaxed) modulus
    - E_∞: High-frequency (unrelaxed) modulus
    - τ: Relaxation time

    For practical use, materials are often specified by:
    - E: Representative modulus (typically E_∞)
    - η: Loss factor at a reference frequency

    The acoustic properties are derived assuming the material behaves
    as an effective fluid within the FDTD scheme (valid for thin
    enclosure walls where flexural waves are less important than
    through-transmission).

    Args:
        name: Material name
        density: Bulk density in kg/m³
        youngs_modulus: Young's modulus E in Pa
        loss_factor: Loss factor η at reference frequency
        poissons_ratio: Poisson's ratio ν (default: 0.3)
        ref_frequency: Reference frequency for loss factor (default: 1 kHz)
        n_poles: Number of relaxation poles (default: 3)

    Attributes:
        density: ρ in kg/m³
        youngs_modulus: E in Pa
        loss_factor: η (dimensionless)
        poissons_ratio: ν (dimensionless)

    Example:
        >>> # Medium Density Fiberboard
        >>> mdf = ViscoelasticSolid(
        ...     name="MDF_medium",
        ...     density=700,
        ...     youngs_modulus=3.5e9,
        ...     loss_factor=0.03,
        ... )
        >>> # Sound speed in MDF
        >>> print(f"c = {mdf.c_inf:.0f} m/s")
    """

    density: float = 700.0  # kg/m³
    youngs_modulus: float = 3.5e9  # Pa
    loss_factor: float = 0.03  # η
    poissons_ratio: float = 0.3  # ν
    ref_frequency: float = 1000.0  # Hz
    num_poles: int = 3

    _poles_cache: list[Pole] | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.youngs_modulus <= 0:
            raise ValueError("youngs_modulus must be positive")
        if self.loss_factor < 0:
            raise ValueError("loss_factor must be non-negative")
        if not -1 < self.poissons_ratio < 0.5:
            raise ValueError("poissons_ratio must be in (-1, 0.5)")
        if self.ref_frequency <= 0:
            raise ValueError("ref_frequency must be positive")

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K = E / (3*(1 - 2*ν))."""
        return self.youngs_modulus / (3 * (1 - 2 * self.poissons_ratio))

    @property
    def shear_modulus(self) -> float:
        """Shear modulus G = E / (2*(1 + ν))."""
        return self.youngs_modulus / (2 * (1 + self.poissons_ratio))

    @property
    def rho_inf(self) -> float:
        """High-frequency density (constant for solids)."""
        return self.density

    @property
    def K_inf(self) -> float:
        """High-frequency bulk modulus."""
        return self.bulk_modulus

    @property
    def longitudinal_modulus(self) -> float:
        """Longitudinal (P-wave) modulus M = K + 4G/3."""
        return self.bulk_modulus + 4 * self.shear_modulus / 3

    @property
    def c_longitudinal(self) -> float:
        """Longitudinal wave speed c_L = √(M/ρ)."""
        return np.sqrt(self.longitudinal_modulus / self.density)

    @property
    def relaxation_time(self) -> float:
        """Estimate relaxation time from loss factor and reference frequency.

        For a single Debye pole at the reference frequency:
        τ ≈ 1 / (2π * f_ref)

        This places the maximum loss near the reference frequency.
        """
        return 1 / (2 * np.pi * self.ref_frequency)

    @property
    def relaxation_strength(self) -> float:
        """Estimate relaxation strength Δ from loss factor.

        For a single Debye relaxation:
        η_max ≈ Δ/2 at ω = 1/τ

        So Δ ≈ 2*η
        """
        return 2 * self.loss_factor

    @property
    def poles(self) -> list[Pole]:
        """Generate Debye poles for viscoelastic response."""
        if self._poles_cache is not None:
            return self._poles_cache

        self._poles_cache = self._generate_poles()
        return self._poles_cache

    def _generate_poles(self) -> list[Pole]:
        """Generate Debye poles to match loss factor spectrum.

        Uses multiple poles spread across frequency range to give
        broadband damping centered at reference frequency.
        """
        poles = []

        # Base relaxation time and strength
        tau_0 = self.relaxation_time
        delta_total = self.relaxation_strength

        # Distribute poles across decades
        if self.num_poles == 1:
            # Single pole at reference frequency
            poles.append(Pole(
                pole_type=PoleType.DEBYE,
                delta_chi=delta_total,
                tau=tau_0,
                target="modulus",
            ))
        else:
            # Multiple poles spanning frequency range
            # Spread poles from 0.1*f_ref to 10*f_ref
            tau_values = tau_0 * np.logspace(-1, 1, self.num_poles)
            weights = self._compute_pole_weights(tau_values)

            for tau, w in zip(tau_values, weights, strict=True):
                if w > 1e-6:
                    poles.append(Pole(
                        pole_type=PoleType.DEBYE,
                        delta_chi=delta_total * w,
                        tau=tau,
                        target="modulus",
                    ))

        return poles

    def _compute_pole_weights(self, tau_values: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute weights for multiple poles to achieve flat loss factor.

        Uses a simple distribution that gives roughly constant loss
        factor across the frequency range spanned by the poles.
        """
        n = len(tau_values)
        # Simple equal weighting for now
        # More sophisticated fitting could optimize for flat η(ω)
        return np.ones(n) / n

    def loss_factor_spectrum(
        self, omega: float | NDArray[np.floating]
    ) -> float | NDArray[np.floating]:
        """Compute loss factor η(ω) = Im(K)/Re(K).

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Loss factor at each frequency
        """
        K = self.effective_modulus(omega)
        return np.imag(K) / np.real(K)

    def quality_factor(
        self, omega: float | NDArray[np.floating]
    ) -> float | NDArray[np.floating]:
        """Compute quality factor Q = 1/η.

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Quality factor at each frequency
        """
        eta = self.loss_factor_spectrum(omega)
        return 1 / (eta + 1e-20)  # Avoid division by zero


@dataclass
class DampingMaterial(ViscoelasticSolid):
    """High-damping viscoelastic material (butyl, sorbothane, etc.).

    These materials have very high loss factors (η > 0.1) and are
    used to damp panel vibrations in speaker enclosures.

    The frequency response is often more complex than simple Zener
    model, but multiple Debye poles provide reasonable approximation.

    Args:
        name: Material name
        density: Bulk density in kg/m³
        youngs_modulus: Young's modulus E in Pa (can be very low ~MPa)
        loss_factor: Peak loss factor η (typically 0.1-1.0)
        poissons_ratio: Poisson's ratio (default: 0.49, nearly incompressible)
        ref_frequency: Reference frequency for loss factor
        n_poles: Number of relaxation poles (default: 5 for broad spectrum)

    Example:
        >>> # Sorbothane damping pad
        >>> sorbothane = DampingMaterial(
        ...     name="sorbothane",
        ...     density=1100,
        ...     youngs_modulus=0.5e6,  # 0.5 MPa (very soft)
        ...     loss_factor=0.50,
        ... )
    """

    poissons_ratio: float = 0.49  # Nearly incompressible
    num_poles: int = 5  # More poles for broadband damping

    @property
    def relaxation_strength(self) -> float:
        """Higher relaxation strength for damping materials.

        For high-loss materials, Δ can be larger than 2η to account
        for the distributed nature of relaxation.
        """
        return min(3 * self.loss_factor, 0.9)  # Cap at 0.9 for stability


@dataclass
class ImpedanceTube:
    """Impedance tube measurement simulation.

    Simulates a two-microphone impedance tube measurement for
    characterizing material absorption coefficient.

    The tube has:
    - Rigid side walls
    - Sample mounted at one end (rigid or open backing)
    - Loudspeaker source at opposite end

    Args:
        length: Tube length in m
        diameter: Tube inner diameter in m
        mic_spacing: Distance between measurement microphones in m
        sample_distance: Distance from microphones to sample in m
        rho_air: Air density (default: 1.2 kg/m³)
        c_air: Speed of sound in air (default: 343 m/s)

    Example:
        >>> tube = ImpedanceTube(
        ...     length=0.5,
        ...     diameter=0.1,
        ...     mic_spacing=0.05,
        ...     sample_distance=0.1,
        ... )
        >>> alpha = tube.measure_absorption(fiberglass, omega)
    """

    length: float = 0.5  # m
    diameter: float = 0.1  # m
    mic_spacing: float = 0.05  # m
    sample_distance: float = 0.1  # m
    rho_air: float = 1.2  # kg/m³
    c_air: float = 343.0  # m/s

    @property
    def Z_air(self) -> float:
        """Air characteristic impedance."""
        return self.rho_air * self.c_air

    @property
    def max_frequency(self) -> float:
        """Maximum valid frequency based on tube diameter.

        f_max < 0.586 * c / d for first cross-mode cutoff
        """
        return 0.586 * self.c_air / self.diameter

    @property
    def min_frequency(self) -> float:
        """Minimum frequency based on microphone spacing.

        f_min > c / (20 * s) for valid transfer function
        """
        return self.c_air / (20 * self.mic_spacing)

    def transfer_function_method(
        self,
        material: AcousticMaterial,
        omega: float | NDArray[np.floating],
    ) -> complex | NDArray[np.complexfloating]:
        """Compute complex reflection coefficient using transfer function method.

        ISO 10534-2 standard method for impedance tube measurements.

        Args:
            material: Material to characterize
            omega: Angular frequency in rad/s

        Returns:
            Complex reflection coefficient R
        """
        omega = np.asarray(omega)

        # For a porous material backed by rigid wall
        if isinstance(material, PorousMaterial) and material.thickness is not None:
            Z_s = material.surface_impedance(omega, backing="rigid")
        else:
            # Semi-infinite material
            Z_s = material.impedance(omega)

        # Reflection coefficient
        R = (Z_s - self.Z_air) / (Z_s + self.Z_air)

        return R

    def absorption_coefficient(
        self,
        material: AcousticMaterial,
        omega: float | NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Compute normal incidence absorption coefficient.

        Args:
            material: Material to characterize
            omega: Angular frequency in rad/s

        Returns:
            Absorption coefficient α (0 to 1)
        """
        R = self.transfer_function_method(material, omega)
        return 1 - np.abs(R) ** 2


# Import PorousMaterial for isinstance check in ImpedanceTube (avoid circular import)
from .porous import PorousMaterial  # noqa: E402

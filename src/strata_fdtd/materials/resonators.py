"""Resonant acoustic absorbers (Helmholtz, membrane, perforated panels).

This module implements resonant absorber models that can be used as
effective materials in FDTD simulations. These include:

- Helmholtz resonators: Tuned cavity absorbers
- Membrane absorbers: Panel absorbers (bass traps)
- Perforated panels: Distributed resonant absorbers

The resonant behavior is captured using Lorentz poles in the ADE
formulation, enabling efficient FDTD simulation.

References:
    - Ingard, "Notes on Sound Absorption Technology" (1994)
    - Kuttruff, "Room Acoustics" (2009)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np
from numpy.typing import NDArray

from .base import AcousticMaterial, Pole, PoleType

# Physical constants
AIR_DENSITY = 1.204  # kg/m³
AIR_SPEED = 343.0  # m/s


@dataclass
class HelmholtzResonator(AcousticMaterial):
    """Helmholtz resonator absorber.

    Models a Helmholtz resonator as an effective material with a
    resonant response at the tuning frequency. The resonance is
    represented by a Lorentz pole in the bulk modulus.

    The resonance frequency is:
        f_0 = c/(2π) * √(S/(V*(L_eff)))

    where S is neck area, V is cavity volume, L_eff is effective neck
    length including end corrections.

    Args:
        name: Absorber name
        neck_radius: Neck radius in m
        neck_length: Physical neck length in m
        cavity_volume: Cavity volume in m³
        loss_factor: Acoustic loss factor (default: 0.1)
        end_correction: End correction factor (default: 0.85)
        rho_air: Air density (default: 1.2 kg/m³)
        c_air: Speed of sound (default: 343 m/s)

    Attributes:
        resonance_frequency: f_0 in Hz
        quality_factor: Q = f_0 / bandwidth

    Example:
        >>> # 100 Hz bass trap
        >>> trap = HelmholtzResonator(
        ...     name="bass_trap_100Hz",
        ...     neck_radius=0.025,  # 25mm radius
        ...     neck_length=0.05,   # 50mm neck
        ...     cavity_volume=0.02, # 20 liters
        ...     loss_factor=0.15,
        ... )
        >>> print(f"f_0 = {trap.resonance_frequency:.1f} Hz")
    """

    neck_radius: float = 0.025  # m
    neck_length: float = 0.05  # m
    cavity_volume: float = 0.01  # m³
    loss_factor: float = 0.1  # Acoustic loss
    end_correction: float = 0.85  # Typically 0.6-0.85
    rho_air: float = AIR_DENSITY
    c_air: float = AIR_SPEED

    _poles_cache: list[Pole] | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.neck_radius <= 0:
            raise ValueError("neck_radius must be positive")
        if self.neck_length <= 0:
            raise ValueError("neck_length must be positive")
        if self.cavity_volume <= 0:
            raise ValueError("cavity_volume must be positive")
        if self.loss_factor < 0:
            raise ValueError("loss_factor must be non-negative")

    @property
    def neck_area(self) -> float:
        """Neck cross-sectional area."""
        return pi * self.neck_radius**2

    @property
    def effective_neck_length(self) -> float:
        """Effective neck length with end corrections.

        L_eff = L + 2 * δ where δ ≈ 0.85 * r for unflanged opening
        """
        return self.neck_length + 2 * self.end_correction * self.neck_radius

    @property
    def resonance_frequency(self) -> float:
        """Helmholtz resonance frequency in Hz."""
        S = self.neck_area
        V = self.cavity_volume
        L_eff = self.effective_neck_length
        return self.c_air / (2 * pi) * np.sqrt(S / (V * L_eff))

    @property
    def omega_0(self) -> float:
        """Angular resonance frequency in rad/s."""
        return 2 * pi * self.resonance_frequency

    @property
    def acoustic_mass(self) -> float:
        """Acoustic mass of neck (kg/m⁴)."""
        return self.rho_air * self.effective_neck_length / self.neck_area

    @property
    def acoustic_compliance(self) -> float:
        """Acoustic compliance of cavity (m³/Pa)."""
        return self.cavity_volume / (self.rho_air * self.c_air**2)

    @property
    def quality_factor_theoretical(self) -> float:
        """Theoretical Q factor (without losses)."""
        # Q ≈ √(M/C) / R where R is acoustic resistance
        # For radiation loss only: Q ≈ 2*V/(S*λ)
        wavelength = self.c_air / self.resonance_frequency
        return 2 * self.cavity_volume / (self.neck_area * wavelength)

    @property
    def gamma(self) -> float:
        """Damping coefficient for Lorentz pole (rad/s)."""
        # γ = ω_0 / Q where Q = 1/(2*η)
        # So γ = 2 * η * ω_0
        return 2 * self.loss_factor * self.omega_0

    @property
    def rho_inf(self) -> float:
        """High-frequency density (air)."""
        return self.rho_air

    @property
    def K_inf(self) -> float:
        """High-frequency bulk modulus (air)."""
        return self.rho_air * self.c_air**2

    @property
    def poles(self) -> list[Pole]:
        """Generate Lorentz pole for resonant response."""
        if self._poles_cache is not None:
            return self._poles_cache

        # Resonance strength (determines absorption bandwidth)
        # Higher delta_chi = broader/stronger absorption
        delta_chi = -0.5  # Negative for absorption

        pole = Pole(
            pole_type=PoleType.LORENTZ,
            delta_chi=delta_chi,
            omega_0=self.omega_0,
            gamma=self.gamma,
            target="modulus",
        )

        self._poles_cache = [pole]
        return self._poles_cache

    def impedance_analytical(
        self, omega: float | NDArray[np.floating]
    ) -> complex | NDArray[np.complexfloating]:
        """Analytical Helmholtz resonator impedance.

        Z = R + i(ωM - 1/(ωC))

        where M is acoustic mass, C is acoustic compliance, R is resistance.

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Complex acoustic impedance
        """
        omega = np.asarray(omega) + 1e-20  # Avoid division by zero

        M = self.acoustic_mass
        C = self.acoustic_compliance
        R = self.loss_factor * self.omega_0 * M  # Resistance from loss factor

        return R + 1j * (omega * M - 1 / (omega * C))

    def absorption_analytical(
        self, omega: float | NDArray[np.floating]
    ) -> float | NDArray[np.floating]:
        """Analytical absorption coefficient at resonance.

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Absorption coefficient (0-1)
        """
        Z = self.impedance_analytical(omega)
        Z_air = self.rho_air * self.c_air
        R = (Z - Z_air) / (Z + Z_air)
        return 1 - np.abs(R) ** 2


@dataclass
class MembraneAbsorber(AcousticMaterial):
    """Membrane (panel) absorber for low frequency absorption.

    Models a vibrating panel backed by an air cavity, commonly used
    as bass traps in room acoustics. The membrane vibration creates
    a resonant absorption peak.

    The resonance frequency is:
        f_0 = (1/(2π)) * √(ρ_air * c² / (m * d))

    where m is panel surface density (kg/m²) and d is cavity depth.

    Args:
        name: Absorber name
        surface_density: Panel mass per unit area in kg/m²
        cavity_depth: Air cavity depth in m
        loss_factor: Combined structural + acoustic loss (default: 0.15)
        panel_area: Panel area in m² (for impedance calculations)
        rho_air: Air density
        c_air: Speed of sound

    Example:
        >>> # 80 Hz bass trap with 6mm MDF panel
        >>> trap = MembraneAbsorber(
        ...     name="bass_trap_80Hz",
        ...     surface_density=4.2,  # 6mm MDF at 700 kg/m³
        ...     cavity_depth=0.15,    # 15cm air gap
        ...     loss_factor=0.15,
        ... )
    """

    surface_density: float = 4.0  # kg/m²
    cavity_depth: float = 0.1  # m
    loss_factor: float = 0.15
    panel_area: float = 1.0  # m²
    rho_air: float = AIR_DENSITY
    c_air: float = AIR_SPEED

    _poles_cache: list[Pole] | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.surface_density <= 0:
            raise ValueError("surface_density must be positive")
        if self.cavity_depth <= 0:
            raise ValueError("cavity_depth must be positive")
        if self.loss_factor < 0:
            raise ValueError("loss_factor must be non-negative")

    @property
    def resonance_frequency(self) -> float:
        """Panel resonance frequency in Hz.

        f_0 = 60 / √(m * d) (simplified formula, m in kg/m², d in cm)
        """
        # More accurate formula:
        m = self.surface_density
        d = self.cavity_depth
        return (1 / (2 * pi)) * np.sqrt(
            self.rho_air * self.c_air**2 / (m * d)
        )

    @property
    def omega_0(self) -> float:
        """Angular resonance frequency."""
        return 2 * pi * self.resonance_frequency

    @property
    def gamma(self) -> float:
        """Damping coefficient."""
        return 2 * self.loss_factor * self.omega_0

    @property
    def rho_inf(self) -> float:
        """High-frequency density."""
        return self.rho_air

    @property
    def K_inf(self) -> float:
        """High-frequency modulus."""
        return self.rho_air * self.c_air**2

    @property
    def poles(self) -> list[Pole]:
        """Generate Lorentz pole for membrane resonance."""
        if self._poles_cache is not None:
            return self._poles_cache

        delta_chi = -0.3  # Absorption strength

        pole = Pole(
            pole_type=PoleType.LORENTZ,
            delta_chi=delta_chi,
            omega_0=self.omega_0,
            gamma=self.gamma,
            target="modulus",
        )

        self._poles_cache = [pole]
        return self._poles_cache


@dataclass
class PerforatedPanel(AcousticMaterial):
    """Perforated panel absorber.

    Models a perforated plate (with many small holes) backed by an
    air cavity. Each hole acts as a small Helmholtz resonator, and
    the panel provides distributed resonant absorption.

    The resonance frequency depends on:
    - Perforation ratio (open area fraction)
    - Hole diameter and plate thickness
    - Backing cavity depth

    Args:
        name: Absorber name
        hole_diameter: Perforation diameter in m
        plate_thickness: Plate thickness in m
        perforation_ratio: Open area fraction (0-1)
        cavity_depth: Backing cavity depth in m
        loss_factor: Acoustic loss factor (default: 0.1)
        rho_air: Air density
        c_air: Speed of sound

    Example:
        >>> # Ceiling tile absorber
        >>> panel = PerforatedPanel(
        ...     name="ceiling_tile",
        ...     hole_diameter=0.003,     # 3mm holes
        ...     plate_thickness=0.012,   # 12mm thick
        ...     perforation_ratio=0.15,  # 15% open area
        ...     cavity_depth=0.2,        # 200mm backing
        ... )
    """

    hole_diameter: float = 0.005  # m
    plate_thickness: float = 0.01  # m
    perforation_ratio: float = 0.1  # Open area fraction
    cavity_depth: float = 0.1  # m
    loss_factor: float = 0.1
    rho_air: float = AIR_DENSITY
    c_air: float = AIR_SPEED

    _poles_cache: list[Pole] | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.hole_diameter <= 0:
            raise ValueError("hole_diameter must be positive")
        if self.plate_thickness <= 0:
            raise ValueError("plate_thickness must be positive")
        if not 0 < self.perforation_ratio < 1:
            raise ValueError("perforation_ratio must be in (0, 1)")
        if self.cavity_depth <= 0:
            raise ValueError("cavity_depth must be positive")

    @property
    def hole_radius(self) -> float:
        """Hole radius."""
        return self.hole_diameter / 2

    @property
    def effective_hole_length(self) -> float:
        """Effective hole length with end corrections."""
        # End correction for each side: δ ≈ 0.85*r * (1 - 1.47*√ε)
        # where ε is perforation ratio
        end_corr = 0.85 * self.hole_radius * (1 - 1.47 * np.sqrt(self.perforation_ratio))
        return self.plate_thickness + 2 * max(end_corr, 0)

    @property
    def resonance_frequency(self) -> float:
        """Perforated panel resonance frequency in Hz.

        f_0 = c/(2π) * √(ε / (t_eff * d))

        where ε is perforation ratio, t_eff is effective thickness,
        d is cavity depth.
        """
        eps = self.perforation_ratio
        t_eff = self.effective_hole_length
        d = self.cavity_depth
        return self.c_air / (2 * pi) * np.sqrt(eps / (t_eff * d))

    @property
    def omega_0(self) -> float:
        """Angular resonance frequency."""
        return 2 * pi * self.resonance_frequency

    @property
    def gamma(self) -> float:
        """Damping coefficient."""
        return 2 * self.loss_factor * self.omega_0

    @property
    def rho_inf(self) -> float:
        """High-frequency density."""
        return self.rho_air

    @property
    def K_inf(self) -> float:
        """High-frequency modulus."""
        return self.rho_air * self.c_air**2

    @property
    def poles(self) -> list[Pole]:
        """Generate Lorentz pole for panel resonance."""
        if self._poles_cache is not None:
            return self._poles_cache

        # Absorption strength scales with perforation ratio
        delta_chi = -0.4 * np.sqrt(self.perforation_ratio)

        pole = Pole(
            pole_type=PoleType.LORENTZ,
            delta_chi=delta_chi,
            omega_0=self.omega_0,
            gamma=self.gamma,
            target="modulus",
        )

        self._poles_cache = [pole]
        return self._poles_cache

    def impedance_analytical(
        self, omega: float | NDArray[np.floating]
    ) -> complex | NDArray[np.complexfloating]:
        """Analytical perforated panel impedance.

        Z = R + i*ω*M - i*Z_air*cot(k*d)/ε

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Complex surface impedance
        """
        omega = np.asarray(omega) + 1e-20

        eps = self.perforation_ratio
        t_eff = self.effective_hole_length
        d = self.cavity_depth
        k = omega / self.c_air

        # Acoustic mass per unit area
        M = self.rho_air * t_eff / eps

        # Viscous resistance (Maa model)
        # R ≈ 32*η*t/(ε*a²) for narrow holes
        eta = 1.81e-5  # Air viscosity
        a = self.hole_radius
        R = 32 * eta * self.plate_thickness / (eps * a**2)

        # Include loss factor
        R *= (1 + self.loss_factor)

        # Cavity impedance
        Z_cav = -1j * self.rho_air * self.c_air / np.tan(k * d) / eps

        return R + 1j * omega * M + Z_cav


@dataclass
class QuarterWaveResonator(AcousticMaterial):
    """Quarter-wave tube resonator.

    A tube closed at one end that resonates at frequencies where
    the tube length is an odd multiple of quarter wavelengths.

    f_n = (2n-1) * c / (4L) for n = 1, 2, 3, ...

    Useful for targeting specific low frequencies where Helmholtz
    resonators would be impractically large.

    Args:
        name: Resonator name
        length: Tube length in m
        diameter: Tube diameter in m
        loss_factor: Acoustic loss factor (default: 0.05)
        n_modes: Number of resonant modes to include (default: 3)
        rho_air: Air density
        c_air: Speed of sound

    Example:
        >>> # 100 Hz quarter-wave absorber
        >>> absorber = QuarterWaveResonator(
        ...     name="qw_100Hz",
        ...     length=0.858,  # c/(4*100) = 0.858m
        ...     diameter=0.1,
        ...     loss_factor=0.08,
        ... )
    """

    length: float = 0.5  # m
    diameter: float = 0.1  # m
    loss_factor: float = 0.05
    n_modes: int = 3
    rho_air: float = AIR_DENSITY
    c_air: float = AIR_SPEED

    _poles_cache: list[Pole] | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.diameter <= 0:
            raise ValueError("diameter must be positive")
        if self.n_modes < 1:
            raise ValueError("n_modes must be >= 1")

    @property
    def area(self) -> float:
        """Tube cross-sectional area."""
        return pi * (self.diameter / 2)**2

    def mode_frequency(self, n: int) -> float:
        """Frequency of nth resonant mode (n = 1, 2, 3, ...).

        f_n = (2n - 1) * c / (4L)
        """
        return (2 * n - 1) * self.c_air / (4 * self.length)

    @property
    def fundamental_frequency(self) -> float:
        """Fundamental (first mode) frequency in Hz."""
        return self.mode_frequency(1)

    @property
    def rho_inf(self) -> float:
        """High-frequency density."""
        return self.rho_air

    @property
    def K_inf(self) -> float:
        """High-frequency modulus."""
        return self.rho_air * self.c_air**2

    @property
    def poles(self) -> list[Pole]:
        """Generate Lorentz poles for each resonant mode."""
        if self._poles_cache is not None:
            return self._poles_cache

        poles = []

        for n in range(1, self.n_modes + 1):
            f_n = self.mode_frequency(n)
            omega_n = 2 * pi * f_n
            gamma_n = 2 * self.loss_factor * omega_n

            # Higher modes have progressively smaller absorption strength
            delta_chi = -0.3 / n

            poles.append(Pole(
                pole_type=PoleType.LORENTZ,
                delta_chi=delta_chi,
                omega_0=omega_n,
                gamma=gamma_n,
                target="modulus",
            ))

        self._poles_cache = poles
        return self._poles_cache

"""Base classes for frequency-dependent acoustic materials.

This module provides the foundation for the ADE (Auxiliary Differential
Equation) material system, enabling frequency-dependent density and
bulk modulus in FDTD simulations.

The ADE approach decomposes frequency-dependent material properties into
a sum of poles (Debye for relaxation, Lorentz for resonances), each of
which adds an auxiliary differential equation to the FDTD update.

Example:
    >>> from strata_fdtd.materials import Pole, PoleType
    >>> # Create a Debye pole for viscous losses
    >>> pole = Pole(
    ...     pole_type=PoleType.DEBYE,
    ...     delta_chi=0.5,
    ...     tau=1e-6,
    ...     target="modulus",
    ... )
    >>> # Get FDTD update coefficients
    >>> a, b = pole.fdtd_coefficients(dt=1e-7)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


class PoleType(Enum):
    """Type of dispersion pole.

    DEBYE: First-order relaxation pole for viscous/thermal losses.
        χ(ω) = Δχ / (1 + iωτ)
        Time domain: τ ∂J/∂t + J = Δχ·f

    LORENTZ: Second-order resonant pole for oscillator models.
        χ(ω) = Δχ·ω₀² / (ω₀² - ω² + iγω)
        Time domain: ∂²J/∂t² + γ∂J/∂t + ω₀²J = Δχ·ω₀²·f
    """

    DEBYE = "debye"
    LORENTZ = "lorentz"


@dataclass
class Pole:
    """A single dispersion pole for ADE material modeling.

    Each pole represents one term in the expansion of a frequency-dependent
    material property. Poles can target either the density ρ(ω) or the
    bulk modulus K(ω).

    For Debye poles (first-order relaxation):
        χ(ω) = Δχ / (1 + iωτ)

        FDTD update:
        J^{n+1} = (τ/Δt · J^n + Δχ · f^{n+1}) / (1 + τ/Δt)

    For Lorentz poles (second-order resonance):
        χ(ω) = Δχ·ω₀² / (ω₀² - ω² + iγω)

        FDTD update:
        J^{n+1} = (a·J^n - b·J^{n-1} + Δχ·ω₀²·Δt²·f^n) / c
        where a = 2 - ω₀²Δt², b = 1 - γΔt/2, c = 1 + γΔt/2

    Args:
        pole_type: Type of pole (DEBYE or LORENTZ)
        delta_chi: Susceptibility strength (dimensionless)
        target: Which property this pole modifies ("density" or "modulus")
        tau: Relaxation time in seconds (Debye poles only)
        omega_0: Resonance angular frequency in rad/s (Lorentz poles only)
        gamma: Damping coefficient in rad/s (Lorentz poles only)

    Attributes:
        pole_type: The type of dispersion model
        delta_chi: The susceptibility strength
        target: Which material property is affected
        tau: Relaxation time (Debye)
        omega_0: Resonance frequency (Lorentz)
        gamma: Damping rate (Lorentz)

    Example:
        >>> # Debye pole for thermal relaxation in density
        >>> debye = Pole(
        ...     pole_type=PoleType.DEBYE,
        ...     delta_chi=0.3,
        ...     tau=1e-5,
        ...     target="density",
        ... )
        >>> # Lorentz pole for resonance in modulus
        >>> lorentz = Pole(
        ...     pole_type=PoleType.LORENTZ,
        ...     delta_chi=0.5,
        ...     omega_0=2*np.pi*1000,  # 1 kHz resonance
        ...     gamma=100,
        ...     target="modulus",
        ... )
    """

    pole_type: PoleType
    delta_chi: float
    target: Literal["density", "modulus"]

    # Debye parameters
    tau: float | None = None

    # Lorentz parameters
    omega_0: float | None = None
    gamma: float | None = None

    def __post_init__(self):
        """Validate pole parameters."""
        if self.pole_type == PoleType.DEBYE:
            if self.tau is None:
                raise ValueError("Debye poles require tau parameter")
            if self.tau <= 0:
                raise ValueError("tau must be positive")
        elif self.pole_type == PoleType.LORENTZ:
            if self.omega_0 is None or self.gamma is None:
                raise ValueError("Lorentz poles require omega_0 and gamma parameters")
            if self.omega_0 <= 0:
                raise ValueError("omega_0 must be positive")
            if self.gamma < 0:
                raise ValueError("gamma must be non-negative")

        if self.target not in ("density", "modulus"):
            raise ValueError("target must be 'density' or 'modulus'")

    def susceptibility(self, omega: float | NDArray[np.floating]) -> complex | NDArray[np.complexfloating]:
        """Compute complex susceptibility at given angular frequency.

        Args:
            omega: Angular frequency in rad/s (can be array)

        Returns:
            Complex susceptibility χ(ω)
        """
        omega = np.asarray(omega)

        if self.pole_type == PoleType.DEBYE:
            return self.delta_chi / (1 + 1j * omega * self.tau)
        else:  # LORENTZ
            return (
                self.delta_chi * self.omega_0**2
                / (self.omega_0**2 - omega**2 + 1j * self.gamma * omega)
            )

    def fdtd_coefficients(self, dt: float) -> tuple[float, ...]:
        """Compute FDTD update coefficients for this pole.

        Args:
            dt: Timestep in seconds

        Returns:
            For Debye: (alpha, beta) where
                J^{n+1} = alpha * J^n + beta * f^{n+1}

            For Lorentz: (a, b, c, d) where
                J^{n+1} = a*J^n + b*J^{n-1} + d*f^n
                (c is the normalization factor, pre-applied to a, b, d)
        """
        if self.pole_type == PoleType.DEBYE:
            tau_dt = self.tau / dt
            denom = 1 + tau_dt
            alpha = tau_dt / denom
            beta = self.delta_chi / denom
            return (alpha, beta)

        else:  # LORENTZ
            w0 = self.omega_0
            g = self.gamma
            dt2 = dt * dt

            c = 1 + g * dt / 2
            a = (2 - w0**2 * dt2) / c
            b = -(1 - g * dt / 2) / c
            d = self.delta_chi * w0**2 * dt2 / c

            return (a, b, d)

    @property
    def is_debye(self) -> bool:
        """Check if this is a Debye pole."""
        return self.pole_type == PoleType.DEBYE

    @property
    def is_lorentz(self) -> bool:
        """Check if this is a Lorentz pole."""
        return self.pole_type == PoleType.LORENTZ

    def __repr__(self) -> str:
        if self.is_debye:
            return (
                f"Pole(DEBYE, Δχ={self.delta_chi:.3g}, τ={self.tau:.2e}s, "
                f"target={self.target})"
            )
        else:
            f0 = self.omega_0 / (2 * np.pi)
            return (
                f"Pole(LORENTZ, Δχ={self.delta_chi:.3g}, f₀={f0:.1f}Hz, "
                f"γ={self.gamma:.2e}, target={self.target})"
            )


@dataclass
class AcousticMaterial(ABC):
    """Abstract base class for frequency-dependent acoustic materials.

    Acoustic materials are characterized by their effective density ρ(ω) and
    bulk modulus K(ω), both of which can be frequency-dependent. This class
    provides the interface for defining materials and computing their
    frequency response.

    The material properties are decomposed as:
        ρ(ω) = ρ∞ · (1 + Σᵢ χᵢ(ω))  for density poles
        K(ω) = K∞ · (1 + Σⱼ χⱼ(ω))  for modulus poles

    where ρ∞ and K∞ are the high-frequency (instantaneous) values and
    χ(ω) are the susceptibility contributions from each pole.

    Subclasses must implement:
        - rho_inf: High-frequency density
        - K_inf: High-frequency bulk modulus
        - poles: List of Pole objects

    Args:
        name: Human-readable material name

    Attributes:
        name: Material identifier
        rho_inf: Instantaneous density in kg/m³
        K_inf: Instantaneous bulk modulus in Pa
        poles: List of dispersion poles

    Example:
        >>> # Get effective properties at 1 kHz
        >>> omega = 2 * np.pi * 1000
        >>> rho = material.effective_density(omega)
        >>> K = material.effective_modulus(omega)
        >>> c = material.phase_velocity(omega)
        >>> Z = material.impedance(omega)
    """

    name: str = "unnamed"

    @property
    @abstractmethod
    def rho_inf(self) -> float:
        """High-frequency (instantaneous) density in kg/m³."""
        ...

    @property
    @abstractmethod
    def K_inf(self) -> float:
        """High-frequency (instantaneous) bulk modulus in Pa."""
        ...

    @property
    @abstractmethod
    def poles(self) -> list[Pole]:
        """List of dispersion poles for this material."""
        ...

    @property
    def density_poles(self) -> list[Pole]:
        """Poles that affect density."""
        return [p for p in self.poles if p.target == "density"]

    @property
    def modulus_poles(self) -> list[Pole]:
        """Poles that affect bulk modulus."""
        return [p for p in self.poles if p.target == "modulus"]

    @property
    def c_inf(self) -> float:
        """High-frequency speed of sound in m/s."""
        return np.sqrt(self.K_inf / self.rho_inf)

    @property
    def Z_inf(self) -> float:
        """High-frequency acoustic impedance in Pa·s/m."""
        return self.rho_inf * self.c_inf

    def effective_density(
        self, omega: float | NDArray[np.floating]
    ) -> complex | NDArray[np.complexfloating]:
        """Compute effective density at given angular frequency.

        ρ(ω) = ρ∞ · (1 + Σᵢ χᵢ(ω))

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Complex effective density. Real part is the inertial density,
            imaginary part represents losses.
        """
        omega = np.asarray(omega)
        result = np.ones_like(omega, dtype=complex)

        for pole in self.density_poles:
            result += pole.susceptibility(omega)

        return self.rho_inf * result

    def effective_modulus(
        self, omega: float | NDArray[np.floating]
    ) -> complex | NDArray[np.complexfloating]:
        """Compute effective bulk modulus at given angular frequency.

        K(ω) = K∞ · (1 + Σⱼ χⱼ(ω))

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Complex effective modulus. Real part is the elastic modulus,
            imaginary part represents losses.
        """
        omega = np.asarray(omega)
        result = np.ones_like(omega, dtype=complex)

        for pole in self.modulus_poles:
            result += pole.susceptibility(omega)

        return self.K_inf * result

    def wavenumber(
        self, omega: float | NDArray[np.floating]
    ) -> complex | NDArray[np.complexfloating]:
        """Compute complex wavenumber at given angular frequency.

        k(ω) = ω / c(ω) = ω · √(ρ(ω) / K(ω))

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Complex wavenumber. Real part is spatial frequency,
            imaginary part is attenuation per meter.
        """
        omega = np.asarray(omega)
        rho = self.effective_density(omega)
        K = self.effective_modulus(omega)
        return omega * np.sqrt(rho / K)

    def phase_velocity(
        self, omega: float | NDArray[np.floating]
    ) -> float | NDArray[np.floating]:
        """Compute phase velocity at given angular frequency.

        c_phase(ω) = ω / Re(k(ω))

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Phase velocity in m/s
        """
        k = self.wavenumber(omega)
        return np.real(omega / k)

    def group_velocity(
        self, omega: float | NDArray[np.floating], domega: float = 1.0
    ) -> float | NDArray[np.floating]:
        """Compute group velocity using finite difference.

        c_group(ω) = dω/dk ≈ Δω / Re(Δk)

        Args:
            omega: Angular frequency in rad/s
            domega: Frequency step for finite difference

        Returns:
            Group velocity in m/s
        """
        omega = np.asarray(omega)
        k1 = self.wavenumber(omega - domega / 2)
        k2 = self.wavenumber(omega + domega / 2)
        return domega / np.real(k2 - k1)

    def impedance(
        self, omega: float | NDArray[np.floating]
    ) -> complex | NDArray[np.complexfloating]:
        """Compute characteristic impedance at given angular frequency.

        Z(ω) = √(ρ(ω) · K(ω))

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Complex characteristic impedance in Pa·s/m
        """
        omega = np.asarray(omega)
        rho = self.effective_density(omega)
        K = self.effective_modulus(omega)
        return np.sqrt(rho * K)

    def attenuation(
        self, omega: float | NDArray[np.floating]
    ) -> float | NDArray[np.floating]:
        """Compute attenuation coefficient at given angular frequency.

        α(ω) = Im(k(ω))

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Attenuation coefficient in Np/m (nepers per meter)
        """
        k = self.wavenumber(omega)
        return np.imag(k)

    def attenuation_db_per_meter(
        self, omega: float | NDArray[np.floating]
    ) -> float | NDArray[np.floating]:
        """Compute attenuation in dB/m.

        α_dB(ω) = 20 · log10(e) · Im(k(ω)) ≈ 8.686 · Im(k(ω))

        Args:
            omega: Angular frequency in rad/s

        Returns:
            Attenuation coefficient in dB/m
        """
        return 8.685889638 * self.attenuation(omega)

    def reflection_coefficient(
        self, omega: float | NDArray[np.floating], Z_other: complex
    ) -> complex | NDArray[np.complexfloating]:
        """Compute reflection coefficient at interface with another medium.

        R = (Z - Z_other) / (Z + Z_other)

        Args:
            omega: Angular frequency in rad/s
            Z_other: Impedance of the other medium

        Returns:
            Complex reflection coefficient
        """
        Z = self.impedance(omega)
        return (Z - Z_other) / (Z + Z_other)

    def absorption_coefficient(
        self, omega: float | NDArray[np.floating], Z_backing: complex
    ) -> float | NDArray[np.floating]:
        """Compute absorption coefficient at interface.

        α_abs = 1 - |R|²

        Args:
            omega: Angular frequency in rad/s
            Z_backing: Impedance of backing medium

        Returns:
            Absorption coefficient (0 to 1)
        """
        R = self.reflection_coefficient(omega, Z_backing)
        return 1 - np.abs(R) ** 2

    def n_poles(self) -> int:
        """Total number of dispersion poles."""
        return len(self.poles)

    def n_density_poles(self) -> int:
        """Number of density poles."""
        return len(self.density_poles)

    def n_modulus_poles(self) -> int:
        """Number of modulus poles."""
        return len(self.modulus_poles)

    def summary(self) -> str:
        """Generate a text summary of material properties."""
        lines = [
            f"Material: {self.name}",
            f"  ρ∞ = {self.rho_inf:.2f} kg/m³",
            f"  K∞ = {self.K_inf:.2e} Pa",
            f"  c∞ = {self.c_inf:.1f} m/s",
            f"  Z∞ = {self.Z_inf:.1f} Pa·s/m",
            f"  Poles: {self.n_poles()} ({self.n_density_poles()} density, "
            f"{self.n_modulus_poles()} modulus)",
        ]

        # Add pole details
        for i, pole in enumerate(self.poles):
            lines.append(f"    [{i}] {pole}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', poles={self.n_poles()})"


@dataclass
class SimpleMaterial(AcousticMaterial):
    """A simple material with constant properties (no dispersion).

    This is useful for representing air, water, or other media with
    negligible frequency dependence over the frequency range of interest.

    Args:
        name: Material name
        rho: Density in kg/m³
        c: Speed of sound in m/s

    Example:
        >>> air = SimpleMaterial(name="air_20C", rho=1.2, c=343.0)
        >>> print(air.Z_inf)  # Impedance
        411.6
    """

    _rho: float = field(default=1.2, repr=False)
    _c: float = field(default=343.0, repr=False)
    _poles: list[Pole] = field(default_factory=list, repr=False)

    @property
    def rho_inf(self) -> float:
        return self._rho

    @property
    def K_inf(self) -> float:
        return self._rho * self._c**2

    @property
    def poles(self) -> list[Pole]:
        return self._poles

    @classmethod
    def from_impedance(cls, name: str, rho: float, Z: float) -> SimpleMaterial:
        """Create material from density and impedance.

        Args:
            name: Material name
            rho: Density in kg/m³
            Z: Acoustic impedance in Pa·s/m

        Returns:
            SimpleMaterial with specified properties
        """
        c = Z / rho
        return cls(name=name, _rho=rho, _c=c)


# Convenience aliases
Material = AcousticMaterial

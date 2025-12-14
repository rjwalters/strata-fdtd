"""Porous acoustic materials using Johnson-Champoux-Allard model.

This module implements the JCA (Johnson-Champoux-Allard) model for
porous absorbers like fiberglass, mineral wool, and acoustic foam.
The model captures frequency-dependent viscous and thermal losses.

The JCA model describes:
- Effective density ρ(ω) with viscous losses in pores
- Effective modulus K(ω) with thermal losses in pores

References:
    - Johnson et al., "Theory of dynamic permeability and tortuosity
      in fluid-saturated porous media" (1987)
    - Champoux & Allard, "Dynamic tortuosity and bulk modulus in
      air-saturated porous media" (1991)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .base import AcousticMaterial, Pole, PoleType

# Physical constants for air at 20°C
AIR_DENSITY = 1.204  # kg/m³
AIR_VISCOSITY = 1.81e-5  # Pa·s (dynamic viscosity)
AIR_PRANDTL = 0.71  # Prandtl number
AIR_GAMMA = 1.4  # Ratio of specific heats
AIR_PRESSURE = 101325  # Pa (atmospheric pressure)
AIR_SPEED = 343.0  # m/s


@dataclass
class PorousMaterial(AcousticMaterial):
    """Porous absorber material using Johnson-Champoux-Allard model.

    The JCA model captures frequency-dependent behavior through five
    macroscopic parameters that describe the pore structure:

    Args:
        name: Material name
        flow_resistivity: Static air flow resistivity σ in Pa·s/m²
            (typical range: 5000-100000)
        porosity: Open porosity φ, volume fraction of air (0-1)
            (typical range: 0.9-0.99)
        tortuosity: Tortuosity α∞, path length ratio (≥1)
            (typical range: 1.0-2.0)
        viscous_length: Viscous characteristic length Λ in meters
            (typical range: 50-500 μm)
        thermal_length: Thermal characteristic length Λ' in meters
            (typical range: 100-1000 μm, often ~2*Λ)
        thickness: Material thickness in meters (for impedance tube)
        n_poles: Number of Debye poles for approximation (default: 6)
        rho_fluid: Density of saturating fluid (default: air)
        eta: Dynamic viscosity of fluid (default: air)
        Pr: Prandtl number (default: air)
        gamma: Ratio of specific heats (default: air)
        P0: Static pressure (default: atmospheric)

    Attributes:
        flow_resistivity: σ in Pa·s/m²
        porosity: φ (dimensionless)
        tortuosity: α∞ (dimensionless)
        viscous_length: Λ in m
        thermal_length: Λ' in m

    Example:
        >>> # Fiberglass insulation (48 kg/m³)
        >>> fiberglass = PorousMaterial(
        ...     name="fiberglass_48",
        ...     flow_resistivity=25000,
        ...     porosity=0.98,
        ...     tortuosity=1.04,
        ...     viscous_length=100e-6,
        ...     thermal_length=200e-6,
        ... )
        >>> # Absorption at 1 kHz
        >>> omega = 2 * np.pi * 1000
        >>> alpha = fiberglass.absorption_coefficient(omega, Z_air)
    """

    flow_resistivity: float = 10000.0  # Pa·s/m²
    porosity: float = 0.95  # φ
    tortuosity: float = 1.05  # α∞
    viscous_length: float = 200e-6  # Λ (m)
    thermal_length: float = 400e-6  # Λ' (m)
    thickness: float | None = None  # Material thickness (m)
    num_poles: int = 6  # Number of approximation poles

    # Fluid properties (default: air at 20°C)
    rho_fluid: float = AIR_DENSITY
    eta: float = AIR_VISCOSITY
    Pr: float = AIR_PRANDTL
    gamma: float = AIR_GAMMA
    P0: float = AIR_PRESSURE

    # Cached poles (computed on first access)
    _poles_cache: list[Pole] | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        if self.flow_resistivity <= 0:
            raise ValueError("flow_resistivity must be positive")
        if not 0 < self.porosity <= 1:
            raise ValueError("porosity must be in (0, 1]")
        if self.tortuosity < 1:
            raise ValueError("tortuosity must be >= 1")
        if self.viscous_length <= 0:
            raise ValueError("viscous_length must be positive")
        if self.thermal_length <= 0:
            raise ValueError("thermal_length must be positive")
        if self.num_poles < 2:
            raise ValueError("num_poles must be >= 2")

    @property
    def rho_inf(self) -> float:
        """High-frequency limit of effective density.

        At high frequencies, inertial effects dominate and:
        ρ∞ = ρ_fluid * α∞ / φ
        """
        return self.rho_fluid * self.tortuosity / self.porosity

    @property
    def K_inf(self) -> float:
        """High-frequency limit of effective bulk modulus.

        At high frequencies, adiabatic process:
        K∞ = γ * P0 / φ
        """
        return self.gamma * self.P0 / self.porosity

    @property
    def omega_viscous(self) -> float:
        """Viscous characteristic frequency (rad/s).

        ω_v = σ * φ / (ρ_f * α∞)
        """
        return (
            self.flow_resistivity * self.porosity
            / (self.rho_fluid * self.tortuosity)
        )

    @property
    def omega_thermal(self) -> float:
        """Thermal characteristic frequency (rad/s).

        ω_t = η / (ρ_f * Pr * Λ'²)
        """
        return self.eta / (self.rho_fluid * self.Pr * self.thermal_length**2)

    @property
    def poles(self) -> list[Pole]:
        """Generate Debye poles for JCA approximation.

        Uses log-spaced frequencies covering the relevant range,
        fitting Debye poles to match the JCA response.
        """
        if self._poles_cache is not None:
            return self._poles_cache

        self._poles_cache = self._fit_jca_poles()
        return self._poles_cache

    def _fit_jca_poles(self) -> list[Pole]:
        """Fit Debye poles to JCA frequency response.

        Uses a simple approach: log-spaced poles covering the
        frequency range from 0.01*ω_v to 100*ω_v.
        """
        poles = []

        # Frequency range for fitting (based on characteristic frequencies)
        omega_min = 0.01 * min(self.omega_viscous, self.omega_thermal)
        omega_max = 100 * max(self.omega_viscous, self.omega_thermal)

        # Log-spaced frequencies for poles
        n_density = self.num_poles // 2
        n_modulus = self.num_poles - n_density

        # Fit density poles (viscous effects)
        tau_density = np.logspace(
            np.log10(1 / omega_max), np.log10(1 / omega_min), n_density
        )
        delta_chi_density = self._compute_density_pole_strengths(tau_density)

        for tau, dchi in zip(tau_density, delta_chi_density, strict=True):
            if abs(dchi) > 1e-6:
                poles.append(Pole(
                    pole_type=PoleType.DEBYE,
                    delta_chi=dchi,
                    tau=tau,
                    target="density",
                ))

        # Fit modulus poles (thermal effects)
        tau_modulus = np.logspace(
            np.log10(1 / omega_max), np.log10(1 / omega_min), n_modulus
        )
        delta_chi_modulus = self._compute_modulus_pole_strengths(tau_modulus)

        for tau, dchi in zip(tau_modulus, delta_chi_modulus, strict=True):
            if abs(dchi) > 1e-6:
                poles.append(Pole(
                    pole_type=PoleType.DEBYE,
                    delta_chi=dchi,
                    tau=tau,
                    target="modulus",
                ))

        return poles

    def _compute_density_pole_strengths(
        self, tau_values: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute pole strengths to match JCA density response.

        Uses a simplified least-squares fit over log-spaced frequencies.
        """
        # Sample frequencies
        omega = np.logspace(1, 6, 100)  # 10 Hz to 1 MHz

        # Target response (normalized by rho_inf)
        target = self._jca_density_normalized(omega)

        # Build design matrix for Debye poles
        # χ_total = Σ δχ_i / (1 + iω*τ_i)
        n_poles = len(tau_values)
        A = np.zeros((len(omega), n_poles), dtype=complex)
        for j, tau in enumerate(tau_values):
            A[:, j] = 1 / (1 + 1j * omega * tau)

        # Least squares fit (real part approximation)
        # Solve for coefficients that minimize |A @ x - target|
        target_real = np.real(target) - 1  # Subtract 1 (the rho_inf normalization)
        target_imag = np.imag(target)

        A_real = np.real(A)
        A_imag = np.imag(A)

        # Stack real and imaginary parts
        A_stacked = np.vstack([A_real, A_imag])
        target_stacked = np.concatenate([target_real, target_imag])

        # Solve with regularization
        delta_chi, _, _, _ = np.linalg.lstsq(A_stacked, target_stacked, rcond=None)

        return delta_chi

    def _compute_modulus_pole_strengths(
        self, tau_values: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute pole strengths to match JCA modulus response."""
        # Sample frequencies
        omega = np.logspace(1, 6, 100)

        # Target response (normalized)
        target = self._jca_modulus_normalized(omega)

        # Build design matrix
        n_poles = len(tau_values)
        A = np.zeros((len(omega), n_poles), dtype=complex)
        for j, tau in enumerate(tau_values):
            A[:, j] = 1 / (1 + 1j * omega * tau)

        # Least squares fit
        target_real = np.real(target) - 1
        target_imag = np.imag(target)

        A_real = np.real(A)
        A_imag = np.imag(A)

        A_stacked = np.vstack([A_real, A_imag])
        target_stacked = np.concatenate([target_real, target_imag])

        delta_chi, _, _, _ = np.linalg.lstsq(A_stacked, target_stacked, rcond=None)

        return delta_chi

    def _jca_density_normalized(
        self, omega: NDArray[np.floating]
    ) -> NDArray[np.complexfloating]:
        """Compute normalized JCA density (ρ/ρ∞).

        Johnson model for dynamic density:
        ρ(ω) = ρ_f * α∞ / φ * [1 + σ*φ/(iω*ρ_f*α∞) * sqrt(1 + iω*4*α∞²*η*ρ_f/(σ²*Λ²*φ²))]
        """
        omega = np.asarray(omega) + 1e-20  # Avoid division by zero

        sigma = self.flow_resistivity
        phi = self.porosity
        alpha = self.tortuosity
        Lambda = self.viscous_length
        rho_f = self.rho_fluid
        eta = self.eta

        # Dimensionless frequency parameter
        omega_tilde = 4 * alpha**2 * eta * rho_f / (sigma**2 * Lambda**2 * phi**2)

        # JCA density correction factor
        F = np.sqrt(1 + 1j * omega * omega_tilde)
        G = 1 + sigma * phi / (1j * omega * rho_f * alpha) * F

        return G

    def _jca_modulus_normalized(
        self, omega: NDArray[np.floating]
    ) -> NDArray[np.complexfloating]:
        """Compute normalized JCA modulus (K/K∞).

        Champoux-Allard model for dynamic bulk modulus:
        K(ω) = γ*P0/φ / [γ - (γ-1)/(1 + 8*η/(iω*Pr*ρ_f*Λ'²) * sqrt(1 + iω*Pr*ρ_f*Λ'²/(16*η)))]
        """
        omega = np.asarray(omega) + 1e-20

        gamma = self.gamma
        eta = self.eta
        Pr = self.Pr
        rho_f = self.rho_fluid
        Lambda_p = self.thermal_length

        # Dimensionless frequency parameter
        omega_tilde_p = Pr * rho_f * Lambda_p**2 / (16 * eta)

        # Champoux-Allard correction factor
        F_p = np.sqrt(1 + 1j * omega * omega_tilde_p)
        G_p = 1 + 8 * eta / (1j * omega * Pr * rho_f * Lambda_p**2) * F_p

        # Normalized modulus
        K_norm = gamma / (gamma - (gamma - 1) / G_p)

        return K_norm

    def jca_density(
        self, omega: float | NDArray[np.floating]
    ) -> complex | NDArray[np.complexfloating]:
        """Compute exact JCA effective density.

        This is the full analytical JCA model, not the pole approximation.
        Useful for validation and comparison.
        """
        return self.rho_inf * self._jca_density_normalized(np.asarray(omega))

    def jca_modulus(
        self, omega: float | NDArray[np.floating]
    ) -> complex | NDArray[np.complexfloating]:
        """Compute exact JCA effective modulus.

        This is the full analytical JCA model, not the pole approximation.
        """
        return self.K_inf * self._jca_modulus_normalized(np.asarray(omega))

    def jca_impedance(
        self, omega: float | NDArray[np.floating]
    ) -> complex | NDArray[np.complexfloating]:
        """Compute exact JCA characteristic impedance."""
        rho = self.jca_density(omega)
        K = self.jca_modulus(omega)
        return np.sqrt(rho * K)

    def jca_wavenumber(
        self, omega: float | NDArray[np.floating]
    ) -> complex | NDArray[np.complexfloating]:
        """Compute exact JCA wavenumber."""
        omega = np.asarray(omega)
        rho = self.jca_density(omega)
        K = self.jca_modulus(omega)
        return omega * np.sqrt(rho / K)

    def surface_impedance(
        self,
        omega: float | NDArray[np.floating],
        backing: str = "rigid",
    ) -> complex | NDArray[np.complexfloating]:
        """Compute surface impedance for finite thickness layer.

        Args:
            omega: Angular frequency in rad/s
            backing: "rigid" for rigid backing, or impedance value

        Returns:
            Surface impedance Z_s
        """
        if self.thickness is None:
            raise ValueError("thickness must be set to compute surface impedance")

        omega = np.asarray(omega)
        Z_c = self.jca_impedance(omega)
        k = self.jca_wavenumber(omega)
        d = self.thickness

        if backing == "rigid":
            # Rigid backing: Z_s = -iZ_c * cot(k*d)
            return -1j * Z_c / np.tan(k * d)
        else:
            # General backing impedance
            Z_b = float(backing)
            return Z_c * (Z_b + 1j * Z_c * np.tan(k * d)) / (Z_c + 1j * Z_b * np.tan(k * d))

    def absorption_coefficient_layer(
        self,
        omega: float | NDArray[np.floating],
        Z_air: float = 413.0,
        backing: str = "rigid",
    ) -> float | NDArray[np.floating]:
        """Compute absorption coefficient for finite thickness layer.

        Args:
            omega: Angular frequency in rad/s
            Z_air: Air impedance (default: 413 Pa·s/m for 20°C)
            backing: "rigid" or backing impedance value

        Returns:
            Normal incidence absorption coefficient (0-1)
        """
        Z_s = self.surface_impedance(omega, backing)
        R = (Z_s - Z_air) / (Z_s + Z_air)
        return 1 - np.abs(R) ** 2

    def validate_pole_fit(
        self, omega: NDArray[np.floating], plot: bool = False
    ) -> dict:
        """Validate pole approximation against exact JCA model.

        Args:
            omega: Angular frequencies to test
            plot: If True, generate comparison plots (requires matplotlib)

        Returns:
            Dict with max errors for density, modulus, impedance
        """
        # Exact JCA
        rho_exact = self.jca_density(omega)
        K_exact = self.jca_modulus(omega)

        # Pole approximation
        rho_approx = self.effective_density(omega)
        K_approx = self.effective_modulus(omega)

        # Relative errors
        rho_error = np.abs(rho_approx - rho_exact) / np.abs(rho_exact)
        K_error = np.abs(K_approx - K_exact) / np.abs(K_exact)

        results = {
            "density_max_error": float(np.max(rho_error)),
            "density_mean_error": float(np.mean(rho_error)),
            "modulus_max_error": float(np.max(K_error)),
            "modulus_mean_error": float(np.mean(K_error)),
        }

        if plot:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            freq = omega / (2 * np.pi)

            # Density comparison
            axes[0, 0].semilogx(freq, np.real(rho_exact), 'b-', label='JCA exact')
            axes[0, 0].semilogx(freq, np.real(rho_approx), 'r--', label='Pole approx')
            axes[0, 0].set_xlabel('Frequency (Hz)')
            axes[0, 0].set_ylabel('Re(ρ) (kg/m³)')
            axes[0, 0].legend()
            axes[0, 0].set_title('Effective Density (Real)')

            axes[0, 1].semilogx(freq, -np.imag(rho_exact), 'b-', label='JCA exact')
            axes[0, 1].semilogx(freq, -np.imag(rho_approx), 'r--', label='Pole approx')
            axes[0, 1].set_xlabel('Frequency (Hz)')
            axes[0, 1].set_ylabel('-Im(ρ) (kg/m³)')
            axes[0, 1].legend()
            axes[0, 1].set_title('Effective Density (Imaginary)')

            # Modulus comparison
            axes[1, 0].semilogx(freq, np.real(K_exact), 'b-', label='JCA exact')
            axes[1, 0].semilogx(freq, np.real(K_approx), 'r--', label='Pole approx')
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('Re(K) (Pa)')
            axes[1, 0].legend()
            axes[1, 0].set_title('Effective Modulus (Real)')

            axes[1, 1].semilogx(freq, np.imag(K_exact), 'b-', label='JCA exact')
            axes[1, 1].semilogx(freq, np.imag(K_approx), 'r--', label='Pole approx')
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('Im(K) (Pa)')
            axes[1, 1].legend()
            axes[1, 1].set_title('Effective Modulus (Imaginary)')

            plt.tight_layout()
            plt.show()

        return results

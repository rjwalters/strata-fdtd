"""
Example: Frequency Sweep - Chirp Source for Bandwidth Testing
=============================================================
Demonstrates using a chirp (frequency sweep) source to characterize
the frequency response of acoustic systems. A linear chirp sweeps
through a range of frequencies, allowing broadband analysis in a
single simulation.

Expected runtime: ~30 seconds on modern hardware (with native backend)
Output: results.h5 (pressure snapshots and probe data)

Grid: 100 × 100 × 100 cells @ 1mm resolution
Domain: 100mm × 100mm × 100mm
Source: Linear chirp from 1 kHz to 20 kHz
Analysis: FFT to extract frequency response

Learning objectives:
- Creating chirp (swept-frequency) sources
- Understanding broadband excitation
- Performing FFT analysis on simulation results
- Measuring system frequency response
"""

import numpy as np

from strata_fdtd import (
    FDTDSolver,
)

# =============================================================================
# Chirp Source Implementation
# =============================================================================
# A chirp is a sinusoid with time-varying frequency:
#   x(t) = A × sin(2π × f(t) × t)
#
# For a linear chirp from f0 to f1 over duration T:
#   f(t) = f0 + (f1 - f0) × t / T
#
# The instantaneous phase is:
#   φ(t) = 2π × ∫f(t)dt = 2π × (f0×t + (f1-f0)×t²/(2T))


class ChirpSource:
    """Linear frequency sweep (chirp) source.

    Sweeps from start_freq to end_freq over the chirp_duration.
    """

    def __init__(
        self,
        position: tuple[float, float, float],
        start_freq: float,
        end_freq: float,
        chirp_duration: float,
        amplitude: float = 1.0,
    ):
        self.position = position
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.chirp_duration = chirp_duration
        self.amplitude = amplitude
        self.source_type = "point"  # For solver compatibility

    @property
    def frequency(self) -> float:
        """Center frequency of the chirp (for HDF5 metadata compatibility)."""
        return (self.start_freq + self.end_freq) / 2

    def waveform(self, t: np.ndarray, dt: float) -> np.ndarray:
        """Generate chirp waveform at given times.

        Args:
            t: Array of time values in seconds
            dt: Timestep in seconds (for reference)

        Returns:
            Array of pressure values at each time
        """
        # Linear chirp phase
        f0 = self.start_freq
        f1 = self.end_freq
        T = self.chirp_duration

        # Smooth amplitude envelope (raised cosine at edges)
        rise_time = 0.0001  # 0.1ms rise time
        envelope = np.ones_like(t)

        # Smooth rise
        rise_mask = t < rise_time
        envelope[rise_mask] = 0.5 * (1 - np.cos(np.pi * t[rise_mask] / rise_time))

        # Smooth fall
        fall_start = T - rise_time
        fall_mask = t > fall_start
        envelope[fall_mask] = 0.5 * (1 + np.cos(np.pi * (t[fall_mask] - fall_start) / rise_time))

        # Zero after chirp ends
        envelope[t > T] = 0

        # Instantaneous phase for linear chirp
        # φ(t) = 2π × (f0×t + (f1-f0)×t²/(2T))
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * T))

        return self.amplitude * envelope * np.sin(phase)


# =============================================================================
# Simulation Setup
# =============================================================================

solver = FDTDSolver(
    shape=(100, 100, 100),
    resolution=1e-3  # 1mm resolution
)

# Chirp parameters
start_freq = 1000   # 1 kHz
end_freq = 20000    # 20 kHz
chirp_duration = 0.002  # 2ms sweep

# Create and add chirp source
source = ChirpSource(
    position=(0.025, 0.05, 0.05),  # 25mm from edge
    start_freq=start_freq,
    end_freq=end_freq,
    chirp_duration=chirp_duration,
    amplitude=1.0,
)

solver.add_source(source)

# Add probes at various distances
probe_distances = [20, 40, 60]  # mm from source

print("=" * 60)
print("FDTD Simulation: Chirp Source (Frequency Sweep)")
print("=" * 60)
print()
print("Chirp parameters:")
print(f"  Start frequency: {start_freq/1e3:.0f} kHz")
print(f"  End frequency: {end_freq/1e3:.0f} kHz")
print(f"  Sweep duration: {chirp_duration*1e3:.0f} ms")
print(f"  Sweep rate: {(end_freq - start_freq) / chirp_duration / 1e6:.1f} MHz/s")
print()

# Calculate wavelengths
c = solver.c
lambda_start = c / start_freq
lambda_end = c / end_freq
print(f"Wavelengths in air (c={c:.0f} m/s):")
print(f"  At {start_freq/1e3:.0f} kHz: λ = {lambda_start*1e3:.1f} mm")
print(f"  At {end_freq/1e3:.0f} kHz: λ = {lambda_end*1e3:.2f} mm")
print(f"  Grid resolution: {solver.dx*1e3:.1f} mm")
print(f"  Points per wavelength (at {end_freq/1e3:.0f} kHz): {lambda_end/solver.dx:.1f}")
print()

# Add probes
source_x = 0.025
for dist in probe_distances:
    probe_x = source_x + dist * 1e-3
    if probe_x < 0.095:
        solver.add_probe(
            f"d{dist}mm",
            position=(probe_x, 0.05, 0.05),
        )
        print(f"  Probe at {dist}mm from source: x={probe_x*1e3:.0f}mm")

# Reference probe very close to source
solver.add_probe(
    "reference",
    position=(source_x + 0.005, 0.05, 0.05),  # 5mm from source
)

print()
print("Solver configuration:")
print(f"  Grid shape: {solver.grid.shape}")
print(f"  Domain: {solver.grid.physical_extent()[0]*1e3:.0f}mm cube")
print(f"  Timestep: {solver.dt*1e9:.3f} ns")
print(f"  Simulation duration: {(chirp_duration + 0.001)*1e3:.0f} ms")
print(f"  Using native backend: {solver.using_native}")
print("=" * 60)
print()

# Run simulation (chirp duration + propagation time)
total_duration = chirp_duration + 0.001  # Add 1ms for propagation
print("Running simulation...")
solver.run(duration=total_duration, output_file="results.h5")

print()
print("=" * 60)
print("✓ Simulation complete!")
print("=" * 60)
print(f"Output saved to: results.h5")
print(f"File size: {__import__('os').path.getsize('results.h5') / 1e6:.1f} MB")
print()

# =============================================================================
# Frequency Analysis
# =============================================================================

print("Frequency analysis:")
print("-" * 40)

# Helper function to compute spectrum
def compute_spectrum(data, dt):
    """Compute FFT spectrum of a signal."""
    n_fft = len(data)
    spectrum = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(n_fft, dt)
    return freqs, spectrum

# Get reference signal and compute its spectrum
ref_data = solver.get_probe_data("reference")["reference"]
ref_freqs, ref_spectrum = compute_spectrum(ref_data, solver.dt)

print(f"Reference signal: {len(ref_data)} samples")
print()

# Analyze each probe
print("Frequency response at each probe:")
print("-" * 40)

for dist in probe_distances:
    probe_name = f"d{dist}mm"
    try:
        data = solver.get_probe_data(probe_name)[probe_name]
        freqs, spectrum = compute_spectrum(data, solver.dt)

        # Find frequency range with significant energy
        # (10% of max reference spectrum)
        threshold = 0.1 * np.max(ref_spectrum)
        significant = spectrum > threshold

        if np.any(significant):
            sig_freqs = freqs[significant]
            min_freq = np.min(sig_freqs)
            max_freq = np.max(sig_freqs)

            # Calculate average magnitude in chirp range
            chirp_mask = (freqs >= start_freq) & (freqs <= end_freq)
            avg_mag = np.mean(spectrum[chirp_mask]) if np.any(chirp_mask) else 0

            print(f"  {probe_name}:")
            print(f"    Frequency range (>10% ref): {min_freq/1e3:.1f} - {max_freq/1e3:.1f} kHz")
            print(f"    Average magnitude: {avg_mag:.4f}")

            # Calculate RMS level
            rms = np.sqrt(np.mean(data**2))
            print(f"    RMS level: {rms:.4f}")
        else:
            print(f"  {probe_name}: Low signal level")

    except (KeyError, ValueError) as e:
        print(f"  {probe_name}: Unable to analyze ({e})")

# Find spectral peaks
print()
print("Spectral analysis summary:")
print("-" * 40)

ref_peak_idx = np.argmax(ref_spectrum)
ref_peak_freq = ref_freqs[ref_peak_idx]
print(f"Reference peak frequency: {ref_peak_freq/1e3:.1f} kHz")

# Check for expected chirp bandwidth
chirp_mask = (ref_freqs >= start_freq) & (ref_freqs <= end_freq)
chirp_energy = np.sum(ref_spectrum[chirp_mask]**2)
total_energy = np.sum(ref_spectrum**2)
chirp_fraction = chirp_energy / total_energy if total_energy > 0 else 0

print(f"Energy in chirp band ({start_freq/1e3:.0f}-{end_freq/1e3:.0f} kHz): {chirp_fraction*100:.1f}%")
print()

print("Key observations:")
print(f"- Chirp excites all frequencies from {start_freq/1e3:.0f} to {end_freq/1e3:.0f} kHz")
print("- Single simulation provides broadband frequency response")
print("- Distance affects arrival time but not frequency content")
print("- FFT analysis reveals the full frequency response")
print("- Useful for testing filters, resonators, and enclosures")
print()
print("Next steps:")
print("1. Upload results.h5 to the FDTD Viewer")
print("2. Watch the chirp propagate through the domain")
print("3. Use FFT tools to analyze probe frequency response")
print("4. Try adding geometry to measure its frequency response")
print()
print("Application examples:")
print("- Measure loudspeaker enclosure frequency response")
print("- Characterize acoustic filters and resonators")
print("- Test room/chamber acoustics")
print("- Identify resonant frequencies of structures")
print()

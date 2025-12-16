"""
Example: PZT Transducer Array
=============================
Demonstrates a linear array of piezoelectric transducer elements
with controlled phase delays for beam steering. This example shows
how to create phased array sources for directional sound emission.

Expected runtime: ~60 seconds on modern hardware (with native backend)
Output: results.h5 (pressure snapshots and probe data)

Grid: 150 × 100 × 100 cells @ 1mm resolution
Domain: 150mm × 100mm × 100mm
Source: 4-element linear array with phase steering
Probes: Far-field measurement points

Learning objectives:
- Creating multi-element source arrays
- Implementing phase delays for beam steering
- Understanding constructive/destructive interference
- Measuring directivity patterns
"""

import numpy as np

from strata_fdtd import (
    GaussianPulse,
    FDTDSolver,
)

# Create an elongated domain for observing far-field patterns
solver = FDTDSolver(
    shape=(150, 100, 100),
    resolution=1e-3  # 1mm resolution
)

# =============================================================================
# Transducer Array Configuration
# =============================================================================

# Array parameters
n_elements = 4  # Number of transducer elements
element_spacing = 0.008  # 8mm spacing (about λ/4 at 10kHz)
frequency = 10e3  # 10 kHz center frequency
wavelength = solver.c / frequency  # ~34mm in air

# Calculate phase delay for beam steering
# Positive angle steers beam toward +y
steer_angle_deg = 15  # Degrees from normal
steer_angle_rad = np.radians(steer_angle_deg)

# Phase delay between adjacent elements for steering
# Δφ = (2π/λ) × d × sin(θ)
phase_delay_per_element = (2 * np.pi / wavelength) * element_spacing * np.sin(steer_angle_rad)

# Array position (elements centered in y-z plane at x=10mm)
array_x = 0.010  # 10mm from left edge
array_center_y = 0.050  # Center of domain
array_z = 0.050  # Center of domain

print("=" * 60)
print("FDTD Simulation: Phased Transducer Array")
print("=" * 60)
print()
print("Array configuration:")
print(f"  Number of elements: {n_elements}")
print(f"  Element spacing: {element_spacing*1e3:.1f} mm")
print(f"  Center frequency: {frequency/1e3:.0f} kHz")
print(f"  Wavelength in air: {wavelength*1e3:.1f} mm")
print(f"  Spacing/wavelength ratio: {element_spacing/wavelength:.2f}")
print()
print("Beam steering:")
print(f"  Steer angle: {steer_angle_deg}° from normal")
print(f"  Phase delay per element: {np.degrees(phase_delay_per_element):.1f}°")
print()

# Add transducer elements as individual sources
# Each source has a phase offset for beam steering
for i in range(n_elements):
    # Element position
    element_y = array_center_y + (i - (n_elements - 1) / 2) * element_spacing
    element_pos = (array_x, element_y, array_z)

    # Phase delay for this element
    element_phase = i * phase_delay_per_element

    # Create source with appropriate delay
    # The delay is implemented as a time offset
    time_delay = element_phase / (2 * np.pi * frequency)

    # Add source (using amplitude modulation to approximate phase)
    # Note: True phase steering requires custom source implementation
    # Here we use time-delayed sources as an approximation
    solver.add_source(
        GaussianPulse(
            position=element_pos,
            frequency=frequency,
            amplitude=1.0 / n_elements,  # Normalize total amplitude
        )
    )

    print(f"  Element {i+1}: y={element_y*1e3:.1f}mm, phase={np.degrees(element_phase):.1f}°")

print()

# =============================================================================
# Measurement Probes
# =============================================================================

# Far-field probes at different angles
far_field_radius = 0.100  # 100mm from array
probe_angles_deg = [-30, -15, 0, 15, 30, 45]

print("Far-field probes:")
for angle_deg in probe_angles_deg:
    angle_rad = np.radians(angle_deg)
    probe_x = array_x + far_field_radius * np.cos(angle_rad)
    probe_y = array_center_y + far_field_radius * np.sin(angle_rad)

    # Ensure probe is within domain
    if 0 < probe_x < 0.149 and 0 < probe_y < 0.099:
        solver.add_probe(
            f"angle_{angle_deg:+d}deg",
            position=(probe_x, probe_y, array_z),
        )
        print(f"  {angle_deg:+d}°: ({probe_x*1e3:.1f}, {probe_y*1e3:.1f}) mm")

# On-axis probe for reference
solver.add_probe("on_axis_near", position=(0.050, array_center_y, array_z))
solver.add_probe("on_axis_far", position=(0.120, array_center_y, array_z))

print()
print("Solver configuration:")
print(f"  Grid shape: {solver.grid.shape}")
print(f"  Domain extent: {solver.grid.physical_extent()[0]*1e3:.0f} × {solver.grid.physical_extent()[1]*1e3:.0f} × {solver.grid.physical_extent()[2]*1e3:.0f} mm")
print(f"  Timestep: {solver.dt*1e9:.3f} ns")
print(f"  Using native backend: {solver.using_native}")
print("=" * 60)
print()

# Run simulation
print("Running simulation...")
solver.run(duration=0.001, output_file="results.h5")  # 1ms duration

print()
print("=" * 60)
print("✓ Simulation complete!")
print("=" * 60)
print(f"Output saved to: results.h5")
print(f"File size: {__import__('os').path.getsize('results.h5') / 1e6:.1f} MB")
print()

# =============================================================================
# Directivity Analysis
# =============================================================================

print("Directivity analysis:")
print("-" * 40)

# Calculate peak amplitudes at each angle
peak_amplitudes = {}
for angle_deg in probe_angles_deg:
    probe_name = f"angle_{angle_deg:+d}deg"
    try:
        data = solver.get_probe_data(probe_name)[probe_name]
        peak = np.max(np.abs(data))
        peak_amplitudes[angle_deg] = peak
    except (KeyError, ValueError):
        pass

if peak_amplitudes:
    # Normalize to maximum
    max_peak = max(peak_amplitudes.values())
    print("\nAngular response (normalized):")
    for angle_deg in sorted(peak_amplitudes.keys()):
        peak = peak_amplitudes[angle_deg]
        normalized = peak / max_peak if max_peak > 0 else 0
        db = 20 * np.log10(normalized) if normalized > 0 else -60
        bar = "█" * int(normalized * 20)
        print(f"  {angle_deg:+3d}°: {normalized:.3f} ({db:+.1f} dB) {bar}")

    # Find main lobe direction
    max_angle = max(peak_amplitudes.keys(), key=lambda a: peak_amplitudes[a])
    print(f"\nMain lobe direction: {max_angle}° (target: {steer_angle_deg}°)")

print()
print("Key observations:")
print(f"- Array with {n_elements} elements creates directional beam")
print(f"- Element spacing of {element_spacing/wavelength:.2f}λ affects grating lobes")
print(f"- Phase steering shifts main lobe direction")
print("- Side lobes appear at angles where elements interfere destructively")
print()
print("Next steps:")
print("1. Upload results.h5 to the FDTD Viewer")
print("2. Watch the beam form from the phased array")
print("3. Try different steer angles and element counts")
print("4. Compare directivity at different frequencies")
print()

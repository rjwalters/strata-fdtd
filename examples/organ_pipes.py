"""
Example: Organ Pipes - Helmholtz Resonator Array
================================================
Demonstrates an array of Helmholtz resonators (similar to organ pipes)
with different resonant frequencies. Each resonator responds selectively
to its tuned frequency, demonstrating acoustic filtering.

Expected runtime: ~90 seconds on modern hardware (with native backend)
Output: results.h5 (pressure snapshots and probe data)

Grid: 120 × 120 × 200 cells @ 1mm resolution
Domain: 120mm × 120mm × 200mm
Resonators: 4 quarter-wave tubes tuned to different frequencies
Source: Broadband pulse to excite multiple resonances

Learning objectives:
- Creating resonant acoustic structures
- Understanding quarter-wave resonators
- Observing frequency-selective response
- Analyzing standing wave patterns
"""

import numpy as np

from strata_fdtd import (
    GaussianPulse,
    FDTDSolver,
    Box,
    Difference,
)

# Create the solver with an elongated domain for the pipes
solver = FDTDSolver(
    shape=(120, 120, 200),
    resolution=1e-3  # 1mm resolution
)

# =============================================================================
# Quarter-Wave Resonator Theory
# =============================================================================
# A tube closed at one end and open at the other resonates at:
#   f_n = (2n-1) × c / (4L)
# where n=1,2,3... and L is the tube length.
#
# For n=1 (fundamental): L = c / (4f)

c = solver.c  # Speed of sound (343 m/s)

# Design resonators for specific frequencies
target_frequencies = [500, 700, 900, 1100]  # Hz
tube_lengths = [c / (4 * f) for f in target_frequencies]  # meters

print("=" * 60)
print("FDTD Simulation: Quarter-Wave Resonator Array")
print("=" * 60)
print()
print("Resonator design (quarter-wave tubes):")
print(f"  Speed of sound: {c:.0f} m/s")
print()
for i, (freq, length) in enumerate(zip(target_frequencies, tube_lengths)):
    print(f"  Tube {i+1}: f={freq} Hz, L={length*1e3:.1f} mm")
print()

# =============================================================================
# Create Resonator Geometry
# =============================================================================

# Tube parameters
tube_width = 0.015  # 15mm inner width
tube_wall = 0.003   # 3mm wall thickness
tube_spacing = 0.030  # 30mm center-to-center

# Position tubes in a row (in x direction)
domain_center_y = 0.060  # 60mm
tube_base_z = 0.010  # 10mm from bottom (open end)

# Create geometry mask
x = np.arange(120) * 1e-3 + 0.5e-3
y = np.arange(120) * 1e-3 + 0.5e-3
z = np.arange(200) * 1e-3 + 0.5e-3
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Start with air everywhere
geometry = np.ones(solver.shape, dtype=bool)

# Add tube walls (solid material)
tube_positions_x = []
for i, length in enumerate(tube_lengths):
    # Calculate tube center position
    tube_x = 0.025 + i * tube_spacing  # Start 25mm from edge
    tube_positions_x.append(tube_x)

    # Outer dimensions (with walls)
    outer_width = tube_width + 2 * tube_wall

    # Tube extends from base to base+length (closed end at top)
    tube_top = tube_base_z + length

    # Create outer box (solid)
    outer_box = (
        (np.abs(X - tube_x) < outer_width / 2) &
        (np.abs(Y - domain_center_y) < outer_width / 2) &
        (Z > tube_base_z) &
        (Z < tube_top)
    )

    # Create inner cavity (air)
    inner_box = (
        (np.abs(X - tube_x) < tube_width / 2) &
        (np.abs(Y - domain_center_y) < tube_width / 2) &
        (Z > tube_base_z) &
        (Z < tube_top - tube_wall)  # Leave wall at closed end
    )

    # Set outer as solid (False)
    geometry[outer_box] = False
    # Carve out inner as air (True)
    geometry[inner_box] = True

# Set geometry in solver
solver.set_geometry(geometry)

print("Geometry created:")
print(f"  Tube inner width: {tube_width*1e3:.0f} mm")
print(f"  Wall thickness: {tube_wall*1e3:.0f} mm")
print(f"  Tube spacing: {tube_spacing*1e3:.0f} mm")
for i, (pos_x, length) in enumerate(zip(tube_positions_x, tube_lengths)):
    print(f"  Tube {i+1} at x={pos_x*1e3:.0f}mm, height={length*1e3:.1f}mm")
print()

# =============================================================================
# Source and Probes
# =============================================================================

# Broadband source to excite all resonators
# Place source below the tube openings
source_pos = (0.060, domain_center_y, 0.005)  # Center x, below tubes
solver.add_source(
    GaussianPulse(
        position=source_pos,
        frequency=800,  # Center frequency between resonators
        bandwidth=1600,  # Wide bandwidth to cover all resonators
        amplitude=1.0,
    )
)

# Add probes inside each tube (near open end to detect resonance)
for i, pos_x in enumerate(tube_positions_x):
    # Probe near open end of tube
    probe_z = tube_base_z + 0.010  # 10mm into tube
    solver.add_probe(
        f"tube_{i+1}_open",
        position=(pos_x, domain_center_y, probe_z),
    )

    # Probe near closed end (pressure antinode)
    probe_z_closed = tube_base_z + tube_lengths[i] - 0.010  # 10mm from closed end
    solver.add_probe(
        f"tube_{i+1}_closed",
        position=(pos_x, domain_center_y, probe_z_closed),
    )

# Reference probe in open space
solver.add_probe("reference", position=(0.100, domain_center_y, 0.005))

print("Probes added:")
for i in range(len(tube_positions_x)):
    print(f"  tube_{i+1}_open: Near open end (velocity antinode)")
    print(f"  tube_{i+1}_closed: Near closed end (pressure antinode)")
print()

print("Solver configuration:")
print(f"  Grid shape: {solver.grid.shape}")
extent = solver.grid.physical_extent()
print(f"  Domain: {extent[0]*1e3:.0f} × {extent[1]*1e3:.0f} × {extent[2]*1e3:.0f} mm")
print(f"  Timestep: {solver.dt*1e9:.3f} ns")
print(f"  Using native backend: {solver.using_native}")
print("=" * 60)
print()

# Run simulation (longer to see resonance build up)
print("Running simulation...")
solver.run(duration=0.005, output_file="results.h5")  # 5ms for resonance

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

for i in range(len(tube_positions_x)):
    target_f = target_frequencies[i]
    probe_name = f"tube_{i+1}_closed"

    try:
        # Get probe data and compute spectrum manually
        data = solver.get_probe_data(probe_name)[probe_name]

        # Compute FFT
        n_fft = len(data)
        spectrum = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(n_fft, solver.dt)

        # Find peak frequency (ignore DC and low frequencies)
        freq_mask = freqs > 100
        peak_idx = np.argmax(spectrum[freq_mask])
        peak_freq = freqs[freq_mask][peak_idx]

        error = (peak_freq - target_f) / target_f * 100

        print(f"  Tube {i+1}: target={target_f} Hz, measured={peak_freq:.0f} Hz ({error:+.1f}%)")
    except (KeyError, ValueError) as e:
        print(f"  Tube {i+1}: Unable to analyze ({e})")

print()
print("Key observations:")
print("- Each tube resonates at its designed quarter-wave frequency")
print("- Closed end has pressure antinode (maximum amplitude)")
print("- Open end has velocity antinode (pressure node)")
print("- Longer tubes = lower frequencies")
print("- Real organ pipes use more complex geometries for tone shaping")
print()
print("Next steps:")
print("1. Upload results.h5 to the FDTD Viewer")
print("2. Watch standing waves form in each tube")
print("3. Use FFT analysis on probe data to verify resonant frequencies")
print("4. Try different tube lengths for musical intervals")
print()

"""
Example: Rectangular Acoustic Waveguide
=======================================
Demonstrates wave propagation in a rectangular duct (waveguide).
The waveguide supports discrete modes that propagate above their
cutoff frequencies.

Expected runtime: ~45 seconds on modern hardware (with native backend)
Output: results.h5 (pressure snapshots and probe data)

Grid: 64 × 64 × 256 cells @ 1mm resolution
Domain: 64mm × 64mm × 256mm (elongated duct)
Source: Gaussian pulse at duct entrance
Walls: Rigid boundaries (perfect reflection)

Learning objectives:
- Understanding waveguide modes and cutoff frequencies
- Observing mode shapes in the transverse plane
- Comparing propagation above and below cutoff
- Rigid boundary conditions
"""

import numpy as np

from strata_fdtd import (
    GaussianPulse,
    FDTDSolver,
)

# =============================================================================
# Waveguide Theory
# =============================================================================
# For a rectangular waveguide with dimensions a × b (in x and y),
# the cutoff frequency for mode (m, n) is:
#
#   f_c(m,n) = (c/2) × sqrt((m/a)² + (n/b)²)
#
# where m, n = 0, 1, 2, ... (but not both zero for acoustic modes).
#
# Only modes with f > f_c propagate; lower frequencies are evanescent.

# Duct dimensions
duct_width_x = 0.050   # 50mm in x
duct_width_y = 0.050   # 50mm in y (square cross-section)
duct_length_z = 0.250  # 250mm long

# Create solver with appropriate shape
# Add some padding for walls
nx = int(duct_width_x / 1e-3) + 14  # 64 cells
ny = int(duct_width_y / 1e-3) + 14  # 64 cells
nz = int(duct_length_z / 1e-3) + 6  # 256 cells

solver = FDTDSolver(
    shape=(nx, ny, nz),
    resolution=1e-3  # 1mm resolution
)

c = solver.c  # Speed of sound

# Calculate cutoff frequencies
a = duct_width_x
b = duct_width_y

# First few modes
modes = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)]
cutoff_freqs = {}

print("=" * 60)
print("FDTD Simulation: Rectangular Acoustic Waveguide")
print("=" * 60)
print()
print("Waveguide dimensions:")
print(f"  Cross-section: {a*1e3:.0f} mm × {b*1e3:.0f} mm")
print(f"  Length: {duct_length_z*1e3:.0f} mm")
print(f"  Speed of sound: {c:.0f} m/s")
print()
print("Mode cutoff frequencies:")
print("  (m,n)   f_cutoff")
print("  -----   --------")

for m, n in modes:
    f_c = (c / 2) * np.sqrt((m / a) ** 2 + (n / b) ** 2)
    cutoff_freqs[(m, n)] = f_c
    print(f"  ({m},{n})    {f_c:.0f} Hz")

print()
print(f"  Plane wave (0,0) propagates at all frequencies")
print(f"  Mode (1,0) cutoff: {cutoff_freqs[(1,0)]:.0f} Hz")
print()

# =============================================================================
# Create Waveguide Geometry
# =============================================================================

# Create geometry mask (True = air, False = solid wall)
x = np.arange(nx) * 1e-3 + 0.5e-3
y = np.arange(ny) * 1e-3 + 0.5e-3
z = np.arange(nz) * 1e-3 + 0.5e-3

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Wall thickness (cells outside duct are solid)
wall_thickness = 0.007  # 7mm walls

# Duct interior boundaries
x_min = wall_thickness
x_max = wall_thickness + duct_width_x
y_min = wall_thickness
y_max = wall_thickness + duct_width_y
z_min = 0.003  # Open entrance
z_max = z_min + duct_length_z

# Interior of waveguide is air
geometry = np.zeros(solver.shape, dtype=bool)  # Start with all solid
interior = (
    (X >= x_min) & (X <= x_max) &
    (Y >= y_min) & (Y <= y_max) &
    (Z >= z_min) & (Z <= z_max)
)
geometry[interior] = True

solver.set_geometry(geometry)

print("Geometry created:")
print(f"  Duct interior: x=[{x_min*1e3:.0f}, {x_max*1e3:.0f}] mm")
print(f"                 y=[{y_min*1e3:.0f}, {y_max*1e3:.0f}] mm")
print(f"                 z=[{z_min*1e3:.0f}, {z_max*1e3:.0f}] mm")
print()

# =============================================================================
# Source Configuration
# =============================================================================

# Source frequency: below (1,0) cutoff (plane wave only)
# and above (1,0) cutoff (higher modes can propagate)
source_freq = 5000  # 5 kHz - above (1,0) cutoff at ~3430 Hz

# Source position at duct entrance, centered
source_x = (x_min + x_max) / 2
source_y = (y_min + y_max) / 2
source_z = z_min + 0.005  # 5mm into duct

solver.add_source(
    GaussianPulse(
        position=(source_x, source_y, source_z),
        frequency=source_freq,
        amplitude=1.0,
    )
)

# Off-center source to excite higher modes
# Placing source at quarter-width excites (1,0) mode
source_x_offset = x_min + duct_width_x * 0.25
solver.add_source(
    GaussianPulse(
        position=(source_x_offset, source_y, source_z),
        frequency=source_freq,
        amplitude=0.5,
    )
)

print("Source configuration:")
print(f"  Frequency: {source_freq} Hz")
print(f"  (1,0) cutoff: {cutoff_freqs[(1,0)]:.0f} Hz → mode {'propagates' if source_freq > cutoff_freqs[(1,0)] else 'evanescent'}")
print(f"  (1,1) cutoff: {cutoff_freqs[(1,1)]:.0f} Hz → mode {'propagates' if source_freq > cutoff_freqs[(1,1)] else 'evanescent'}")
print(f"  Center source at ({source_x*1e3:.1f}, {source_y*1e3:.1f}, {source_z*1e3:.1f}) mm")
print(f"  Offset source at ({source_x_offset*1e3:.1f}, {source_y*1e3:.1f}, {source_z*1e3:.1f}) mm")
print()

# =============================================================================
# Measurement Probes
# =============================================================================

# Probes along the duct centerline
z_positions = [0.020, 0.060, 0.120, 0.180, 0.240]  # meters

print("Probes along centerline:")
for i, z_pos in enumerate(z_positions):
    if z_pos < z_max:
        solver.add_probe(
            f"center_z{int(z_pos*1e3)}mm",
            position=(source_x, source_y, z_pos),
        )
        print(f"  z={z_pos*1e3:.0f}mm")

# Cross-section probes at mid-duct
z_mid = (z_min + z_max) / 2
cross_section_points = [
    ("center", source_x, source_y),
    ("corner", x_min + 0.005, y_min + 0.005),
    ("edge_x", x_min + 0.005, source_y),
    ("edge_y", source_x, y_min + 0.005),
]

print()
print("Cross-section probes at z={}mm:".format(int(z_mid * 1e3)))
for name, px, py in cross_section_points:
    solver.add_probe(
        f"cross_{name}",
        position=(px, py, z_mid),
    )
    print(f"  {name}: ({px*1e3:.0f}, {py*1e3:.0f}) mm")

print()
print("Solver configuration:")
print(f"  Grid shape: {solver.grid.shape}")
extent = solver.grid.physical_extent()
print(f"  Domain: {extent[0]*1e3:.0f} × {extent[1]*1e3:.0f} × {extent[2]*1e3:.0f} mm")
print(f"  Timestep: {solver.dt*1e9:.3f} ns")
print(f"  Using native backend: {solver.using_native}")
print("=" * 60)
print()

# Run simulation
print("Running simulation...")
solver.run(duration=0.001, output_file="results.h5")  # 1ms

print()
print("=" * 60)
print("✓ Simulation complete!")
print("=" * 60)
print(f"Output saved to: results.h5")
print(f"File size: {__import__('os').path.getsize('results.h5') / 1e6:.1f} MB")
print()

# =============================================================================
# Analysis
# =============================================================================

print("Propagation analysis:")
print("-" * 40)

# Calculate expected propagation time along duct
propagation_distance = z_positions[-1] - z_positions[0]
expected_time = propagation_distance / c

print(f"  Distance from first to last probe: {propagation_distance*1e3:.0f} mm")
print(f"  Expected propagation time: {expected_time*1e6:.0f} μs")
print()

# Compare amplitudes at different z positions
print("Amplitude vs position:")
for i, z_pos in enumerate(z_positions):
    probe_name = f"center_z{int(z_pos*1e3)}mm"
    try:
        data = solver.get_probe_data(probe_name)[probe_name]
        peak = np.max(np.abs(data))
        print(f"  z={z_pos*1e3:.0f}mm: peak amplitude = {peak:.4f}")
    except (KeyError, ValueError):
        pass

print()
print("Key observations:")
print("- Plane wave (0,0) propagates at all frequencies")
print("- Higher modes (m,n) only propagate above their cutoff")
print("- Mode shapes show characteristic pressure patterns")
print("- Off-center source excites asymmetric modes")
print("- Rigid walls create standing wave patterns transversely")
print()
print("Next steps:")
print("1. Upload results.h5 to the FDTD Viewer")
print("2. Watch wave propagation along the duct")
print("3. Look at cross-sections to see mode patterns")
print("4. Try frequencies below (1,0) cutoff to see evanescent behavior")
print()

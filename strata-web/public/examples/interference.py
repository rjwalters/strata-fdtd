"""
Two-Source Interference Pattern
================================

This example demonstrates wave interference:
- Two coherent sinusoidal sources
- Constructive and destructive interference
- Standing wave pattern formation
- Phase relationship effects

Estimated runtime: ~2 minutes on modern laptop
Output size: ~80 MB
"""

from strata_fdtd import UniformGrid, Scene, SinusoidalSource, FDTDSolver
import numpy as np

# Create uniform grid
# Shape: 150 x 150 x 100 cells
# Resolution: 1 mm per cell
# Total domain: 15 cm x 15 cm x 10 cm
grid = UniformGrid(
    shape=(150, 150, 100),
    resolution=1e-3  # 1 mm
)

# Create scene
scene = Scene(grid)

# Define two sources separated by 5 wavelengths
# Frequency: 10 kHz
# Wavelength in air: λ = c/f = 343/10000 = 34.3 mm
frequency = 10e3  # 10 kHz
wavelength = 343 / frequency  # ~34.3 mm
source_separation = 5 * wavelength  # ~17 cm... too large, use 3λ
source_separation = 3 * wavelength  # ~10 cm

# Source positions
# Both at same z, separated in x direction, centered in y
source1_pos = (0.075 - source_separation/2, 0.075, 0.02)
source2_pos = (0.075 + source_separation/2, 0.075, 0.02)

# Add first source (in phase)
source1 = SinusoidalSource(
    frequency=frequency,
    position=source1_pos,
    amplitude=500,  # 500 Pa
    phase=0  # Reference phase
)
scene.add_source(source1)

# Add second source (in phase for constructive interference pattern)
source2 = SinusoidalSource(
    frequency=frequency,
    position=source2_pos,
    amplitude=500,  # 500 Pa
    phase=0  # Same phase as source1
)
scene.add_source(source2)

# Add probe array to capture interference pattern
# Line of probes between sources
n_probes_between = 7
for i in range(n_probes_between):
    x = source1_pos[0] + (source2_pos[0] - source1_pos[0]) * i / (n_probes_between - 1)
    scene.add_probe(position=(x, 0.075, 0.02))

# Line of probes in front of sources (perpendicular bisector)
# This will show maxima at center and periodic maxima/minima
n_probes_front = 9
for i in range(n_probes_front):
    y_offset = -0.03 + 0.06 * i / (n_probes_front - 1)  # -3cm to +3cm
    scene.add_probe(position=(0.075, 0.075 + y_offset, 0.05))

# Line of probes far from sources (far-field pattern)
for i in range(n_probes_front):
    y_offset = -0.03 + 0.06 * i / (n_probes_front - 1)
    scene.add_probe(position=(0.075, 0.075 + y_offset, 0.08))

# Create solver
solver = FDTDSolver(
    grid=grid,
    scene=scene,
    duration=3e-3,  # 3 milliseconds (30 periods)
    pml_thickness=10  # 10-cell PML boundary
)

# Calculate interference pattern parameters
# Fringe spacing at distance d from sources: Δy = λd / s
# where s is source separation
fringe_spacing_near = wavelength * 0.03 / source_separation  # At z=5cm
fringe_spacing_far = wavelength * 0.06 / source_separation   # At z=8cm

# Print simulation info
print(f"Grid: {solver.grid.shape}")
print(f"Source frequency: {frequency/1e3:.1f} kHz")
print(f"Wavelength: {wavelength*1e3:.2f} mm")
print(f"Source separation: {source_separation*1e3:.1f} mm ({source_separation/wavelength:.1f}λ)")
print(f"Expected fringe spacing (near): {fringe_spacing_near*1e3:.2f} mm")
print(f"Expected fringe spacing (far): {fringe_spacing_far*1e3:.2f} mm")
print(f"Timestep: {solver.dt:.2e} s")
print(f"Number of steps: {solver.num_steps}")
print(f"Expected runtime: ~{solver.estimate_runtime():.0f} seconds")

# Note: This script is designed to be executed by fdtd-compute CLI
# Example: fdtd-compute interference.py
# Visualize pressure field to see interference fringes

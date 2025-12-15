"""
Acoustic Lens Focusing
=======================

This example demonstrates acoustic strata_fdtd lens design:
- Graded index strata_fdtd lens
- Plane wave to focused beam conversion
- Focal point analysis
- Beam width measurement

Estimated runtime: ~3 minutes on modern laptop
Output size: ~150 MB
"""

from strata_fdtd import UniformGrid, Scene, PlaneWaveSource, FDTDSolver, GradedMaterial
import numpy as np

# Create uniform grid
# Shape: 200 x 200 x 150 cells
# Resolution: 0.5 mm per cell
# Total domain: 10 cm x 10 cm x 7.5 cm
grid = UniformGrid(
    shape=(200, 200, 150),
    resolution=0.5e-3  # 0.5 mm
)

# Create scene
scene = Scene(grid)

# Define graded-index lens
# Lens diameter: 4 cm
# Lens thickness: 2 cm
# Speed of sound varies from 343 m/s (edge) to 500 m/s (center)
# This creates focusing effect

lens_center = np.array([0.05, 0.05, 0.0375])  # Center of domain
lens_radius = 0.02  # 2 cm radius
lens_thickness = 0.01  # 1 cm half-thickness

def lens_region(x, y, z):
    """Define lens region with graded index"""
    r = np.sqrt((x - lens_center[0])**2 + (y - lens_center[1])**2)
    z_dist = abs(z - lens_center[2])
    return (r <= lens_radius) and (z_dist <= lens_thickness)

def sound_speed_profile(x, y, z):
    """Graded sound speed - slower at edges for focusing"""
    r = np.sqrt((x - lens_center[0])**2 + (y - lens_center[1])**2)
    # Parabolic profile: faster in center
    r_normalized = r / lens_radius
    c = 343 + 157 * (1 - r_normalized**2)  # 343 to 500 m/s
    return c

# Add graded material
lens_material = GradedMaterial(
    sound_speed_func=sound_speed_profile,
    density=1.2  # Air density
)
scene.add_material(
    material=lens_material,
    region=lens_region
)

# Add plane wave source from left
# Frequency: 40 kHz
source = PlaneWaveSource(
    frequency=40e3,  # 40 kHz
    direction=(0, 0, 1),  # Propagating in +z direction
    amplitude=1000,  # 1000 Pa
    position=(0.05, 0.05, 0.01)  # Starts before lens
)
scene.add_source(source)

# Add probe array to measure focal region
# Probes along beam axis past the lens
probe_positions = [
    (0.05, 0.05, 0.045),  # Just after lens
    (0.05, 0.05, 0.050),  # Expected focal point
    (0.05, 0.05, 0.055),  # Past focal point
    (0.05, 0.05, 0.060),  # Far field
]

# Also add transverse probes at focal plane
for offset in [-0.01, -0.005, 0, 0.005, 0.01]:
    scene.add_probe(position=(0.05 + offset, 0.05, 0.050))

for pos in probe_positions:
    scene.add_probe(position=pos)

# Create solver
solver = FDTDSolver(
    grid=grid,
    scene=scene,
    duration=1.5e-4,  # 150 microseconds
    pml_thickness=15  # 15-cell PML boundary
)

# Print simulation info
print(f"Grid: {solver.grid.shape}")
print(f"Lens diameter: {lens_radius*2*1000:.1f} mm")
print(f"Source frequency: {source.frequency/1e3:.1f} kHz")
print(f"Wavelength in air: {343/source.frequency*1000:.2f} mm")
print(f"Timestep: {solver.dt:.2e} s")
print(f"Number of steps: {solver.num_steps}")
print(f"Expected runtime: ~{solver.estimate_runtime():.0f} seconds")

# Note: This script is designed to be executed by fdtd-compute CLI
# Example: fdtd-compute acoustic-lens.py

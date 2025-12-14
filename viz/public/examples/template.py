"""
[Example Title]
===============

This example demonstrates [key concepts]:
- [Concept 1]
- [Concept 2]
- [Concept 3]

Estimated runtime: ~X minutes on modern laptop
Output size: ~XX MB

[Optional: Additional context, physics background, or references]
"""

from strata_fdtd import UniformGrid, Scene, FDTDSolver
# Add other imports as needed:
# from strata_fdtd import GaussianPulse, SinusoidalSource, PlaneWaveSource
# from strata_fdtd import PZT5Material, GradedMaterial, etc.
import numpy as np

# =============================================================================
# GRID SETUP
# =============================================================================

# Create grid
# Explain why you chose these dimensions and resolution
grid = UniformGrid(
    shape=(100, 100, 100),    # [NX, NY, NZ] cells
    resolution=1e-3           # [meters] cell size
)

# Print grid info
print(f"Grid shape: {grid.shape}")
print(f"Resolution: {grid.resolution*1e3:.2f} mm")
print(f"Physical extent: {grid.physical_extent()}")
print()

# =============================================================================
# SCENE CONFIGURATION
# =============================================================================

# Create scene
scene = Scene(grid)

# Add sources
# Explain what type of source and why
# Example:
# source = GaussianPulse(
#     frequency=40e3,  # 40 kHz - explain choice
#     position=(0.01, 0.05, 0.05),  # explain position
#     amplitude=1000  # 1000 Pa - explain amplitude
# )
# scene.add_source(source)

# Add materials (if needed)
# Explain material properties and regions
# Example:
# material = PZT5Material()
# scene.add_material(
#     material=material,
#     region=lambda x, y, z: [condition for material region]
# )

# Add probes
# Explain what you're measuring and why these positions
# Example:
# scene.add_probe(position=(0.09, 0.05, 0.05))  # Explain position choice

# =============================================================================
# SOLVER SETUP
# =============================================================================

# Create solver
solver = FDTDSolver(
    grid=grid,
    scene=scene,
    duration=1e-4,  # [seconds] - explain duration choice
    pml_thickness=10,  # [cells] - or use boundary_condition='rigid' for resonance
)

# =============================================================================
# SIMULATION INFO
# =============================================================================

# Print useful information for user
print("=" * 60)
print("SIMULATION PARAMETERS")
print("=" * 60)
print(f"Timestep: {solver.dt:.2e} s")
print(f"Number of steps: {solver.num_steps}")
print(f"CFL number: {solver.cfl:.3f} (should be < {solver.cfl_limit:.3f})")

# Calculate and print wavelength if using a frequency source
# Example:
# c = 343  # Speed of sound in air (m/s)
# frequency = 40e3  # Your source frequency
# wavelength = c / frequency
# cells_per_wavelength = wavelength / grid.resolution
# print(f"\nSource frequency: {frequency/1e3:.1f} kHz")
# print(f"Wavelength: {wavelength*1e3:.2f} mm")
# print(f"Cells per wavelength: {cells_per_wavelength:.1f}")
# print(f"(Recommended: at least 10-15 cells per wavelength)")

print(f"\nEstimated runtime: ~{solver.estimate_runtime():.0f} seconds")
print(f"Memory usage: ~{solver.estimate_memory()/1e6:.0f} MB")
print("=" * 60)
print()

# =============================================================================
# EXPECTED RESULTS
# =============================================================================

# Document what the user should observe in the results
# Example:
"""
Expected results:
- Probe 1 should show [expected behavior]
- Pressure field should exhibit [expected pattern]
- [Any other observable phenomena]

Analysis suggestions:
- Plot probe data as time series
- Compute FFT to observe frequency content
- Visualize pressure field at different time steps
- [Other analysis approaches]
"""

# =============================================================================
# EXECUTION
# =============================================================================

# Note: This script is designed to be executed by fdtd-compute CLI
# The CLI will automatically run the solver and save results
#
# To run this example:
#   fdtd-compute template.py
#
# To customize:
#   cp template.py my-simulation.py
#   # Edit my-simulation.py with your parameters
#   fdtd-compute my-simulation.py

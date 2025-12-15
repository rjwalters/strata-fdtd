"""
Basic Gaussian Pulse Propagation
=================================

This example demonstrates fundamental FDTD concepts:
- Creating a uniform grid
- Adding a Gaussian pulse source
- Placing a probe to measure pressure
- Running the simulation

Estimated runtime: ~1 minute on modern laptop
Output size: ~50 MB
"""

from strata_fdtd import UniformGrid, Scene, GaussianPulse, FDTDSolver
import numpy as np

# Create uniform grid
# Shape: 100 x 100 x 100 cells
# Resolution: 1 mm per cell
# Total domain: 10 cm x 10 cm x 10 cm
grid = UniformGrid(
    shape=(100, 100, 100),
    resolution=1e-3  # 1 mm
)

# Create scene
scene = Scene(grid)

# Add Gaussian pulse source
# Frequency: 40 kHz (ultrasound)
# Position: Center of left face
source = GaussianPulse(
    frequency=40e3,  # 40 kHz
    position=(0.01, 0.05, 0.05),  # Near left edge
    amplitude=1000  # 1000 Pa
)
scene.add_source(source)

# Add probe to measure pressure
# Position: Center of right face
scene.add_probe(
    position=(0.09, 0.05, 0.05)  # Near right edge
)

# Create solver
solver = FDTDSolver(
    grid=grid,
    scene=scene,
    duration=1e-4,  # 100 microseconds
    pml_thickness=10  # 10-cell PML boundary
)

# Print simulation info
print(f"Grid: {solver.grid.shape}")
print(f"Timestep: {solver.dt:.2e} s")
print(f"Number of steps: {solver.num_steps}")
print(f"Expected runtime: ~{solver.estimate_runtime():.0f} seconds")

# Note: This script is designed to be executed by fdtd-compute CLI
# Example: fdtd-compute basic-pulse.py

"""
Nonuniform Grid Example
========================

This example demonstrates variable resolution grids:
- Fine resolution near source and probe
- Coarse resolution in propagation region
- Memory and computation savings
- Maintaining accuracy where needed

Estimated runtime: ~2 minutes on modern laptop
Output size: ~60 MB (reduced due to adaptive grid)
"""

from strata_fdtd import NonuniformGrid, Scene, GaussianPulse, FDTDSolver
import numpy as np

# Create nonuniform grid using geometric stretch
# Fine resolution (0.5 mm) at center, grows to ~2.5 mm at edges
# This reduces cell count while maintaining accuracy near source/probe
grid = NonuniformGrid.from_stretch(
    shape=(100, 100, 200),
    base_resolution=0.5e-3,  # 0.5 mm finest cells
    stretch_x=1.0,           # Uniform in x
    stretch_y=1.0,           # Uniform in y
    stretch_z=1.05,          # 5% growth per cell in z
    center_fine=True         # Finest cells at center
)

# Print grid statistics
print(f"Grid shape: {grid.shape}")
print(f"Min cell spacing: {grid.min_spacing*1e3:.3f} mm")
print(f"Max cell spacing: {grid.max_spacing*1e3:.3f} mm")
print(f"Stretch ratio (z): {grid.stretch_ratio[2]:.2f}")
print(f"Physical extent: {grid.physical_extent()}")
print(f"Memory saved vs uniform: ~{(1 - grid.cell_count_ratio)*100:.0f}%")
print()

# Create scene
scene = Scene(grid)

# Add Gaussian pulse source at center (fine resolution region)
# Frequency: 50 kHz
source = GaussianPulse(
    frequency=50e3,  # 50 kHz
    position=(0.05, 0.05, 0.05),  # Center of domain
    amplitude=1000  # 1000 Pa
)
scene.add_source(source)

# Add probes at various positions
# Near source (fine resolution)
scene.add_probe(position=(0.05, 0.05, 0.052))  # 2 mm from source

# Intermediate distance (medium resolution)
scene.add_probe(position=(0.05, 0.05, 0.07))   # 2 cm from source

# Far field (coarse resolution)
scene.add_probe(position=(0.05, 0.05, 0.09))   # 4 cm from source

# Create solver
# The CFL condition uses the minimum cell spacing
solver = FDTDSolver(
    grid=grid,
    scene=scene,
    duration=1.5e-4,  # 150 microseconds
    pml_thickness=10  # 10-cell PML boundary
)

# Compare with equivalent uniform grid
uniform_cell_count = 100 * 100 * 200
nonuniform_cell_count = solver.grid.total_cells
memory_ratio = nonuniform_cell_count / uniform_cell_count

# Print simulation info
print(f"Timestep: {solver.dt:.2e} s (set by min spacing)")
print(f"Number of steps: {solver.num_steps}")
print(f"Uniform grid cells: {uniform_cell_count:,}")
print(f"Nonuniform grid cells (effective): {nonuniform_cell_count:,}")
print(f"Cell count ratio: {memory_ratio:.2%}")
print(f"Expected runtime: ~{solver.estimate_runtime():.0f} seconds")
print()

# Demonstrate grid spacing variation
print("Grid spacing along z-axis (selected cells):")
z_indices = [0, 25, 50, 75, 100, 125, 150, 175, 199]
for i in z_indices:
    print(f"  Cell {i:3d}: dz = {grid.dz[i]*1e3:.3f} mm")

# Note: This script is designed to be executed by fdtd-compute CLI
# Example: fdtd-compute nonuniform-grid.py
# Compare results with uniform grid to verify accuracy

"""
Example: Nonuniform Grid - Adaptive Resolution
==============================================
Demonstrates memory-efficient simulations using nonuniform grids.
High resolution is concentrated near the source and probe regions,
with coarser resolution in bulk regions where less detail is needed.

Expected runtime: ~40 seconds on modern hardware (with native backend)
Output: results.h5 (pressure snapshots and probe data)

Comparison:
- Uniform 1mm grid: 100×100×100 = 1M cells
- Nonuniform grid: same physical domain with fewer cells in coarse regions
- Memory savings depend on stretch ratio

Learning objectives:
- Creating nonuniform grids with geometric stretch
- Understanding the CFL condition with variable cell sizes
- Balancing resolution and computational cost
- Piecewise-constant resolution for specific regions
"""

import numpy as np

from strata_fdtd import (
    GaussianPulse,
    FDTDSolver,
    NonuniformGrid,
)

# =============================================================================
# Option 1: Geometric Stretch Grid
# =============================================================================
# Cells grow geometrically from the center outward.
# Good for point sources where resolution matters most near the source.

print("=" * 60)
print("FDTD Simulation: Nonuniform Grid Examples")
print("=" * 60)
print()

# Create a grid with fine resolution at center, coarser at edges
# The stretch ratio determines how quickly cells grow
grid_stretched = NonuniformGrid.from_stretch(
    shape=(80, 80, 80),  # Fewer cells than uniform grid
    base_resolution=1e-3,  # 1mm at the finest
    stretch_x=1.02,  # 2% growth per cell in x
    stretch_y=1.02,  # 2% growth per cell in y
    stretch_z=1.02,  # 2% growth per cell in z
    center_fine=True,  # Finest cells at center
)

print("Grid with Geometric Stretch:")
print(f"  Shape: {grid_stretched.shape}")
print(f"  Min cell size: {grid_stretched.min_spacing*1e3:.2f} mm")
print(f"  Max cell size: {grid_stretched.max_spacing*1e3:.2f} mm")
print(f"  Stretch ratio: {grid_stretched.stretch_ratio}")
extent = grid_stretched.physical_extent()
print(f"  Domain extent: {extent[0]*1e3:.1f} × {extent[1]*1e3:.1f} × {extent[2]*1e3:.1f} mm")
print()

# Create solver with stretched grid
solver1 = FDTDSolver(grid=grid_stretched)

# Add source at center (where resolution is finest)
center = tuple(e/2 for e in extent)
solver1.add_source(
    GaussianPulse(
        position=center,
        frequency=40e3,
        amplitude=1.0,
    )
)

# Add probes
solver1.add_probe("center", position=center)
solver1.add_probe("edge", position=(extent[0]*0.9, center[1], center[2]))

print(f"Solver info (stretched grid):")
print(f"  Timestep: {solver1.dt*1e9:.3f} ns")
print(f"  CFL uses min spacing: {grid_stretched.min_spacing*1e3:.2f} mm")
print(f"  Using native backend: {solver1.using_native}")
print()

# =============================================================================
# Option 2: Piecewise Regions Grid
# =============================================================================
# Different resolution in different regions.
# Good when you know exactly where fine resolution is needed.

# Define regions: (start, end, resolution)
# Fine resolution in the middle (0.03-0.07m), coarse at edges
x_regions = [
    (0, 0.03, 2e-3),     # Coarse: 2mm cells
    (0.03, 0.07, 1e-3),  # Fine: 1mm cells (region of interest)
    (0.07, 0.10, 2e-3),  # Coarse: 2mm cells
]

grid_regions = NonuniformGrid.from_regions(
    x_regions=x_regions,
    y_regions=[(0, 0.10, 1.5e-3)],  # Uniform 1.5mm in y
    z_regions=[(0, 0.10, 1.5e-3)],  # Uniform 1.5mm in z
)

print("Grid with Piecewise Regions:")
print(f"  Shape: {grid_regions.shape}")
print(f"  Min cell size: {grid_regions.min_spacing*1e3:.2f} mm")
print(f"  Max cell size: {grid_regions.max_spacing*1e3:.2f} mm")
extent2 = grid_regions.physical_extent()
print(f"  Domain extent: {extent2[0]*1e3:.1f} × {extent2[1]*1e3:.1f} × {extent2[2]*1e3:.1f} mm")
print()

# Create solver with region-based grid
solver2 = FDTDSolver(grid=grid_regions)

# Source in fine region
solver2.add_source(
    GaussianPulse(
        position=(0.05, 0.05, 0.05),  # In fine region
        frequency=40e3,
        amplitude=1.0,
    )
)

solver2.add_probe("fine_region", position=(0.06, 0.05, 0.05))
solver2.add_probe("coarse_region", position=(0.02, 0.05, 0.05))

print(f"Solver info (region-based grid):")
print(f"  Timestep: {solver2.dt*1e9:.3f} ns")
print(f"  Using native backend: {solver2.using_native}")
print()

# =============================================================================
# Compare with uniform grid
# =============================================================================

solver_uniform = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)
solver_uniform.add_source(
    GaussianPulse(position=(0.05, 0.05, 0.05), frequency=40e3)
)
solver_uniform.add_probe("center", position=(0.05, 0.05, 0.05))

# Calculate cell counts for comparison
uniform_cells = np.prod(solver_uniform.shape)
stretched_cells = np.prod(solver1.shape)
region_cells = np.prod(solver2.shape)

print("Comparison with Uniform Grid (100×100×100 @ 1mm):")
print(f"  Uniform grid: {solver_uniform.shape} = {uniform_cells:,} cells")
print(f"  Stretched grid: {solver1.shape} = {stretched_cells:,} cells")
print(f"  Region-based grid: {solver2.shape} = {region_cells:,} cells")

# Calculate relative savings
print(f"  Stretched cell savings: {(1 - stretched_cells/uniform_cells)*100:.1f}%")
print(f"  Region-based cell savings: {(1 - region_cells/uniform_cells)*100:.1f}%")
print()

# =============================================================================
# Run simulation with stretched grid
# =============================================================================

print("=" * 60)
print("Running simulation with stretched grid...")
print("=" * 60)
solver1.run(duration=0.0004, output_file="results.h5")  # 0.4ms

print()
print("=" * 60)
print("✓ Simulation complete!")
print("=" * 60)
print(f"Output saved to: results.h5")
print(f"File size: {__import__('os').path.getsize('results.h5') / 1e6:.1f} MB")
print()

# Analysis
print("Resolution analysis:")
print("-" * 40)
print("The nonuniform grid provides:")
print("  - High resolution (1mm) at the center where the source is located")
print("  - Gradually coarser resolution toward the edges")
print("  - Same physical accuracy near sources and probes")
print("  - Reduced memory and computation in bulk regions")
print()
print("When to use nonuniform grids:")
print("  - Point sources: fine resolution near source, coarse far away")
print("  - Boundary layers: fine resolution near surfaces")
print("  - Waveguides: fine cross-section, coarse along propagation direction")
print("  - Multi-scale problems: detailed geometry in one region, bulk in another")
print()
print("Trade-offs:")
print("  - CFL timestep limited by smallest cell (no speedup in dt)")
print("  - Memory savings can be significant for large domains")
print("  - Wave propagation through coarse regions may have reduced accuracy")
print()
print("Next steps:")
print("1. Upload results.h5 to the FDTD Viewer")
print("2. Observe wave propagation through variable-resolution regions")
print("3. Compare probe signals in fine vs coarse regions")
print()

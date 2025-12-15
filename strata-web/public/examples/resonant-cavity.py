"""
Resonant Cavity Modes
======================

This example demonstrates acoustic resonance:
- Rectangular cavity with rigid walls
- Excitation at fundamental frequency
- Observation of standing wave patterns
- Mode shape visualization

Estimated runtime: ~2 minutes on modern laptop
Output size: ~80 MB
"""

from strata_fdtd import UniformGrid, Scene, SinusoidalSource, FDTDSolver
import numpy as np

# Create uniform grid for rectangular cavity
# Shape: 100 x 100 x 150 cells
# Resolution: 1 mm per cell
# Cavity dimensions: 10 cm x 10 cm x 15 cm
grid = UniformGrid(
    shape=(100, 100, 150),
    resolution=1e-3  # 1 mm
)

# Create scene
scene = Scene(grid)

# Calculate fundamental frequency for cavity
# For rectangular cavity: f = c/(2*L) where L is longest dimension
# L = 0.15 m, c = 343 m/s (air)
fundamental_freq = 343 / (2 * 0.15)  # ~1143 Hz

# Add sinusoidal source at fundamental frequency
# Position: Near one corner to excite mode
source = SinusoidalSource(
    frequency=fundamental_freq,
    position=(0.01, 0.05, 0.05),  # Near left wall
    amplitude=100  # 100 Pa
)
scene.add_source(source)

# Add probe array to capture standing wave pattern
# Probes along the cavity length (z-axis)
probe_positions = [
    (0.05, 0.05, 0.025),   # 1/6 length
    (0.05, 0.05, 0.050),   # 2/6 length
    (0.05, 0.05, 0.075),   # 3/6 length (center)
    (0.05, 0.05, 0.100),   # 4/6 length
    (0.05, 0.05, 0.125),   # 5/6 length
]

for pos in probe_positions:
    scene.add_probe(position=pos)

# Create solver with rigid boundary conditions
# No PML - use rigid walls to create resonance
solver = FDTDSolver(
    grid=grid,
    scene=scene,
    duration=0.01,  # 10 milliseconds (several periods)
    boundary_condition='rigid'  # Rigid walls for resonance
)

# Print simulation info
print(f"Grid: {solver.grid.shape}")
print(f"Cavity dimensions: {grid.physical_extent()}")
print(f"Fundamental frequency: {fundamental_freq:.1f} Hz")
print(f"Period: {1/fundamental_freq*1000:.2f} ms")
print(f"Timestep: {solver.dt:.2e} s")
print(f"Number of steps: {solver.num_steps}")
print(f"Expected runtime: ~{solver.estimate_runtime():.0f} seconds")

# Note: This script is designed to be executed by fdtd-compute CLI
# Example: fdtd-compute resonant-cavity.py

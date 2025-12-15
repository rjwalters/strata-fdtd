"""
Doppler Effect Demonstration
=============================

This example demonstrates the Doppler effect:
- Moving sinusoidal source
- Frequency shift observation
- Stationary observers at different positions
- Time-domain and frequency analysis

Estimated runtime: ~2 minutes on modern laptop
Output size: ~100 MB
"""

from strata_fdtd import UniformGrid, Scene, MovingSource, FDTDSolver
import numpy as np

# Create uniform grid
# Shape: 150 x 100 x 100 cells
# Resolution: 1 mm per cell
# Total domain: 15 cm x 10 cm x 10 cm
grid = UniformGrid(
    shape=(150, 100, 100),
    resolution=1e-3  # 1 mm
)

# Create scene
scene = Scene(grid)

# Define source motion
# Source moves along x-axis at 50 m/s (about Mach 0.15)
# Starts at x=2cm, moves to x=13cm over 2.2 milliseconds
source_velocity = 50  # m/s
start_position = np.array([0.02, 0.05, 0.05])
end_position = np.array([0.13, 0.05, 0.05])
motion_duration = np.linalg.norm(end_position - start_position) / source_velocity

# Create moving source
# Frequency: 5 kHz
# This will demonstrate frequency shift for stationary observers
source = MovingSource(
    frequency=5e3,  # 5 kHz
    start_position=start_position,
    end_position=end_position,
    velocity=source_velocity,
    amplitude=1000  # 1000 Pa
)
scene.add_source(source)

# Add stationary probes
# Observer ahead of source (will hear higher frequency)
# Observer behind source (will hear lower frequency)
# Observer at midpoint (will hear transition)

probe_positions = [
    (0.03, 0.05, 0.05),  # Behind starting point
    (0.05, 0.05, 0.05),  # Early in path
    (0.075, 0.05, 0.05), # Midpoint - will see transition
    (0.10, 0.05, 0.05),  # Late in path
    (0.12, 0.05, 0.05),  # Ahead of ending point
]

for pos in probe_positions:
    scene.add_probe(position=pos)

# Also add off-axis probes to show wavefront shape
for offset in [-0.02, 0.02]:
    scene.add_probe(position=(0.075, 0.05 + offset, 0.05))

# Create solver
solver = FDTDSolver(
    grid=grid,
    scene=scene,
    duration=motion_duration * 1.5,  # Run a bit longer than motion
    pml_thickness=10  # 10-cell PML boundary
)

# Calculate expected frequency shifts
c = 343  # Speed of sound in air (m/s)
f0 = 5e3  # Source frequency (Hz)
f_ahead = f0 * c / (c - source_velocity)  # Frequency heard ahead
f_behind = f0 * c / (c + source_velocity)  # Frequency heard behind

# Print simulation info
print(f"Grid: {solver.grid.shape}")
print(f"Source frequency: {f0/1e3:.1f} kHz")
print(f"Source velocity: {source_velocity:.1f} m/s (Mach {source_velocity/c:.3f})")
print(f"Motion duration: {motion_duration*1e3:.2f} ms")
print(f"Expected frequency ahead: {f_ahead/1e3:.2f} kHz (+{(f_ahead-f0)/f0*100:.1f}%)")
print(f"Expected frequency behind: {f_behind/1e3:.2f} kHz ({(f_behind-f0)/f0*100:.1f}%)")
print(f"Timestep: {solver.dt:.2e} s")
print(f"Number of steps: {solver.num_steps}")
print(f"Expected runtime: ~{solver.estimate_runtime():.0f} seconds")

# Note: This script is designed to be executed by fdtd-compute CLI
# Example: fdtd-compute doppler-effect.py
# Analyze output with FFT to observe frequency shifts

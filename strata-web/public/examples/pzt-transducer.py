"""
Piezoelectric Transducer Design
================================

This example demonstrates piezoelectric transducer modeling:
- Using PZT-5 piezoelectric material
- Sinusoidal excitation for ultrasonic waves
- Multiple probes for radiation pattern analysis
- Near-field pressure distribution

Estimated runtime: ~3 minutes on modern laptop
Output size: ~120 MB
"""

from strata_fdtd import UniformGrid, Scene, SinusoidalSource, FDTDSolver, PZT5Material
import numpy as np

# Create uniform grid
# Shape: 150 x 150 x 200 cells
# Resolution: 0.5 mm per cell
# Total domain: 7.5 cm x 7.5 cm x 10 cm
grid = UniformGrid(
    shape=(150, 150, 200),
    resolution=0.5e-3  # 0.5 mm
)

# Create scene
scene = Scene(grid)

# Define PZT-5 transducer disk
# Diameter: 10 mm, Thickness: 2 mm
# Center position at z = 20 mm
transducer_material = PZT5Material()
scene.add_material(
    material=transducer_material,
    region=lambda x, y, z: (
        (x - 0.0375)**2 + (y - 0.0375)**2 <= (0.005)**2 and  # Radius 5mm
        0.019 <= z <= 0.021  # Thickness 2mm at z=20mm
    )
)

# Add sinusoidal source
# Frequency: 1 MHz (typical ultrasound)
# Applied to transducer surface
source = SinusoidalSource(
    frequency=1e6,  # 1 MHz
    position=(0.0375, 0.0375, 0.020),  # Center of transducer
    amplitude=10000  # 10 kPa
)
scene.add_source(source)

# Add probe array for radiation pattern
# 5 probes along axial direction
probe_positions = [
    (0.0375, 0.0375, 0.03),   # 10 mm from transducer
    (0.0375, 0.0375, 0.04),   # 20 mm from transducer
    (0.0375, 0.0375, 0.05),   # 30 mm from transducer
    (0.0375, 0.0375, 0.06),   # 40 mm from transducer
    (0.0375, 0.0375, 0.07),   # 50 mm from transducer
]

for pos in probe_positions:
    scene.add_probe(position=pos)

# Create solver
solver = FDTDSolver(
    grid=grid,
    scene=scene,
    duration=5e-6,  # 5 microseconds (5 cycles)
    pml_thickness=15  # 15-cell PML boundary
)

# Print simulation info
print(f"Grid: {solver.grid.shape}")
print(f"Timestep: {solver.dt:.2e} s")
print(f"Number of steps: {solver.num_steps}")
print(f"Transducer frequency: {source.frequency/1e6:.1f} MHz")
print(f"Wavelength in water: {1500/source.frequency*1000:.2f} mm")
print(f"Expected runtime: ~{solver.estimate_runtime():.0f} seconds")

# Note: This script is designed to be executed by fdtd-compute CLI
# Example: fdtd-compute pzt-transducer.py

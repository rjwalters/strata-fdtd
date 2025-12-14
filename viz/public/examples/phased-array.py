"""
Phased Array Beam Steering
===========================

This example demonstrates ultrasound phased array imaging:
- Array of transducer elements
- Electronic beam steering via time delays
- Focused beam formation
- Angle-dependent directivity

Estimated runtime: ~4 minutes on modern laptop
Output size: ~180 MB
"""

from strata_fdtd import UniformGrid, Scene, SinusoidalSource, FDTDSolver
import numpy as np

# Create uniform grid
# Shape: 200 x 150 x 200 cells
# Resolution: 0.5 mm per cell
# Total domain: 10 cm x 7.5 cm x 10 cm
grid = UniformGrid(
    shape=(200, 150, 200),
    resolution=0.5e-3  # 0.5 mm
)

# Create scene
scene = Scene(grid)

# Define phased array parameters
# 16-element linear array
# Element spacing: 0.5 mm (pitch)
# Frequency: 5 MHz (medical ultrasound)
n_elements = 16
element_pitch = 0.5e-3  # 0.5 mm spacing
frequency = 5e6  # 5 MHz
wavelength = 1500 / frequency  # ~0.3 mm in water (assuming water medium)

# Steering angle: 20 degrees
steering_angle = 20 * np.pi / 180  # radians
c = 1500  # Speed of sound in water (m/s)

# Calculate time delays for beam steering
# Delay for element n: τ_n = n * d * sin(θ) / c
# where d is element pitch, θ is steering angle
delays = np.zeros(n_elements)
for n in range(n_elements):
    # Relative delay compared to center element
    n_centered = n - n_elements / 2
    delays[n] = n_centered * element_pitch * np.sin(steering_angle) / c

# Normalize delays (make all positive)
delays = delays - delays.min()

# Array center position
array_center = np.array([0.05, 0.0375, 0.02])

# Add sources for each array element
for n in range(n_elements):
    # Element position
    x_offset = (n - n_elements / 2) * element_pitch
    element_pos = array_center + np.array([x_offset, 0, 0])

    # Add source with appropriate phase delay
    # Phase = -2π * f * τ
    phase = -2 * np.pi * frequency * delays[n]

    source = SinusoidalSource(
        frequency=frequency,
        position=tuple(element_pos),
        amplitude=5000,  # 5 kPa per element
        phase=phase  # Time delay implemented as phase shift
    )
    scene.add_source(source)

# Add probe array to measure beam pattern
# Probes in a fan pattern in front of array
focal_distance = 0.03  # 3 cm from array
n_angles = 11
angles = np.linspace(-30, 30, n_angles) * np.pi / 180

for angle in angles:
    # Position at focal distance and angle
    x = array_center[0] + focal_distance * np.sin(angle)
    z = array_center[2] + focal_distance * np.cos(angle)
    scene.add_probe(position=(x, array_center[1], z))

# Also add probes along beam axis
for z_offset in [0.01, 0.02, 0.03, 0.04, 0.05]:
    # Calculate x position for steering angle
    x = array_center[0] + z_offset * np.tan(steering_angle)
    scene.add_probe(position=(x, array_center[1], array_center[2] + z_offset))

# Create solver
solver = FDTDSolver(
    grid=grid,
    scene=scene,
    duration=2e-5,  # 20 microseconds (100 cycles)
    pml_thickness=15  # 15-cell PML boundary
)

# Print simulation info
print(f"Grid: {solver.grid.shape}")
print(f"Array: {n_elements} elements, pitch = {element_pitch*1e3:.2f} mm")
print(f"Frequency: {frequency/1e6:.1f} MHz")
print(f"Wavelength in water: {wavelength*1e3:.3f} mm")
print(f"Steering angle: {steering_angle*180/np.pi:.1f}°")
print(f"Max delay: {delays.max()*1e9:.2f} ns")
print(f"Focal distance: {focal_distance*1e2:.1f} cm")
print(f"Timestep: {solver.dt:.2e} s")
print(f"Number of steps: {solver.num_steps}")
print(f"Expected runtime: ~{solver.estimate_runtime():.0f} seconds")

# Note: This script is designed to be executed by fdtd-compute CLI
# Example: fdtd-compute phased-array.py
# Visualize to see focused, steered beam

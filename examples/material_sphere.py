"""
Example: Material Sphere - Scattering from a Water Sphere
========================================================
Demonstrates acoustic scattering from a material object.
A Gaussian pulse propagates through air and interacts with
a water sphere, producing reflection and transmission effects.

Expected runtime: ~45 seconds on modern hardware (with native backend)
Output: results.h5 (pressure snapshots and probe data)

Grid: 100 × 100 × 100 cells @ 1mm resolution
Domain: 100mm × 100mm × 100mm
Source: 40 kHz Gaussian pulse
Object: 20mm diameter water sphere at center
Probes: Forward (transmission) and backward (reflection) measurement

Learning objectives:
- Defining material properties (density, speed of sound)
- Creating spherical geometry for material regions
- Understanding acoustic scattering phenomena
- Measuring reflection and transmission
"""

import numpy as np

from strata_fdtd import (
    GaussianPulse,
    FDTDSolver,
    Sphere,
)
from strata_fdtd.materials import WATER_20C, SimpleMaterial

# Create the FDTD solver
solver = FDTDSolver(
    shape=(100, 100, 100),
    resolution=1e-3  # 1mm in meters
)

# Create a water sphere at the center of the domain
# Water has different acoustic properties than air:
# - Speed of sound: ~1480 m/s (vs 343 m/s in air)
# - Density: ~1000 kg/m³ (vs 1.2 kg/m³ in air)
sphere_center = (0.05, 0.05, 0.05)  # Center of domain (50mm)
sphere_radius = 0.01  # 10mm radius = 20mm diameter

# Create the sphere geometry
sphere = Sphere(center=sphere_center, radius=sphere_radius)

# Create a mask for the sphere region
# We'll evaluate the SDF at each grid point
x = np.arange(100) * 1e-3 + 0.5e-3  # Cell centers
y = np.arange(100) * 1e-3 + 0.5e-3
z = np.arange(100) * 1e-3 + 0.5e-3
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# SDF: negative inside, positive outside
# Stack grid points into (N, 3) array for SDF evaluation
points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
sdf_values = sphere.sdf(points).reshape(X.shape)
sphere_mask = sdf_values < 0  # True inside the sphere

# Register water as a material and assign to the sphere region
# Using SimpleMaterial for a constant-property material
water = SimpleMaterial(
    name="water",
    _rho=1000.0,  # kg/m³
    _c=1480.0,  # m/s (speed of sound)
)

mat_id = solver.register_material(water)
solver.set_material_region(sphere_mask, mat_id)

# Add a Gaussian pulse source to the left of the sphere
source_x = 0.015  # 15mm from left edge
solver.add_source(
    GaussianPulse(
        position=(source_x, 0.05, 0.05),  # Left of sphere
        frequency=40e3,  # 40 kHz
        amplitude=1.0,
    )
)

# Add probes to measure incident, reflected, and transmitted waves
# Incident probe: between source and sphere
solver.add_probe(
    "incident",
    position=(0.025, 0.05, 0.05),  # 25mm from left (before sphere)
)

# Reflection probe: same location as incident (will see both)
# For reflection analysis, we'll use a probe closer to the source
solver.add_probe(
    "reflection",
    position=(0.020, 0.05, 0.05),  # 20mm from left
)

# Transmission probe: after the sphere
solver.add_probe(
    "transmission",
    position=(0.075, 0.05, 0.05),  # 75mm from left (after sphere)
)

# Reference probe: to the side (no sphere interaction)
solver.add_probe(
    "reference",
    position=(0.075, 0.08, 0.05),  # Off-axis, same x as transmission
)

# Print simulation info
print("=" * 60)
print("FDTD Simulation: Scattering from Water Sphere")
print("=" * 60)
print(f"Grid shape: {solver.grid.shape}")
print(f"Grid resolution: {solver.dx*1e3:.2f} mm")
print(f"Domain extent: {solver.grid.physical_extent()[0]*1e3:.1f} mm cube")
print(f"Timestep: {solver.dt*1e9:.3f} ns")
print(f"Using native backend: {solver.using_native}")
print()
print("Material configuration:")
print(f"  Background (air):")
print(f"    Speed of sound: {solver.c:.0f} m/s")
print(f"    Density: {solver.rho:.1f} kg/m³")
print(f"  Sphere (water):")
print(f"    Speed of sound: {water.c_inf:.0f} m/s")
print(f"    Density: {water.rho_inf:.0f} kg/m³")
print(f"    Diameter: {sphere_radius*2*1e3:.0f} mm")
print(f"    Center: ({sphere_center[0]*1e3:.0f}, {sphere_center[1]*1e3:.0f}, {sphere_center[2]*1e3:.0f}) mm")
print()
print("Acoustic impedance:")
Z_air = solver.rho * solver.c
Z_water = water.rho_inf * water.c_inf
R = (Z_water - Z_air) / (Z_water + Z_air)
T = 2 * Z_air / (Z_water + Z_air)
print(f"  Z_air = {Z_air:.1f} Pa·s/m")
print(f"  Z_water = {Z_water:.0f} Pa·s/m")
print(f"  Reflection coefficient: R = {R:.3f} ({abs(R)*100:.1f}% amplitude)")
print(f"  Transmission coefficient: T = {T:.3f}")
print("=" * 60)
print()

# Run the simulation
print("Running simulation...")
solver.run(duration=0.0005, output_file="results.h5")  # 0.5ms

print()
print("=" * 60)
print("✓ Simulation complete!")
print("=" * 60)
print(f"Output saved to: results.h5")
print(f"File size: {__import__('os').path.getsize('results.h5') / 1e6:.1f} MB")
print()

# Analysis
print("Signal analysis:")
print("-" * 40)

# Get probe data (returns dict, extract arrays)
incident = solver.get_probe_data("incident")["incident"]
transmission = solver.get_probe_data("transmission")["transmission"]
reference = solver.get_probe_data("reference")["reference"]

# Peak amplitudes
inc_peak = np.max(np.abs(incident))
trans_peak = np.max(np.abs(transmission))
ref_peak = np.max(np.abs(reference))

print(f"  Incident peak amplitude: {inc_peak:.4f}")
print(f"  Transmission peak amplitude: {trans_peak:.4f}")
print(f"  Reference peak amplitude: {ref_peak:.4f}")
print()

if ref_peak > 0:
    # Compare transmission to reference (ratio shows attenuation through sphere)
    trans_ratio = trans_peak / ref_peak
    print(f"  Transmission/Reference ratio: {trans_ratio:.3f} ({trans_ratio*100:.1f}%)")
    print(f"  (Ratio < 1 indicates energy redistribution due to scattering)")

print()
print("Key observations:")
print("- The water sphere has much higher acoustic impedance than air")
print("- Most energy is reflected at the air-water boundary")
print("- Some energy is transmitted through the sphere")
print("- Diffraction around the sphere creates complex interference patterns")
print()
print("Next steps:")
print("1. Upload results.h5 to the FDTD Viewer")
print("2. Watch the wavefront interact with the sphere")
print("3. Observe reflection, transmission, and diffraction")
print()

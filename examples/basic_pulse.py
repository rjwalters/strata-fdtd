"""
Example: Basic Gaussian Pulse
==============================
A simple Gaussian pulse propagating through air in a 100mm cube.
This demonstrates the fundamental FDTD workflow: grid creation,
source/probe placement, and simulation execution.

Expected runtime: ~30 seconds on modern hardware (with native backend)
Output: results.h5 (pressure snapshots and probe data)

Grid: 100 × 100 × 100 cells @ 1mm resolution
Domain: 100mm × 100mm × 100mm
Source: 40 kHz Gaussian pulse at x=25mm
Probe: Pressure measurement at x=75mm
"""

from strata_fdtd import (
    GaussianPulse,
    FDTDSolver,
)

# Create the FDTD solver
# - 100x100x100 cells
# - 1mm resolution
# - Total domain: 100mm x 100mm x 100mm
# - CFL condition automatically determines timestep
solver = FDTDSolver(
    shape=(100, 100, 100),
    resolution=1e-3  # 1mm in meters
)

# Add a Gaussian pulse source
# - 40 kHz center frequency
# - Located 25mm from the origin
solver.add_source(
    GaussianPulse(
        position=(0.025, 0.05, 0.05),  # x=25mm, y=50mm, z=50mm (meters)
        frequency=40e3,  # 40 kHz
        amplitude=1.0,
    )
)

# Add a probe to measure pressure
# - Located 75mm from the origin (50mm from source)
solver.add_probe(
    "downstream",
    position=(0.075, 0.05, 0.05),  # x=75mm, y=50mm, z=50mm (meters)
)

# Print simulation info
print("=" * 60)
print("FDTD Simulation: Basic Gaussian Pulse")
print("=" * 60)
print(f"Grid shape: {solver.grid.shape}")
print(f"Grid resolution: {solver.dx*1e3:.2f} mm")
print(f"Domain extent: {solver.grid.physical_extent()[0]*1e3:.1f} mm cube")
print(f"Timestep: {solver.dt*1e9:.3f} ns")
print(f"Total duration: 1.0 ms")
print(f"Using native backend: {solver.using_native}")
print("=" * 60)
print()

# Run the simulation
# Duration is in seconds (0.001s = 1ms)
print("Running simulation...")
solver.run(duration=0.001, output_file="results.h5")

print()
print("=" * 60)
print("✓ Simulation complete!")
print("=" * 60)
print(f"Output saved to: results.h5")
print(f"File size: {__import__('os').path.getsize('results.h5') / 1e6:.1f} MB")
print()
print("Next steps:")
print("1. Upload results.h5 to the FDTD Viewer")
print("2. Scrub through time to see wave propagation")
print("3. Analyze probe data (time series and FFT)")
print()
print("Or analyze in Python:")
print("  python")
print("  >>> import h5py")
print("  >>> f = h5py.File('results.h5', 'r')")
print("  >>> pressure = f['probes/downstream/pressure'][:]")
print("  >>> time = f['probes/downstream/time'][:]")

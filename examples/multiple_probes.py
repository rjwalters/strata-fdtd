"""
Example: Multiple Probes - Distance-dependent Arrival Times
==========================================================
Demonstrates multiple probe placement to observe wave propagation
and distance-dependent arrival times. A Gaussian pulse is emitted
from a central source, and probes at varying distances record the
pressure waveform.

Expected runtime: ~30 seconds on modern hardware (with native backend)
Output: results.h5 (pressure snapshots and probe data)

Grid: 100 × 100 × 100 cells @ 1mm resolution
Domain: 100mm × 100mm × 100mm
Source: 40 kHz Gaussian pulse at center
Probes: 5 probes at increasing distances from source

Learning objectives:
- Understanding wave propagation and arrival times
- Using multiple probes for spatial analysis
- Calculating propagation speed from time-of-flight
"""

from strata_fdtd import (
    GaussianPulse,
    FDTDSolver,
)

# Create the FDTD solver
# 100x100x100 cells @ 1mm resolution = 100mm cube
solver = FDTDSolver(
    shape=(100, 100, 100),
    resolution=1e-3  # 1mm in meters
)

# Source position: center of the domain
source_pos = (0.05, 0.05, 0.05)  # 50mm in all directions (meters)

# Add a Gaussian pulse source at the center
solver.add_source(
    GaussianPulse(
        position=source_pos,  # Center of domain (meters)
        frequency=40e3,  # 40 kHz
        amplitude=1.0,
    )
)

# Add multiple probes at increasing distances from the source
# All probes along the x-axis from the source
probe_distances = [10, 20, 30, 40, 45]  # distances in mm

for dist in probe_distances:
    probe_x = 0.05 + dist * 1e-3  # Convert to meters
    solver.add_probe(
        f"probe_{dist}mm",
        position=(probe_x, 0.05, 0.05),  # Along x-axis from source (meters)
    )

# Print simulation info
print("=" * 60)
print("FDTD Simulation: Multiple Probes")
print("=" * 60)
print(f"Grid shape: {solver.grid.shape}")
print(f"Grid resolution: {solver.dx*1e3:.2f} mm")
print(f"Domain extent: {solver.grid.physical_extent()[0]*1e3:.1f} mm cube")
print(f"Timestep: {solver.dt*1e9:.3f} ns")
print(f"Speed of sound: {solver.c:.1f} m/s")
print(f"Total duration: 0.5 ms")
print(f"Using native backend: {solver.using_native}")
print()
print("Probe configuration:")
print(f"  Source position: ({source_pos[0]*1e3:.0f}, {source_pos[1]*1e3:.0f}, {source_pos[2]*1e3:.0f}) mm")
for dist in probe_distances:
    expected_arrival = dist * 1e-3 / solver.c * 1e6  # microseconds
    print(f"  probe_{dist}mm: {dist}mm from source, expected arrival ~{expected_arrival:.0f} μs")
print("=" * 60)
print()

# Run the simulation
print("Running simulation...")
solver.run(duration=0.0005, output_file="results.h5")  # 0.5ms duration

print()
print("=" * 60)
print("✓ Simulation complete!")
print("=" * 60)
print(f"Output saved to: results.h5")
print(f"File size: {__import__('os').path.getsize('results.h5') / 1e6:.1f} MB")
print()

# Analyze arrival times
print("Arrival time analysis:")
print("-" * 40)
import numpy as np

# Get probe data and find arrival times (first significant peak)
arrival_times = {}
for dist in probe_distances:
    probe_name = f"probe_{dist}mm"
    data = solver.get_probe_data(probe_name)[probe_name]  # Returns dict, extract array
    time = np.arange(len(data)) * solver.dt

    # Find first significant peak (threshold = 10% of max)
    threshold = 0.1 * np.max(np.abs(data))
    above_threshold = np.where(np.abs(data) > threshold)[0]
    if len(above_threshold) > 0:
        arrival_idx = above_threshold[0]
        arrival_time_us = time[arrival_idx] * 1e6  # Convert to microseconds
        arrival_times[dist] = arrival_time_us

        # Expected arrival time based on speed of sound
        expected_time_us = dist * 1e-3 / solver.c * 1e6

        print(f"  {probe_name}: arrival at {arrival_time_us:.1f} μs (expected: {expected_time_us:.1f} μs)")

# Calculate measured speed of sound from probe pairs
if len(arrival_times) >= 2:
    print()
    print("Speed of sound verification:")
    print("-" * 40)
    distances = sorted(arrival_times.keys())
    for i in range(1, len(distances)):
        d1, d2 = distances[i-1], distances[i]
        t1, t2 = arrival_times[d1], arrival_times[d2]
        if t2 > t1:
            delta_d = (d2 - d1) * 1e-3  # meters
            delta_t = (t2 - t1) * 1e-6  # seconds
            measured_c = delta_d / delta_t
            error = abs(measured_c - solver.c) / solver.c * 100
            print(f"  From {d1}mm to {d2}mm: c = {measured_c:.1f} m/s (error: {error:.1f}%)")

print()
print("Next steps:")
print("1. Upload results.h5 to the FDTD Viewer")
print("2. Scrub through time to see the expanding wavefront")
print("3. View probe data to compare arrival times")
print()
print("Or analyze in Python:")
print("  python")
print("  >>> import h5py")
print("  >>> f = h5py.File('results.h5', 'r')")
print("  >>> for name in f['probes'].keys(): print(name)")
print("  >>> probe_data = f['probes/probe_20mm/pressure'][:]")
print()

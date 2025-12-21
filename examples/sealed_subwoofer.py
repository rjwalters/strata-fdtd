#!/usr/bin/env python3
"""
Example: Sealed Subwoofer Simulation
====================================

Demonstrates membrane source excitation in a sealed loudspeaker cabinet.
This is the simplest realistic loudspeaker simulation:
- Single 12" (305mm) woofer driver
- Sealed enclosure (~72 liters internal volume)
- Low frequency operation (predictable physics)

The simulation uses a Gaussian pulse to excite the driver membrane,
allowing measurement of the system's impulse response. FFT analysis
reveals the expected sealed-box frequency response with -12dB/octave
rolloff below the system resonance frequency.

Expected runtime: ~3-5 minutes on modern hardware (with native backend)
Output: sealed_subwoofer_results.h5

Grid: Variable based on resolution (default 10mm)
Domain: ~1.6m x 1.6m x 1.6m (cabinet + surrounding air)

Physical Design:
- Enclosure: 40cm W x 45cm D x 50cm H
- Wall thickness: 19mm MDF
- Driver: 12" woofer centered on front baffle
- Internal volume: ~72 liters
"""

import argparse

import numpy as np

from strata_fdtd import (
    PML,
    CircularMembraneSource,
    FDTDSolver,
    GaussianPulse,
    LoudspeakerEnclosure,
)


def build_sealed_subwoofer(waveform):
    """Build sealed subwoofer enclosure geometry.

    Args:
        waveform: Waveform object for the driver membrane.

    Returns:
        Tuple of (enclosure, driver_info) where:
        - enclosure: LoudspeakerEnclosure object
        - driver_info: Dict with driver position and diameter
    """
    # Enclosure dimensions (meters)
    # Typical home theater subwoofer proportions
    width = 0.40  # 40cm
    depth = 0.45  # 45cm
    height = 0.50  # 50cm
    wall_thickness = 0.019  # 19mm MDF

    enc = LoudspeakerEnclosure(
        external_size=(width, depth, height),
        wall_thickness=wall_thickness,
    )

    # 12" woofer centered on front baffle
    # Position is (x, z) on the baffle face - centered horizontally,
    # positioned slightly above center vertically for aesthetics
    driver_diameter = 0.305  # 12" = 305mm
    driver_x = width / 2  # Centered horizontally
    driver_z = height * 0.45  # Slightly below center

    # mounting_depth must be at least wall_thickness to create proper cutout
    # Adding extra 1cm so membrane source has clear air path
    mounting_depth = wall_thickness + 0.01

    enc.add_driver(
        position=(driver_x, driver_z),
        diameter=driver_diameter,
        mounting_depth=mounting_depth,
        baffle_face="front",
        name="woofer",
        waveform=waveform,
        mode=(0, 1),  # Fundamental piston mode
    )

    driver_info = {
        "x": driver_x,
        "z": driver_z,
        "diameter": driver_diameter,
        "mounting_depth": mounting_depth,
    }

    return enc, driver_info


def run_simulation(enc, driver_info, resolution=0.01, duration=0.005, output_file=None):
    """Run FDTD simulation of sealed subwoofer.

    Args:
        enc: LoudspeakerEnclosure object
        driver_info: Driver position and diameter info
        resolution: Grid resolution in meters (default 10mm)
        duration: Simulation duration in seconds (default 5ms)
        output_file: Output HDF5 file path (optional)

    Returns:
        FDTDSolver object after simulation
    """
    width, depth, height = enc.external_size

    # Add air space around cabinet for radiation
    # Need enough space for waves to propagate, probes, and PML absorption
    air_margin = 0.60  # 60cm of air on each side
    pml_depth = 10  # PML cells

    # Calculate grid dimensions
    domain_x = width + 2 * air_margin
    domain_y = depth + 2 * air_margin
    domain_z = height + 2 * air_margin

    grid_nx = int(np.ceil(domain_x / resolution))
    grid_ny = int(np.ceil(domain_y / resolution))
    grid_nz = int(np.ceil(domain_z / resolution))

    print(f"Grid dimensions: {grid_nx} x {grid_ny} x {grid_nz}")
    print(f"Domain size: {domain_x*100:.1f} x {domain_y*100:.1f} x {domain_z*100:.1f} cm")

    # Create solver
    solver = FDTDSolver(
        shape=(grid_nx, grid_ny, grid_nz),
        resolution=resolution,
    )

    print(f"Timestep: {solver.dt*1e6:.3f} us")
    print(f"Using native backend: {solver.using_native}")

    # Build and set geometry
    # Cabinet is positioned with its corner at (air_margin, air_margin, air_margin)
    geometry = enc.build()

    # Translate geometry to center it in domain
    from strata_fdtd.geometry import Translate
    positioned_geometry = Translate(geometry, (air_margin, air_margin, air_margin))

    # Manually voxelize and invert: SDF returns True=solid, solver expects True=air
    sdf_mask = positioned_geometry.voxelize(solver._grid)
    solver.geometry = ~sdf_mask

    # Add PML absorbing boundaries
    solver.add_boundary(PML(depth=pml_depth, axis='all'))

    # Get membrane sources from enclosure
    # These are automatically positioned at driver locations
    sources = enc.get_membrane_sources()
    print(f"Adding {len(sources)} membrane source(s)")

    # Wall thickness is needed for proper source positioning
    wall_thickness = enc.wall_thickness

    for source in sources:
        # Position membrane just inside the cabinet cavity
        # (get_membrane_sources returns y=0, but membrane should be at inner baffle surface)
        membrane_y = air_margin + wall_thickness + 0.01  # 1cm inside cavity

        # Reduce radius slightly to avoid edge effects where membrane meets cabinet walls
        effective_radius = source.radius - 2 * resolution

        translated_source = CircularMembraneSource(
            center=(
                source.center[0] + air_margin,
                membrane_y,
                source.center[2] + air_margin,
            ),
            radius=effective_radius,
            normal_axis=source.normal_axis,
            waveform=source.waveform,
            mode=source.mode,
        )
        solver.add_source(translated_source)

    # Add measurement probes
    # On-axis probe: 50cm in front of driver
    driver_3d = (
        air_margin + driver_info["x"],
        air_margin,  # Front baffle at y=0 + offset
        air_margin + driver_info["z"],
    )

    # Probe 50cm in front (negative y direction from front baffle)
    probe_50cm_y = air_margin - 0.50  # 50cm in front of cabinet
    if probe_50cm_y > pml_depth * resolution:  # Ensure probe is not in PML
        solver.add_probe(
            "50cm_on_axis",
            position=(driver_3d[0], probe_50cm_y, driver_3d[2]),
        )

    # Probe 30cm in front (closer)
    probe_30cm_y = air_margin - 0.30
    if probe_30cm_y > pml_depth * resolution:
        solver.add_probe(
            "30cm_on_axis",
            position=(driver_3d[0], probe_30cm_y, driver_3d[2]),
        )

    # Probe 25cm to the side (for polar pattern reference)
    probe_side_x = air_margin + width + 0.25
    solver.add_probe(
        "25cm_90deg",
        position=(probe_side_x, air_margin + depth / 2, driver_3d[2]),
    )

    # Run simulation
    n_steps = int(np.ceil(duration / solver.dt))
    print(f"Running {n_steps} timesteps ({duration*1000:.1f} ms)")
    print()

    hdf5_saved = False
    try:
        if output_file:
            solver.run(duration=duration, output_file=output_file)
            hdf5_saved = True
        else:
            solver.run(duration=duration)
    except AttributeError as e:
        # Workaround: HDF5 writer doesn't fully support membrane sources yet
        # Run without HDF5 output if there's an attribute error
        if "position" in str(e) or "frequency" in str(e):
            print("Note: HDF5 output disabled (membrane source metadata not fully supported)")
            print("Running simulation without file output...")
            solver.run(duration=duration)
        else:
            raise

    return solver, hdf5_saved


def main():
    """Main entry point for sealed subwoofer demo."""
    parser = argparse.ArgumentParser(
        description="Sealed subwoofer simulation with membrane source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/sealed_subwoofer.py
    python examples/sealed_subwoofer.py --resolution 0.005 --duration 0.01
    python examples/sealed_subwoofer.py --output my_results.h5
        """,
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.01,
        help="Grid resolution in meters (default: 0.01 = 10mm)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.005,
        help="Simulation duration in seconds (default: 0.005 = 5ms)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sealed_subwoofer_results.h5",
        help="Output HDF5 file (default: sealed_subwoofer_results.h5)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="Gaussian pulse center frequency in Hz (default: 50)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Sealed Subwoofer Simulation Demo")
    print("=" * 60)
    print()

    # Physical parameters
    print("Physical configuration:")
    print("  Enclosure: 40cm W x 45cm D x 50cm H (sealed)")
    print("  Wall thickness: 19mm MDF")
    print("  Driver: 12\" (305mm) woofer on front baffle")
    print("  Internal volume: ~72 liters")
    print()

    # Simulation parameters
    print("Simulation parameters:")
    print(f"  Resolution: {args.resolution*1000:.1f} mm")
    print(f"  Duration: {args.duration*1000:.1f} ms")
    print(f"  Source frequency: {args.frequency:.0f} Hz")
    print()

    # Create waveform - Gaussian pulse centered at specified frequency
    # Position is not used for membrane sources (only for point sources)
    waveform = GaussianPulse(
        position=(0, 0, 0),  # Not used for membrane sources
        frequency=args.frequency,
        amplitude=1.0,
    )

    print("Building enclosure geometry...")
    enc, driver_info = build_sealed_subwoofer(waveform)

    print("Initializing simulation...")
    print("-" * 40)

    solver, hdf5_saved = run_simulation(
        enc,
        driver_info,
        resolution=args.resolution,
        duration=args.duration,
        output_file=args.output,
    )

    print("-" * 40)
    print()
    print("=" * 60)
    print("Simulation complete!")
    print("=" * 60)
    print()

    # Report probe data summary
    max_pressure = np.max(np.abs(solver.p))
    print(f"Peak pressure in domain: {max_pressure:.6e} Pa")
    print()

    # Check if output file was saved
    import os
    if hdf5_saved and os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / 1e6
        print(f"Output saved to: {args.output}")
        print(f"File size: {size_mb:.1f} MB")
        print()
        print("Next steps:")
        print("1. Upload the HDF5 file to the FDTD Viewer for visualization")
        print("2. Examine probe data for impulse response")
        print("3. Compute FFT to see frequency response")
        print()
        print("Analysis in Python:")
        print("  import h5py")
        print("  import numpy as np")
        print("  from scipy import fft")
        print()
        print("  f = h5py.File('" + args.output + "', 'r')")
        print("  pressure = f['probes/30cm_on_axis/pressure'][:]")
        print("  time = f['probes/30cm_on_axis/time'][:]")
        print("  dt = time[1] - time[0]")
        print()
        print("  # Frequency response via FFT")
        print("  spectrum = np.abs(fft.rfft(pressure))")
        print("  freq = fft.rfftfreq(len(pressure), dt)")
        print("  spectrum_db = 20 * np.log10(spectrum / np.max(spectrum) + 1e-10)")
    else:
        print("Note: HDF5 output was not saved (membrane source metadata limitation).")
        print("Probe data is available in memory via solver._probes.")
    print()


if __name__ == "__main__":
    main()

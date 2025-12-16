#!/usr/bin/env python3
"""
Generate sample HDF5 simulation files for Viewer Mode demo.

This script creates small reference HDF5 files that users can load
in the Viewer without running simulations first.

Usage:
    python scripts/generate_sample_hdf5.py --all
    python scripts/generate_sample_hdf5.py --sample simple-pulse
    python scripts/generate_sample_hdf5.py --list

Output files are written to strata-web/public/samples/
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for local development
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from strata_fdtd import FDTDSolver, GaussianPulse


def get_output_dir() -> Path:
    """Get the output directory for sample files."""
    output_dir = project_root / "strata-web" / "public" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_simple_pulse() -> Path:
    """Generate a minimal HDF5 file with a simple Gaussian pulse.

    This creates a small file (~1-2MB) suitable for bundling in the repo.

    Configuration:
    - 40x40x40 grid @ 2mm resolution = 80mm cube
    - Single Gaussian pulse source at center
    - 2 probes (near and far field)
    - 50 timesteps with snapshots every 5 steps
    - ~0.2ms simulation time

    Returns:
        Path to the generated file
    """
    print("Generating simple-pulse.h5...")
    print("  Grid: 40x40x40 @ 2mm = 80mm cube")

    # Create a small grid for fast simulation
    solver = FDTDSolver(
        shape=(40, 40, 40),
        resolution=2e-3  # 2mm resolution
    )

    # Add a Gaussian pulse at center
    center = (0.04, 0.04, 0.04)  # 40mm = center of 80mm cube
    solver.add_source(
        GaussianPulse(
            position=center,
            frequency=20e3,  # 20 kHz
            amplitude=1.0,
        )
    )

    # Add probes at different distances
    solver.add_probe("near_field", position=(0.05, 0.04, 0.04))  # 10mm from source
    solver.add_probe("far_field", position=(0.07, 0.04, 0.04))   # 30mm from source

    # Run simulation
    output_path = get_output_dir() / "simple-pulse.h5"
    print(f"  Running simulation (50 timesteps)...")

    # Run for a short duration with snapshot saving
    solver.run(
        duration=0.0002,  # 0.2ms
        output_file=str(output_path),
        snapshot_interval=5,  # Save every 5 steps
    )

    # Get file size
    file_size = output_path.stat().st_size
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size / 1e6:.2f} MB")

    return output_path


def generate_reflection() -> Path:
    """Generate HDF5 with material boundary for reflection demo.

    Configuration:
    - 60x40x40 grid @ 2mm resolution
    - Pulse source on one side
    - Reflective geometry
    - 3 probes for incident/reflected/transmitted

    Returns:
        Path to the generated file
    """
    print("Generating reflection.h5...")
    print("  Grid: 60x40x40 @ 2mm")

    import numpy as np

    solver = FDTDSolver(
        shape=(60, 40, 40),
        resolution=2e-3  # 2mm
    )

    # Add source on left side
    solver.add_source(
        GaussianPulse(
            position=(0.02, 0.04, 0.04),  # 20mm from left
            frequency=20e3,
            amplitude=1.0,
        )
    )

    # Create a wall at center (geometry mask: True=air, False=solid)
    x = np.arange(60) * 2e-3 + 1e-3
    y = np.arange(40) * 2e-3 + 1e-3
    z = np.arange(40) * 2e-3 + 1e-3
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Wall from x=56mm to x=64mm (thin wall)
    geometry = np.ones(solver.shape, dtype=bool)  # All air
    wall_mask = (X > 0.056) & (X < 0.064)
    geometry[wall_mask] = False  # Wall is solid

    solver.set_geometry(geometry)

    # Add probes
    solver.add_probe("incident", position=(0.04, 0.04, 0.04))    # Before wall
    solver.add_probe("reflected", position=(0.03, 0.04, 0.04))   # For reflected wave
    solver.add_probe("transmitted", position=(0.10, 0.04, 0.04)) # After wall

    # Run simulation
    output_path = get_output_dir() / "reflection.h5"
    print(f"  Running simulation...")

    solver.run(
        duration=0.0004,  # 0.4ms
        output_file=str(output_path),
        snapshot_interval=5,
    )

    file_size = output_path.stat().st_size
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size / 1e6:.2f} MB")

    return output_path


SAMPLES = {
    "simple-pulse": {
        "generator": generate_simple_pulse,
        "description": "Simple Gaussian pulse propagation (smallest file)",
    },
    "reflection": {
        "generator": generate_reflection,
        "description": "Pulse reflection from a wall",
    },
}


def list_samples():
    """Print available sample configurations."""
    print("Available sample configurations:")
    print("-" * 50)
    for name, config in SAMPLES.items():
        print(f"  {name}: {config['description']}")
    print()
    print("Use --sample NAME to generate a specific sample")
    print("Use --all to generate all samples")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample HDF5 simulation files for Viewer Mode demo."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all sample files",
    )
    parser.add_argument(
        "--sample",
        choices=list(SAMPLES.keys()),
        help="Generate a specific sample file",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available sample configurations",
    )

    args = parser.parse_args()

    if args.list:
        list_samples()
        return 0

    if not args.all and not args.sample:
        parser.print_help()
        print()
        list_samples()
        return 1

    print("=" * 60)
    print("Sample HDF5 Generator")
    print("=" * 60)
    print()

    generated = []

    if args.all:
        for name, config in SAMPLES.items():
            try:
                path = config["generator"]()
                generated.append((name, path))
                print()
            except Exception as e:
                print(f"  ERROR: {e}")
                print()
    elif args.sample:
        config = SAMPLES[args.sample]
        try:
            path = config["generator"]()
            generated.append((args.sample, path))
        except Exception as e:
            print(f"  ERROR: {e}")
            return 1

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, path in generated:
        size = path.stat().st_size
        print(f"  {name}: {path.name} ({size / 1e6:.2f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

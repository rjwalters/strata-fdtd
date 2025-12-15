#!/usr/bin/env python3
"""Acoustic strata_fdtd unit cell using CSG operations.

This example demonstrates how to create a unit cell for an acoustic strata_fdtd
structure. Metamaterials exhibit exotic properties not found in nature by using
carefully designed sub-wavelength structures. This example showcases:

1. **Periodic Structure**: Repeating unit cell with spherical cavities
2. **Connecting Channels**: Cylindrical passages linking cavities
3. **CSG Composition**: Efficient union of many primitives
4. **Metamaterial Physics**: Locally resonant structures

Physics:
    Acoustic metamaterials can achieve:
    - Negative effective mass density
    - Negative effective bulk modulus
    - Sound focusing and cloaking
    - Sub-wavelength imaging

    This example creates a simple "locally resonant" unit cell where:
    - Cavities act as Helmholtz resonators
    - Connecting channels couple adjacent resonators
    - Array behavior creates frequency-dependent effective properties

Design Considerations:
    - Unit cell size << wavelength (typically λ/4 or smaller)
    - Cavity resonance frequency sets strata_fdtd response
    - Channel dimensions control coupling strength
    - Periodic boundary conditions for simulation

Applications:
    - Acoustic absorbers and barriers
    - Waveguides and filters
    - Transformation acoustics
    - Phononic crystals
"""

import numpy as np

# Import grid for voxelization
from strata_fdtd.grid import UniformGrid

# Import SDF primitives and CSG operations
from strata_fdtd.sdf import Cylinder, Difference, Sphere, Union


def create_metamaterial_unit_cell(
    cell_size=0.020,
    cavity_radius=0.006,
    channel_radius=0.002,
    wall_thickness=0.001,
):
    """Create a strata_fdtd unit cell with connected cavities.

    The unit cell consists of:
    - Central spherical cavity
    - Six connecting channels (±x, ±y, ±z directions)
    - Designed for periodic tiling

    Args:
        cell_size: Size of cubic unit cell in meters
        cavity_radius: Radius of spherical cavity in meters
        channel_radius: Radius of connecting channels in meters
        wall_thickness: Thickness of cavity walls in meters

    Returns:
        tuple: (geometry, grid, info_dict) where geometry is the CSG tree,
               grid is voxelization grid, and info_dict contains design parameters
    """
    cell_center = (cell_size / 2, cell_size / 2, cell_size / 2)

    # --- 1. Create central cavity ---
    # Outer sphere
    outer_sphere = Sphere(center=cell_center, radius=cavity_radius)

    # Inner sphere (hollow cavity)
    inner_radius = cavity_radius - wall_thickness
    inner_sphere = Sphere(center=cell_center, radius=inner_radius)

    # Hollow cavity
    cavity = Difference(outer_sphere, inner_sphere)

    # Calculate cavity volume (for resonance frequency)
    cavity_volume = (4 / 3) * np.pi * inner_radius ** 3

    # --- 2. Create connecting channels in all 6 directions ---
    channels = []

    # Channel extends from cavity surface to cell boundary
    channel_start = cavity_radius
    channel_end = cell_size / 2

    # +X direction
    channels.append(
        Cylinder(
            p1=(cell_center[0] + channel_start, cell_center[1], cell_center[2]),
            p2=(cell_size, cell_center[1], cell_center[2]),
            radius=channel_radius,
        )
    )

    # -X direction
    channels.append(
        Cylinder(
            p1=(cell_center[0] - channel_start, cell_center[1], cell_center[2]),
            p2=(0, cell_center[1], cell_center[2]),
            radius=channel_radius,
        )
    )

    # +Y direction
    channels.append(
        Cylinder(
            p1=(cell_center[0], cell_center[1] + channel_start, cell_center[2]),
            p2=(cell_center[0], cell_size, cell_center[2]),
            radius=channel_radius,
        )
    )

    # -Y direction
    channels.append(
        Cylinder(
            p1=(cell_center[0], cell_center[1] - channel_start, cell_center[2]),
            p2=(cell_center[0], 0, cell_center[2]),
            radius=channel_radius,
        )
    )

    # +Z direction
    channels.append(
        Cylinder(
            p1=(cell_center[0], cell_center[1], cell_center[2] + channel_start),
            p2=(cell_center[0], cell_center[1], cell_size),
            radius=channel_radius,
        )
    )

    # -Z direction
    channels.append(
        Cylinder(
            p1=(cell_center[0], cell_center[1], cell_center[2] - channel_start),
            p2=(cell_center[0], cell_center[1], 0),
            radius=channel_radius,
        )
    )

    # --- 3. Combine cavity and channels ---
    # Union of all channels
    all_channels = Union(*channels)

    # Create openings in cavity for channels
    cavity_with_openings = Difference(cavity, all_channels)

    # Final unit cell (cavity + channel structures)
    # The channels themselves are hollow (air), so we add their walls
    for _channel in channels:
        # Create a slightly larger cylinder for the wall
        # Extract channel parameters (this is simplified - in practice we'd store these)
        pass  # For this example, we'll keep channels as simple openings

    # For simplicity, just use the cavity with openings
    # In a real strata_fdtd, you might add channel walls, mounting structure, etc.
    unit_cell = cavity_with_openings

    # --- 4. Calculate resonance properties ---
    c = 343  # Speed of sound (m/s)
    channel_area = np.pi * channel_radius ** 2
    channel_length = channel_end - channel_start

    # Effective length with end corrections (multiple channels)
    # Each channel acts as a Helmholtz neck
    L_eff = channel_length + 1.2 * channel_radius  # End correction

    # Resonance frequency (approximate, assuming single dominant mode)
    # With 6 channels, effective neck area is 6 × channel_area
    f0 = (c / (2 * np.pi)) * np.sqrt((6 * channel_area) / (cavity_volume * L_eff))

    # --- 5. Create voxelization grid ---
    # High resolution needed for small features
    resolution = min(channel_radius / 5, cell_size / 100)  # At least 5 cells across channel
    n_cells = int(np.ceil(cell_size / resolution))

    grid = UniformGrid(shape=(n_cells, n_cells, n_cells), resolution=resolution)

    # Information dictionary
    info = {
        "cell_size": cell_size,
        "cavity_radius": cavity_radius,
        "cavity_volume": cavity_volume,
        "channel_radius": channel_radius,
        "channel_length": channel_length,
        "resonance_frequency": f0,
        "resolution": resolution,
    }

    print("Metamaterial Unit Cell Created:")
    print(f"  Cell size: {cell_size*1000:.1f} mm cubic")
    print(f"  Cavity radius: {cavity_radius*1000:.1f} mm (volume: {cavity_volume*1e9:.2f} mm³)")
    print(f"  Channel radius: {channel_radius*1000:.1f} mm")
    print("  Number of channels: 6 (±x, ±y, ±z)")
    print(f"  Estimated resonance: {f0:.0f} Hz")
    print(f"  Wavelength at f₀: {c/f0*1000:.1f} mm")
    print(f"  Cell size / wavelength: {cell_size / (c/f0):.2f} (sub-wavelength)")
    print(f"  Voxel grid: {n_cells}³ cells ({resolution*1000:.3f} mm resolution)")

    return unit_cell, grid, info


def create_array(unit_cell_geometry, cell_size, nx=3, ny=3, nz=3):
    """Create an array of unit cells by translation.

    This demonstrates how to tile the unit cell to create a larger strata_fdtd structure.

    Args:
        unit_cell_geometry: The unit cell CSG geometry
        cell_size: Size of one unit cell
        nx, ny, nz: Number of cells in each direction

    Returns:
        Union of all translated unit cells
    """
    print(f"\nCreating {nx}×{ny}×{nz} array of unit cells...")

    # Generate all unit cells with appropriate translations
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Translation for this cell
                offset = (i * cell_size, j * cell_size, k * cell_size)
                # Translate the unit cell
                translated_cell = unit_cell_geometry.translate(offset)
                cells.append(translated_cell)

    # Union all cells
    array = Union(*cells)

    total_size = (nx * cell_size, ny * cell_size, nz * cell_size)
    print(f"  Total structure size: {total_size[0]*1000:.1f} × {total_size[1]*1000:.1f} × {total_size[2]*1000:.1f} mm")
    print(f"  Total cells: {len(cells)}")

    return array


def visualize_cross_section(geometry, grid, slice_axis="z", slice_position=0.5):
    """Visualize a 2D cross-section through the unit cell.

    Args:
        geometry: The CSG geometry
        grid: The voxelization grid
        slice_axis: Which axis to slice perpendicular to ("x", "y", or "z")
        slice_position: Position along axis (0 to 1)
    """
    import matplotlib.pyplot as plt

    print("\nVoxelizing geometry (this may take a moment)...")
    voxel_mask = geometry.voxelize(grid)

    # Extract slice
    if slice_axis == "z":
        slice_idx = int(slice_position * grid.shape[2])
        cross_section = voxel_mask[:, :, slice_idx]
        xlabel, ylabel = "X (mm)", "Y (mm)"
        extent = [0, grid.shape[0] * grid.resolution * 1000,
                  0, grid.shape[1] * grid.resolution * 1000]
    elif slice_axis == "y":
        slice_idx = int(slice_position * grid.shape[1])
        cross_section = voxel_mask[:, slice_idx, :]
        xlabel, ylabel = "X (mm)", "Z (mm)"
        extent = [0, grid.shape[0] * grid.resolution * 1000,
                  0, grid.shape[2] * grid.resolution * 1000]
    elif slice_axis == "x":
        slice_idx = int(slice_position * grid.shape[0])
        cross_section = voxel_mask[slice_idx, :, :]
        xlabel, ylabel = "Y (mm)", "Z (mm)"
        extent = [0, grid.shape[1] * grid.resolution * 1000,
                  0, grid.shape[2] * grid.resolution * 1000]
    else:
        raise ValueError(f"Invalid slice_axis: {slice_axis}")

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(cross_section.T, origin="lower", cmap="RdYlBu_r",
               extent=extent, aspect="equal")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Metamaterial Unit Cell - {slice_axis.upper()} Cross-Section")
    plt.colorbar(label="Air (blue) / Solid (red)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_metamaterial_properties(info):
    """Analyze and display strata_fdtd properties.

    Args:
        info: Info dictionary from create_metamaterial_unit_cell()
    """
    print("\n" + "=" * 60)
    print("Metamaterial Properties Analysis")
    print("=" * 60)

    f0 = info["resonance_frequency"]
    c = 343  # m/s
    wavelength = c / f0

    print("\nAcoustic Properties:")
    print(f"  Resonance frequency: {f0:.0f} Hz")
    print(f"  Wavelength at resonance: {wavelength*1000:.1f} mm")
    print(f"  Cell size: {info['cell_size']*1000:.1f} mm")
    print(f"  Cell/wavelength ratio: {info['cell_size']/wavelength:.3f}")

    # Sub-wavelength criterion
    if info['cell_size'] < wavelength / 4:
        print("  ✓ Sub-wavelength structure (cell < λ/4)")
    else:
        print(f"  ✗ Not sub-wavelength (cell should be < {wavelength/4*1000:.1f} mm)")

    print("\nGeometric Parameters:")
    print(f"  Cavity volume: {info['cavity_volume']*1e9:.2f} mm³")
    print(f"  Channel radius: {info['channel_radius']*1000:.1f} mm")
    print(f"  Channel length: {info['channel_length']*1000:.1f} mm")
    print(f"  Fill factor: {info['cavity_volume'] / info['cell_size']**3:.2%}")

    print("\nSimulation Considerations:")
    print("  - Use periodic boundary conditions for unit cell simulation")
    print("  - Excite with plane wave at normal incidence")
    print("  - Measure transmission/reflection coefficients")
    print("  - Extract effective parameters from S-parameters")
    print("  - Look for bandgaps and resonance features")


def main():
    """Main function demonstrating strata_fdtd unit cell example."""
    print("=" * 60)
    print("Acoustic Metamaterial Unit Cell - CSG Example")
    print("=" * 60)

    # Create a standard unit cell
    print("\n--- Standard Unit Cell ---")
    geometry, grid, info = create_metamaterial_unit_cell(
        cell_size=0.020,  # 20mm unit cell
        cavity_radius=0.006,  # 6mm cavity
        channel_radius=0.002,  # 2mm channels
        wall_thickness=0.001,  # 1mm walls
    )

    # Analyze properties
    analyze_metamaterial_properties(info)

    # Voxelize
    print("\n" + "=" * 60)
    print("Voxelizing unit cell...")
    voxel_mask = geometry.voxelize(grid)
    air_cells = np.sum(voxel_mask)
    total_cells = np.prod(grid.shape)

    print("Voxelization complete:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Air cells: {air_cells:,} ({100 * air_cells / total_cells:.1f}%)")
    print(f"  Solid cells: {total_cells - air_cells:,}")

    # Optional: Create array
    print("\n" + "=" * 60)
    try:
        create_array(geometry, info["cell_size"], nx=3, ny=3, nz=1)
        print("  Array created successfully!")
        print("  Note: Large arrays can take significant time to voxelize")
    except Exception as e:
        print(f"  Array creation: {e}")

    # Optional: Visualize
    try:
        print("\nGenerating cross-section visualization...")
        visualize_cross_section(geometry, grid, slice_axis="z", slice_position=0.5)
        print("Close the plot window to continue...")
    except ImportError:
        print("  (matplotlib not available, skipping visualization)")
    except Exception as e:
        print(f"  (visualization failed: {e})")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  - Metamaterial unit cells are sub-wavelength periodic structures")
    print("  - Resonant cavities provide frequency-dependent response")
    print("  - Connecting channels couple adjacent cells")
    print("  - Arrays create bulk strata_fdtd with exotic properties")
    print("\nNext Steps:")
    print("  - Simulate single unit cell with periodic boundaries")
    print("  - Extract effective material parameters")
    print("  - Optimize geometry for specific frequency response")
    print("  - Create arrays for testing bulk properties")
    print("  - Explore different cavity shapes (cubic, cylindrical)")
    print("  - Add multiple resonances for broadband response")


if __name__ == "__main__":
    main()

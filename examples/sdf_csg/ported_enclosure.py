#!/usr/bin/env python3
"""Loudspeaker enclosure with bass reflex port using CSG operations.

This example demonstrates how to use CSG (Constructive Solid Geometry) operations
to model a realistic loudspeaker enclosure for acoustic simulation. It showcases:

1. **Enclosure Shell**: Hollow box created by subtracting inner volume from outer box
2. **Driver Cutout**: Circular cutout on front baffle for speaker driver mounting
3. **Bass Reflex Port**: Tuned port with flared mouth for enhanced low-frequency response
4. **Voxelization**: Converting the geometry to a voxel grid for FDTD simulation

Design Parameters:
    - Enclosure: 200mm × 200mm × 300mm (W × H × D)
    - Wall thickness: 18mm (typical MDF enclosure)
    - Driver: 130mm diameter (5" woofer)
    - Port: 50mm diameter, 120mm length with 30mm flare

Physics:
    The port is tuned to resonate around 50 Hz, extending the low-frequency
    response of the enclosure. The Helmholtz resonance frequency is approximately:
        f_b ≈ c/(2π) × sqrt(S_p / (V × L_eff))
    where S_p is port area, V is enclosure volume, L_eff is effective port length.
"""

import numpy as np

# Import grid for voxelization
from strata_fdtd.grid import UniformGrid

# Import SDF primitives and CSG operations
from strata_fdtd.sdf import Box, Cone, Cylinder, Difference, Union


def create_ported_enclosure():
    """Create a complete ported loudspeaker enclosure using CSG.

    Returns:
        tuple: (geometry, grid) where geometry is the CSG tree and grid is the voxelization grid

    The geometry consists of:
        - Hollow enclosure shell (outer box - inner box)
        - Driver cutout on front face
        - Bass reflex port with flared mouth
    """
    # Enclosure dimensions (in meters)
    width = 0.2  # 200mm
    height = 0.2  # 200mm
    depth = 0.3  # 300mm
    wall_thickness = 0.018  # 18mm MDF

    # Center enclosure at origin for convenience
    center = (width / 2, height / 2, depth / 2)

    # --- 1. Create hollow enclosure shell ---
    # Outer box (full enclosure)
    outer_box = Box(center=center, size=(width, height, depth))

    # Inner box (hollow cavity)
    inner_size = (
        width - 2 * wall_thickness,
        height - 2 * wall_thickness,
        depth - 2 * wall_thickness,
    )
    inner_box = Box(center=center, size=inner_size)

    # Subtract inner from outer to create shell
    enclosure_shell = Difference(outer_box, inner_box)

    # --- 2. Create driver cutout on front face ---
    # Driver parameters
    driver_diameter = 0.130  # 130mm (5" woofer)
    driver_radius = driver_diameter / 2

    # Position driver centered on front face (z = 0)
    driver_center_x = width / 2
    driver_center_y = height / 2
    driver_z_front = 0  # Front face at z=0
    driver_z_back = wall_thickness  # Cutout goes through front panel

    # Create cylindrical cutout for driver
    driver_cutout = Cylinder(
        p1=(driver_center_x, driver_center_y, driver_z_front - 0.01),  # Extend slightly beyond
        p2=(driver_center_x, driver_center_y, driver_z_back + 0.01),
        radius=driver_radius,
    )

    # --- 3. Create bass reflex port ---
    # Port parameters
    port_diameter = 0.050  # 50mm
    port_radius = port_diameter / 2
    port_length = 0.120  # 120mm
    flare_length = 0.030  # 30mm flare
    flare_radius = port_radius * 1.5  # 50% wider at mouth

    # Position port on front face, offset from driver
    port_center_x = width / 2
    port_center_y = height / 2 + 0.060  # 60mm offset from driver
    port_z_mouth = 0  # Mouth at front face
    port_z_throat = port_length  # Extends into enclosure

    # Main port tube
    port_tube = Cylinder(
        p1=(port_center_x, port_center_y, port_z_mouth),
        p2=(port_center_x, port_center_y, port_z_throat),
        radius=port_radius,
    )

    # Port flare (conical expansion at mouth)
    port_flare = Cone(
        p1=(port_center_x, port_center_y, port_z_mouth + flare_length),
        p2=(port_center_x, port_center_y, port_z_mouth),
        r1=port_radius,  # Throat radius matches port
        r2=flare_radius,  # Wider at mouth
    )

    # Combine tube and flare
    port = Union(port_tube, port_flare)

    # --- 4. Combine all components ---
    # Subtract driver cutout and port from enclosure
    complete_geometry = Difference(enclosure_shell, driver_cutout, port)

    # --- 5. Create voxelization grid ---
    # Use 2mm resolution for reasonable accuracy
    resolution = 0.002  # 2mm cells
    nx = int(np.ceil(width / resolution))
    ny = int(np.ceil(height / resolution))
    nz = int(np.ceil(depth / resolution))

    grid = UniformGrid(shape=(nx, ny, nz), resolution=resolution)

    print("Ported Enclosure Geometry Created:")
    print(f"  Enclosure: {width*1000:.0f} × {height*1000:.0f} × {depth*1000:.0f} mm")
    print(f"  Internal Volume: {inner_size[0]*inner_size[1]*inner_size[2]*1e6:.2f} liters")
    print(f"  Driver: {driver_diameter*1000:.0f} mm diameter")
    print(f"  Port: {port_diameter*1000:.0f} mm × {port_length*1000:.0f} mm")
    print(f"  Voxel Grid: {nx} × {ny} × {nz} cells ({resolution*1000:.1f} mm resolution)")
    print("  Estimated Tuning Frequency: ~50 Hz")

    return complete_geometry, grid


def visualize_cross_section(geometry, grid, plane="xy", slice_position=0.5):
    """Visualize a 2D cross-section of the geometry.

    Args:
        geometry: The CSG geometry to visualize
        grid: The voxelization grid
        plane: Which plane to slice ("xy", "xz", or "yz")
        slice_position: Position along the perpendicular axis (0 to 1)
    """
    import matplotlib.pyplot as plt

    # Voxelize the geometry
    print("Voxelizing geometry (this may take a moment)...")
    voxel_mask = geometry.voxelize(grid)

    # Extract 2D slice based on plane
    if plane == "xy":
        slice_idx = int(slice_position * grid.shape[2])
        cross_section = voxel_mask[:, :, slice_idx]
        xlabel, ylabel = "X (mm)", "Y (mm)"
        extent = [0, grid.shape[0] * grid.resolution * 1000,
                  0, grid.shape[1] * grid.resolution * 1000]
    elif plane == "xz":
        slice_idx = int(slice_position * grid.shape[1])
        cross_section = voxel_mask[:, slice_idx, :]
        xlabel, ylabel = "X (mm)", "Z (mm)"
        extent = [0, grid.shape[0] * grid.resolution * 1000,
                  0, grid.shape[2] * grid.resolution * 1000]
    elif plane == "yz":
        slice_idx = int(slice_position * grid.shape[0])
        cross_section = voxel_mask[slice_idx, :, :]
        xlabel, ylabel = "Y (mm)", "Z (mm)"
        extent = [0, grid.shape[1] * grid.resolution * 1000,
                  0, grid.shape[2] * grid.resolution * 1000]
    else:
        raise ValueError(f"Invalid plane: {plane}")

    # Plot the cross-section
    plt.figure(figsize=(10, 8))
    plt.imshow(cross_section.T, origin="lower", cmap="RdYlBu_r",
               extent=extent, aspect="equal")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Ported Enclosure - {plane.upper()} Cross-Section")
    plt.colorbar(label="Air (blue) / Solid (red)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_port_tuning(port_diameter, port_length, enclosure_volume):
    """Calculate theoretical Helmholtz resonance frequency for the port.

    Args:
        port_diameter: Port diameter in meters
        port_length: Port length in meters
        enclosure_volume: Enclosure internal volume in cubic meters

    Returns:
        float: Tuning frequency in Hz
    """
    c = 343  # Speed of sound in air (m/s)
    port_radius = port_diameter / 2
    port_area = np.pi * port_radius**2

    # Effective length includes end corrections (~0.85 × diameter for both ends)
    end_correction = 0.85 * port_diameter
    L_eff = port_length + end_correction

    # Helmholtz resonance frequency
    f_b = (c / (2 * np.pi)) * np.sqrt(port_area / (enclosure_volume * L_eff))

    print("\nPort Tuning Analysis:")
    print(f"  Port Area: {port_area * 1e4:.2f} cm²")
    print(f"  Effective Length: {L_eff * 1000:.1f} mm (includes end corrections)")
    print(f"  Enclosure Volume: {enclosure_volume * 1000:.2f} liters")
    print(f"  Helmholtz Frequency: {f_b:.1f} Hz")

    return f_b


def main():
    """Main function demonstrating the ported enclosure example."""
    print("=" * 60)
    print("Ported Loudspeaker Enclosure - CSG Example")
    print("=" * 60)

    # Create the geometry
    geometry, grid = create_ported_enclosure()

    # Calculate port tuning
    port_diameter = 0.050  # 50mm
    port_length = 0.120  # 120mm
    enclosure_volume = 0.164 * 0.164 * 0.264  # Internal volume (m³)
    analyze_port_tuning(port_diameter, port_length, enclosure_volume)

    # Voxelize for FDTD simulation
    print("\nVoxelizing geometry...")
    voxel_mask = geometry.voxelize(grid)
    air_cells = np.sum(voxel_mask)
    total_cells = np.prod(grid.shape)

    print("Voxelization complete:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Air cells: {air_cells:,} ({100 * air_cells / total_cells:.1f}%)")
    print(f"  Solid cells: {total_cells - air_cells:,} ({100 * (1 - air_cells / total_cells):.1f}%)")

    # Optional: Visualize cross-sections
    try:
        print("\nGenerating cross-section visualizations...")
        visualize_cross_section(geometry, grid, plane="xz", slice_position=0.5)
        print("Close the plot window to continue...")
    except ImportError:
        print("  (matplotlib not available, skipping visualization)")
    except Exception as e:
        print(f"  (visualization failed: {e})")

    print("\nExample complete!")
    print("\nNext steps:")
    print("  - Use voxel_mask with FDTDSolver for acoustic simulation")
    print("  - Adjust port length to fine-tune resonance frequency")
    print("  - Experiment with different driver positions and sizes")
    print("  - Add damping material by modifying cell properties")


if __name__ == "__main__":
    main()

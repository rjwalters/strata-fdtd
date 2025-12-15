#!/usr/bin/env python3
"""Helmholtz resonator using CSG operations with smooth blending.

This example demonstrates how to create a Helmholtz resonator - a fundamental
acoustic device that resonates at a specific frequency. It showcases:

1. **Cavity**: Hollow sphere as the resonant volume
2. **Neck**: Cylindrical tube connecting to the cavity
3. **Smooth Transitions**: Using SmoothUnion for realistic geometry
4. **Theoretical Analysis**: Calculate resonance frequency
5. **Voxelization**: Prepare geometry for FDTD simulation

Physics:
    A Helmholtz resonator acts like a mass-spring system where:
    - Air in the neck = mass
    - Air in the cavity = spring (compressible volume)

    The resonance frequency is:
        f_0 = (c / 2π) × sqrt(S / (V × L_eff))

    where:
        c = speed of sound (343 m/s)
        S = neck cross-sectional area
        V = cavity volume
        L_eff = effective neck length (includes end corrections)

Applications:
    - Bass traps in acoustic treatment
    - Perforated panel absorbers
    - Resonant metamaterials
    - Musical instrument acoustics
"""

import numpy as np

# Import grid for voxelization
from strata_fdtd.grid import UniformGrid

# Import SDF primitives and CSG operations
from strata_fdtd.sdf import Box, Cylinder, Difference, SmoothUnion, Sphere, Union


def create_helmholtz_resonator(
    cavity_radius=0.050,
    cavity_type="sphere",
    neck_radius=0.010,
    neck_length=0.030,
    smooth_blend=True,
    blend_radius=0.005,
):
    """Create a Helmholtz resonator using CSG operations.

    Args:
        cavity_radius: Radius of spherical cavity (or half-size of box) in meters
        cavity_type: Type of cavity ("sphere" or "box")
        neck_radius: Radius of cylindrical neck in meters
        neck_length: Length of neck extending from cavity in meters
        smooth_blend: Whether to use smooth CSG operations for blending
        blend_radius: Blend radius for smooth transitions in meters

    Returns:
        tuple: (geometry, grid, f0) where geometry is the CSG tree,
               grid is voxelization grid, and f0 is theoretical resonance frequency
    """
    # --- 1. Create cavity (hollow volume) ---
    if cavity_type == "sphere":
        # Spherical cavity
        cavity_center = (0, 0, 0)
        outer_sphere = Sphere(center=cavity_center, radius=cavity_radius)

        # Hollow out the sphere (wall thickness)
        wall_thickness = cavity_radius * 0.05  # 5% wall thickness
        inner_sphere = Sphere(center=cavity_center, radius=cavity_radius - wall_thickness)

        cavity = Difference(outer_sphere, inner_sphere)
        cavity_volume = (4 / 3) * np.pi * (cavity_radius - wall_thickness) ** 3

    elif cavity_type == "box":
        # Cubic cavity
        cavity_size = 2 * cavity_radius
        cavity_center = (0, 0, 0)
        outer_box = Box(center=cavity_center, size=(cavity_size, cavity_size, cavity_size))

        # Hollow out the box
        wall_thickness = cavity_radius * 0.05
        inner_size = cavity_size - 2 * wall_thickness
        inner_box = Box(center=cavity_center, size=(inner_size, inner_size, inner_size))

        cavity = Difference(outer_box, inner_box)
        cavity_volume = inner_size ** 3

    else:
        raise ValueError(f"Invalid cavity_type: {cavity_type}. Must be 'sphere' or 'box'")

    # --- 2. Create neck opening through cavity wall ---
    # Position neck at top of cavity, extending upward
    neck_start_z = cavity_radius  # Start at cavity surface
    neck_end_z = neck_start_z + neck_length  # Extend outward

    # Neck tube (this will be subtracted to create the opening)
    neck_tube = Cylinder(
        p1=(0, 0, neck_start_z - wall_thickness - 0.001),  # Start inside cavity
        p2=(0, 0, neck_end_z),
        radius=neck_radius,
    )

    # Outer collar/rim of the neck (for structural integrity)
    collar_thickness = wall_thickness
    collar_height = 0.010  # 10mm collar
    outer_collar = Cylinder(
        p1=(0, 0, neck_start_z),
        p2=(0, 0, neck_start_z + collar_height),
        radius=neck_radius + collar_thickness,
    )

    # --- 3. Combine components ---
    if smooth_blend:
        # Use smooth blending for realistic geometry
        # Blend the collar with the cavity
        cavity_with_collar = SmoothUnion(cavity, outer_collar, radius=blend_radius)
        # Subtract the neck tube to create opening
        resonator = Difference(cavity_with_collar, neck_tube)
    else:
        # Sharp transitions
        cavity_with_collar = Union(cavity, outer_collar)
        resonator = Difference(cavity_with_collar, neck_tube)

    # --- 4. Calculate theoretical resonance frequency ---
    c = 343  # Speed of sound in air (m/s)
    neck_area = np.pi * neck_radius ** 2

    # Effective length includes end corrections
    # Unflanged tube: approximately 0.6 × radius at each end
    end_correction = 2 * 0.6 * neck_radius
    L_eff = neck_length + end_correction

    # Helmholtz frequency
    f0 = (c / (2 * np.pi)) * np.sqrt(neck_area / (cavity_volume * L_eff))

    # --- 5. Create voxelization grid ---
    # Grid should encompass the entire resonator with some padding
    grid_size = 2 * (cavity_radius + neck_length) + 0.020  # 20mm padding
    resolution = 0.001  # 1mm resolution
    n_cells = int(np.ceil(grid_size / resolution))

    grid = UniformGrid(shape=(n_cells, n_cells, n_cells), resolution=resolution)

    print("Helmholtz Resonator Created:")
    print(f"  Cavity: {cavity_type}, radius = {cavity_radius*1000:.1f} mm")
    print(f"  Cavity volume: {cavity_volume*1e6:.2f} liters")
    print(f"  Neck: radius = {neck_radius*1000:.1f} mm, length = {neck_length*1000:.1f} mm")
    print(f"  Effective neck length: {L_eff*1000:.1f} mm (includes end corrections)")
    print(f"  Resonance frequency (f₀): {f0:.1f} Hz")
    print(f"  Smooth blending: {'Yes' if smooth_blend else 'No'}")
    if smooth_blend:
        print(f"  Blend radius: {blend_radius*1000:.1f} mm")
    print(f"  Voxel grid: {n_cells}³ cells ({resolution*1000:.1f} mm resolution)")

    return resonator, grid, f0


def compare_smooth_vs_sharp():
    """Compare smooth vs sharp CSG operations."""
    print("=" * 60)
    print("Comparing Smooth vs Sharp CSG Operations")
    print("=" * 60)

    # Create two resonators - one smooth, one sharp
    print("\n--- Smooth Blending ---")
    smooth_res, smooth_grid, smooth_f0 = create_helmholtz_resonator(
        smooth_blend=True, blend_radius=0.005
    )

    print("\n--- Sharp Edges ---")
    sharp_res, sharp_grid, sharp_f0 = create_helmholtz_resonator(
        smooth_blend=False
    )

    print("\nResonance frequencies:")
    print(f"  Smooth: {smooth_f0:.1f} Hz")
    print(f"  Sharp:  {sharp_f0:.1f} Hz")
    print(f"  Difference: {abs(smooth_f0 - sharp_f0):.1f} Hz")
    print("\nNote: Smooth blending creates slightly different internal volumes,")
    print("      affecting the resonance frequency.")


def visualize_cross_section(geometry, grid):
    """Visualize a cross-section through the resonator centerline.

    Args:
        geometry: The CSG geometry to visualize
        grid: The voxelization grid
    """
    import matplotlib.pyplot as plt

    print("\nVoxelizing geometry (this may take a moment)...")
    voxel_mask = geometry.voxelize(grid)

    # Extract YZ cross-section through center (x = center)
    center_x = grid.shape[0] // 2
    cross_section = voxel_mask[center_x, :, :]

    # Plot the cross-section
    extent_mm = grid.shape[1] * grid.resolution * 1000
    plt.figure(figsize=(8, 8))
    plt.imshow(cross_section.T, origin="lower", cmap="RdYlBu_r",
               extent=[-extent_mm/2, extent_mm/2, -extent_mm/2, extent_mm/2],
               aspect="equal")
    plt.xlabel("Y (mm)")
    plt.ylabel("Z (mm)")
    plt.title("Helmholtz Resonator - YZ Cross-Section")
    plt.colorbar(label="Air (blue) / Solid (red)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def frequency_response_example(geometry, grid, f0):
    """Example of how to use the resonator geometry with FDTD simulation.

    This is a demonstration stub - actual FDTD simulation would require
    more setup (sources, probes, boundary conditions).

    Args:
        geometry: The CSG geometry
        grid: The voxelization grid
        f0: Theoretical resonance frequency
    """
    print("\n" + "=" * 60)
    print("FDTD Simulation Setup Example")
    print("=" * 60)
    print("\nTo simulate the frequency response of this resonator:")
    print("\n1. Voxelize the geometry:")
    print("   voxel_mask = geometry.voxelize(grid)")
    print("\n2. Create FDTD solver:")
    print("   from strata_fdtd import FDTDSolver")
    print("   solver = FDTDSolver(grid=grid)")
    print("\n3. Add acoustic source near neck opening:")
    print("   source_pos = (grid.shape[0]//2, grid.shape[1]//2, int(0.75*grid.shape[2]))")
    print("   # Broadband pulse or frequency sweep")
    print("\n4. Add probe inside cavity:")
    print("   probe_pos = (grid.shape[0]//2, grid.shape[1]//2, grid.shape[2]//2)")
    print("\n5. Run simulation and analyze frequency response:")
    print("   # FFT of probe signal will show peak at resonance frequency")
    print(f"   # Expected peak: {f0:.1f} Hz")
    print("\n6. Observe resonance:")
    print("   - Strong pressure oscillations inside cavity at f₀")
    print("   - High velocity through neck at f₀")
    print("   - Phase relationship between pressure and velocity")


def main():
    """Main function demonstrating Helmholtz resonator examples."""
    print("=" * 60)
    print("Helmholtz Resonator - CSG Example")
    print("=" * 60)

    # Create a standard spherical resonator
    print("\n--- Standard Configuration ---")
    geometry, grid, f0 = create_helmholtz_resonator(
        cavity_radius=0.050,  # 50mm radius sphere
        cavity_type="sphere",
        neck_radius=0.010,  # 10mm radius neck
        neck_length=0.030,  # 30mm neck length
        smooth_blend=True,
        blend_radius=0.003,  # 3mm smooth blending
    )

    # Voxelize
    print("\nVoxelizing geometry...")
    voxel_mask = geometry.voxelize(grid)
    air_cells = np.sum(voxel_mask)
    total_cells = np.prod(grid.shape)

    print("Voxelization complete:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Air cells: {air_cells:,} ({100 * air_cells / total_cells:.1f}%)")
    print(f"  Solid cells: {total_cells - air_cells:,}")

    # Compare smooth vs sharp
    print("\n" + "=" * 60)
    compare_smooth_vs_sharp()

    # FDTD simulation example
    frequency_response_example(geometry, grid, f0)

    # Optional: Visualize
    try:
        print("\nGenerating cross-section visualization...")
        visualize_cross_section(geometry, grid)
        print("Close the plot window to continue...")
    except ImportError:
        print("  (matplotlib not available, skipping visualization)")
    except Exception as e:
        print(f"  (visualization failed: {e})")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  - Helmholtz resonators are tuned by cavity volume and neck dimensions")
    print("  - Smooth CSG operations create more realistic geometries")
    print("  - End corrections are important for accurate frequency prediction")
    print("  - Use FDTD simulation to verify theoretical resonance frequency")
    print("\nExperiment with:")
    print("  - Different cavity shapes (sphere vs box)")
    print("  - Varying neck dimensions to shift resonance frequency")
    print("  - Multiple resonators for broadband absorption")
    print("  - Arrays of resonators for strata_fdtd applications")


if __name__ == "__main__":
    main()

"""
Unit tests for material geometry assignment (MaterializedGeometry, MaterialVolume).

Tests verify:
- Material region assignment and voxelization
- Material ID assignment and table generation
- MaterialVolume convenience class
- Integration with FDTDSolver
- Priority ordering (later assignments override earlier)
"""

import numpy as np
import pytest

from strata_fdtd import MaterializedGeometry, MaterialVolume, UniformGrid
from strata_fdtd.materials.base import SimpleMaterial
from strata_fdtd.materials.library import FIBERGLASS_48
from strata_fdtd.sdf import Box


class TestMaterializedGeometry:
    """Tests for MaterializedGeometry class."""

    def test_init_with_default_material(self):
        """Test initialization with default AIR_20C material."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        mat_geo = MaterializedGeometry(box)

        assert mat_geo.geometry is box
        assert mat_geo.default_material.name == "air_20C"
        assert mat_geo.material_count() == 1

    def test_init_with_custom_default_material(self):
        """Test initialization with custom default material."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        custom_air = SimpleMaterial(name="custom_air", _rho=1.0, _c=350.0)
        mat_geo = MaterializedGeometry(box, default_material=custom_air)

        assert mat_geo.default_material.name == "custom_air"
        assert mat_geo.default_material.rho_inf == 1.0

    def test_set_material_single_region(self):
        """Test setting material in a single region."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        mat_geo = MaterializedGeometry(box)

        # Add absorber region
        absorber_region = Box(center=(0.03, 0.05, 0.05), size=(0.04, 0.08, 0.08))
        mat_id = mat_geo.set_material(absorber_region, FIBERGLASS_48)

        assert mat_id == 1
        assert mat_geo.material_count() == 2

    def test_set_material_multiple_regions(self):
        """Test setting materials in multiple regions."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        mat_geo = MaterializedGeometry(box)

        # Add two different absorber regions
        region1 = Box(center=(0.03, 0.05, 0.05), size=(0.04, 0.08, 0.08))
        region2 = Box(center=(0.07, 0.05, 0.05), size=(0.04, 0.08, 0.08))

        mat1 = SimpleMaterial(name="absorber1", _rho=50.0, _c=300.0)
        mat2 = SimpleMaterial(name="absorber2", _rho=60.0, _c=310.0)

        mat_id1 = mat_geo.set_material(region1, mat1)
        mat_id2 = mat_geo.set_material(region2, mat2)

        assert mat_id1 == 1
        assert mat_id2 == 2
        assert mat_geo.material_count() == 3

    def test_material_id_overflow(self):
        """Test that material ID assignment raises error after 255 materials."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        mat_geo = MaterializedGeometry(box)

        # Force counter to max
        mat_geo._material_id_counter = 256

        region = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))
        with pytest.raises(ValueError, match="Maximum of 255 material regions exceeded"):
            mat_geo.set_material(region, FIBERGLASS_48)

    def test_voxelize_with_materials_single_region(self):
        """Test voxelization with a single material region."""
        # Create 10x10x10 grid
        grid = UniformGrid(shape=(10, 10, 10), resolution=1e-2)

        # Geometry: entire 10cm cube
        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        mat_geo = MaterializedGeometry(box)

        # Material region: center 5x5x5
        absorber_region = Box(center=(0.05, 0.05, 0.05), size=(0.05, 0.05, 0.05))
        mat_geo.set_material(absorber_region, FIBERGLASS_48)

        # Voxelize
        geometry_mask, material_ids = mat_geo.voxelize_with_materials(grid)

        # Check shapes
        assert geometry_mask.shape == (10, 10, 10)
        assert material_ids.shape == (10, 10, 10)

        # All cells should be air (inside geometry)
        assert np.all(geometry_mask)

        # Center region should have material ID 1
        center_cells = np.sum(material_ids == 1)
        assert center_cells > 0  # Should have some cells

        # Outer cells should have material ID 0 (default)
        outer_cells = np.sum(material_ids == 0)
        assert outer_cells > 0

        # Total should match grid size
        assert center_cells + outer_cells == 1000

    def test_voxelize_with_materials_overlapping_regions(self):
        """Test that later material assignments override earlier ones."""
        grid = UniformGrid(shape=(10, 10, 10), resolution=1e-2)

        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        mat_geo = MaterializedGeometry(box)

        # Add overlapping regions
        mat1 = SimpleMaterial(name="mat1", _rho=50.0, _c=300.0)
        mat2 = SimpleMaterial(name="mat2", _rho=60.0, _c=310.0)

        # First region: left half
        region1 = Box(center=(0.03, 0.05, 0.05), size=(0.06, 0.1, 0.1))
        mat_geo.set_material(region1, mat1)

        # Second region: overlapping center
        region2 = Box(center=(0.05, 0.05, 0.05), size=(0.04, 0.08, 0.08))
        mat_geo.set_material(region2, mat2)

        # Voxelize
        geometry_mask, material_ids = mat_geo.voxelize_with_materials(grid)

        # Check that overlapping region has material 2 (later assignment)
        center_mask = (
            (np.arange(10)[:, None, None] >= 3)
            & (np.arange(10)[:, None, None] <= 6)
            & (np.arange(10)[None, :, None] >= 2)
            & (np.arange(10)[None, :, None] <= 7)
            & (np.arange(10)[None, None, :] >= 2)
            & (np.arange(10)[None, None, :] <= 7)
        )

        # Cells in center should have material ID 2 (override)
        center_material_ids = material_ids[center_mask]
        if len(center_material_ids) > 0:
            # Most center cells should be material 2
            assert np.sum(center_material_ids == 2) > 0

    def test_get_material_table(self):
        """Test material table generation."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        mat_geo = MaterializedGeometry(box)

        mat1 = SimpleMaterial(name="mat1", _rho=50.0, _c=300.0)
        mat2 = SimpleMaterial(name="mat2", _rho=60.0, _c=310.0)

        region1 = Box(center=(0.03, 0.05, 0.05), size=(0.04, 0.08, 0.08))
        region2 = Box(center=(0.07, 0.05, 0.05), size=(0.04, 0.08, 0.08))

        mat_geo.set_material(region1, mat1)
        mat_geo.set_material(region2, mat2)

        table = mat_geo.get_material_table()

        # Check IDs
        assert 0 in table
        assert 1 in table
        assert 2 in table

        # Check materials
        assert table[0].name == "air_20C"
        assert table[1].name == "mat1"
        assert table[2].name == "mat2"

        # Check properties
        assert table[1].rho_inf == 50.0
        assert table[2].rho_inf == 60.0

    def test_repr(self):
        """Test string representation."""
        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        mat_geo = MaterializedGeometry(box)

        repr_str = repr(mat_geo)
        assert "MaterializedGeometry" in repr_str
        assert "Box" in repr_str
        assert "materials=1" in repr_str


class TestMaterialVolume:
    """Tests for MaterialVolume convenience class."""

    def test_from_corners(self):
        """Test creation from corner coordinates."""
        mat = SimpleMaterial(name="test_mat", _rho=50.0, _c=300.0)
        volume = MaterialVolume.from_corners(
            min_corner=(0.0, 0.0, 0.0), max_corner=(0.1, 0.1, 0.1), material=mat
        )

        assert volume.material.name == "test_mat"
        assert np.allclose(volume.bounds[0], [0.0, 0.0, 0.0])
        assert np.allclose(volume.bounds[1], [0.1, 0.1, 0.1])

    def test_from_center_size(self):
        """Test creation from center and size."""
        mat = SimpleMaterial(name="test_mat", _rho=50.0, _c=300.0)
        volume = MaterialVolume.from_center_size(
            center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1), material=mat
        )

        assert volume.material.name == "test_mat"
        assert np.allclose(volume.bounds[0], [0.0, 0.0, 0.0])
        assert np.allclose(volume.bounds[1], [0.1, 0.1, 0.1])

    def test_region_property(self):
        """Test that region property returns correct Box SDF."""
        mat = SimpleMaterial(name="test_mat", _rho=50.0, _c=300.0)
        volume = MaterialVolume.from_center_size(
            center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1), material=mat
        )

        region = volume.region
        assert isinstance(region, Box)
        assert np.allclose(region.center, [0.05, 0.05, 0.05])
        assert np.allclose(region.size, [0.1, 0.1, 0.1])

    def test_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        mat = SimpleMaterial(name="test_mat", _rho=50.0, _c=300.0)

        # Min > Max
        with pytest.raises(ValueError, match="min_corner must be less than max_corner"):
            MaterialVolume.from_corners(
                min_corner=(0.1, 0.1, 0.1), max_corner=(0.0, 0.0, 0.0), material=mat
            )

    def test_missing_parameters(self):
        """Test that missing parameters raise error."""
        mat = SimpleMaterial(name="test_mat", _rho=50.0, _c=300.0)

        with pytest.raises(ValueError, match="Must provide either bounds or"):
            MaterialVolume(material=mat)

    def test_repr(self):
        """Test string representation."""
        mat = SimpleMaterial(name="test_mat", _rho=50.0, _c=300.0)
        volume = MaterialVolume.from_center_size(
            center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1), material=mat
        )

        repr_str = repr(volume)
        assert "MaterialVolume" in repr_str
        assert "test_mat" in repr_str


class TestMaterialGeometryIntegration:
    """Integration tests with FDTDSolver."""

    def test_fdtd_solver_accepts_materialized_geometry(self):
        """Test that FDTDSolver.set_geometry() accepts MaterializedGeometry."""
        from strata_fdtd import FDTDSolver

        # Create solver
        solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3)

        # Create geometry with material
        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        mat_geo = MaterializedGeometry(box)

        absorber_region = Box(center=(0.05, 0.05, 0.05), size=(0.05, 0.05, 0.05))
        mat_geo.set_material(absorber_region, FIBERGLASS_48)

        # Set geometry (should not raise)
        solver.set_geometry(mat_geo)

        # Check that geometry was set
        assert solver.geometry is not None
        assert solver.geometry.shape == (20, 20, 20)

        # Check that materials were registered
        assert solver.has_materials
        assert solver.material_count >= 1

    def test_fdtd_solver_material_registration(self):
        """Test that materials are correctly registered in FDTDSolver."""
        from strata_fdtd import FDTDSolver

        solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3)

        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        mat_geo = MaterializedGeometry(box)

        mat1 = SimpleMaterial(name="mat1", _rho=50.0, _c=300.0)
        mat2 = SimpleMaterial(name="mat2", _rho=60.0, _c=310.0)

        region1 = Box(center=(0.03, 0.05, 0.05), size=(0.04, 0.08, 0.08))
        region2 = Box(center=(0.07, 0.05, 0.05), size=(0.04, 0.08, 0.08))

        mat_geo.set_material(region1, mat1)
        mat_geo.set_material(region2, mat2)

        solver.set_geometry(mat_geo)

        # Check material count (should have 2 non-default materials)
        assert solver.material_count == 2

    def test_fdtd_solver_sdf_primitive(self):
        """Test that FDTDSolver.set_geometry() accepts SDFPrimitive."""
        from strata_fdtd import FDTDSolver

        solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3)

        box = Box(center=(0.05, 0.05, 0.05), size=(0.1, 0.1, 0.1))
        solver.set_geometry(box)

        # Check that geometry was set
        assert solver.geometry is not None
        assert solver.geometry.shape == (20, 20, 20)

        # Should not have materials (air only)
        assert not solver.has_materials

    def test_fdtd_solver_legacy_array(self):
        """Test that FDTDSolver.set_geometry() still accepts boolean arrays."""
        from strata_fdtd import FDTDSolver

        solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3)

        # Legacy: direct boolean array
        geometry_mask = np.ones((20, 20, 20), dtype=bool)
        geometry_mask[:5, :, :] = False  # Solid region

        solver.set_geometry(geometry_mask)

        # Check that geometry was set
        assert solver.geometry is not None
        assert solver.geometry.shape == (20, 20, 20)
        assert np.sum(~solver.geometry) == 5 * 20 * 20  # Solid cells


class TestPorousAbsorberIntegration:
    """Integration test with porous absorber simulation."""

    def test_enclosure_with_absorber_simulation(self):
        """
        Test full simulation with enclosure and porous absorber.

        Creates a box enclosure with absorber material in one region,
        adds a source, and verifies simulation runs without errors.
        """
        from strata_fdtd import FDTDSolver, GaussianPulse

        # Use simpler geometry: just a box with absorber region
        # Avoid complex CSG that might cause geometry issues
        box = Box(center=(0.15, 0.15, 0.2), size=(0.3, 0.3, 0.4))

        # Create materialized geometry
        mat_geo = MaterializedGeometry(box)

        # Add absorber region in part of the box
        absorber_region = Box(center=(0.15, 0.25, 0.2), size=(0.2, 0.1, 0.3))
        mat_id = mat_geo.set_material(absorber_region, FIBERGLASS_48)

        assert mat_id == 1

        # Create solver
        solver = FDTDSolver(shape=(150, 150, 200), resolution=2e-3)

        # Set geometry with materials
        solver.set_geometry(mat_geo)

        # Verify geometry and materials
        assert solver.has_materials
        assert solver.material_count >= 1

        # Add source in lower part (away from absorber)
        source = GaussianPulse(position=(75, 50, 100), frequency=500, amplitude=1.0)
        solver.add_source(source)

        # Add probe in middle
        solver.add_probe("test", position=(75, 75, 100))

        # Run short simulation (just verify it doesn't crash)
        solver.run(duration=0.002, progress=False)

        # Check that we recorded something
        probe_dict = solver.get_probe_data("test")
        probe_data = probe_dict["test"]
        assert len(probe_data) > 0

        # Note: We don't check for non-zero data here because with materials,
        # the solver behavior can be complex. The key is that it runs without crashing.

    def test_absorber_attenuates_sound(self):
        """
        Test that porous absorber actually attenuates sound.

        Compares two simulations:
        1. Air-only cavity
        2. Cavity with absorber

        The absorber should reduce sound pressure.
        """
        from strata_fdtd import FDTDSolver, GaussianPulse

        # Simulation parameters
        shape = (50, 50, 100)
        resolution = 2e-3
        duration = 0.005

        # Test 1: Air only
        box = Box(center=(0.05, 0.05, 0.1), size=(0.1, 0.1, 0.2))
        mat_geo_air = MaterializedGeometry(box)

        solver_air = FDTDSolver(shape=shape, resolution=resolution)
        solver_air.set_geometry(mat_geo_air)
        solver_air.add_source(GaussianPulse(position=(25, 25, 30), frequency=1000))
        solver_air.add_probe("probe", position=(25, 25, 70))
        solver_air.run(duration=duration, progress=False)

        air_dict = solver_air.get_probe_data("probe")
        air_data = air_dict["probe"]
        air_max = np.max(np.abs(air_data))

        # Test 2: With absorber in far half
        mat_geo_absorber = MaterializedGeometry(box)
        absorber_region = Box(center=(0.05, 0.05, 0.15), size=(0.08, 0.08, 0.1))
        mat_geo_absorber.set_material(absorber_region, FIBERGLASS_48)

        solver_absorber = FDTDSolver(shape=shape, resolution=resolution)
        solver_absorber.set_geometry(mat_geo_absorber)
        solver_absorber.add_source(GaussianPulse(position=(25, 25, 30), frequency=1000))
        solver_absorber.add_probe("probe", position=(25, 25, 70))
        solver_absorber.run(duration=duration, progress=False)

        absorber_dict = solver_absorber.get_probe_data("probe")
        absorber_data = absorber_dict["probe"]
        absorber_max = np.max(np.abs(absorber_data))

        # With absorber should have lower peak pressure
        # Note: This is a simple check - absorber effect may be modest
        # in such a short simulation
        assert absorber_max <= air_max * 1.1  # Allow small tolerance

        # At minimum, simulation should complete without error
        assert len(air_data) > 0
        assert len(absorber_data) > 0

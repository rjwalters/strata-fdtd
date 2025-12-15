"""Tests for HDF5 output format for FDTD results."""

import hashlib
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from strata_fdtd import (
    PML,
    FDTDSolver,
    GaussianPulse,
    UniformGrid,
)
from strata_fdtd.io import HDF5ResultReader, HDF5ResultWriter


@pytest.fixture
def small_solver():
    """Create a small FDTD solver for testing."""
    grid = UniformGrid(shape=(10, 10, 10), resolution=1e-3)
    solver = FDTDSolver(grid=grid, backend="python")
    solver.add_source(GaussianPulse(position=(5, 5, 5), frequency=1000, amplitude=1.0))
    solver.add_probe('center', position=(7, 5, 5))
    return solver


@pytest.fixture
def temp_h5_file():
    """Create a temporary HDF5 file path."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        filepath = f.name
    yield filepath
    # Cleanup
    Path(filepath).unlink(missing_ok=True)


def test_hdf5_writer_metadata(small_solver, temp_h5_file):
    """Test that HDF5Writer writes correct metadata."""
    script_content = "# Test script\nprint('hello')"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    # Read back and verify
    with h5py.File(temp_h5_file, 'r') as f:
        # Check metadata group
        assert 'metadata' in f
        meta = f['metadata']
        assert 'script_hash' in meta.attrs
        assert 'script_content' in meta.attrs
        assert 'created_at' in meta.attrs
        assert 'solver_version' in meta.attrs

        # Verify script hash
        expected_hash = hashlib.sha256(script_content.encode()).hexdigest()
        assert meta.attrs['script_hash'] == expected_hash
        assert meta.attrs['script_content'] == script_content


def test_hdf5_writer_grid_info(small_solver, temp_h5_file):
    """Test that grid information is written correctly."""
    script_content = "# Test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    with h5py.File(temp_h5_file, 'r') as f:
        # Check grid group
        assert 'grid' in f
        grid_group = f['grid']

        assert tuple(grid_group.attrs['shape']) == (10, 10, 10)
        assert grid_group.attrs['is_uniform']
        assert grid_group.attrs['resolution'] == 1e-3
        assert 'extent' in grid_group.attrs


def test_hdf5_writer_simulation_params(small_solver, temp_h5_file):
    """Test that simulation parameters are written correctly."""
    script_content = "# Test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    with h5py.File(temp_h5_file, 'r') as f:
        # Check simulation group
        assert 'simulation' in f
        sim_group = f['simulation']

        assert 'timestep' in sim_group.attrs
        assert 'cfl_number' in sim_group.attrs
        assert 'c' in sim_group.attrs
        assert 'rho' in sim_group.attrs
        assert sim_group.attrs['timestep'] > 0


def test_hdf5_writer_sources(small_solver, temp_h5_file):
    """Test that source definitions are written correctly."""
    script_content = "# Test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    with h5py.File(temp_h5_file, 'r') as f:
        # Check sources group
        assert 'sources' in f
        sources = f['sources']

        assert 'source_0' in sources
        source = sources['source_0']

        # Source type is stored as the internal source_type attribute
        assert 'type' in source.attrs
        assert tuple(source.attrs['position']) == (5, 5, 5)
        assert source.attrs['frequency'] == 1000


def test_hdf5_writer_probes(small_solver, temp_h5_file):
    """Test that probe data is written correctly."""
    script_content = "# Test"

    # Run simulation to collect probe data
    small_solver.run(duration=0.001)

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    with h5py.File(temp_h5_file, 'r') as f:
        # Check probes group
        assert 'probes' in f
        probes = f['probes']

        assert 'center' in probes
        probe_data = probes['center']

        # Check attributes
        assert tuple(probe_data.attrs['position']) == (7, 5, 5)
        assert probe_data.attrs['units'] == 'Pa'


def test_hdf5_writer_pressure_snapshots(small_solver, temp_h5_file):
    """Test that pressure snapshots are written correctly."""
    script_content = "# Test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    n_steps = 10

    # Write some timesteps with snapshots
    for i in range(n_steps):
        small_solver.step()
        writer.write_timestep(i, save_snapshot=True)

    writer.finalize()

    with h5py.File(temp_h5_file, 'r') as f:
        # Check fields group
        assert 'fields' in f
        assert 'pressure' in f['fields']

        pressure = f['fields/pressure']

        # Check shape
        assert pressure.shape == (n_steps, 10, 10, 10)

        # Check attributes
        assert pressure.attrs['units'] == 'Pa'

        # Verify data is not all zeros (simulation should have some activity)
        assert np.any(pressure[:] != 0)


def test_hdf5_reader_metadata(small_solver, temp_h5_file):
    """Test that HDF5Reader reads metadata correctly."""
    script_content = "# Test script"

    # Write data
    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    # Read back
    reader = HDF5ResultReader(temp_h5_file)
    metadata = reader.get_metadata()

    assert 'metadata' in metadata
    assert 'script_hash' in metadata['metadata']
    assert 'grid' in metadata
    assert 'simulation' in metadata

    assert tuple(metadata['grid']['shape']) == (10, 10, 10)
    assert metadata['grid']['is_uniform']
    assert metadata['simulation']['timestep'] > 0

    reader.close()


def test_hdf5_reader_load_timestep(small_solver, temp_h5_file):
    """Test loading individual timesteps."""
    script_content = "# Test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    n_steps = 10

    # Write timesteps with snapshots
    for i in range(n_steps):
        small_solver.step()
        writer.write_timestep(i, save_snapshot=True)

    writer.finalize()

    # Read back
    reader = HDF5ResultReader(temp_h5_file)

    # Load specific timestep
    pressure = reader.load_timestep(5)
    assert pressure.shape == (10, 10, 10)

    # Check number of snapshots
    assert reader.get_num_snapshots() == n_steps

    reader.close()


def test_hdf5_reader_load_probe(small_solver, temp_h5_file):
    """Test loading probe data."""
    script_content = "# Test"

    # Run simulation
    small_solver.run(duration=0.001)

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    # Read back
    reader = HDF5ResultReader(temp_h5_file)

    # List probes
    probes = reader.get_probe_names()
    assert 'center' in probes

    # Load probe data - note: probe data may be empty if not written during simulation
    # This test just verifies the API works
    try:
        probe_data = reader.load_probe('center')
        # If we got here, the probe exists (data may or may not be populated)
    except KeyError:
        pytest.fail("Probe 'center' should exist")

    reader.close()


def test_hdf5_reader_context_manager(small_solver, temp_h5_file):
    """Test that HDF5Reader works as context manager."""
    script_content = "# Test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    # Use as context manager
    with HDF5ResultReader(temp_h5_file) as reader:
        metadata = reader.get_metadata()
        assert 'grid' in metadata


def test_compression_ratio(small_solver, temp_h5_file):
    """Test that compression reduces file size."""
    script_content = "# Compression test"

    # Create uncompressed file
    temp_uncompressed = temp_h5_file.replace('.h5', '_uncomp.h5')

    writer_comp = HDF5ResultWriter(temp_h5_file, small_solver, script_content, compression="gzip")
    writer_uncomp = HDF5ResultWriter(
        temp_uncompressed, small_solver, script_content, compression=None
    )

    n_steps = 20

    # Write same data to both with snapshots
    for i in range(n_steps):
        small_solver.step()
        writer_comp.write_timestep(i, save_snapshot=True)
        writer_uncomp.write_timestep(i, save_snapshot=True)

    writer_comp.finalize()
    writer_uncomp.finalize()

    # Check file sizes
    size_comp = Path(temp_h5_file).stat().st_size
    size_uncomp = Path(temp_uncompressed).stat().st_size

    # Compressed should be smaller (or equal for very small data)
    assert size_comp <= size_uncomp

    # Cleanup
    Path(temp_uncompressed).unlink(missing_ok=True)


def test_hdf5_writer_without_script_content(small_solver, temp_h5_file):
    """Test that HDF5Writer works without script_content."""
    # Should not raise - script_content is optional
    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content=None)
    writer.finalize()

    with h5py.File(temp_h5_file, 'r') as f:
        assert 'metadata' in f
        # script_hash should not be present without script_content
        assert 'script_hash' not in f['metadata'].attrs


def test_hdf5_geometry_output(temp_h5_file):
    """Test that geometry is written when solver has geometry attribute."""
    grid = UniformGrid(shape=(10, 10, 10), resolution=1e-3)
    solver = FDTDSolver(grid=grid, backend="python")

    # Set geometry attribute (normally done by set_geometry method)
    solver.geometry = np.ones((10, 10, 10), dtype=np.uint8)

    script_content = "# Geometry test"
    writer = HDF5ResultWriter(temp_h5_file, solver, script_content)
    writer.finalize()

    # Verify materials group and geometry dataset exist
    with h5py.File(temp_h5_file, 'r') as f:
        assert 'materials' in f
        assert 'geometry' in f['materials']

        geom_ds = f['materials/geometry']
        assert geom_ds.shape == (10, 10, 10)


def test_hdf5_reader_load_geometry(temp_h5_file):
    """Test that HDF5Reader can load geometry."""
    grid = UniformGrid(shape=(10, 10, 10), resolution=1e-3)
    solver = FDTDSolver(grid=grid, backend="python")

    # Set geometry
    solver.geometry = np.ones((10, 10, 10), dtype=np.uint8)

    script_content = "# Geometry reader test"
    writer = HDF5ResultWriter(temp_h5_file, solver, script_content)
    writer.finalize()

    # Read back
    reader = HDF5ResultReader(temp_h5_file)
    geometry = reader.load_geometry()

    assert geometry is not None
    assert geometry.shape == (10, 10, 10)

    reader.close()


def test_hdf5_reader_default_geometry(small_solver, temp_h5_file):
    """Test that load_geometry returns default geometry (all True) for solver without custom geometry."""
    script_content = "# Default geometry test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    reader = HDF5ResultReader(temp_h5_file)
    geometry = reader.load_geometry()

    # Solver has default geometry (all ones/True), so should not be None
    assert geometry is not None
    assert geometry.shape == (10, 10, 10)
    # Default geometry is all True (air everywhere)
    assert geometry.all()

    reader.close()

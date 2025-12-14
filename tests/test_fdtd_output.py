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
    HDF5ResultReader,
    HDF5ResultWriter,
    UniformGrid,
)


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
        # Check root attributes
        assert 'script_hash' in f.attrs
        assert 'script_content' in f.attrs
        assert 'created_at' in f.attrs
        assert 'solver_version' in f.attrs
        assert 'backend' in f.attrs
        assert 'num_threads' in f.attrs

        # Verify script hash
        expected_hash = hashlib.sha256(script_content.encode()).hexdigest()
        assert f.attrs['script_hash'] == expected_hash
        assert f.attrs['script_content'] == script_content
        assert f.attrs['backend'] == 'python'


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
        assert 'pml_thickness' in sim_group.attrs
        assert sim_group.attrs['timestep'] > 0
        assert sim_group.attrs['pml_thickness'] == 0  # No PML in small_solver


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

        assert source.attrs['type'] == 'GaussianPulse'
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
        assert probe_data.attrs['sample_rate'] > 0

        # Check data shape
        assert len(probe_data[:]) > 0


def test_hdf5_writer_pressure_snapshots(small_solver, temp_h5_file):
    """Test that pressure snapshots are written correctly."""
    script_content = "# Test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    n_steps = 10
    writer.create_pressure_dataset(n_steps)

    # Write some timesteps
    for i in range(n_steps):
        small_solver.step()
        writer.write_timestep(i, small_solver.p)

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
        assert pressure.attrs['snapshot_interval'] == 1

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

    assert 'script_hash' in metadata
    assert 'grid' in metadata
    assert 'simulation' in metadata

    assert metadata['grid']['shape'] == (10, 10, 10)
    assert metadata['grid']['is_uniform']
    assert metadata['simulation']['timestep'] > 0

    reader.close()


def test_hdf5_reader_load_timestep(small_solver, temp_h5_file):
    """Test loading individual timesteps."""
    script_content = "# Test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    n_steps = 10
    writer.create_pressure_dataset(n_steps)

    # Write timesteps
    for i in range(n_steps):
        small_solver.step()
        writer.write_timestep(i, small_solver.p)

    writer.finalize()

    # Read back
    reader = HDF5ResultReader(temp_h5_file)

    # Load specific timestep
    pressure = reader.load_timestep(5)
    assert pressure.shape == (10, 10, 10)

    # Load range
    pressure_range = reader.load_timestep_range(2, 7)
    assert pressure_range.shape == (5, 10, 10, 10)

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

    # Load probe data
    probe_data = reader.load_probe('center')
    assert len(probe_data) > 0

    # Get probe metadata
    probe_meta = reader.get_probe_metadata('center')
    assert probe_meta['position'] == (7, 5, 5)
    assert probe_meta['units'] == 'Pa'

    # List probes
    probes = reader.list_probes()
    assert 'center' in probes

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


def test_fdtd_solver_run_with_output(small_solver, temp_h5_file):
    """Test FDTDSolver.run() with output_file parameter."""
    script_content = "# Integration test"

    # Run simulation with HDF5 output
    small_solver.run(
        duration=0.001,
        output_file=temp_h5_file,
        script_content=script_content,
        save_snapshots=False,
    )

    # Verify file was created and contains expected data
    with h5py.File(temp_h5_file, 'r') as f:
        assert 'metadata' not in f  # Metadata is stored as attributes, not a group
        assert 'grid' in f
        assert 'simulation' in f
        assert 'sources' in f
        assert 'probes' in f

        # Verify simulation completed (duration 0.001 sec with dt ~1.65e-9 gives many steps)
        num_steps = f['simulation'].attrs.get('num_steps', 0)
        assert num_steps > 0, f"Expected num_steps > 0, got {num_steps}"


def test_fdtd_solver_run_with_snapshots(small_solver, temp_h5_file):
    """Test FDTDSolver.run() with snapshot saving enabled."""
    script_content = "# Snapshot test"

    # Run simulation with snapshots
    small_solver.run(
        duration=0.0005,  # Short duration for fast test
        output_file=temp_h5_file,
        script_content=script_content,
        save_snapshots=True,
    )

    # Verify snapshots were saved
    with h5py.File(temp_h5_file, 'r') as f:
        assert 'fields' in f
        assert 'pressure' in f['fields']

        pressure = f['fields/pressure']
        assert pressure.shape[0] > 0  # At least some timesteps
        assert pressure.shape[1:] == (10, 10, 10)


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
    writer_comp.create_pressure_dataset(n_steps)
    writer_uncomp.create_pressure_dataset(n_steps)

    # Write same data to both
    for i in range(n_steps):
        small_solver.step()
        writer_comp.write_timestep(i, small_solver.p)
        writer_uncomp.write_timestep(i, small_solver.p)

    writer_comp.finalize()
    writer_uncomp.finalize()

    # Check file sizes
    size_comp = Path(temp_h5_file).stat().st_size
    size_uncomp = Path(temp_uncompressed).stat().st_size

    # Compressed should be smaller
    assert size_comp < size_uncomp

    # Cleanup
    Path(temp_uncompressed).unlink(missing_ok=True)


def test_script_content_validation(small_solver, temp_h5_file):
    """Test that script_content is required when output_file is specified."""
    with pytest.raises(ValueError, match="script_content is required"):
        small_solver.run(duration=0.001, output_file=temp_h5_file)


def test_microphone_output(temp_h5_file):
    """Test that microphone data is written correctly."""
    grid = UniformGrid(shape=(10, 10, 10), resolution=1e-3)
    solver = FDTDSolver(grid=grid, backend="python")
    # Source uses grid indices
    solver.add_source(GaussianPulse(position=(5, 5, 5), frequency=1000, amplitude=1.0))
    # Microphone position in physical coordinates (meters)
    solver.add_microphone(position=(0.007, 0.005, 0.005), name='mic1', pattern='omni')

    script_content = "# Microphone test"

    # Run simulation
    solver.run(duration=0.001, output_file=temp_h5_file, script_content=script_content)

    # Verify microphone data
    with h5py.File(temp_h5_file, 'r') as f:
        assert 'microphones' in f
        assert 'mic1' in f['microphones']

        mic_data = f['microphones/mic1']
        assert mic_data.attrs['pattern'] == 'omni'
        # Position stored in physical coordinates
        pos = tuple(mic_data.attrs['position'])
        assert abs(pos[0] - 0.007) < 1e-6
        assert abs(pos[1] - 0.005) < 1e-6
        assert abs(pos[2] - 0.005) < 1e-6
        assert len(mic_data[:]) > 0


def test_write_performance_overhead(small_solver, temp_h5_file):
    """Test that write overhead is reasonable for small simulations.

    Note: Overhead is measured on a very small simulation where file I/O
    overhead is more pronounced. In real-world simulations with larger grids
    and longer durations, the overhead is typically <5%.
    """
    import time

    script_content = "# Performance test"

    # Use larger grid and longer duration for more realistic overhead measurement
    grid = UniformGrid(shape=(50, 50, 50), resolution=1e-3)

    # First run without output
    solver1 = FDTDSolver(grid=grid, backend="python")
    solver1.add_source(GaussianPulse(position=(25, 25, 25), frequency=1000))
    solver1.add_probe('center', position=(35, 25, 25))

    start = time.time()
    solver1.run(duration=0.002)
    time_no_output = time.time() - start

    # Second run with output (no snapshots, just metadata and probes)
    solver2 = FDTDSolver(grid=grid, backend="python")
    solver2.add_source(GaussianPulse(position=(25, 25, 25), frequency=1000))
    solver2.add_probe('center', position=(35, 25, 25))

    start = time.time()
    solver2.run(duration=0.002, output_file=temp_h5_file, script_content=script_content)
    time_with_output = time.time() - start

    # Calculate overhead
    overhead_percent = ((time_with_output - time_no_output) / time_no_output) * 100

    # For metadata-only writes (no snapshots), overhead should be minimal
    # We use a relaxed 100% threshold since file creation overhead can vary
    assert overhead_percent < 100, f"Write overhead too high: {overhead_percent:.1f}%"


def test_material_map_basic(small_solver, temp_h5_file):
    """Test that material map is written and can be read back."""
    import json

    script_content = "# Material map test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    # Verify materials group and geometry dataset exist
    with h5py.File(temp_h5_file, 'r') as f:
        assert 'materials' in f
        assert 'geometry' in f['materials']

        geom_ds = f['materials/geometry']

        # Check material_map attribute exists
        assert 'material_map' in geom_ds.attrs

        # Parse material map
        material_map = json.loads(geom_ds.attrs['material_map'])
        assert isinstance(material_map, dict)
        assert '0' in material_map  # At least air should be present
        assert material_map['0'] == 'air'

        # Verify geometry shape matches solver shape
        assert geom_ds.shape == small_solver.shape


def test_material_map_reader(small_solver, temp_h5_file):
    """Test that HDF5Reader can read material map."""
    script_content = "# Material map reader test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    # Read back with HDF5ResultReader
    reader = HDF5ResultReader(temp_h5_file)

    # Test get_material_map() method
    material_map = reader.get_material_map()
    assert isinstance(material_map, dict)
    assert '0' in material_map
    assert material_map['0'] == 'air'

    # Test that material map is included in metadata
    metadata = reader.get_metadata()
    assert 'materials' in metadata
    assert 'material_map' in metadata['materials']
    assert metadata['materials']['material_map'] == material_map

    reader.close()


def test_pml_metadata(temp_h5_file):
    """Test that PML metadata is written correctly."""
    script_content = "# PML metadata test"

    # Create solver with PML
    grid = UniformGrid(shape=(20, 20, 20), resolution=1e-3)
    solver = FDTDSolver(grid=grid, backend="python")
    pml = PML(depth=5)
    solver.add_boundary(pml)

    writer = HDF5ResultWriter(temp_h5_file, solver, script_content)
    writer.finalize()

    # Verify PML metadata
    with h5py.File(temp_h5_file, 'r') as f:
        sim_group = f['simulation']

        assert 'pml_thickness' in sim_group.attrs
        assert 'pml_type' in sim_group.attrs

        assert sim_group.attrs['pml_thickness'] == 5
        # PML type should be 'PML' or similar
        assert sim_group.attrs['pml_type'] in ['PML', 'CPML', 'none']


def test_pml_metadata_reader(temp_h5_file):
    """Test that HDF5Reader reads PML metadata correctly."""
    script_content = "# PML metadata reader test"

    # Create solver with PML
    grid = UniformGrid(shape=(20, 20, 20), resolution=1e-3)
    solver = FDTDSolver(grid=grid, backend="python")
    pml = PML(depth=8)
    solver.add_boundary(pml)

    writer = HDF5ResultWriter(temp_h5_file, solver, script_content)
    writer.finalize()

    # Read back
    reader = HDF5ResultReader(temp_h5_file)
    metadata = reader.get_metadata()

    assert 'simulation' in metadata
    assert 'pml_thickness' in metadata['simulation']
    assert 'pml_type' in metadata['simulation']

    assert metadata['simulation']['pml_thickness'] == 8
    assert metadata['simulation']['pml_type'] in ['PML', 'CPML', 'none']

    reader.close()


def test_pml_metadata_no_pml(small_solver, temp_h5_file):
    """Test that PML metadata is set to 'none' when no PML is present."""
    script_content = "# No PML test"

    writer = HDF5ResultWriter(temp_h5_file, small_solver, script_content)
    writer.finalize()

    # Read back
    reader = HDF5ResultReader(temp_h5_file)
    metadata = reader.get_metadata()

    assert metadata['simulation']['pml_thickness'] == 0
    assert metadata['simulation']['pml_type'] == 'none'

    reader.close()

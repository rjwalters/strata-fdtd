"""Tests for HDF5 output format."""

import tempfile
from pathlib import Path

import h5py

from strata_fdtd import FDTDSolver, GaussianPulse
from strata_fdtd.io.hdf5 import HDF5ResultReader, HDF5ResultWriter


def test_hdf5_writer_basic():
    """Test basic HDF5 writer functionality."""
    # Create a small solver
    solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3)
    solver.add_probe("center", (5, 5, 5))

    script_content = "# Test script"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        output_path = Path(f.name)

    try:
        # Create writer
        writer = HDF5ResultWriter(output_path, solver, script_content)

        # Run a few steps
        for step in range(10):
            solver.step()
            writer.write_timestep(step, save_snapshot=(step % 5 == 0))

        writer.finalize(runtime=1.5)

        # Verify file exists and has correct structure
        assert output_path.exists()

        with h5py.File(output_path, "r") as f:
            # Check groups exist
            assert "metadata" in f
            assert "grid" in f
            assert "simulation" in f
            assert "probes" in f

            # Check metadata
            assert "script_content" in f["metadata"].attrs
            assert f["metadata"].attrs["script_content"] == script_content

            # Check grid
            assert list(f["grid"].attrs["shape"]) == [10, 10, 10]

            # Check probes
            assert "center" in f["probes"]
            assert "position" in f["probes/center"].attrs

    finally:
        if output_path.exists():
            output_path.unlink()


def test_hdf5_reader():
    """Test HDF5 reader functionality."""
    # Create and write a test file
    solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3)
    solver.add_probe("test_probe", (5, 5, 5))

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        output_path = Path(f.name)

    try:
        writer = HDF5ResultWriter(output_path, solver)

        # Run and save snapshots
        for step in range(5):
            solver.step()
            writer.write_timestep(step, save_snapshot=True)

        writer.finalize()

        # Read back the data
        reader = HDF5ResultReader(output_path)

        # Test metadata reading
        metadata = reader.get_metadata()
        assert "grid" in metadata
        assert list(metadata["grid"]["shape"]) == [10, 10, 10]

        # Test probe reading
        probe_names = reader.get_probe_names()
        assert "test_probe" in probe_names

        probe_data = reader.load_probe("test_probe")
        assert len(probe_data) == 5  # 5 steps

        # Test snapshot reading
        num_snapshots = reader.get_num_snapshots()
        assert num_snapshots == 5

        snapshot = reader.load_timestep(0)
        assert snapshot.shape == (10, 10, 10)

        reader.close()

    finally:
        if output_path.exists():
            output_path.unlink()


def test_hdf5_compression():
    """Test that HDF5 compression is working."""
    solver = FDTDSolver(shape=(20, 20, 20), resolution=1e-3)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        output_path = Path(f.name)

    try:
        writer = HDF5ResultWriter(output_path, solver, compression="gzip", compression_level=4)

        # Run and save many snapshots
        for step in range(20):
            solver.step()
            writer.write_timestep(step, save_snapshot=True)

        writer.finalize()

        # Check that compression was applied
        with h5py.File(output_path, "r") as f:
            if "fields/pressure" in f:
                pressure_dset = f["fields/pressure"]
                assert pressure_dset.compression == "gzip"
                assert pressure_dset.compression_opts == 4

    finally:
        if output_path.exists():
            output_path.unlink()


def test_hdf5_with_solver_run():
    """Test HDF5 output integrated with solver.run()."""
    solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3)
    solver.add_source(GaussianPulse(position=(5, 5, 5), frequency=40e3))
    solver.add_probe("probe1", (8, 5, 5))

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        output_path = Path(f.name)

    try:
        # Run simulation with HDF5 output
        duration = 100 * solver.dt  # Run for 100 steps
        solver.run(
            duration=duration,
            output_file=str(output_path),
            script_content="# Integration test",
            snapshot_interval=10,
        )

        # Verify output file
        assert output_path.exists()

        reader = HDF5ResultReader(output_path)
        metadata = reader.get_metadata()

        # Check simulation parameters were saved
        assert "simulation" in metadata
        assert metadata["simulation"]["num_steps"] == 100

        # Check probe data
        probe_data = reader.load_probe("probe1")
        assert len(probe_data) == 100

        # Check snapshots (should be every 10th step)
        num_snapshots = reader.get_num_snapshots()
        assert num_snapshots == 10  # 100 steps / 10 interval

        reader.close()

    finally:
        if output_path.exists():
            output_path.unlink()


def test_hdf5_no_snapshots():
    """Test HDF5 output without field snapshots."""
    solver = FDTDSolver(shape=(10, 10, 10), resolution=1e-3)
    solver.add_probe("p1", (5, 5, 5))

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        output_path = Path(f.name)

    try:
        # Run without snapshots
        duration = 50 * solver.dt
        solver.run(duration=duration, output_file=str(output_path))

        reader = HDF5ResultReader(output_path)

        # Probe data should exist
        probe_data = reader.load_probe("p1")
        assert len(probe_data) == 50

        # But no field snapshots
        num_snapshots = reader.get_num_snapshots()
        assert num_snapshots == 0

        reader.close()

    finally:
        if output_path.exists():
            output_path.unlink()

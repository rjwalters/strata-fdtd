"""HDF5 output format for FDTD simulation results.

This module provides streaming writers and readers for FDTD results in a
standardized HDF5 format optimized for:
- Efficient storage with compression
- Streaming during computation
- Fast random access for visualization
- Complete reproducibility metadata
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .fdtd import FDTDSolver


class HDF5ResultWriter:
    """Streaming writer for FDTD simulation results.

    Creates an HDF5 file with a standardized schema including:
    - Simulation metadata (grid, materials, timestep, etc.)
    - Source script for reproducibility
    - Field snapshots (pressure, velocity) with compression
    - Probe time series
    - Source and geometry information

    Example:
        >>> writer = HDF5ResultWriter("results.h5", solver, script_content)
        >>> for step in range(num_steps):
        ...     solver.step()
        ...     writer.write_timestep(step)
        >>> writer.finalize(runtime_seconds=123.4)
    """

    def __init__(
        self,
        filename: str | Path,
        solver: "FDTDSolver",
        script_content: str | None = None,
        compression: str = "gzip",
        compression_level: int = 4,
    ):
        """Initialize HDF5 writer.

        Args:
            filename: Output file path
            solver: FDTD solver instance
            script_content: Source script for reproducibility
            compression: Compression algorithm ('gzip', 'lzf', None)
            compression_level: Compression level (0-9 for gzip)
        """
        self.filename = Path(filename)
        self.solver = solver
        self.file = h5py.File(filename, "w")
        self.compression = compression
        self.compression_opts = compression_level if compression == "gzip" else None

        self._write_metadata(script_content)
        self._create_datasets()

    def _write_metadata(self, script_content: str | None):
        """Write simulation metadata to HDF5 attributes."""
        solver = self.solver

        # Top-level metadata
        meta = self.file.create_group("metadata")
        if script_content:
            script_hash = hashlib.sha256(script_content.encode()).hexdigest()
            meta.attrs["script_hash"] = script_hash
            meta.attrs["script_content"] = script_content
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["solver_version"] = "0.1.0"  # TODO: Get from package version

        # Grid information
        grid_group = self.file.create_group("grid")
        grid_group.attrs["shape"] = list(solver.shape)

        if hasattr(solver.grid, "is_uniform") and solver.grid.is_uniform:
            grid_group.attrs["is_uniform"] = True
            grid_group.attrs["resolution"] = solver.dx
        else:
            grid_group.attrs["is_uniform"] = False
            # For nonuniform grids, store coordinate arrays
            grid_group.create_dataset("x_coords", data=solver.grid.x_coords)
            grid_group.create_dataset("y_coords", data=solver.grid.y_coords)
            grid_group.create_dataset("z_coords", data=solver.grid.z_coords)

        extent = [
            solver.shape[0] * solver.dx,
            solver.shape[1] * solver.dx,
            solver.shape[2] * solver.dx,
        ]
        grid_group.attrs["extent"] = extent

        # Simulation parameters
        sim_group = self.file.create_group("simulation")
        sim_group.attrs["timestep"] = solver.dt
        # Calculate CFL number: CFL = c * dt / min_spacing
        cfl_number = solver.c * solver.dt / solver.grid.min_spacing
        sim_group.attrs["cfl_number"] = cfl_number
        sim_group.attrs["c"] = solver.c
        sim_group.attrs["rho"] = solver.rho

        # Source information
        sources_group = self.file.create_group("sources")
        for i, source in enumerate(solver._sources):
            src = sources_group.create_group(f"source_{i}")
            src.attrs["type"] = source.source_type
            # Handle both tuple and array positions
            pos = source.position
            if hasattr(pos, "__iter__"):
                src.attrs["position"] = list(pos)
            else:
                src.attrs["position"] = [pos, pos, pos]
            src.attrs["frequency"] = source.frequency
            if hasattr(source, "bandwidth"):
                src.attrs["bandwidth"] = source.bandwidth

    def _create_datasets(self):
        """Create HDF5 datasets for field snapshots and probes."""
        solver = self.solver

        # Create fields group
        self.file.create_group("fields")

        # Prepare for pressure field snapshots
        # We'll create the dataset when we know num_steps
        self.pressure_dataset = None
        self._next_snapshot_idx = 0

        # Create probe datasets
        probes_group = self.file.create_group("probes")
        for name, probe in solver._probes.items():
            # We'll resize these as we collect data
            dataset = probes_group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                chunks=True,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            dataset.attrs["position"] = list(probe.position)
            dataset.attrs["units"] = "Pa"

        # Store material map if available
        if hasattr(solver, "geometry"):
            materials_group = self.file.create_group("materials")
            materials_group.create_dataset(
                "geometry",
                data=solver.geometry.astype(np.uint8),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

    def write_timestep(self, step: int, save_snapshot: bool = False):
        """Write data for current timestep.

        Args:
            step: Current timestep number
            save_snapshot: If True, save pressure field snapshot
        """
        solver = self.solver

        # Update probe data
        probes_group = self.file["probes"]
        for name, probe in solver._probes.items():
            dataset = probes_group[name]
            # Resize and append current value
            current_data = probe.get_data()
            if len(current_data) > dataset.shape[0]:
                dataset.resize((len(current_data),))
                dataset[-1] = current_data[-1]

        # Save pressure snapshot if requested
        if save_snapshot:
            if self.pressure_dataset is None:
                # Create dataset on first snapshot
                fields_group = self.file["fields"]
                # Use chunking for efficient time-slice access
                chunk_shape = (1,) + solver.shape
                self.pressure_dataset = fields_group.create_dataset(
                    "pressure",
                    shape=(1,) + solver.shape,
                    maxshape=(None,) + solver.shape,
                    dtype=np.float32,
                    chunks=chunk_shape,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
                self.pressure_dataset.attrs["units"] = "Pa"
                self.pressure_dataset.attrs["snapshot_interval"] = 1

            # Resize and write pressure field
            idx = self._next_snapshot_idx
            if idx >= self.pressure_dataset.shape[0]:
                self.pressure_dataset.resize((idx + 1,) + solver.shape)
            self.pressure_dataset[idx] = solver.p
            self._next_snapshot_idx += 1

    def finalize(self, runtime: float | None = None, **extra_metadata):
        """Write final metadata and close file.

        Args:
            runtime: Total simulation runtime in seconds
            **extra_metadata: Additional metadata to store
        """
        solver = self.solver

        # Update simulation metadata with final values
        sim_group = self.file["simulation"]
        sim_group.attrs["num_steps"] = solver.step_count
        sim_group.attrs["total_time"] = solver.time

        if runtime is not None:
            self.file["metadata"].attrs["total_runtime_seconds"] = runtime

        # Add any extra metadata
        if extra_metadata:
            for key, value in extra_metadata.items():
                self.file["metadata"].attrs[key] = value

        # Flush and close
        self.file.flush()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.finalize()


class HDF5ResultReader:
    """Reader for FDTD simulation results from HDF5 files.

    Provides efficient access to simulation data including:
    - Metadata and configuration
    - Pressure field snapshots
    - Probe time series
    - Source and geometry information

    Example:
        >>> reader = HDF5ResultReader("results.h5")
        >>> metadata = reader.get_metadata()
        >>> probe_data = reader.load_probe("probe_0")
        >>> snapshot = reader.load_timestep(100)
    """

    def __init__(self, filename: str | Path):
        """Initialize reader.

        Args:
            filename: Path to HDF5 results file
        """
        self.filename = Path(filename)
        self.file = h5py.File(filename, "r")

    def get_metadata(self) -> dict[str, Any]:
        """Extract all simulation metadata.

        Returns:
            Dict with metadata, grid, simulation parameters, etc.
        """
        metadata = {}

        # Top-level metadata
        if "metadata" in self.file:
            metadata["metadata"] = dict(self.file["metadata"].attrs)

        # Grid information
        if "grid" in self.file:
            grid = dict(self.file["grid"].attrs)
            if not grid.get("is_uniform", True):
                # Include coordinate arrays for nonuniform grids
                grid["x_coords"] = self.file["grid/x_coords"][:]
                grid["y_coords"] = self.file["grid/y_coords"][:]
                grid["z_coords"] = self.file["grid/z_coords"][:]
            metadata["grid"] = grid

        # Simulation parameters
        if "simulation" in self.file:
            metadata["simulation"] = dict(self.file["simulation"].attrs)

        # Source information
        if "sources" in self.file:
            sources = []
            for name in self.file["sources"]:
                src_data = dict(self.file[f"sources/{name}"].attrs)
                sources.append(src_data)
            metadata["sources"] = sources

        # Probe information
        if "probes" in self.file:
            probes = {}
            for name in self.file["probes"]:
                probe_attrs = dict(self.file[f"probes/{name}"].attrs)
                probes[name] = probe_attrs
            metadata["probes"] = probes

        return metadata

    def load_timestep(self, step: int) -> NDArray[np.floating]:
        """Load pressure field for specific timestep.

        Args:
            step: Timestep index

        Returns:
            3D pressure field array
        """
        if "fields/pressure" not in self.file:
            raise ValueError("No pressure field data in file")

        return self.file["fields/pressure"][step]

    def load_probe(self, probe_name: str) -> NDArray[np.floating]:
        """Load probe time series.

        Args:
            probe_name: Name of probe to load

        Returns:
            1D array of pressure values over time
        """
        if f"probes/{probe_name}" not in self.file:
            available = list(self.file["probes"].keys()) if "probes" in self.file else []
            raise KeyError(
                f"Probe '{probe_name}' not found. Available: {available}"
            )

        return self.file[f"probes/{probe_name}"][:]

    def get_probe_names(self) -> list[str]:
        """Get list of available probe names."""
        if "probes" not in self.file:
            return []
        return list(self.file["probes"].keys())

    def get_num_snapshots(self) -> int:
        """Get number of saved pressure field snapshots."""
        if "fields/pressure" not in self.file:
            return 0
        return self.file["fields/pressure"].shape[0]

    def load_geometry(self) -> NDArray[np.bool_] | None:
        """Load geometry mask if available.

        Returns:
            3D boolean geometry array, or None if not present
        """
        if "materials/geometry" not in self.file:
            return None

        return self.file["materials/geometry"][:].astype(bool)

    def close(self):
        """Close the HDF5 file."""
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

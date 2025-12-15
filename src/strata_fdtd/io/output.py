"""HDF5 output format for FDTD simulation results.

This module provides standardized HDF5 storage for FDTD simulations with:
- Complete metadata (grid, simulation params, solver config)
- Material geometry and material map
- PML boundary metadata (thickness and type)
- Pressure field snapshots (compressed, chunked for efficient access)
- Probe time series with position metadata
- Source definitions for reproducibility
- Embedded source script with hash

File structure:
    results_{hash}.h5
    ├─ /metadata (attributes: script_hash, created_at, solver_version, backend, num_threads)
    ├─ /grid (group)
    │  └─ attributes: shape, is_uniform, resolution (or x/y/z_coords), extent
    ├─ /materials (group)
    │  └─ /geometry (dataset: uint8 material IDs)
    │     └─ attributes: material_map (JSON string mapping IDs to names)
    ├─ /simulation (group)
    │  └─ attributes: timestep, cfl_number, pml_thickness, pml_type, num_steps, total_time
    ├─ /fields/pressure (dataset, chunked, compressed)
    ├─ /sources/* (groups with attributes)
    └─ /probes/* (datasets with attributes)

Material Map Format:
    The material_map attribute is a JSON string mapping material IDs to names:
    {"0": "air", "1": "pzt5", "2": "aluminum"}

PML Metadata:
    - pml_thickness: Number of PML cells (0 if no PML)
    - pml_type: Type of PML ("PML", "CPML", or "none")

Example:
    >>> from strata_fdtd import FDTDSolver, HDF5ResultWriter, HDF5ResultReader
    >>> solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)
    >>> # ... add sources, probes, run simulation ...
    >>> writer = HDF5ResultWriter("results.h5", solver, script_content=open(__file__).read())
    >>> writer.finalize()
    >>>
    >>> # Read back material map and metadata
    >>> reader = HDF5ResultReader("results.h5")
    >>> material_map = reader.get_material_map()
    >>> metadata = reader.get_metadata()
    >>> print(f"PML thickness: {metadata['simulation']['pml_thickness']} cells")
    >>> print(f"PML type: {metadata['simulation']['pml_type']}")
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import h5py
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .fdtd import FDTDSolver


class HDF5ResultWriter:
    """Streaming writer for FDTD simulation results.

    Writes results to HDF5 file during simulation with compression and
    efficient chunking for time-series data.

    Args:
        filename: Output HDF5 file path
        solver: FDTDSolver instance to extract metadata from
        script_content: Source script content for reproducibility
        compression: Compression algorithm ("gzip" or None)
        compression_level: Compression level (1-9 for gzip, default 4)

    Example:
        >>> solver = FDTDSolver(shape=(100, 100, 100), resolution=1e-3)
        >>> writer = HDF5ResultWriter("results.h5", solver, script_content=open(__file__).read())
        >>> # During simulation:
        >>> for step in range(num_steps):
        >>>     solver.step()
        >>>     writer.write_timestep(step, solver.p)
        >>> writer.finalize()
    """

    def __init__(
        self,
        filename: str,
        solver: FDTDSolver,
        script_content: str,
        compression: str | None = "gzip",
        compression_level: int = 4,
    ):
        self.filename = filename
        self.solver = solver
        self.script_content = script_content
        self.compression = compression
        self.compression_level = compression_level

        # Create file and write initial metadata
        self.file = h5py.File(filename, 'w')
        self._write_metadata(solver, script_content)
        self._write_grid(solver)
        self._write_simulation_params(solver)
        self._write_materials(solver)

        # Pressure field dataset will be created on first write
        self._pressure_dataset = None
        self._current_step = 0
        self._start_time = time.time()

    def _write_metadata(self, solver: FDTDSolver, script_content: str) -> None:
        """Write top-level metadata attributes."""
        # Compute script hash for reproducibility
        script_hash = hashlib.sha256(script_content.encode()).hexdigest()

        # Get solver backend info
        backend = "python"
        if solver.using_native:
            backend = "native"
        elif solver.using_gpu:
            backend = "gpu"

        num_threads = 1
        if solver.using_native:
            from . import _kernels
            num_threads = _kernels.get_num_threads()

        # Write metadata as root attributes
        self.file.attrs['script_hash'] = script_hash
        self.file.attrs['script_content'] = script_content
        self.file.attrs['created_at'] = datetime.now(timezone.utc).isoformat()
        self.file.attrs['solver_version'] = "0.1.0"
        self.file.attrs['backend'] = backend
        self.file.attrs['num_threads'] = num_threads

    def _write_grid(self, solver: FDTDSolver) -> None:
        """Write grid information."""
        grid_group = self.file.create_group('grid')

        # Basic grid properties
        grid_group.attrs['shape'] = solver.shape
        grid_group.attrs['is_uniform'] = solver.grid.is_uniform

        if solver.grid.is_uniform:
            # Uniform grid: store single resolution value
            grid_group.attrs['resolution'] = solver.dx
        else:
            # Nonuniform grid: store coordinate arrays
            grid_group.create_dataset('x_coords', data=solver.grid.x_coords, compression='gzip')
            grid_group.create_dataset('y_coords', data=solver.grid.y_coords, compression='gzip')
            grid_group.create_dataset('z_coords', data=solver.grid.z_coords, compression='gzip')

        # Physical extent
        extent = solver.grid.physical_extent()
        grid_group.attrs['extent'] = extent

    def _write_simulation_params(self, solver: FDTDSolver) -> None:
        """Write simulation parameters as attributes."""
        sim_group = self.file.create_group('simulation')

        sim_group.attrs['timestep'] = solver.dt
        sim_group.attrs['cfl_number'] = 0.5  # Standard CFL for stability

        # PML metadata if applicable
        # Look for PML in boundaries list
        pml = None
        if hasattr(solver, '_boundaries'):
            from .boundaries import PML
            for boundary in solver._boundaries:
                if isinstance(boundary, PML):
                    pml = boundary
                    break

        if pml is not None:
            sim_group.attrs['pml_thickness'] = pml.depth
            # Store PML type if available
            if hasattr(pml, 'type'):
                sim_group.attrs['pml_type'] = pml.type
            else:
                sim_group.attrs['pml_type'] = 'PML'  # Default type
        else:
            sim_group.attrs['pml_thickness'] = 0
            sim_group.attrs['pml_type'] = 'none'

    def _write_materials(self, solver: FDTDSolver) -> None:
        """Write material geometry and material map."""
        import json

        materials_group = self.file.create_group('materials')

        # Create material map from solver's material dictionary
        material_map = {}
        if hasattr(solver, '_materials') and solver._materials:
            for material_id, material in solver._materials.items():
                if hasattr(material, 'name'):
                    material_map[str(material_id)] = material.name
                else:
                    material_map[str(material_id)] = f"material_{material_id}"
        else:
            # Default mapping: 0 = air
            material_map["0"] = "air"

        # Store geometry with material map
        if hasattr(solver, '_material_id'):
            geom_ds = materials_group.create_dataset(
                'geometry',
                data=solver._material_id.astype(np.uint8),
                compression=self.compression,
                compression_opts=self.compression_level if self.compression == "gzip" else None,
            )
        else:
            # Fallback: create all-air geometry if _material_id not available
            geom_ds = materials_group.create_dataset(
                'geometry',
                data=np.zeros(solver.shape, dtype=np.uint8),
                compression=self.compression,
                compression_opts=self.compression_level if self.compression == "gzip" else None,
            )

        # Add material map as JSON string attribute
        geom_ds.attrs['material_map'] = json.dumps(material_map)

    def create_pressure_dataset(self, num_steps: int) -> None:
        """Create pressure field dataset with chunking and compression.

        Args:
            num_steps: Total number of timesteps to allocate
        """
        fields_group = self.file.create_group('fields')

        nx, ny, nz = self.solver.shape

        # Create chunked dataset: one timestep per chunk for efficient scrubbing
        chunk_shape = (1, nx, ny, nz)

        self._pressure_dataset = fields_group.create_dataset(
            'pressure',
            shape=(num_steps, nx, ny, nz),
            dtype=np.float32,
            chunks=chunk_shape,
            compression=self.compression,
            compression_opts=self.compression_level if self.compression == "gzip" else None,
        )

        self._pressure_dataset.attrs['units'] = 'Pa'
        self._pressure_dataset.attrs['snapshot_interval'] = 1

    def write_timestep(self, step: int, pressure: NDArray[np.floating]) -> None:
        """Write pressure field for a single timestep.

        Args:
            step: Timestep index
            pressure: Pressure field array with shape matching solver grid
        """
        if self._pressure_dataset is None:
            raise RuntimeError("Call create_pressure_dataset() before writing timesteps")

        self._pressure_dataset[step] = pressure.astype(np.float32)
        self._current_step = step + 1

    def write_sources(self) -> None:
        """Write source definitions to file."""
        if not self.solver._sources:
            return

        sources_group = self.file.create_group('sources')

        for i, source in enumerate(self.solver._sources):
            source_group = sources_group.create_group(f'source_{i}')
            source_group.attrs['type'] = type(source).__name__
            source_group.attrs['position'] = source.position

            # Write source-specific parameters
            if hasattr(source, 'frequency'):
                source_group.attrs['frequency'] = source.frequency
            if hasattr(source, 'amplitude'):
                source_group.attrs['amplitude'] = source.amplitude
            if hasattr(source, 'phase'):
                source_group.attrs['phase'] = source.phase

    def write_probes(self) -> None:
        """Write probe time series and metadata."""
        if not self.solver._probes:
            return

        probes_group = self.file.create_group('probes')

        for name, probe in self.solver._probes.items():
            data = probe.get_data()

            dataset = probes_group.create_dataset(
                name,
                data=data,
                compression='gzip',
                compression_opts=4,
            )

            dataset.attrs['position'] = probe.position
            dataset.attrs['units'] = 'Pa'
            dataset.attrs['sample_rate'] = 1.0 / self.solver.dt

    def write_microphones(self) -> None:
        """Write microphone time series and metadata."""
        if not self.solver._microphones:
            return

        mics_group = self.file.create_group('microphones')

        for name, mic in self.solver._microphones.items():
            # Microphones use get_waveform() instead of get_data()
            data = mic.get_waveform()

            dataset = mics_group.create_dataset(
                name,
                data=data,
                compression='gzip',
                compression_opts=4,
            )

            dataset.attrs['position'] = mic.position
            dataset.attrs['units'] = 'Pa'
            dataset.attrs['sample_rate'] = 1.0 / self.solver.dt
            dataset.attrs['pattern'] = mic._pattern_name

            if mic.is_directional():
                dataset.attrs['direction'] = mic.direction

    def finalize(self) -> None:
        """Write final metadata and close file.

        Call this after simulation completes to write probe data and
        update runtime statistics.
        """
        # Write sources and probes
        self.write_sources()
        self.write_probes()
        self.write_microphones()

        # Update simulation metadata with final values
        sim_group = self.file['simulation']
        # Use solver's step count (accurate even when snapshots not saved)
        sim_group.attrs['num_steps'] = self.solver.step_count
        sim_group.attrs['total_time'] = self.solver._time

        # Write total runtime
        runtime = time.time() - self._start_time
        self.file.attrs['total_runtime_seconds'] = runtime

        # Close file
        self.file.close()


class HDF5ResultReader:
    """Reader for analyzing FDTD simulation results.

    Provides efficient access to stored simulation data with methods for
    loading specific timesteps, probes, and metadata.

    Args:
        filename: Path to HDF5 results file

    Example:
        >>> reader = HDF5ResultReader("results.h5")
        >>> metadata = reader.get_metadata()
        >>> print(f"Simulation ran for {metadata['simulation']['total_time']:.6f} seconds")
        >>>
        >>> # Load specific timestep
        >>> pressure = reader.load_timestep(100)
        >>>
        >>> # Load probe data
        >>> probe_data = reader.load_probe('center')
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.file = h5py.File(filename, 'r')

    def get_metadata(self) -> dict:
        """Extract all metadata from file.

        Returns:
            Dictionary containing metadata, grid, simulation parameters, and materials
        """
        metadata = {}

        # Root attributes
        metadata['script_hash'] = self.file.attrs['script_hash']
        metadata['created_at'] = self.file.attrs['created_at']
        metadata['solver_version'] = self.file.attrs['solver_version']
        metadata['backend'] = self.file.attrs['backend']
        metadata['num_threads'] = self.file.attrs['num_threads']
        metadata['total_runtime_seconds'] = self.file.attrs.get('total_runtime_seconds', 0)

        # Grid info
        grid = {}
        grid_group = self.file['grid']
        grid['shape'] = tuple(grid_group.attrs['shape'])
        grid['is_uniform'] = grid_group.attrs['is_uniform']
        grid['extent'] = tuple(grid_group.attrs['extent'])

        if grid['is_uniform']:
            grid['resolution'] = grid_group.attrs['resolution']
        else:
            grid['x_coords'] = grid_group['x_coords'][:]
            grid['y_coords'] = grid_group['y_coords'][:]
            grid['z_coords'] = grid_group['z_coords'][:]

        metadata['grid'] = grid

        # Simulation params
        sim = {}
        sim_group = self.file['simulation']
        sim['timestep'] = sim_group.attrs['timestep']
        sim['cfl_number'] = sim_group.attrs['cfl_number']
        sim['pml_thickness'] = sim_group.attrs['pml_thickness']
        sim['pml_type'] = sim_group.attrs.get('pml_type', 'none')  # Default to 'none' for old files
        sim['num_steps'] = sim_group.attrs.get('num_steps', 0)
        sim['total_time'] = sim_group.attrs.get('total_time', 0)

        metadata['simulation'] = sim

        # Materials info (if available)
        if 'materials' in self.file:
            materials = {}
            materials['material_map'] = self.get_material_map()
            metadata['materials'] = materials

        return metadata

    def load_timestep(self, step: int) -> NDArray[np.floating]:
        """Load pressure field for a single timestep.

        Args:
            step: Timestep index

        Returns:
            Pressure field array with shape (nx, ny, nz)
        """
        return self.file['fields/pressure'][step]

    def load_timestep_range(self, start: int, end: int) -> NDArray[np.floating]:
        """Load pressure fields for a range of timesteps.

        Args:
            start: Starting timestep index (inclusive)
            end: Ending timestep index (exclusive)

        Returns:
            Pressure field array with shape (end-start, nx, ny, nz)
        """
        return self.file['fields/pressure'][start:end]

    def load_probe(self, probe_name: str) -> NDArray[np.floating]:
        """Load probe time series.

        Args:
            probe_name: Name of probe

        Returns:
            Pressure time series array
        """
        return self.file[f'probes/{probe_name}'][:]

    def load_microphone(self, mic_name: str) -> NDArray[np.floating]:
        """Load microphone time series.

        Args:
            mic_name: Name of microphone

        Returns:
            Pressure time series array
        """
        return self.file[f'microphones/{mic_name}'][:]

    def get_probe_metadata(self, probe_name: str) -> dict:
        """Get metadata for a specific probe.

        Args:
            probe_name: Name of probe

        Returns:
            Dictionary with position, units, sample_rate
        """
        probe = self.file[f'probes/{probe_name}']
        return {
            'position': tuple(probe.attrs['position']),
            'units': probe.attrs['units'],
            'sample_rate': probe.attrs['sample_rate'],
        }

    def get_microphone_metadata(self, mic_name: str) -> dict:
        """Get metadata for a specific microphone.

        Args:
            mic_name: Name of microphone

        Returns:
            Dictionary with position, units, sample_rate, pattern, direction
        """
        mic = self.file[f'microphones/{mic_name}']
        metadata = {
            'position': tuple(mic.attrs['position']),
            'units': mic.attrs['units'],
            'sample_rate': mic.attrs['sample_rate'],
            'pattern': mic.attrs['pattern'],
        }

        if 'direction' in mic.attrs:
            metadata['direction'] = tuple(mic.attrs['direction'])

        return metadata

    def list_probes(self) -> list[str]:
        """List all probe names in file.

        Returns:
            List of probe names
        """
        if 'probes' not in self.file:
            return []
        return list(self.file['probes'].keys())

    def list_microphones(self) -> list[str]:
        """List all microphone names in file.

        Returns:
            List of microphone names
        """
        if 'microphones' not in self.file:
            return []
        return list(self.file['microphones'].keys())

    def get_script_content(self) -> str:
        """Get embedded source script content.

        Returns:
            Source script as string
        """
        return self.file.attrs['script_content']

    def get_material_map(self) -> dict[str, str]:
        """Get material map from geometry dataset.

        Returns:
            Dictionary mapping material IDs (as strings) to material names.
            Returns empty dict if materials group doesn't exist.

        Example:
            >>> reader = HDF5ResultReader("results.h5")
            >>> material_map = reader.get_material_map()
            >>> print(material_map)
            {'0': 'air', '1': 'pzt5'}
        """
        import json

        if 'materials' not in self.file:
            return {}

        if 'geometry' not in self.file['materials']:
            return {}

        geom_ds = self.file['materials/geometry']
        if 'material_map' in geom_ds.attrs:
            return json.loads(geom_ds.attrs['material_map'])
        else:
            return {}

    def close(self) -> None:
        """Close HDF5 file."""
        self.file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

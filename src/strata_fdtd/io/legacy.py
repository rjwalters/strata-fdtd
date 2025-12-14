"""
Input/Output utilities for FDTD simulation data.

This module provides export functions for simulation results in formats
suitable for visualization (Three.js) and analysis.

Supported formats:
    - Binary: Compact float16 arrays for field snapshots
    - JSON: Metadata, probe data, geometry descriptions
    - NumPy: Native .npy/.npz for Python interoperability
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .fdtd import FDTDSolver


def export_snapshot(
    pressure: NDArray[np.floating],
    path: str | Path,
    format: str = "float16",
    downsample: int = 1,
) -> dict[str, Any]:
    """Export pressure field snapshot to binary file.

    Args:
        pressure: 3D pressure field array
        path: Output file path
        format: Data format ('float16', 'float32', 'uint8')
        downsample: Downsampling factor (1 = no downsampling)

    Returns:
        Metadata dict with shape, dtype, and file info
    """
    path = Path(path)

    # Downsample if requested
    if downsample > 1:
        pressure = pressure[::downsample, ::downsample, ::downsample]

    # Convert to requested format
    if format == "float16":
        data = pressure.astype(np.float16)
    elif format == "float32":
        data = pressure.astype(np.float32)
    elif format == "uint8":
        # Normalize to 0-255 range
        p_min, p_max = pressure.min(), pressure.max()
        if p_max - p_min > 0:
            normalized = (pressure - p_min) / (p_max - p_min)
        else:
            normalized = np.zeros_like(pressure)
        data = (normalized * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Write binary data
    data.tofile(path)

    # Return metadata
    metadata = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "format": format,
        "downsample": downsample,
        "file": str(path),
        "bytes": data.nbytes,
    }

    if format == "uint8":
        metadata["value_range"] = [float(p_min), float(p_max)]

    return metadata


def export_velocity_snapshot(
    vx: NDArray[np.floating],
    vy: NDArray[np.floating],
    vz: NDArray[np.floating],
    path_prefix: str | Path,
    format: str = "float16",
    downsample: int = 1,
    interleaved: bool = True,
) -> dict[str, Any]:
    """Export velocity field snapshot to binary file(s).

    Args:
        vx, vy, vz: 3D velocity component arrays (cell-centered)
        path_prefix: Output file path prefix (without extension)
        format: Data format ('float16' or 'float32')
        downsample: Downsampling factor (1 = no downsampling)
        interleaved: If True, pack as [vx,vy,vz,vx,vy,vz,...] in single file.
            If False, write separate files for each component.

    Returns:
        Metadata dict with shape, dtype, and file info
    """
    path_prefix = Path(path_prefix)

    # Downsample if requested
    if downsample > 1:
        vx = vx[::downsample, ::downsample, ::downsample]
        vy = vy[::downsample, ::downsample, ::downsample]
        vz = vz[::downsample, ::downsample, ::downsample]

    # Convert to requested format
    np_dtype = np.float16 if format == "float16" else np.float32
    vx = vx.astype(np_dtype)
    vy = vy.astype(np_dtype)
    vz = vz.astype(np_dtype)

    if interleaved:
        # Pack as interleaved [vx, vy, vz, vx, vy, vz, ...]
        # Shape: (nx*ny*nz, 3) -> flatten to (nx*ny*nz*3,)
        interleaved_data = np.stack([vx.ravel(), vy.ravel(), vz.ravel()], axis=1)
        interleaved_data = interleaved_data.ravel()

        file_path = Path(str(path_prefix) + ".bin")
        interleaved_data.tofile(file_path)

        return {
            "shape": list(vx.shape),
            "dtype": str(np_dtype),
            "format": "interleaved",
            "downsample": downsample,
            "components": 3,
            "file": str(file_path),
            "bytes": interleaved_data.nbytes,
        }
    else:
        # Write separate files for each component
        files = {}
        total_bytes = 0

        for name, data in [("vx", vx), ("vy", vy), ("vz", vz)]:
            file_path = Path(str(path_prefix) + f"_{name}.bin")
            data.tofile(file_path)
            files[name] = str(file_path)
            total_bytes += data.nbytes

        return {
            "shape": list(vx.shape),
            "dtype": str(np_dtype),
            "format": "separate",
            "downsample": downsample,
            "components": 3,
            "files": files,
            "bytes": total_bytes,
        }


def export_geometry(
    geometry: NDArray[np.bool_],
    path: str | Path,
    format: str = "binary",
) -> dict[str, Any]:
    """Export geometry mask for visualization.

    Args:
        geometry: 3D boolean mask (True=air, False=solid)
        path: Output file path
        format: 'binary' (packed bits) or 'json' (run-length encoded)

    Returns:
        Metadata dict with shape and encoding info
    """
    path = Path(path)

    if format == "binary":
        # Pack boolean array as bits
        packed = np.packbits(geometry.flatten())
        packed.tofile(path)

        return {
            "shape": list(geometry.shape),
            "format": "packed_bits",
            "file": str(path),
            "bytes": packed.nbytes,
        }

    elif format == "json":
        # Run-length encode for potentially smaller size
        flat = geometry.flatten()
        runs = []
        current_val = flat[0]
        count = 1

        for val in flat[1:]:
            if val == current_val:
                count += 1
            else:
                runs.append({"v": bool(current_val), "n": count})
                current_val = val
                count = 1
        runs.append({"v": bool(current_val), "n": count})

        data = {
            "shape": list(geometry.shape),
            "format": "rle",
            "runs": runs,
        }

        with open(path, "w") as f:
            json.dump(data, f)

        return {
            "shape": list(geometry.shape),
            "format": "rle_json",
            "file": str(path),
            "runs": len(runs),
        }

    else:
        raise ValueError(f"Unknown format: {format}")


def export_probe_data(
    solver: "FDTDSolver",
    path: str | Path,
    probe_names: list[str] | None = None,
) -> dict[str, Any]:
    """Export probe time series data to JSON.

    Args:
        solver: FDTD solver with recorded probe data
        path: Output file path
        probe_names: Specific probes to export (None = all)

    Returns:
        Metadata dict with probe info
    """
    path = Path(path)

    probe_data = solver.get_probe_data()
    if probe_names is not None:
        probe_data = {k: v for k, v in probe_data.items() if k in probe_names}

    # Convert to JSON-serializable format
    output = {
        "sample_rate": solver.get_sample_rate(),
        "dt": solver.dt,
        "duration": solver.time,
        "n_samples": solver.step_count,
        "probes": {},
    }

    for name, data in probe_data.items():
        probe = solver._probes[name]
        output["probes"][name] = {
            "position": list(probe.position),
            "data": data.tolist(),
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    return {
        "file": str(path),
        "n_probes": len(output["probes"]),
        "n_samples": output["n_samples"],
        "sample_rate": output["sample_rate"],
    }


def export_metadata(
    solver: "FDTDSolver",
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Export solver metadata to JSON.

    Args:
        solver: FDTD solver instance
        path: Output file path
        extra: Additional metadata to include
    """
    path = Path(path)

    metadata = {
        "grid": {
            "shape": list(solver.shape),
            "resolution": solver.dx,
            "physical_size": [s * solver.dx for s in solver.shape],
        },
        "physics": {
            "c": solver.c,
            "rho": solver.rho,
        },
        "simulation": {
            "dt": solver.dt,
            "cfl_limit": solver.dx / (solver.c * np.sqrt(3)),
            "current_time": solver.time,
            "step_count": solver.step_count,
        },
        "probes": {
            name: {"position": list(probe.position)}
            for name, probe in solver._probes.items()
        },
        "sources": [
            {
                "type": source.source_type,
                "position": (
                    list(source.position)
                    if isinstance(source.position, tuple)
                    else source.position
                ),
                "frequency": source.frequency,
                "bandwidth": source.bandwidth,
            }
            for source in solver._sources
        ],
    }

    if extra:
        metadata["extra"] = extra

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def export_simulation(
    solver: "FDTDSolver",
    output_dir: str | Path,
    prefix: str = "sim",
    include_snapshots: bool = True,
    include_velocity: bool | None = None,
    snapshot_format: str = "float16",
    snapshot_downsample: int = 1,
) -> dict[str, Any]:
    """Export complete simulation results.

    Creates a directory with:
        - {prefix}_metadata.json: Solver configuration
        - {prefix}_probes.json: All probe time series
        - {prefix}_geometry.bin: Geometry mask
        - {prefix}_snapshot_{time}.bin: Field snapshots (if enabled)
        - {prefix}_velocity_{time}.bin: Velocity snapshots (if enabled)

    Args:
        solver: FDTD solver with completed simulation
        output_dir: Output directory path
        prefix: Filename prefix
        include_snapshots: Whether to export field snapshots
        include_velocity: Whether to export velocity snapshots.
            If None (default), exports velocity only if solver captured them.
        snapshot_format: Format for snapshots ('float16', 'float32')
        snapshot_downsample: Downsampling factor for snapshots

    Returns:
        Dict with paths to all exported files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # Export metadata
    metadata_path = output_dir / f"{prefix}_metadata.json"
    export_metadata(solver, metadata_path)
    files["metadata"] = str(metadata_path)

    # Export probe data
    probe_path = output_dir / f"{prefix}_probes.json"
    export_probe_data(solver, probe_path)
    files["probes"] = str(probe_path)

    # Export geometry
    geometry_path = output_dir / f"{prefix}_geometry.bin"
    export_geometry(solver.geometry, geometry_path)
    files["geometry"] = str(geometry_path)

    # Export pressure snapshots
    if include_snapshots:
        snapshots = solver.get_snapshots()
        files["snapshots"] = []
        for i, (time, pressure) in enumerate(snapshots):
            snap_path = output_dir / f"{prefix}_snapshot_{i:04d}.bin"
            snap_info = export_snapshot(
                pressure,
                snap_path,
                format=snapshot_format,
                downsample=snapshot_downsample,
            )
            snap_info["time"] = time
            files["snapshots"].append(snap_info)

    # Export velocity snapshots
    velocity_snapshots = solver.get_velocity_snapshots()
    should_export_velocity = (
        include_velocity if include_velocity is not None else len(velocity_snapshots) > 0
    )

    if should_export_velocity and velocity_snapshots:
        files["velocitySnapshots"] = []
        for i, (time, vx, vy, vz) in enumerate(velocity_snapshots):
            vel_path_prefix = output_dir / f"{prefix}_velocity_{i:04d}"
            vel_info = export_velocity_snapshot(
                vx,
                vy,
                vz,
                vel_path_prefix,
                format=snapshot_format,
                downsample=snapshot_downsample,
                interleaved=True,
            )
            vel_info["time"] = time
            files["velocitySnapshots"].append(vel_info)

    # Write manifest
    manifest_path = output_dir / f"{prefix}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(files, f, indent=2)

    files["manifest"] = str(manifest_path)
    return files


def load_probe_data(path: str | Path) -> dict[str, Any]:
    """Load probe data from exported JSON.

    Args:
        path: Path to probe data JSON file

    Returns:
        Dict with probe data and metadata
    """
    path = Path(path)

    with open(path) as f:
        data = json.load(f)

    # Convert lists back to numpy arrays
    for probe_name in data["probes"]:
        data["probes"][probe_name]["data"] = np.array(
            data["probes"][probe_name]["data"], dtype=np.float32
        )

    return data


def load_snapshot(
    path: str | Path,
    shape: tuple[int, int, int],
    dtype: str = "float16",
) -> NDArray[np.floating]:
    """Load pressure field snapshot from binary file.

    Args:
        path: Path to binary snapshot file
        shape: Expected array shape (nx, ny, nz)
        dtype: Data type ('float16', 'float32', 'uint8')

    Returns:
        3D pressure field array
    """
    path = Path(path)

    np_dtype = np.dtype(dtype)
    data = np.fromfile(path, dtype=np_dtype)

    return data.reshape(shape)


def load_geometry(
    path: str | Path,
    shape: tuple[int, int, int],
    format: str = "binary",
) -> NDArray[np.bool_]:
    """Load geometry mask from exported file.

    Args:
        path: Path to geometry file
        shape: Expected array shape (nx, ny, nz)
        format: File format ('binary' or 'json')

    Returns:
        3D boolean geometry mask
    """
    path = Path(path)

    if format == "binary":
        packed = np.fromfile(path, dtype=np.uint8)
        flat = np.unpackbits(packed)
        # Trim to exact size (unpackbits pads to multiple of 8)
        n_elements = shape[0] * shape[1] * shape[2]
        return flat[:n_elements].reshape(shape).astype(bool)

    elif format == "json":
        with open(path) as f:
            data = json.load(f)

        # Decode run-length encoding
        flat = []
        for run in data["runs"]:
            flat.extend([run["v"]] * run["n"])

        return np.array(flat, dtype=bool).reshape(shape)

    else:
        raise ValueError(f"Unknown format: {format}")

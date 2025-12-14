"""I/O and data management for FDTD results."""

# HDF5 I/O
from strata_fdtd.io.hdf5 import (
    HDF5ResultWriter,
    HDF5ResultReader,
)

# Output management
from strata_fdtd.io.output import (
    HDF5ResultWriter as OutputHDF5Writer,
    HDF5ResultReader as OutputHDF5Reader,
)

__all__ = [
    "HDF5ResultWriter",
    "HDF5ResultReader",
]

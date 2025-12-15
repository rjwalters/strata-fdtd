"""I/O and data management for FDTD results."""

# HDF5 I/O
from strata_fdtd.io.hdf5 import (
    HDF5ResultReader,
    HDF5ResultWriter,
)
from strata_fdtd.io.output import (
    HDF5ResultReader as OutputHDF5Reader,
)

# Output management
from strata_fdtd.io.output import (
    HDF5ResultWriter as OutputHDF5Writer,
)

__all__ = [
    "HDF5ResultWriter",
    "HDF5ResultReader",
    "OutputHDF5Writer",
    "OutputHDF5Reader",
]

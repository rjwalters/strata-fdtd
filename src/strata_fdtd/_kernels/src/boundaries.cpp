/**
 * @file boundaries.cpp
 * @brief Rigid boundary condition implementations
 *
 * Stub implementation for infrastructure setup.
 * Full optimized implementation in issue #47.
 */

#include "boundaries.hpp"

namespace fdtd {

BoundaryCells precompute_boundary_cells(
    const bool* __restrict__ geometry,
    const GridShape& shape
) {
    BoundaryCells bc;
    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;

    // Pre-compute vx boundary cells
    // vx[i,j,k] is between cells (i,j,k) and (i+1,j,k)
    // Zero if either cell is solid
    for (int i = 0; i < nx - 1; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                int curr = idx(i, j, k, ny, nz);
                int next = idx(i + 1, j, k, ny, nz);
                if (!geometry[curr] || !geometry[next]) {
                    bc.vx_indices.push_back(curr);
                }
            }
        }
    }

    // Pre-compute vy boundary cells
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny - 1; j++) {
            for (int k = 0; k < nz; k++) {
                int curr = idx(i, j, k, ny, nz);
                int next = idx(i, j + 1, k, ny, nz);
                if (!geometry[curr] || !geometry[next]) {
                    bc.vy_indices.push_back(curr);
                }
            }
        }
    }

    // Pre-compute vz boundary cells
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz - 1; k++) {
                int curr = idx(i, j, k, ny, nz);
                int next = idx(i, j, k + 1, ny, nz);
                if (!geometry[curr] || !geometry[next]) {
                    bc.vz_indices.push_back(curr);
                }
            }
        }
    }

    return bc;
}

void apply_rigid_boundaries(
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const BoundaryCells& bc
) {
    // Zero vx at boundary cells
    FDTD_PARALLEL_FOR
    for (size_t i = 0; i < bc.vx_indices.size(); i++) {
        vx[bc.vx_indices[i]] = 0.0f;
    }

    // Zero vy at boundary cells
    FDTD_PARALLEL_FOR
    for (size_t i = 0; i < bc.vy_indices.size(); i++) {
        vy[bc.vy_indices[i]] = 0.0f;
    }

    // Zero vz at boundary cells
    FDTD_PARALLEL_FOR
    for (size_t i = 0; i < bc.vz_indices.size(); i++) {
        vz[bc.vz_indices[i]] = 0.0f;
    }
}

void apply_rigid_boundaries_direct(
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const bool* __restrict__ geometry,
    const GridShape& shape
) {
    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;

    // vx boundaries
    FDTD_PARALLEL_FOR
    for (int i = 0; i < nx - 1; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                int curr = idx(i, j, k, ny, nz);
                int next = idx(i + 1, j, k, ny, nz);
                if (!geometry[curr] || !geometry[next]) {
                    vx[curr] = 0.0f;
                }
            }
        }
    }

    // vy boundaries
    FDTD_PARALLEL_FOR
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny - 1; j++) {
            for (int k = 0; k < nz; k++) {
                int curr = idx(i, j, k, ny, nz);
                int next = idx(i, j + 1, k, ny, nz);
                if (!geometry[curr] || !geometry[next]) {
                    vy[curr] = 0.0f;
                }
            }
        }
    }

    // vz boundaries
    FDTD_PARALLEL_FOR
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz - 1; k++) {
                int curr = idx(i, j, k, ny, nz);
                int next = idx(i, j, k + 1, ny, nz);
                if (!geometry[curr] || !geometry[next]) {
                    vz[curr] = 0.0f;
                }
            }
        }
    }
}

}  // namespace fdtd

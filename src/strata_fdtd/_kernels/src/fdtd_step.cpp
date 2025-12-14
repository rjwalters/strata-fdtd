/**
 * @file fdtd_step.cpp
 * @brief Core FDTD time-stepping kernel implementations
 *
 * Optimized implementation with OpenMP parallelization and SIMD vectorization.
 * Uses collapse(2) for better thread utilization on multi-core systems.
 *
 * Issue #46: Fused FDTD step kernel implementation
 */

#include "fdtd_step.hpp"

namespace fdtd {

void update_velocity(
    const float* __restrict__ p,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const GridShape& shape,
    float coeff_v
) {
    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;

    // Update vx: vx[i,j,k] += coeff_v * (p[i+1,j,k] - p[i,j,k])
    // vx[i] is at face between cells i and i+1
    // Use collapse(2) to get enough parallel work for small grids
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx - 1; i++) {
        for (int j = 0; j < ny; j++) {
            // Pre-compute base indices for this row
            const int base_curr = i * ny_nz + j * nz;
            const int base_next = (i + 1) * ny_nz + j * nz;
            // Inner loop vectorizes with SIMD
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 0; k < nz; k++) {
                vx[base_curr + k] += coeff_v * (p[base_next + k] - p[base_curr + k]);
            }
        }
    }

    // Update vy: vy[i,j,k] += coeff_v * (p[i,j+1,k] - p[i,j,k])
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny - 1; j++) {
            const int base_curr = i * ny_nz + j * nz;
            const int base_next = i * ny_nz + (j + 1) * nz;
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 0; k < nz; k++) {
                vy[base_curr + k] += coeff_v * (p[base_next + k] - p[base_curr + k]);
            }
        }
    }

    // Update vz: vz[i,j,k] += coeff_v * (p[i,j,k+1] - p[i,j,k])
    // Note: k loop has nz-1 iterations, still vectorizable
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            const int base = i * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 0; k < nz - 1; k++) {
                vz[base + k] += coeff_v * (p[base + k + 1] - p[base + k]);
            }
        }
    }
}

void update_pressure(
    float* __restrict__ p,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    const bool* __restrict__ geometry,
    const GridShape& shape,
    float coeff_p
) {
    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;

    // Compute divergence and update pressure in a single fused pass
    // div = (vx[i] - vx[i-1]) + (vy[j] - vy[j-1]) + (vz[k] - vz[k-1])
    //
    // Split into boundary cases and interior for better vectorization:
    // - i=0, j=0, k=0 planes have special handling
    // - Interior can be fully vectorized

    // Interior cells (i>0, j>0, k>0) - fully vectorizable
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            const int base = i * ny_nz + j * nz;
            const int base_im1 = (i - 1) * ny_nz + j * nz;
            const int base_jm1 = i * ny_nz + (j - 1) * nz;
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 1; k < nz; k++) {
                const int curr = base + k;
                if (!geometry[curr]) {
                    p[curr] = 0.0f;
                    continue;
                }
                // X divergence: vx[i] - vx[i-1]
                const float div_x = vx[curr] - vx[base_im1 + k];
                // Y divergence: vy[j] - vy[j-1]
                const float div_y = vy[curr] - vy[base_jm1 + k];
                // Z divergence: vz[k] - vz[k-1]
                const float div_z = vz[curr] - vz[curr - 1];

                p[curr] += coeff_p * (div_x + div_y + div_z);
            }
        }
    }

    // Handle boundary planes (i=0, j=0, k=0)
    // These have fewer iterations, so overhead is minimal

    // k=0 plane for interior i,j
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            const int curr = i * ny_nz + j * nz;  // k=0
            if (!geometry[curr]) {
                p[curr] = 0.0f;
                continue;
            }
            const float div_x = vx[curr] - vx[(i - 1) * ny_nz + j * nz];
            const float div_y = vy[curr] - vy[i * ny_nz + (j - 1) * nz];
            const float div_z = vz[curr];  // vz[-1] = 0 at boundary
            p[curr] += coeff_p * (div_x + div_y + div_z);
        }
    }

    // j=0 plane for interior i, all k
#if FDTD_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 1; i < nx; i++) {
        const int base = i * ny_nz;  // j=0
        const int base_im1 = (i - 1) * ny_nz;
        // k=0 case
        {
            const int curr = base;
            if (geometry[curr]) {
                const float div_x = vx[curr] - vx[base_im1];
                const float div_y = vy[curr];  // vy[-1] = 0
                const float div_z = vz[curr];  // vz[-1] = 0
                p[curr] += coeff_p * (div_x + div_y + div_z);
            } else {
                p[curr] = 0.0f;
            }
        }
        // k>0 cases
#if FDTD_HAS_OPENMP
        #pragma omp simd
#endif
        for (int k = 1; k < nz; k++) {
            const int curr = base + k;
            if (!geometry[curr]) {
                p[curr] = 0.0f;
                continue;
            }
            const float div_x = vx[curr] - vx[base_im1 + k];
            const float div_y = vy[curr];  // vy[-1] = 0
            const float div_z = vz[curr] - vz[curr - 1];
            p[curr] += coeff_p * (div_x + div_y + div_z);
        }
    }

    // i=0 plane for all j, all k
#if FDTD_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < ny; j++) {
        const int base = j * nz;  // i=0
        const int base_jm1 = (j > 0) ? (j - 1) * nz : -1;

        for (int k = 0; k < nz; k++) {
            const int curr = base + k;
            if (!geometry[curr]) {
                p[curr] = 0.0f;
                continue;
            }
            const float div_x = vx[curr];  // vx[-1] = 0 at boundary
            const float div_y = (j > 0) ? (vy[curr] - vy[base_jm1 + k]) : vy[curr];
            const float div_z = (k > 0) ? (vz[curr] - vz[curr - 1]) : vz[curr];
            p[curr] += coeff_p * (div_x + div_y + div_z);
        }
    }
}

void fdtd_step(
    float* __restrict__ p,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const bool* __restrict__ geometry,
    const GridShape& shape,
    const FDTDCoeffs& coeffs
) {
    // Phase 1: Update velocities from pressure gradient
    update_velocity(p, vx, vy, vz, shape, coeffs.coeff_v);

    // Phase 2: Update pressure from velocity divergence
    // (Rigid boundaries handled implicitly via geometry mask)
    update_pressure(p, vx, vy, vz, geometry, shape, coeffs.coeff_p);
}

// =============================================================================
// Nonuniform grid kernels (Issue #131)
// =============================================================================

void update_velocity_nonuniform(
    const float* __restrict__ p,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const GridShape& shape,
    const NonuniformGridData& grid_data,
    float coeff_v_base
) {
    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;

    // Pointers to spacing arrays
    const float* inv_dx_face = grid_data.inv_dx_face.data();
    const float* inv_dy_face = grid_data.inv_dy_face.data();
    const float* inv_dz_face = grid_data.inv_dz_face.data();

    // Update vx: vx[i,j,k] += coeff_v_base * inv_dx_face[i] * (p[i+1,j,k] - p[i,j,k])
    // vx[i] is at face between cells i and i+1
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx - 1; i++) {
        for (int j = 0; j < ny; j++) {
            const int base_curr = i * ny_nz + j * nz;
            const int base_next = (i + 1) * ny_nz + j * nz;
            const float coeff = coeff_v_base * inv_dx_face[i];
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 0; k < nz; k++) {
                vx[base_curr + k] += coeff * (p[base_next + k] - p[base_curr + k]);
            }
        }
    }

    // Update vy: vy[i,j,k] += coeff_v_base * inv_dy_face[j] * (p[i,j+1,k] - p[i,j,k])
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny - 1; j++) {
            const int base_curr = i * ny_nz + j * nz;
            const int base_next = i * ny_nz + (j + 1) * nz;
            const float coeff = coeff_v_base * inv_dy_face[j];
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 0; k < nz; k++) {
                vy[base_curr + k] += coeff * (p[base_next + k] - p[base_curr + k]);
            }
        }
    }

    // Update vz: vz[i,j,k] += coeff_v_base * inv_dz_face[k] * (p[i,j,k+1] - p[i,j,k])
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            const int base = i * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 0; k < nz - 1; k++) {
                const float coeff = coeff_v_base * inv_dz_face[k];
                vz[base + k] += coeff * (p[base + k + 1] - p[base + k]);
            }
        }
    }
}

void update_pressure_nonuniform(
    float* __restrict__ p,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    const bool* __restrict__ geometry,
    const GridShape& shape,
    const NonuniformGridData& grid_data,
    float coeff_p_base
) {
    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;

    // Pointers to spacing arrays
    const float* inv_dx_cell = grid_data.inv_dx_cell.data();
    const float* inv_dy_cell = grid_data.inv_dy_cell.data();
    const float* inv_dz_cell = grid_data.inv_dz_cell.data();

    // Interior cells (i>0, j>0, k>0)
    // div = inv_dx_cell[i] * (vx[i] - vx[i-1]) + inv_dy_cell[j] * (vy[j] - vy[j-1])
    //     + inv_dz_cell[k] * (vz[k] - vz[k-1])
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            const int base = i * ny_nz + j * nz;
            const int base_im1 = (i - 1) * ny_nz + j * nz;
            const int base_jm1 = i * ny_nz + (j - 1) * nz;
            const float inv_dx = inv_dx_cell[i];
            const float inv_dy = inv_dy_cell[j];
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 1; k < nz; k++) {
                const int curr = base + k;
                if (!geometry[curr]) {
                    p[curr] = 0.0f;
                    continue;
                }
                const float div_x = inv_dx * (vx[curr] - vx[base_im1 + k]);
                const float div_y = inv_dy * (vy[curr] - vy[base_jm1 + k]);
                const float div_z = inv_dz_cell[k] * (vz[curr] - vz[curr - 1]);
                p[curr] += coeff_p_base * (div_x + div_y + div_z);
            }
        }
    }

    // Handle boundary planes (i=0, j=0, k=0)

    // k=0 plane for interior i,j
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            const int curr = i * ny_nz + j * nz;  // k=0
            if (!geometry[curr]) {
                p[curr] = 0.0f;
                continue;
            }
            const float div_x = inv_dx_cell[i] * (vx[curr] - vx[(i - 1) * ny_nz + j * nz]);
            const float div_y = inv_dy_cell[j] * (vy[curr] - vy[i * ny_nz + (j - 1) * nz]);
            const float div_z = inv_dz_cell[0] * vz[curr];  // vz[-1] = 0 at boundary
            p[curr] += coeff_p_base * (div_x + div_y + div_z);
        }
    }

    // j=0 plane for interior i, all k
#if FDTD_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 1; i < nx; i++) {
        const int base = i * ny_nz;  // j=0
        const int base_im1 = (i - 1) * ny_nz;
        // k=0 case
        {
            const int curr = base;
            if (geometry[curr]) {
                const float div_x = inv_dx_cell[i] * (vx[curr] - vx[base_im1]);
                const float div_y = inv_dy_cell[0] * vy[curr];  // vy[-1] = 0
                const float div_z = inv_dz_cell[0] * vz[curr];  // vz[-1] = 0
                p[curr] += coeff_p_base * (div_x + div_y + div_z);
            } else {
                p[curr] = 0.0f;
            }
        }
        // k>0 cases
#if FDTD_HAS_OPENMP
        #pragma omp simd
#endif
        for (int k = 1; k < nz; k++) {
            const int curr = base + k;
            if (!geometry[curr]) {
                p[curr] = 0.0f;
                continue;
            }
            const float div_x = inv_dx_cell[i] * (vx[curr] - vx[base_im1 + k]);
            const float div_y = inv_dy_cell[0] * vy[curr];  // vy[-1] = 0
            const float div_z = inv_dz_cell[k] * (vz[curr] - vz[curr - 1]);
            p[curr] += coeff_p_base * (div_x + div_y + div_z);
        }
    }

    // i=0 plane for all j, all k
#if FDTD_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < ny; j++) {
        const int base = j * nz;  // i=0
        const int base_jm1 = (j > 0) ? (j - 1) * nz : -1;

        for (int k = 0; k < nz; k++) {
            const int curr = base + k;
            if (!geometry[curr]) {
                p[curr] = 0.0f;
                continue;
            }
            const float div_x = inv_dx_cell[0] * vx[curr];  // vx[-1] = 0 at boundary
            const float div_y = (j > 0)
                ? inv_dy_cell[j] * (vy[curr] - vy[base_jm1 + k])
                : inv_dy_cell[0] * vy[curr];
            const float div_z = (k > 0)
                ? inv_dz_cell[k] * (vz[curr] - vz[curr - 1])
                : inv_dz_cell[0] * vz[curr];
            p[curr] += coeff_p_base * (div_x + div_y + div_z);
        }
    }
}

void fdtd_step_nonuniform(
    float* __restrict__ p,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const bool* __restrict__ geometry,
    const GridShape& shape,
    const NonuniformGridData& grid_data,
    const FDTDCoeffsBase& coeffs
) {
    // Phase 1: Update velocities from pressure gradient
    update_velocity_nonuniform(p, vx, vy, vz, shape, grid_data, coeffs.coeff_v_base);

    // Phase 2: Update pressure from velocity divergence
    update_pressure_nonuniform(p, vx, vy, vz, geometry, shape, grid_data, coeffs.coeff_p_base);
}

}  // namespace fdtd

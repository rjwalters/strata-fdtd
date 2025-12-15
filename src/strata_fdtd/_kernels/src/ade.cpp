/**
 * @file ade.cpp
 * @brief ADE (Auxiliary Differential Equation) kernel implementations
 *
 * Native C++ kernels for frequency-dependent acoustic materials using the
 * ADE formulation with Debye and Lorentz poles.
 *
 * Performance optimizations:
 * - OpenMP parallelization over grid cells
 * - SIMD vectorization for inner loops
 * - Efficient material ID lookup
 *
 * Issue #139: Native C++ kernels for ADE material updates
 */

#include "ade.hpp"
#include <cstring>  // for memcpy

namespace fdtd {

// =============================================================================
// Debye pole update kernels
// =============================================================================

void update_ade_density_debye(
    float* __restrict__ J,
    const float* __restrict__ pressure,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape
) {
    if (material_data.debye_poles.empty()) return;

    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;
    const int grid_size = nx * ny_nz;

    // Process each Debye density pole
    for (const auto& pole : material_data.debye_poles) {
        if (pole.target != 0) continue;  // Skip modulus poles

        const int mat_id = pole.material_id;
        const float alpha = pole.coeffs.alpha;
        const float beta = pole.coeffs.beta;
        float* J_pole = J + pole.field_index * grid_size;

#if FDTD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                const int base = i * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
                #pragma omp simd
#endif
                for (int k = 0; k < nz; k++) {
                    const int idx = base + k;
                    if (material_id[idx] == mat_id) {
                        // J^{n+1} = alpha * J^n + beta * pressure
                        J_pole[idx] = alpha * J_pole[idx] + beta * pressure[idx];
                    }
                }
            }
        }
    }
}

void update_ade_modulus_debye(
    float* __restrict__ J,
    const float* __restrict__ divergence,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape
) {
    if (material_data.debye_poles.empty()) return;

    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;
    const int grid_size = nx * ny_nz;

    // Process each Debye modulus pole
    for (const auto& pole : material_data.debye_poles) {
        if (pole.target != 1) continue;  // Skip density poles

        const int mat_id = pole.material_id;
        const float alpha = pole.coeffs.alpha;
        const float beta = pole.coeffs.beta;
        float* J_pole = J + pole.field_index * grid_size;

#if FDTD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                const int base = i * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
                #pragma omp simd
#endif
                for (int k = 0; k < nz; k++) {
                    const int idx = base + k;
                    if (material_id[idx] == mat_id) {
                        J_pole[idx] = alpha * J_pole[idx] + beta * divergence[idx];
                    }
                }
            }
        }
    }
}

// =============================================================================
// Lorentz pole update kernels
// =============================================================================

void update_ade_density_lorentz(
    float* __restrict__ J,
    float* __restrict__ J_prev,
    const float* __restrict__ pressure,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape
) {
    if (material_data.lorentz_poles.empty()) return;

    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;
    const int grid_size = nx * ny_nz;

    // Process each Lorentz density pole
    for (const auto& pole : material_data.lorentz_poles) {
        if (pole.target != 0) continue;  // Skip modulus poles

        const int mat_id = pole.material_id;
        const float a = pole.coeffs.a;
        const float b = pole.coeffs.b;
        const float d = pole.coeffs.d;
        float* J_pole = J + pole.field_index * grid_size;
        float* J_prev_pole = J_prev + pole.field_index * grid_size;

#if FDTD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                const int base = i * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
                #pragma omp simd
#endif
                for (int k = 0; k < nz; k++) {
                    const int idx = base + k;
                    if (material_id[idx] == mat_id) {
                        // J^{n+1} = a*J^n + b*J^{n-1} + d*source
                        const float J_old = J_pole[idx];
                        const float J_prev_old = J_prev_pole[idx];
                        const float J_new = a * J_old + b * J_prev_old + d * pressure[idx];

                        // Shift history
                        J_prev_pole[idx] = J_old;
                        J_pole[idx] = J_new;
                    }
                }
            }
        }
    }
}

void update_ade_modulus_lorentz(
    float* __restrict__ J,
    float* __restrict__ J_prev,
    const float* __restrict__ divergence,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape
) {
    if (material_data.lorentz_poles.empty()) return;

    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;
    const int grid_size = nx * ny_nz;

    // Process each Lorentz modulus pole
    for (const auto& pole : material_data.lorentz_poles) {
        if (pole.target != 1) continue;  // Skip density poles

        const int mat_id = pole.material_id;
        const float a = pole.coeffs.a;
        const float b = pole.coeffs.b;
        const float d = pole.coeffs.d;
        float* J_pole = J + pole.field_index * grid_size;
        float* J_prev_pole = J_prev + pole.field_index * grid_size;

#if FDTD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                const int base = i * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
                #pragma omp simd
#endif
                for (int k = 0; k < nz; k++) {
                    const int idx = base + k;
                    if (material_id[idx] == mat_id) {
                        const float J_old = J_pole[idx];
                        const float J_prev_old = J_prev_pole[idx];
                        const float J_new = a * J_old + b * J_prev_old + d * divergence[idx];

                        J_prev_pole[idx] = J_old;
                        J_pole[idx] = J_new;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Correction kernels
// =============================================================================

void apply_ade_velocity_correction(
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const float* __restrict__ J_debye,
    const float* __restrict__ J_lorentz,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape,
    float dt,
    float inv_dx
) {
    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;
    const int grid_size = nx * ny_nz;

    // Apply Debye density pole corrections using gradient of J
    for (const auto& pole : material_data.debye_poles) {
        if (pole.target != 0) continue;  // Only density poles

        const int mat_id = pole.material_id;
        const float rho_inf = material_data.rho_inf[mat_id];
        // Coefficient includes 1/dx for the gradient computation
        const float coeff = -dt / rho_inf * inv_dx;
        const float* J_pole = J_debye + pole.field_index * grid_size;

        // Apply gradient-based correction to vx (faces between cells i and i+1)
        // vx[i,j,k] += coeff * (J[i+1,j,k] - J[i,j,k])
        // Only apply where BOTH adjacent cells are in the material
#if FDTD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < nx - 1; i++) {
            for (int j = 0; j < ny; j++) {
                const int base = i * ny_nz + j * nz;
                const int base_ip1 = (i + 1) * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
                #pragma omp simd
#endif
                for (int k = 0; k < nz; k++) {
                    const int idx = base + k;
                    const int idx_ip1 = base_ip1 + k;
                    // Check both adjacent cells are in this material
                    if (material_id[idx] == mat_id && material_id[idx_ip1] == mat_id) {
                        const float dJ_dx = J_pole[idx_ip1] - J_pole[idx];
                        vx[idx] += coeff * dJ_dx;
                    }
                }
            }
        }

        // Apply gradient-based correction to vy (faces between cells j and j+1)
        // vy[i,j,k] += coeff * (J[i,j+1,k] - J[i,j,k])
#if FDTD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny - 1; j++) {
                const int base = i * ny_nz + j * nz;
                const int base_jp1 = i * ny_nz + (j + 1) * nz;
#if FDTD_HAS_OPENMP
                #pragma omp simd
#endif
                for (int k = 0; k < nz; k++) {
                    const int idx = base + k;
                    const int idx_jp1 = base_jp1 + k;
                    if (material_id[idx] == mat_id && material_id[idx_jp1] == mat_id) {
                        const float dJ_dy = J_pole[idx_jp1] - J_pole[idx];
                        vy[idx] += coeff * dJ_dy;
                    }
                }
            }
        }

        // Apply gradient-based correction to vz (faces between cells k and k+1)
        // vz[i,j,k] += coeff * (J[i,j,k+1] - J[i,j,k])
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
                    const int idx = base + k;
                    const int idx_kp1 = base + k + 1;
                    if (material_id[idx] == mat_id && material_id[idx_kp1] == mat_id) {
                        const float dJ_dz = J_pole[idx_kp1] - J_pole[idx];
                        vz[idx] += coeff * dJ_dz;
                    }
                }
            }
        }
    }

    // Apply Lorentz density pole corrections (same gradient-based structure)
    for (const auto& pole : material_data.lorentz_poles) {
        if (pole.target != 0) continue;  // Only density poles

        const int mat_id = pole.material_id;
        const float rho_inf = material_data.rho_inf[mat_id];
        const float coeff = -dt / rho_inf * inv_dx;
        const float* J_pole = J_lorentz + pole.field_index * grid_size;

        // vx: gradient in x direction
#if FDTD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < nx - 1; i++) {
            for (int j = 0; j < ny; j++) {
                const int base = i * ny_nz + j * nz;
                const int base_ip1 = (i + 1) * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
                #pragma omp simd
#endif
                for (int k = 0; k < nz; k++) {
                    const int idx = base + k;
                    const int idx_ip1 = base_ip1 + k;
                    if (material_id[idx] == mat_id && material_id[idx_ip1] == mat_id) {
                        const float dJ_dx = J_pole[idx_ip1] - J_pole[idx];
                        vx[idx] += coeff * dJ_dx;
                    }
                }
            }
        }

        // vy: gradient in y direction
#if FDTD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny - 1; j++) {
                const int base = i * ny_nz + j * nz;
                const int base_jp1 = i * ny_nz + (j + 1) * nz;
#if FDTD_HAS_OPENMP
                #pragma omp simd
#endif
                for (int k = 0; k < nz; k++) {
                    const int idx = base + k;
                    const int idx_jp1 = base_jp1 + k;
                    if (material_id[idx] == mat_id && material_id[idx_jp1] == mat_id) {
                        const float dJ_dy = J_pole[idx_jp1] - J_pole[idx];
                        vy[idx] += coeff * dJ_dy;
                    }
                }
            }
        }

        // vz: gradient in z direction
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
                    const int idx = base + k;
                    const int idx_kp1 = base + k + 1;
                    if (material_id[idx] == mat_id && material_id[idx_kp1] == mat_id) {
                        const float dJ_dz = J_pole[idx_kp1] - J_pole[idx];
                        vz[idx] += coeff * dJ_dz;
                    }
                }
            }
        }
    }
}

void apply_ade_pressure_correction(
    float* __restrict__ p,
    const float* __restrict__ J_debye,
    const float* __restrict__ J_lorentz,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape,
    float dt
) {
    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;
    const int grid_size = nx * ny_nz;

    // Apply Debye modulus pole corrections
    for (const auto& pole : material_data.debye_poles) {
        if (pole.target != 1) continue;  // Only modulus poles

        const int mat_id = pole.material_id;
        const float K_inf = material_data.K_inf[mat_id];
        const float coeff = -K_inf * dt;
        const float* J_pole = J_debye + pole.field_index * grid_size;

#if FDTD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                const int base = i * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
                #pragma omp simd
#endif
                for (int k = 0; k < nz; k++) {
                    const int idx = base + k;
                    if (material_id[idx] == mat_id) {
                        p[idx] += coeff * J_pole[idx];
                    }
                }
            }
        }
    }

    // Apply Lorentz modulus pole corrections
    for (const auto& pole : material_data.lorentz_poles) {
        if (pole.target != 1) continue;

        const int mat_id = pole.material_id;
        const float K_inf = material_data.K_inf[mat_id];
        const float coeff = -K_inf * dt;
        const float* J_pole = J_lorentz + pole.field_index * grid_size;

#if FDTD_HAS_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                const int base = i * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
                #pragma omp simd
#endif
                for (int k = 0; k < nz; k++) {
                    const int idx = base + k;
                    if (material_id[idx] == mat_id) {
                        p[idx] += coeff * J_pole[idx];
                    }
                }
            }
        }
    }
}

// =============================================================================
// Utility kernels
// =============================================================================

void compute_divergence(
    float* __restrict__ divergence,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    const GridShape& shape,
    float inv_dx
) {
    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;

    // x-divergence for i > 0: (vx[i] - vx[i-1]) / dx
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 1; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            const int base = i * ny_nz + j * nz;
            const int base_im1 = (i - 1) * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 0; k < nz; k++) {
                divergence[base + k] = (vx[base + k] - vx[base_im1 + k]) * inv_dx;
            }
        }
    }

    // x-divergence for i = 0: vx[0] / dx (vx[-1] = 0)
#if FDTD_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < ny; j++) {
        const int base = j * nz;  // i = 0
#if FDTD_HAS_OPENMP
        #pragma omp simd
#endif
        for (int k = 0; k < nz; k++) {
            divergence[base + k] = vx[base + k] * inv_dx;
        }
    }

    // y-divergence for j > 0: add (vy[j] - vy[j-1]) / dx
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            const int base = i * ny_nz + j * nz;
            const int base_jm1 = i * ny_nz + (j - 1) * nz;
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 0; k < nz; k++) {
                divergence[base + k] += (vy[base + k] - vy[base_jm1 + k]) * inv_dx;
            }
        }
    }

    // y-divergence for j = 0
#if FDTD_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        const int base = i * ny_nz;  // j = 0
#if FDTD_HAS_OPENMP
        #pragma omp simd
#endif
        for (int k = 0; k < nz; k++) {
            divergence[base + k] += vy[base + k] * inv_dx;
        }
    }

    // z-divergence for k > 0: add (vz[k] - vz[k-1]) / dx
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            const int base = i * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 1; k < nz; k++) {
                divergence[base + k] += (vz[base + k] - vz[base + k - 1]) * inv_dx;
            }
        }
    }

    // z-divergence for k = 0
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            const int base = i * ny_nz + j * nz;  // k = 0
            divergence[base] += vz[base] * inv_dx;
        }
    }
}

void compute_divergence_nonuniform(
    float* __restrict__ divergence,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    const GridShape& shape,
    const NonuniformGridData& grid_data
) {
    const int nx = shape.nx;
    const int ny = shape.ny;
    const int nz = shape.nz;
    const int ny_nz = ny * nz;

    // Pointers to spacing arrays
    const float* inv_dx_cell = grid_data.inv_dx_cell.data();
    const float* inv_dy_cell = grid_data.inv_dy_cell.data();
    const float* inv_dz_cell = grid_data.inv_dz_cell.data();

    // x-divergence for i > 0: inv_dx_cell[i] * (vx[i] - vx[i-1])
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 1; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            const int base = i * ny_nz + j * nz;
            const int base_im1 = (i - 1) * ny_nz + j * nz;
            const float inv_dx = inv_dx_cell[i];
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 0; k < nz; k++) {
                divergence[base + k] = (vx[base + k] - vx[base_im1 + k]) * inv_dx;
            }
        }
    }

    // x-divergence for i = 0: inv_dx_cell[0] * vx[0] (vx[-1] = 0)
#if FDTD_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < ny; j++) {
        const int base = j * nz;  // i = 0
        const float inv_dx = inv_dx_cell[0];
#if FDTD_HAS_OPENMP
        #pragma omp simd
#endif
        for (int k = 0; k < nz; k++) {
            divergence[base + k] = vx[base + k] * inv_dx;
        }
    }

    // y-divergence for j > 0: add inv_dy_cell[j] * (vy[j] - vy[j-1])
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            const int base = i * ny_nz + j * nz;
            const int base_jm1 = i * ny_nz + (j - 1) * nz;
            const float inv_dy = inv_dy_cell[j];
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 0; k < nz; k++) {
                divergence[base + k] += (vy[base + k] - vy[base_jm1 + k]) * inv_dy;
            }
        }
    }

    // y-divergence for j = 0
#if FDTD_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        const int base = i * ny_nz;  // j = 0
        const float inv_dy = inv_dy_cell[0];
#if FDTD_HAS_OPENMP
        #pragma omp simd
#endif
        for (int k = 0; k < nz; k++) {
            divergence[base + k] += vy[base + k] * inv_dy;
        }
    }

    // z-divergence for k > 0: add inv_dz_cell[k] * (vz[k] - vz[k-1])
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            const int base = i * ny_nz + j * nz;
#if FDTD_HAS_OPENMP
            #pragma omp simd
#endif
            for (int k = 1; k < nz; k++) {
                divergence[base + k] += (vz[base + k] - vz[base + k - 1]) * inv_dz_cell[k];
            }
        }
    }

    // z-divergence for k = 0
#if FDTD_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            const int base = i * ny_nz + j * nz;  // k = 0
            divergence[base] += vz[base] * inv_dz_cell[0];
        }
    }
}

}  // namespace fdtd

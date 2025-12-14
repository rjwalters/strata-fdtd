#pragma once
/**
 * @file fdtd_types.hpp
 * @brief Common type definitions for FDTD kernels
 */

#include <cstdint>
#include <cstddef>
#include <vector>

// OpenMP configuration - include outside namespace to avoid scope issues
#if FDTD_HAS_OPENMP
    #include <omp.h>
    #define FDTD_PARALLEL_FOR _Pragma("omp parallel for")
    #define FDTD_PARALLEL_FOR_SIMD _Pragma("omp parallel for simd")
    #define FDTD_SIMD _Pragma("omp simd")
#else
    #define FDTD_PARALLEL_FOR
    #define FDTD_PARALLEL_FOR_SIMD
    #define FDTD_SIMD
#endif

namespace fdtd {

// Grid dimensions type
struct GridShape {
    int nx;
    int ny;
    int nz;

    int size() const { return nx * ny * nz; }
};

// Inline helper for 3D -> 1D index conversion (row-major, C-contiguous)
// Layout: [i, j, k] -> i * ny * nz + j * nz + k
inline int idx(int i, int j, int k, int ny, int nz) {
    return i * ny * nz + j * nz + k;
}

inline int idx(int i, int j, int k, const GridShape& shape) {
    return i * shape.ny * shape.nz + j * shape.nz + k;
}

// FDTD coefficients
struct FDTDCoeffs {
    float coeff_v;  // Velocity update coefficient: -dt / (rho * dx)
    float coeff_p;  // Pressure update coefficient: -rho * c^2 * dt / dx
};

// Nonuniform grid spacing data
// For velocity updates: inv_dx_face has (n-1) elements for each axis
// For pressure updates: inv_dx_cell has n elements for each axis
struct NonuniformGridData {
    // Face spacing (1/dx) for velocity updates
    std::vector<float> inv_dx_face;  // size: nx-1
    std::vector<float> inv_dy_face;  // size: ny-1
    std::vector<float> inv_dz_face;  // size: nz-1

    // Cell spacing (1/dx) for pressure updates
    std::vector<float> inv_dx_cell;  // size: nx
    std::vector<float> inv_dy_cell;  // size: ny
    std::vector<float> inv_dz_cell;  // size: nz

    // Check if data is valid for a given grid shape
    bool is_valid(const GridShape& shape) const {
        return inv_dx_face.size() == static_cast<size_t>(shape.nx - 1) &&
               inv_dy_face.size() == static_cast<size_t>(shape.ny - 1) &&
               inv_dz_face.size() == static_cast<size_t>(shape.nz - 1) &&
               inv_dx_cell.size() == static_cast<size_t>(shape.nx) &&
               inv_dy_cell.size() == static_cast<size_t>(shape.ny) &&
               inv_dz_cell.size() == static_cast<size_t>(shape.nz);
    }
};

// Base coefficients for nonuniform grids (without the 1/dx factor)
struct FDTDCoeffsBase {
    float coeff_v_base;  // Base velocity coefficient: -dt / rho
    float coeff_p_base;  // Base pressure coefficient: -rho * c^2 * dt
};

}  // namespace fdtd

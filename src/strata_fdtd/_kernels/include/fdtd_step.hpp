#pragma once
/**
 * @file fdtd_step.hpp
 * @brief Core FDTD time-stepping kernels
 *
 * Implements velocity and pressure updates for the acoustic wave equation
 * on a staggered Yee grid.
 */

#include "fdtd_types.hpp"

namespace fdtd {

/**
 * @brief Update velocity fields from pressure gradient
 *
 * Implements: v += coeff_v * grad(p)
 *
 * Uses staggered grid layout where:
 * - vx[i,j,k] is at face (i+1/2, j, k)
 * - vy[i,j,k] is at face (i, j+1/2, k)
 * - vz[i,j,k] is at face (i, j, k+1/2)
 *
 * @param p Pressure field (nx * ny * nz)
 * @param vx X-velocity field (nx * ny * nz)
 * @param vy Y-velocity field (nx * ny * nz)
 * @param vz Z-velocity field (nx * ny * nz)
 * @param shape Grid dimensions
 * @param coeff_v Velocity update coefficient
 */
void update_velocity(
    const float* __restrict__ p,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const GridShape& shape,
    float coeff_v
);

/**
 * @brief Update pressure field from velocity divergence
 *
 * Implements: p += coeff_p * div(v)
 *
 * @param p Pressure field (nx * ny * nz), modified in place
 * @param vx X-velocity field (nx * ny * nz)
 * @param vy Y-velocity field (nx * ny * nz)
 * @param vz Z-velocity field (nx * ny * nz)
 * @param geometry Boolean mask (true=air, false=solid)
 * @param shape Grid dimensions
 * @param coeff_p Pressure update coefficient
 */
void update_pressure(
    float* __restrict__ p,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    const bool* __restrict__ geometry,
    const GridShape& shape,
    float coeff_p
);

/**
 * @brief Fused FDTD step: velocity update + pressure update
 *
 * Combines velocity and pressure updates for better cache utilization.
 * This is the main kernel to call for time stepping.
 *
 * @param p Pressure field
 * @param vx X-velocity field
 * @param vy Y-velocity field
 * @param vz Z-velocity field
 * @param geometry Boolean mask (true=air, false=solid)
 * @param shape Grid dimensions
 * @param coeffs FDTD coefficients
 */
void fdtd_step(
    float* __restrict__ p,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const bool* __restrict__ geometry,
    const GridShape& shape,
    const FDTDCoeffs& coeffs
);

// =============================================================================
// Nonuniform grid kernels (Issue #131)
// =============================================================================

/**
 * @brief Update velocity fields for nonuniform grids
 *
 * Implements: v += coeff_v_base * inv_dx_face * (p[i+1] - p[i])
 *
 * Uses per-face spacing arrays for variable cell sizes.
 *
 * @param p Pressure field (nx * ny * nz)
 * @param vx X-velocity field (nx * ny * nz)
 * @param vy Y-velocity field (nx * ny * nz)
 * @param vz Z-velocity field (nx * ny * nz)
 * @param shape Grid dimensions
 * @param grid_data Precomputed spacing arrays
 * @param coeff_v_base Base velocity coefficient (-dt / rho)
 */
void update_velocity_nonuniform(
    const float* __restrict__ p,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const GridShape& shape,
    const NonuniformGridData& grid_data,
    float coeff_v_base
);

/**
 * @brief Update pressure field for nonuniform grids
 *
 * Implements: p += coeff_p_base * inv_dx_cell * div(v)
 *
 * Uses per-cell spacing arrays for variable cell sizes.
 *
 * @param p Pressure field (nx * ny * nz), modified in place
 * @param vx X-velocity field (nx * ny * nz)
 * @param vy Y-velocity field (nx * ny * nz)
 * @param vz Z-velocity field (nx * ny * nz)
 * @param geometry Boolean mask (true=air, false=solid)
 * @param shape Grid dimensions
 * @param grid_data Precomputed spacing arrays
 * @param coeff_p_base Base pressure coefficient (-rho * c^2 * dt)
 */
void update_pressure_nonuniform(
    float* __restrict__ p,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    const bool* __restrict__ geometry,
    const GridShape& shape,
    const NonuniformGridData& grid_data,
    float coeff_p_base
);

/**
 * @brief Fused FDTD step for nonuniform grids
 *
 * @param p Pressure field
 * @param vx X-velocity field
 * @param vy Y-velocity field
 * @param vz Z-velocity field
 * @param geometry Boolean mask (true=air, false=solid)
 * @param shape Grid dimensions
 * @param grid_data Precomputed spacing arrays
 * @param coeffs Base FDTD coefficients
 */
void fdtd_step_nonuniform(
    float* __restrict__ p,
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const bool* __restrict__ geometry,
    const GridShape& shape,
    const NonuniformGridData& grid_data,
    const FDTDCoeffsBase& coeffs
);

}  // namespace fdtd

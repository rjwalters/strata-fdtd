#pragma once
/**
 * @file pml.hpp
 * @brief Perfectly Matched Layer (PML) absorbing boundary conditions
 *
 * Pre-computes exponential decay factors for efficient application.
 */

#include "fdtd_types.hpp"
#include <vector>
#include <cmath>

namespace fdtd {

/**
 * @brief Pre-computed PML decay factors
 *
 * Stores exp(-sigma * dt) for each axis, avoiding repeated exp() calls.
 */
struct PMLData {
    std::vector<float> decay_x;  // exp(-sigma_x * dt) for each x
    std::vector<float> decay_y;  // exp(-sigma_y * dt) for each y
    std::vector<float> decay_z;  // exp(-sigma_z * dt) for each z

    bool has_x() const { return !decay_x.empty(); }
    bool has_y() const { return !decay_y.empty(); }
    bool has_z() const { return !decay_z.empty(); }
};

/**
 * @brief Initialize PML decay factors from sigma profiles
 *
 * @param sigma_x Conductivity profile for x-axis (or nullptr)
 * @param sigma_y Conductivity profile for y-axis (or nullptr)
 * @param sigma_z Conductivity profile for z-axis (or nullptr)
 * @param shape Grid dimensions
 * @param dt Timestep in seconds
 * @return PMLData with pre-computed decay factors
 */
PMLData initialize_pml(
    const float* sigma_x,
    const float* sigma_y,
    const float* sigma_z,
    const GridShape& shape,
    float dt
);

/**
 * @brief Apply PML damping to velocity fields
 *
 * Multiplies each velocity component by the corresponding decay factor.
 *
 * @param vx X-velocity field
 * @param vy Y-velocity field
 * @param vz Z-velocity field
 * @param pml Pre-computed PML decay factors
 * @param shape Grid dimensions
 */
void apply_pml_velocity(
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const PMLData& pml,
    const GridShape& shape
);

/**
 * @brief Apply PML damping to pressure field
 *
 * @param p Pressure field
 * @param pml Pre-computed PML decay factors
 * @param shape Grid dimensions
 */
void apply_pml_pressure(
    float* __restrict__ p,
    const PMLData& pml,
    const GridShape& shape
);

}  // namespace fdtd

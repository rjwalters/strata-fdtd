#pragma once
/**
 * @file boundaries.hpp
 * @brief Rigid boundary condition handling
 *
 * Pre-computes boundary cell indices for efficient enforcement.
 */

#include "fdtd_types.hpp"
#include <vector>

namespace fdtd {

/**
 * @brief Pre-computed boundary cell indices
 *
 * Stores flat indices of cells where velocity should be zeroed
 * due to adjacent solid geometry.
 */
struct BoundaryCells {
    std::vector<int> vx_indices;  // Cells where vx should be zeroed
    std::vector<int> vy_indices;  // Cells where vy should be zeroed
    std::vector<int> vz_indices;  // Cells where vz should be zeroed

    bool empty() const {
        return vx_indices.empty() && vy_indices.empty() && vz_indices.empty();
    }

    size_t total_size() const {
        return vx_indices.size() + vy_indices.size() + vz_indices.size();
    }
};

/**
 * @brief Pre-compute boundary cell indices from geometry
 *
 * For each velocity component, identifies cells at solid/air interfaces
 * where velocity should be zeroed.
 *
 * @param geometry Boolean mask (true=air, false=solid)
 * @param shape Grid dimensions
 * @return BoundaryCells structure with pre-computed indices
 */
BoundaryCells precompute_boundary_cells(
    const bool* __restrict__ geometry,
    const GridShape& shape
);

/**
 * @brief Apply rigid boundary conditions using pre-computed indices
 *
 * Zeros velocity at all boundary cells. Much faster than per-step
 * mask creation when geometry is sparse.
 *
 * @param vx X-velocity field
 * @param vy Y-velocity field
 * @param vz Z-velocity field
 * @param bc Pre-computed boundary cell indices
 */
void apply_rigid_boundaries(
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const BoundaryCells& bc
);

/**
 * @brief Apply rigid boundaries using geometry mask (slower fallback)
 *
 * Direct implementation that checks geometry each step.
 * Use precompute_boundary_cells + apply_rigid_boundaries for better performance.
 *
 * @param vx X-velocity field
 * @param vy Y-velocity field
 * @param vz Z-velocity field
 * @param geometry Boolean mask (true=air, false=solid)
 * @param shape Grid dimensions
 */
void apply_rigid_boundaries_direct(
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    const bool* __restrict__ geometry,
    const GridShape& shape
);

}  // namespace fdtd

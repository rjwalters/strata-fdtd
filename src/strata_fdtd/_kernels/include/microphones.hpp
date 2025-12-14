#pragma once
/**
 * @file microphones.hpp
 * @brief Native kernel for batch microphone recording with trilinear interpolation
 *
 * Provides accelerated microphone recording for FDTD simulations with
 * OpenMP parallelization across microphones and SIMD vectorization
 * for trilinear interpolation.
 *
 * Issue #102: Native C++ kernel for high microphone count performance
 */

#include "fdtd_types.hpp"
#include <vector>
#include <cstdint>

namespace fdtd {

/**
 * @brief Pre-computed interpolation data for a batch of microphones.
 *
 * Stores packed corner indices and weights for efficient batch recording.
 * Memory layout is optimized for cache-friendly access during parallel
 * processing of multiple microphones.
 *
 * For n_mics microphones:
 * - flat_indices: [n_mics * 8] linearized 3D indices for each corner
 * - weights: [n_mics * 8] interpolation weights for each corner
 *
 * Corner ordering (same as Python implementation):
 * 0: (i0, j0, k0), 1: (i1, j0, k0), 2: (i0, j1, k0), 3: (i1, j1, k0)
 * 4: (i0, j0, k1), 5: (i1, j0, k1), 6: (i0, j1, k1), 7: (i1, j1, k1)
 */
struct MicrophoneData {
    std::vector<int> flat_indices;    // [n_mics * 8] linearized corner indices
    std::vector<float> weights;        // [n_mics * 8] interpolation weights
    int n_mics;                        // Number of microphones

    MicrophoneData() : n_mics(0) {}

    bool empty() const { return n_mics == 0; }
    int total_indices() const { return n_mics * 8; }
};

/**
 * @brief Pre-compute microphone interpolation data from grid positions.
 *
 * Takes physical microphone positions and computes the linearized corner
 * indices and trilinear interpolation weights for batch recording.
 *
 * @param grid_positions Flattened grid coordinates [n_mics * 3] as (gx, gy, gz) triples
 * @param n_mics Number of microphones
 * @param shape Grid dimensions
 * @return MicrophoneData with pre-computed indices and weights
 */
MicrophoneData precompute_microphone_data(
    const float* grid_positions,
    int n_mics,
    const GridShape& shape
);

/**
 * @brief Record pressure at all microphones using batch trilinear interpolation.
 *
 * Uses OpenMP parallelization across microphones for large arrays.
 * Falls back to serial execution for small microphone counts.
 *
 * @param pressure 3D pressure field [nx * ny * nz] (row-major)
 * @param mic_data Pre-computed microphone interpolation data
 * @param shape Grid dimensions
 * @param output Output buffer [n_mics] for recorded pressure values
 */
void record_microphones_batch(
    const float* pressure,
    const MicrophoneData& mic_data,
    const GridShape& shape,
    float* output
);

}  // namespace fdtd

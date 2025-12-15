/**
 * @file microphones.cpp
 * @brief Native kernel implementations for batch microphone recording
 *
 * Optimized implementation with OpenMP parallelization and SIMD vectorization
 * for efficient recording of large microphone arrays.
 *
 * Issue #102: Native C++ kernel for high microphone count performance
 */

#include "microphones.hpp"
#include <cmath>

namespace fdtd {

MicrophoneData precompute_microphone_data(
    const float* grid_positions,
    int n_mics,
    const GridShape& shape
) {
    MicrophoneData data;
    data.n_mics = n_mics;
    data.flat_indices.resize(n_mics * 8);
    data.weights.resize(n_mics * 8);

    const int ny_nz = shape.ny * shape.nz;

    for (int m = 0; m < n_mics; m++) {
        // Extract grid position for this microphone
        const float gx = grid_positions[m * 3 + 0];
        const float gy = grid_positions[m * 3 + 1];
        const float gz = grid_positions[m * 3 + 2];

        // Integer grid indices (lower corner of cell)
        const int i0 = static_cast<int>(gx);
        const int j0 = static_cast<int>(gy);
        const int k0 = static_cast<int>(gz);
        const int i1 = i0 + 1;
        const int j1 = j0 + 1;
        const int k1 = k0 + 1;

        // Fractional position within cell [0, 1)
        const float fx = gx - i0;
        const float fy = gy - j0;
        const float fz = gz - k0;

        // Compute weights for 8 corners
        const float w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
        const float w100 = fx * (1.0f - fy) * (1.0f - fz);
        const float w010 = (1.0f - fx) * fy * (1.0f - fz);
        const float w110 = fx * fy * (1.0f - fz);
        const float w001 = (1.0f - fx) * (1.0f - fy) * fz;
        const float w101 = fx * (1.0f - fy) * fz;
        const float w011 = (1.0f - fx) * fy * fz;
        const float w111 = fx * fy * fz;

        // Compute linearized indices for 8 corners
        const int base = m * 8;
        data.flat_indices[base + 0] = i0 * ny_nz + j0 * shape.nz + k0;
        data.flat_indices[base + 1] = i1 * ny_nz + j0 * shape.nz + k0;
        data.flat_indices[base + 2] = i0 * ny_nz + j1 * shape.nz + k0;
        data.flat_indices[base + 3] = i1 * ny_nz + j1 * shape.nz + k0;
        data.flat_indices[base + 4] = i0 * ny_nz + j0 * shape.nz + k1;
        data.flat_indices[base + 5] = i1 * ny_nz + j0 * shape.nz + k1;
        data.flat_indices[base + 6] = i0 * ny_nz + j1 * shape.nz + k1;
        data.flat_indices[base + 7] = i1 * ny_nz + j1 * shape.nz + k1;

        // Store weights
        data.weights[base + 0] = w000;
        data.weights[base + 1] = w100;
        data.weights[base + 2] = w010;
        data.weights[base + 3] = w110;
        data.weights[base + 4] = w001;
        data.weights[base + 5] = w101;
        data.weights[base + 6] = w011;
        data.weights[base + 7] = w111;
    }

    return data;
}

void record_microphones_batch(
    const float* __restrict__ pressure,
    const MicrophoneData& mic_data,
    const GridShape& shape,
    float* __restrict__ output
) {
    const int n_mics = mic_data.n_mics;
    const int* __restrict__ indices = mic_data.flat_indices.data();
    const float* __restrict__ weights = mic_data.weights.data();

    // Use OpenMP parallelization for large microphone counts
    // For small counts (<16), serial execution avoids thread overhead
#if FDTD_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(n_mics >= 16)
#endif
    for (int m = 0; m < n_mics; m++) {
        const int base = m * 8;

        // Trilinear interpolation with explicit unrolling for SIMD
        // The compiler can vectorize this accumulation pattern
        float sum = 0.0f;

        // Unroll the 8-point interpolation for better vectorization
        sum += weights[base + 0] * pressure[indices[base + 0]];
        sum += weights[base + 1] * pressure[indices[base + 1]];
        sum += weights[base + 2] * pressure[indices[base + 2]];
        sum += weights[base + 3] * pressure[indices[base + 3]];
        sum += weights[base + 4] * pressure[indices[base + 4]];
        sum += weights[base + 5] * pressure[indices[base + 5]];
        sum += weights[base + 6] * pressure[indices[base + 6]];
        sum += weights[base + 7] * pressure[indices[base + 7]];

        output[m] = sum;
    }
}

}  // namespace fdtd

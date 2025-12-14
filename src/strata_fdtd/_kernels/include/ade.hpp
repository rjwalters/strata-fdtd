#pragma once
/**
 * @file ade.hpp
 * @brief ADE (Auxiliary Differential Equation) kernels for dispersive materials
 *
 * Implements native C++ kernels for frequency-dependent acoustic materials
 * using the ADE formulation with Debye and Lorentz poles.
 *
 * Issue #139: Native C++ kernels for ADE material updates
 */

#include "fdtd_types.hpp"
#include <vector>

namespace fdtd {

/**
 * @brief Coefficients for a single Debye pole.
 *
 * Debye update: J^{n+1} = alpha * J^n + beta * source
 */
struct DebyePoleCoeffs {
    float alpha;    // Decay coefficient (tau/dt) / (1 + tau/dt)
    float beta;     // Source coefficient delta_chi / (1 + tau/dt)
};

/**
 * @brief Coefficients for a single Lorentz pole.
 *
 * Lorentz update: J^{n+1} = a*J^n + b*J^{n-1} + d*source
 */
struct LorentzPoleCoeffs {
    float a;    // Current field coefficient
    float b;    // Previous field coefficient
    float d;    // Source coefficient
};

/**
 * @brief Pre-computed material data for native ADE kernels.
 *
 * Contains all pole coefficients and material properties needed
 * for efficient ADE updates across all materials in the simulation.
 */
struct ADEMaterialData {
    // Material properties for each material ID (index 0 is unused/air)
    std::vector<float> rho_inf;    // High-frequency density per material
    std::vector<float> K_inf;      // High-frequency bulk modulus per material

    // Debye poles: coefficients and auxiliary field indices
    // Each Debye pole has: material_id, target (0=density, 1=modulus), coeffs
    struct DebyePole {
        int material_id;
        int target;           // 0 = density, 1 = modulus
        DebyePoleCoeffs coeffs;
        int field_index;      // Index into auxiliary field arrays
    };
    std::vector<DebyePole> debye_poles;

    // Lorentz poles: coefficients and auxiliary field indices
    struct LorentzPole {
        int material_id;
        int target;           // 0 = density, 1 = modulus
        LorentzPoleCoeffs coeffs;
        int field_index;      // Index into J and J_prev arrays
    };
    std::vector<LorentzPole> lorentz_poles;

    // Total counts for allocation
    int n_debye_fields = 0;    // Number of Debye auxiliary fields
    int n_lorentz_fields = 0;  // Number of Lorentz auxiliary fields

    // Check if any materials are registered
    bool has_materials() const {
        return !debye_poles.empty() || !lorentz_poles.empty();
    }

    // Count poles by type
    int n_density_debye() const {
        int count = 0;
        for (const auto& pole : debye_poles) {
            if (pole.target == 0) count++;
        }
        return count;
    }

    int n_modulus_debye() const {
        int count = 0;
        for (const auto& pole : debye_poles) {
            if (pole.target == 1) count++;
        }
        return count;
    }

    int n_density_lorentz() const {
        int count = 0;
        for (const auto& pole : lorentz_poles) {
            if (pole.target == 0) count++;
        }
        return count;
    }

    int n_modulus_lorentz() const {
        int count = 0;
        for (const auto& pole : lorentz_poles) {
            if (pole.target == 1) count++;
        }
        return count;
    }
};

// =============================================================================
// Debye pole update kernels
// =============================================================================

/**
 * @brief Update Debye pole auxiliary fields for density.
 *
 * For each Debye density pole: J^{n+1} = alpha * J^n + beta * pressure
 *
 * Called before velocity update.
 *
 * @param J Array of auxiliary fields [n_fields * grid_size]
 * @param pressure Pressure field
 * @param material_id Per-cell material ID array
 * @param material_data Pre-computed material coefficients
 * @param shape Grid dimensions
 */
void update_ade_density_debye(
    float* __restrict__ J,
    const float* __restrict__ pressure,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape
);

/**
 * @brief Update Debye pole auxiliary fields for modulus.
 *
 * For each Debye modulus pole: J^{n+1} = alpha * J^n + beta * divergence
 *
 * Called before pressure update.
 *
 * @param J Array of auxiliary fields [n_fields * grid_size]
 * @param divergence Velocity divergence field
 * @param material_id Per-cell material ID array
 * @param material_data Pre-computed material coefficients
 * @param shape Grid dimensions
 */
void update_ade_modulus_debye(
    float* __restrict__ J,
    const float* __restrict__ divergence,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape
);

// =============================================================================
// Lorentz pole update kernels
// =============================================================================

/**
 * @brief Update Lorentz pole auxiliary fields for density.
 *
 * For each Lorentz density pole: J^{n+1} = a*J^n + b*J^{n-1} + d*pressure
 * Also shifts history: J_prev = J before update.
 *
 * @param J Array of current auxiliary fields [n_fields * grid_size]
 * @param J_prev Array of previous auxiliary fields [n_fields * grid_size]
 * @param pressure Pressure field
 * @param material_id Per-cell material ID array
 * @param material_data Pre-computed material coefficients
 * @param shape Grid dimensions
 */
void update_ade_density_lorentz(
    float* __restrict__ J,
    float* __restrict__ J_prev,
    const float* __restrict__ pressure,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape
);

/**
 * @brief Update Lorentz pole auxiliary fields for modulus.
 *
 * For each Lorentz modulus pole: J^{n+1} = a*J^n + b*J^{n-1} + d*divergence
 *
 * @param J Array of current auxiliary fields [n_fields * grid_size]
 * @param J_prev Array of previous auxiliary fields [n_fields * grid_size]
 * @param divergence Velocity divergence field
 * @param material_id Per-cell material ID array
 * @param material_data Pre-computed material coefficients
 * @param shape Grid dimensions
 */
void update_ade_modulus_lorentz(
    float* __restrict__ J,
    float* __restrict__ J_prev,
    const float* __restrict__ divergence,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape
);

// =============================================================================
// Correction kernels
// =============================================================================

/**
 * @brief Apply ADE velocity corrections from density poles using gradient of J.
 *
 * For density dispersion, the auxiliary field J represents a polarization-like
 * term that modifies the effective density. The velocity correction requires
 * computing the gradient of J at each velocity face location:
 *
 *   vx[i] += -dt/ρ_inf * (J[i+1] - J[i]) / dx  = coeff * dJ/dx at face i
 *   vy[j] += -dt/ρ_inf * (J[j+1] - J[j]) / dy  = coeff * dJ/dy at face j
 *   vz[k] += -dt/ρ_inf * (J[k+1] - J[k]) / dz  = coeff * dJ/dz at face k
 *
 * The correction is only applied at faces where BOTH adjacent cells are in
 * the material region, ensuring proper treatment at material interfaces.
 *
 * @param vx X-velocity field (modified in place)
 * @param vy Y-velocity field (modified in place)
 * @param vz Z-velocity field (modified in place)
 * @param J_debye Debye auxiliary fields
 * @param J_lorentz Lorentz auxiliary fields
 * @param material_id Per-cell material ID array
 * @param material_data Pre-computed material coefficients
 * @param shape Grid dimensions
 * @param dt Timestep
 * @param inv_dx Inverse grid spacing (1/dx)
 */
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
);

/**
 * @brief Apply ADE pressure corrections from modulus poles.
 *
 * Adds dispersive contribution: p += coeff * J for each modulus pole
 *
 * @param p Pressure field (modified in place)
 * @param J_debye Debye auxiliary fields
 * @param J_lorentz Lorentz auxiliary fields
 * @param material_id Per-cell material ID array
 * @param material_data Pre-computed material coefficients
 * @param shape Grid dimensions
 * @param dt Timestep
 */
void apply_ade_pressure_correction(
    float* __restrict__ p,
    const float* __restrict__ J_debye,
    const float* __restrict__ J_lorentz,
    const uint8_t* __restrict__ material_id,
    const ADEMaterialData& material_data,
    const GridShape& shape,
    float dt
);

// =============================================================================
// Utility kernels
// =============================================================================

/**
 * @brief Compute velocity divergence field.
 *
 * div = (vx[i] - vx[i-1]) + (vy[j] - vy[j-1]) + (vz[k] - vz[k-1])
 *
 * @param divergence Output divergence field (modified in place)
 * @param vx X-velocity field
 * @param vy Y-velocity field
 * @param vz Z-velocity field
 * @param shape Grid dimensions
 * @param inv_dx Inverse grid spacing (1/dx)
 */
void compute_divergence(
    float* __restrict__ divergence,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    const GridShape& shape,
    float inv_dx
);

/**
 * @brief Compute velocity divergence field for nonuniform grids.
 *
 * Uses per-cell inverse spacing arrays:
 * div = inv_dx_cell[i] * (vx[i] - vx[i-1])
 *     + inv_dy_cell[j] * (vy[j] - vy[j-1])
 *     + inv_dz_cell[k] * (vz[k] - vz[k-1])
 *
 * @param divergence Output divergence field (modified in place)
 * @param vx X-velocity field
 * @param vy Y-velocity field
 * @param vz Z-velocity field
 * @param shape Grid dimensions
 * @param grid_data Precomputed spacing arrays
 */
void compute_divergence_nonuniform(
    float* __restrict__ divergence,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    const GridShape& shape,
    const NonuniformGridData& grid_data
);

}  // namespace fdtd

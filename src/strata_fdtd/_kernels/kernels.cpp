/**
 * @file fdtd_kernels.cpp
 * @brief pybind11 bindings for FDTD acceleration kernels
 *
 * Exposes C++ FDTD kernels to Python with NumPy array support.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "fdtd_types.hpp"
#include "fdtd_step.hpp"
#include "boundaries.hpp"
#include "pml.hpp"
#include "microphones.hpp"
#include "ade.hpp"

namespace py = pybind11;

// Helper to get raw pointer from numpy array with validation
inline float* get_float_ptr(py::array_t<float, py::array::c_style>& arr, const std::string& name) {
    auto info = arr.request();
    if (info.ndim != 3) {
        throw std::runtime_error(name + " must be 3-dimensional");
    }
    return static_cast<float*>(info.ptr);
}

inline const float* get_float_ptr_const(const py::array_t<float, py::array::c_style>& arr, const std::string& name) {
    auto info = arr.request();
    if (info.ndim != 3) {
        throw std::runtime_error(name + " must be 3-dimensional");
    }
    return static_cast<const float*>(info.ptr);
}

inline const bool* get_bool_ptr_const(const py::array_t<bool, py::array::c_style>& arr, const std::string& name) {
    auto info = arr.request();
    if (info.ndim != 3) {
        throw std::runtime_error(name + " must be 3-dimensional");
    }
    return static_cast<const bool*>(info.ptr);
}

// Extract shape from numpy array
fdtd::GridShape get_shape(const py::array& arr) {
    auto info = arr.request();
    if (info.ndim != 3) {
        throw std::runtime_error("Array must be 3-dimensional");
    }
    return fdtd::GridShape{
        static_cast<int>(info.shape[0]),
        static_cast<int>(info.shape[1]),
        static_cast<int>(info.shape[2])
    };
}

// Verify arrays have matching shapes
void check_shapes_match(const fdtd::GridShape& a, const fdtd::GridShape& b,
                        const std::string& name_a, const std::string& name_b) {
    if (a.nx != b.nx || a.ny != b.ny || a.nz != b.nz) {
        throw std::runtime_error(name_a + " and " + name_b + " must have the same shape");
    }
}

// Helper to copy numpy array to std::vector<float>
inline std::vector<float> array_to_vector(const py::array_t<float>& arr) {
    auto info = arr.request();
    const float* data = static_cast<const float*>(info.ptr);
    return std::vector<float>(data, data + info.size);
}

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "FDTD acoustic simulation kernels (C++ accelerated)";

    // Version and build info
    m.attr("__version__") = FDTD_KERNELS_VERSION;
#if FDTD_HAS_OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif

    // Get number of OpenMP threads
    m.def("get_num_threads", []() -> int {
#if FDTD_HAS_OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }, "Get the number of OpenMP threads available");

    m.def("set_num_threads", [](int n) {
#if FDTD_HAS_OPENMP
        omp_set_num_threads(n);
#else
        if (n != 1) {
            throw std::runtime_error("OpenMP not available, cannot set thread count");
        }
#endif
    }, "Set the number of OpenMP threads", py::arg("n"));

    // ==========================================================================
    // Core FDTD step functions
    // ==========================================================================

    m.def("update_velocity",
        [](const py::array_t<float, py::array::c_style>& p,
           py::array_t<float, py::array::c_style>& vx,
           py::array_t<float, py::array::c_style>& vy,
           py::array_t<float, py::array::c_style>& vz,
           float coeff_v) {

            auto shape = get_shape(p);
            check_shapes_match(shape, get_shape(vx), "p", "vx");
            check_shapes_match(shape, get_shape(vy), "p", "vy");
            check_shapes_match(shape, get_shape(vz), "p", "vz");

            const float* p_ptr = get_float_ptr_const(p, "p");
            float* vx_ptr = get_float_ptr(vx, "vx");
            float* vy_ptr = get_float_ptr(vy, "vy");
            float* vz_ptr = get_float_ptr(vz, "vz");

            fdtd::update_velocity(p_ptr, vx_ptr, vy_ptr, vz_ptr, shape, coeff_v);
        },
        R"doc(
        Update velocity fields from pressure gradient.

        Args:
            p: Pressure field (nx, ny, nz) float32, read-only
            vx: X-velocity field (nx, ny, nz) float32, modified in place
            vy: Y-velocity field (nx, ny, nz) float32, modified in place
            vz: Z-velocity field (nx, ny, nz) float32, modified in place
            coeff_v: Velocity update coefficient
        )doc",
        py::arg("p"), py::arg("vx"), py::arg("vy"), py::arg("vz"),
        py::arg("coeff_v")
    );

    m.def("update_pressure",
        [](py::array_t<float, py::array::c_style>& p,
           const py::array_t<float, py::array::c_style>& vx,
           const py::array_t<float, py::array::c_style>& vy,
           const py::array_t<float, py::array::c_style>& vz,
           const py::array_t<bool, py::array::c_style>& geometry,
           float coeff_p) {

            auto shape = get_shape(p);
            check_shapes_match(shape, get_shape(vx), "p", "vx");
            check_shapes_match(shape, get_shape(vy), "p", "vy");
            check_shapes_match(shape, get_shape(vz), "p", "vz");
            check_shapes_match(shape, get_shape(geometry), "p", "geometry");

            float* p_ptr = get_float_ptr(p, "p");
            const float* vx_ptr = get_float_ptr_const(vx, "vx");
            const float* vy_ptr = get_float_ptr_const(vy, "vy");
            const float* vz_ptr = get_float_ptr_const(vz, "vz");
            const bool* geo_ptr = get_bool_ptr_const(geometry, "geometry");

            fdtd::update_pressure(p_ptr, vx_ptr, vy_ptr, vz_ptr, geo_ptr, shape, coeff_p);
        },
        R"doc(
        Update pressure field from velocity divergence.

        Args:
            p: Pressure field (nx, ny, nz) float32, modified in place
            vx: X-velocity field (nx, ny, nz) float32, read-only
            vy: Y-velocity field (nx, ny, nz) float32, read-only
            vz: Z-velocity field (nx, ny, nz) float32, read-only
            geometry: Boolean mask (nx, ny, nz), True=air, False=solid
            coeff_p: Pressure update coefficient
        )doc",
        py::arg("p"), py::arg("vx"), py::arg("vy"), py::arg("vz"),
        py::arg("geometry"), py::arg("coeff_p")
    );

    m.def("fdtd_step",
        [](py::array_t<float, py::array::c_style>& p,
           py::array_t<float, py::array::c_style>& vx,
           py::array_t<float, py::array::c_style>& vy,
           py::array_t<float, py::array::c_style>& vz,
           const py::array_t<bool, py::array::c_style>& geometry,
           float coeff_v, float coeff_p) {

            auto shape = get_shape(p);
            check_shapes_match(shape, get_shape(vx), "p", "vx");
            check_shapes_match(shape, get_shape(vy), "p", "vy");
            check_shapes_match(shape, get_shape(vz), "p", "vz");
            check_shapes_match(shape, get_shape(geometry), "p", "geometry");

            float* p_ptr = get_float_ptr(p, "p");
            float* vx_ptr = get_float_ptr(vx, "vx");
            float* vy_ptr = get_float_ptr(vy, "vy");
            float* vz_ptr = get_float_ptr(vz, "vz");
            const bool* geo_ptr = get_bool_ptr_const(geometry, "geometry");

            fdtd::FDTDCoeffs coeffs{coeff_v, coeff_p};
            fdtd::fdtd_step(p_ptr, vx_ptr, vy_ptr, vz_ptr, geo_ptr, shape, coeffs);
        },
        R"doc(
        Perform one FDTD timestep (velocity + pressure update).

        NOTE: This fused function does NOT apply rigid boundaries between
        velocity and pressure updates. For proper energy conservation with
        rigid boundaries, use update_velocity() + apply_rigid_boundaries() +
        update_pressure() separately.

        Args:
            p: Pressure field (nx, ny, nz) float32, modified in place
            vx: X-velocity field (nx, ny, nz) float32, modified in place
            vy: Y-velocity field (nx, ny, nz) float32, modified in place
            vz: Z-velocity field (nx, ny, nz) float32, modified in place
            geometry: Boolean mask (nx, ny, nz), True=air, False=solid
            coeff_v: Velocity update coefficient
            coeff_p: Pressure update coefficient
        )doc",
        py::arg("p"), py::arg("vx"), py::arg("vy"), py::arg("vz"),
        py::arg("geometry"), py::arg("coeff_v"), py::arg("coeff_p")
    );

    // ==========================================================================
    // Boundary conditions
    // ==========================================================================

    py::class_<fdtd::BoundaryCells>(m, "BoundaryCells",
        "Pre-computed boundary cell indices for rigid boundaries")
        .def(py::init<>())
        .def_readonly("vx_indices", &fdtd::BoundaryCells::vx_indices)
        .def_readonly("vy_indices", &fdtd::BoundaryCells::vy_indices)
        .def_readonly("vz_indices", &fdtd::BoundaryCells::vz_indices)
        .def("empty", &fdtd::BoundaryCells::empty)
        .def("total_size", &fdtd::BoundaryCells::total_size);

    m.def("precompute_boundary_cells",
        [](const py::array_t<bool, py::array::c_style>& geometry) {
            auto shape = get_shape(geometry);
            const bool* geo_ptr = get_bool_ptr_const(geometry, "geometry");
            return fdtd::precompute_boundary_cells(geo_ptr, shape);
        },
        R"doc(
        Pre-compute boundary cell indices from geometry mask.

        Args:
            geometry: Boolean mask (nx, ny, nz), True=air, False=solid

        Returns:
            BoundaryCells object with pre-computed indices
        )doc",
        py::arg("geometry")
    );

    m.def("apply_rigid_boundaries",
        [](py::array_t<float, py::array::c_style>& vx,
           py::array_t<float, py::array::c_style>& vy,
           py::array_t<float, py::array::c_style>& vz,
           const fdtd::BoundaryCells& bc) {
            float* vx_ptr = get_float_ptr(vx, "vx");
            float* vy_ptr = get_float_ptr(vy, "vy");
            float* vz_ptr = get_float_ptr(vz, "vz");
            fdtd::apply_rigid_boundaries(vx_ptr, vy_ptr, vz_ptr, bc);
        },
        R"doc(
        Apply rigid boundary conditions using pre-computed indices.

        Args:
            vx: X-velocity field, modified in place
            vy: Y-velocity field, modified in place
            vz: Z-velocity field, modified in place
            bc: Pre-computed BoundaryCells from precompute_boundary_cells()
        )doc",
        py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("bc")
    );

    // ==========================================================================
    // PML boundaries
    // ==========================================================================

    py::class_<fdtd::PMLData>(m, "PMLData",
        "Pre-computed PML decay factors")
        .def(py::init<>())
        .def("has_x", &fdtd::PMLData::has_x)
        .def("has_y", &fdtd::PMLData::has_y)
        .def("has_z", &fdtd::PMLData::has_z);

    m.def("initialize_pml",
        [](const py::object& sigma_x_obj,
           const py::object& sigma_y_obj,
           const py::object& sigma_z_obj,
           int nx, int ny, int nz, float dt) {

            const float* sigma_x = nullptr;
            const float* sigma_y = nullptr;
            const float* sigma_z = nullptr;

            py::array_t<float> sigma_x_arr, sigma_y_arr, sigma_z_arr;

            if (!sigma_x_obj.is_none()) {
                sigma_x_arr = sigma_x_obj.cast<py::array_t<float>>();
                sigma_x = sigma_x_arr.data();
            }
            if (!sigma_y_obj.is_none()) {
                sigma_y_arr = sigma_y_obj.cast<py::array_t<float>>();
                sigma_y = sigma_y_arr.data();
            }
            if (!sigma_z_obj.is_none()) {
                sigma_z_arr = sigma_z_obj.cast<py::array_t<float>>();
                sigma_z = sigma_z_arr.data();
            }

            fdtd::GridShape shape{nx, ny, nz};
            return fdtd::initialize_pml(sigma_x, sigma_y, sigma_z, shape, dt);
        },
        R"doc(
        Initialize PML with pre-computed decay factors.

        Args:
            sigma_x: Conductivity profile for x-axis (1D array or None)
            sigma_y: Conductivity profile for y-axis (1D array or None)
            sigma_z: Conductivity profile for z-axis (1D array or None)
            nx, ny, nz: Grid dimensions
            dt: Timestep in seconds

        Returns:
            PMLData object with pre-computed decay factors
        )doc",
        py::arg("sigma_x"), py::arg("sigma_y"), py::arg("sigma_z"),
        py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("dt")
    );

    m.def("apply_pml_velocity",
        [](py::array_t<float, py::array::c_style>& vx,
           py::array_t<float, py::array::c_style>& vy,
           py::array_t<float, py::array::c_style>& vz,
           const fdtd::PMLData& pml) {
            auto shape = get_shape(vx);
            float* vx_ptr = get_float_ptr(vx, "vx");
            float* vy_ptr = get_float_ptr(vy, "vy");
            float* vz_ptr = get_float_ptr(vz, "vz");
            fdtd::apply_pml_velocity(vx_ptr, vy_ptr, vz_ptr, pml, shape);
        },
        "Apply PML damping to velocity fields",
        py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("pml")
    );

    m.def("apply_pml_pressure",
        [](py::array_t<float, py::array::c_style>& p,
           const fdtd::PMLData& pml) {
            auto shape = get_shape(p);
            float* p_ptr = get_float_ptr(p, "p");
            fdtd::apply_pml_pressure(p_ptr, pml, shape);
        },
        "Apply PML damping to pressure field",
        py::arg("p"), py::arg("pml")
    );

    // ==========================================================================
    // Microphone recording (Issue #102)
    // ==========================================================================

    py::class_<fdtd::MicrophoneData>(m, "MicrophoneData",
        "Pre-computed interpolation data for batch microphone recording")
        .def(py::init<>())
        .def_readonly("n_mics", &fdtd::MicrophoneData::n_mics)
        .def("empty", &fdtd::MicrophoneData::empty)
        .def("total_indices", &fdtd::MicrophoneData::total_indices);

    m.def("precompute_microphone_data",
        [](const py::array_t<float, py::array::c_style>& grid_positions,
           int nx, int ny, int nz) {
            auto info = grid_positions.request();
            if (info.ndim != 1 || info.shape[0] % 3 != 0) {
                throw std::runtime_error(
                    "grid_positions must be 1D array with length divisible by 3"
                );
            }
            int n_mics = static_cast<int>(info.shape[0]) / 3;
            const float* pos_ptr = static_cast<const float*>(info.ptr);
            fdtd::GridShape shape{nx, ny, nz};
            return fdtd::precompute_microphone_data(pos_ptr, n_mics, shape);
        },
        R"doc(
        Pre-compute microphone interpolation data from grid positions.

        Args:
            grid_positions: Flattened grid coordinates [n_mics * 3] as
                           (gx, gy, gz) triples in grid units (not meters)
            nx, ny, nz: Grid dimensions

        Returns:
            MicrophoneData object with pre-computed indices and weights
        )doc",
        py::arg("grid_positions"), py::arg("nx"), py::arg("ny"), py::arg("nz")
    );

    m.def("record_microphones_batch",
        [](const py::array_t<float, py::array::c_style>& pressure,
           const fdtd::MicrophoneData& mic_data,
           py::array_t<float, py::array::c_style>& output) {
            auto shape = get_shape(pressure);
            const float* p_ptr = get_float_ptr_const(pressure, "pressure");

            auto out_info = output.request();
            if (out_info.ndim != 1 || out_info.shape[0] < mic_data.n_mics) {
                throw std::runtime_error(
                    "output must be 1D array with length >= n_mics"
                );
            }
            float* out_ptr = static_cast<float*>(out_info.ptr);

            fdtd::record_microphones_batch(p_ptr, mic_data, shape, out_ptr);
        },
        R"doc(
        Record pressure at all microphones using batch trilinear interpolation.

        Args:
            pressure: 3D pressure field (nx, ny, nz) float32, read-only
            mic_data: Pre-computed MicrophoneData from precompute_microphone_data()
            output: Output buffer (n_mics,) float32, modified in place

        Note:
            Uses OpenMP parallelization for microphone counts >= 16.
        )doc",
        py::arg("pressure"), py::arg("mic_data"), py::arg("output")
    );

    // ==========================================================================
    // Nonuniform grid support (Issue #131)
    // ==========================================================================

    py::class_<fdtd::NonuniformGridData>(m, "NonuniformGridData",
        "Pre-computed spacing arrays for nonuniform FDTD grids")
        .def(py::init<>())
        .def_readonly("inv_dx_face", &fdtd::NonuniformGridData::inv_dx_face)
        .def_readonly("inv_dy_face", &fdtd::NonuniformGridData::inv_dy_face)
        .def_readonly("inv_dz_face", &fdtd::NonuniformGridData::inv_dz_face)
        .def_readonly("inv_dx_cell", &fdtd::NonuniformGridData::inv_dx_cell)
        .def_readonly("inv_dy_cell", &fdtd::NonuniformGridData::inv_dy_cell)
        .def_readonly("inv_dz_cell", &fdtd::NonuniformGridData::inv_dz_cell)
        .def("is_valid", &fdtd::NonuniformGridData::is_valid,
             "Check if data is valid for a given grid shape");

    m.def("create_nonuniform_grid_data",
        [](const py::array_t<float>& inv_dx_face,
           const py::array_t<float>& inv_dy_face,
           const py::array_t<float>& inv_dz_face,
           const py::array_t<float>& inv_dx_cell,
           const py::array_t<float>& inv_dy_cell,
           const py::array_t<float>& inv_dz_cell) {
            fdtd::NonuniformGridData data;
            data.inv_dx_face = array_to_vector(inv_dx_face);
            data.inv_dy_face = array_to_vector(inv_dy_face);
            data.inv_dz_face = array_to_vector(inv_dz_face);
            data.inv_dx_cell = array_to_vector(inv_dx_cell);
            data.inv_dy_cell = array_to_vector(inv_dy_cell);
            data.inv_dz_cell = array_to_vector(inv_dz_cell);
            return data;
        },
        R"doc(
        Create NonuniformGridData from spacing arrays.

        Args:
            inv_dx_face: 1/dx for velocity x-updates (nx-1 elements)
            inv_dy_face: 1/dy for velocity y-updates (ny-1 elements)
            inv_dz_face: 1/dz for velocity z-updates (nz-1 elements)
            inv_dx_cell: 1/dx for pressure updates (nx elements)
            inv_dy_cell: 1/dy for pressure updates (ny elements)
            inv_dz_cell: 1/dz for pressure updates (nz elements)

        Returns:
            NonuniformGridData object with copied spacing arrays
        )doc",
        py::arg("inv_dx_face"), py::arg("inv_dy_face"), py::arg("inv_dz_face"),
        py::arg("inv_dx_cell"), py::arg("inv_dy_cell"), py::arg("inv_dz_cell")
    );

    m.def("update_velocity_nonuniform",
        [](const py::array_t<float, py::array::c_style>& p,
           py::array_t<float, py::array::c_style>& vx,
           py::array_t<float, py::array::c_style>& vy,
           py::array_t<float, py::array::c_style>& vz,
           const fdtd::NonuniformGridData& grid_data,
           float coeff_v_base) {

            auto shape = get_shape(p);
            check_shapes_match(shape, get_shape(vx), "p", "vx");
            check_shapes_match(shape, get_shape(vy), "p", "vy");
            check_shapes_match(shape, get_shape(vz), "p", "vz");

            if (!grid_data.is_valid(shape)) {
                throw std::runtime_error(
                    "NonuniformGridData is not valid for the given array shape"
                );
            }

            const float* p_ptr = get_float_ptr_const(p, "p");
            float* vx_ptr = get_float_ptr(vx, "vx");
            float* vy_ptr = get_float_ptr(vy, "vy");
            float* vz_ptr = get_float_ptr(vz, "vz");

            fdtd::update_velocity_nonuniform(
                p_ptr, vx_ptr, vy_ptr, vz_ptr, shape, grid_data, coeff_v_base
            );
        },
        R"doc(
        Update velocity fields for nonuniform grids.

        Args:
            p: Pressure field (nx, ny, nz) float32, read-only
            vx: X-velocity field (nx, ny, nz) float32, modified in place
            vy: Y-velocity field (nx, ny, nz) float32, modified in place
            vz: Z-velocity field (nx, ny, nz) float32, modified in place
            grid_data: Pre-computed NonuniformGridData
            coeff_v_base: Base velocity coefficient (-dt / rho)
        )doc",
        py::arg("p"), py::arg("vx"), py::arg("vy"), py::arg("vz"),
        py::arg("grid_data"), py::arg("coeff_v_base")
    );

    m.def("update_pressure_nonuniform",
        [](py::array_t<float, py::array::c_style>& p,
           const py::array_t<float, py::array::c_style>& vx,
           const py::array_t<float, py::array::c_style>& vy,
           const py::array_t<float, py::array::c_style>& vz,
           const py::array_t<bool, py::array::c_style>& geometry,
           const fdtd::NonuniformGridData& grid_data,
           float coeff_p_base) {

            auto shape = get_shape(p);
            check_shapes_match(shape, get_shape(vx), "p", "vx");
            check_shapes_match(shape, get_shape(vy), "p", "vy");
            check_shapes_match(shape, get_shape(vz), "p", "vz");
            check_shapes_match(shape, get_shape(geometry), "p", "geometry");

            if (!grid_data.is_valid(shape)) {
                throw std::runtime_error(
                    "NonuniformGridData is not valid for the given array shape"
                );
            }

            float* p_ptr = get_float_ptr(p, "p");
            const float* vx_ptr = get_float_ptr_const(vx, "vx");
            const float* vy_ptr = get_float_ptr_const(vy, "vy");
            const float* vz_ptr = get_float_ptr_const(vz, "vz");
            const bool* geo_ptr = get_bool_ptr_const(geometry, "geometry");

            fdtd::update_pressure_nonuniform(
                p_ptr, vx_ptr, vy_ptr, vz_ptr, geo_ptr, shape, grid_data, coeff_p_base
            );
        },
        R"doc(
        Update pressure field for nonuniform grids.

        Args:
            p: Pressure field (nx, ny, nz) float32, modified in place
            vx: X-velocity field (nx, ny, nz) float32, read-only
            vy: Y-velocity field (nx, ny, nz) float32, read-only
            vz: Z-velocity field (nx, ny, nz) float32, read-only
            geometry: Boolean mask (nx, ny, nz), True=air, False=solid
            grid_data: Pre-computed NonuniformGridData
            coeff_p_base: Base pressure coefficient (-rho * c^2 * dt)
        )doc",
        py::arg("p"), py::arg("vx"), py::arg("vy"), py::arg("vz"),
        py::arg("geometry"), py::arg("grid_data"), py::arg("coeff_p_base")
    );

    m.def("fdtd_step_nonuniform",
        [](py::array_t<float, py::array::c_style>& p,
           py::array_t<float, py::array::c_style>& vx,
           py::array_t<float, py::array::c_style>& vy,
           py::array_t<float, py::array::c_style>& vz,
           const py::array_t<bool, py::array::c_style>& geometry,
           const fdtd::NonuniformGridData& grid_data,
           float coeff_v_base, float coeff_p_base) {

            auto shape = get_shape(p);
            check_shapes_match(shape, get_shape(vx), "p", "vx");
            check_shapes_match(shape, get_shape(vy), "p", "vy");
            check_shapes_match(shape, get_shape(vz), "p", "vz");
            check_shapes_match(shape, get_shape(geometry), "p", "geometry");

            if (!grid_data.is_valid(shape)) {
                throw std::runtime_error(
                    "NonuniformGridData is not valid for the given array shape"
                );
            }

            float* p_ptr = get_float_ptr(p, "p");
            float* vx_ptr = get_float_ptr(vx, "vx");
            float* vy_ptr = get_float_ptr(vy, "vy");
            float* vz_ptr = get_float_ptr(vz, "vz");
            const bool* geo_ptr = get_bool_ptr_const(geometry, "geometry");

            fdtd::FDTDCoeffsBase coeffs{coeff_v_base, coeff_p_base};
            fdtd::fdtd_step_nonuniform(
                p_ptr, vx_ptr, vy_ptr, vz_ptr, geo_ptr, shape, grid_data, coeffs
            );
        },
        R"doc(
        Perform one FDTD timestep for nonuniform grids.

        NOTE: This fused function does NOT apply rigid boundaries between
        velocity and pressure updates. For proper energy conservation with
        rigid boundaries, use update_velocity_nonuniform() +
        apply_rigid_boundaries() + update_pressure_nonuniform() separately.

        Args:
            p: Pressure field (nx, ny, nz) float32, modified in place
            vx: X-velocity field (nx, ny, nz) float32, modified in place
            vy: Y-velocity field (nx, ny, nz) float32, modified in place
            vz: Z-velocity field (nx, ny, nz) float32, modified in place
            geometry: Boolean mask (nx, ny, nz), True=air, False=solid
            grid_data: Pre-computed NonuniformGridData
            coeff_v_base: Base velocity coefficient (-dt / rho)
            coeff_p_base: Base pressure coefficient (-rho * c^2 * dt)
        )doc",
        py::arg("p"), py::arg("vx"), py::arg("vy"), py::arg("vz"),
        py::arg("geometry"), py::arg("grid_data"),
        py::arg("coeff_v_base"), py::arg("coeff_p_base")
    );

    // ==========================================================================
    // ADE (Auxiliary Differential Equation) material kernels (Issue #139)
    // ==========================================================================

    py::class_<fdtd::DebyePoleCoeffs>(m, "DebyePoleCoeffs",
        "Coefficients for a Debye pole: J^{n+1} = alpha * J^n + beta * source")
        .def(py::init<>())
        .def_readwrite("alpha", &fdtd::DebyePoleCoeffs::alpha)
        .def_readwrite("beta", &fdtd::DebyePoleCoeffs::beta);

    py::class_<fdtd::LorentzPoleCoeffs>(m, "LorentzPoleCoeffs",
        "Coefficients for a Lorentz pole: J^{n+1} = a*J^n + b*J^{n-1} + d*source")
        .def(py::init<>())
        .def_readwrite("a", &fdtd::LorentzPoleCoeffs::a)
        .def_readwrite("b", &fdtd::LorentzPoleCoeffs::b)
        .def_readwrite("d", &fdtd::LorentzPoleCoeffs::d);

    py::class_<fdtd::ADEMaterialData::DebyePole>(m, "ADEDebyePole",
        "Debye pole data for ADE material")
        .def(py::init<>())
        .def_readwrite("material_id", &fdtd::ADEMaterialData::DebyePole::material_id)
        .def_readwrite("target", &fdtd::ADEMaterialData::DebyePole::target)
        .def_readwrite("coeffs", &fdtd::ADEMaterialData::DebyePole::coeffs)
        .def_readwrite("field_index", &fdtd::ADEMaterialData::DebyePole::field_index);

    py::class_<fdtd::ADEMaterialData::LorentzPole>(m, "ADELorentzPole",
        "Lorentz pole data for ADE material")
        .def(py::init<>())
        .def_readwrite("material_id", &fdtd::ADEMaterialData::LorentzPole::material_id)
        .def_readwrite("target", &fdtd::ADEMaterialData::LorentzPole::target)
        .def_readwrite("coeffs", &fdtd::ADEMaterialData::LorentzPole::coeffs)
        .def_readwrite("field_index", &fdtd::ADEMaterialData::LorentzPole::field_index);

    py::class_<fdtd::ADEMaterialData>(m, "ADEMaterialData",
        "Pre-computed material data for native ADE kernels")
        .def(py::init<>())
        .def_readwrite("rho_inf", &fdtd::ADEMaterialData::rho_inf)
        .def_readwrite("K_inf", &fdtd::ADEMaterialData::K_inf)
        .def_readwrite("debye_poles", &fdtd::ADEMaterialData::debye_poles)
        .def_readwrite("lorentz_poles", &fdtd::ADEMaterialData::lorentz_poles)
        .def_readwrite("n_debye_fields", &fdtd::ADEMaterialData::n_debye_fields)
        .def_readwrite("n_lorentz_fields", &fdtd::ADEMaterialData::n_lorentz_fields)
        .def("has_materials", &fdtd::ADEMaterialData::has_materials)
        .def("n_density_debye", &fdtd::ADEMaterialData::n_density_debye)
        .def("n_modulus_debye", &fdtd::ADEMaterialData::n_modulus_debye)
        .def("n_density_lorentz", &fdtd::ADEMaterialData::n_density_lorentz)
        .def("n_modulus_lorentz", &fdtd::ADEMaterialData::n_modulus_lorentz);

    m.def("update_ade_density_debye",
        [](py::array_t<float, py::array::c_style>& J,
           const py::array_t<float, py::array::c_style>& pressure,
           const py::array_t<uint8_t, py::array::c_style>& material_id,
           const fdtd::ADEMaterialData& material_data) {
            auto shape = get_shape(pressure);
            check_shapes_match(shape, get_shape(material_id), "pressure", "material_id");

            float* J_ptr = static_cast<float*>(J.request().ptr);
            const float* p_ptr = get_float_ptr_const(pressure, "pressure");
            const uint8_t* mat_ptr = static_cast<const uint8_t*>(material_id.request().ptr);

            fdtd::update_ade_density_debye(J_ptr, p_ptr, mat_ptr, material_data, shape);
        },
        R"doc(
        Update Debye auxiliary fields for density poles.

        Args:
            J: Debye auxiliary field array [n_fields * grid_size]
            pressure: Pressure field (nx, ny, nz)
            material_id: Per-cell material ID array (nx, ny, nz) uint8
            material_data: Pre-computed ADEMaterialData
        )doc",
        py::arg("J"), py::arg("pressure"), py::arg("material_id"),
        py::arg("material_data")
    );

    m.def("update_ade_modulus_debye",
        [](py::array_t<float, py::array::c_style>& J,
           const py::array_t<float, py::array::c_style>& divergence,
           const py::array_t<uint8_t, py::array::c_style>& material_id,
           const fdtd::ADEMaterialData& material_data) {
            auto shape = get_shape(divergence);
            float* J_ptr = static_cast<float*>(J.request().ptr);
            const float* div_ptr = get_float_ptr_const(divergence, "divergence");
            const uint8_t* mat_ptr = static_cast<const uint8_t*>(material_id.request().ptr);

            fdtd::update_ade_modulus_debye(J_ptr, div_ptr, mat_ptr, material_data, shape);
        },
        R"doc(
        Update Debye auxiliary fields for modulus poles.

        Args:
            J: Debye auxiliary field array [n_fields * grid_size]
            divergence: Velocity divergence field (nx, ny, nz)
            material_id: Per-cell material ID array (nx, ny, nz) uint8
            material_data: Pre-computed ADEMaterialData
        )doc",
        py::arg("J"), py::arg("divergence"), py::arg("material_id"),
        py::arg("material_data")
    );

    m.def("update_ade_density_lorentz",
        [](py::array_t<float, py::array::c_style>& J,
           py::array_t<float, py::array::c_style>& J_prev,
           const py::array_t<float, py::array::c_style>& pressure,
           const py::array_t<uint8_t, py::array::c_style>& material_id,
           const fdtd::ADEMaterialData& material_data) {
            auto shape = get_shape(pressure);
            float* J_ptr = static_cast<float*>(J.request().ptr);
            float* J_prev_ptr = static_cast<float*>(J_prev.request().ptr);
            const float* p_ptr = get_float_ptr_const(pressure, "pressure");
            const uint8_t* mat_ptr = static_cast<const uint8_t*>(material_id.request().ptr);

            fdtd::update_ade_density_lorentz(J_ptr, J_prev_ptr, p_ptr, mat_ptr, material_data, shape);
        },
        R"doc(
        Update Lorentz auxiliary fields for density poles.

        Args:
            J: Current Lorentz auxiliary field array
            J_prev: Previous Lorentz auxiliary field array
            pressure: Pressure field (nx, ny, nz)
            material_id: Per-cell material ID array
            material_data: Pre-computed ADEMaterialData
        )doc",
        py::arg("J"), py::arg("J_prev"), py::arg("pressure"),
        py::arg("material_id"), py::arg("material_data")
    );

    m.def("update_ade_modulus_lorentz",
        [](py::array_t<float, py::array::c_style>& J,
           py::array_t<float, py::array::c_style>& J_prev,
           const py::array_t<float, py::array::c_style>& divergence,
           const py::array_t<uint8_t, py::array::c_style>& material_id,
           const fdtd::ADEMaterialData& material_data) {
            auto shape = get_shape(divergence);
            float* J_ptr = static_cast<float*>(J.request().ptr);
            float* J_prev_ptr = static_cast<float*>(J_prev.request().ptr);
            const float* div_ptr = get_float_ptr_const(divergence, "divergence");
            const uint8_t* mat_ptr = static_cast<const uint8_t*>(material_id.request().ptr);

            fdtd::update_ade_modulus_lorentz(J_ptr, J_prev_ptr, div_ptr, mat_ptr, material_data, shape);
        },
        R"doc(
        Update Lorentz auxiliary fields for modulus poles.

        Args:
            J: Current Lorentz auxiliary field array
            J_prev: Previous Lorentz auxiliary field array
            divergence: Velocity divergence field
            material_id: Per-cell material ID array
            material_data: Pre-computed ADEMaterialData
        )doc",
        py::arg("J"), py::arg("J_prev"), py::arg("divergence"),
        py::arg("material_id"), py::arg("material_data")
    );

    m.def("apply_ade_velocity_correction",
        [](py::array_t<float, py::array::c_style>& vx,
           py::array_t<float, py::array::c_style>& vy,
           py::array_t<float, py::array::c_style>& vz,
           const py::array_t<float, py::array::c_style>& J_debye,
           const py::array_t<float, py::array::c_style>& J_lorentz,
           const py::array_t<uint8_t, py::array::c_style>& material_id,
           const fdtd::ADEMaterialData& material_data,
           float dt,
           float inv_dx) {
            auto shape = get_shape(vx);
            float* vx_ptr = get_float_ptr(vx, "vx");
            float* vy_ptr = get_float_ptr(vy, "vy");
            float* vz_ptr = get_float_ptr(vz, "vz");
            const float* J_debye_ptr = static_cast<const float*>(J_debye.request().ptr);
            const float* J_lorentz_ptr = static_cast<const float*>(J_lorentz.request().ptr);
            const uint8_t* mat_ptr = static_cast<const uint8_t*>(material_id.request().ptr);

            fdtd::apply_ade_velocity_correction(
                vx_ptr, vy_ptr, vz_ptr, J_debye_ptr, J_lorentz_ptr,
                mat_ptr, material_data, shape, dt, inv_dx
            );
        },
        R"doc(
        Apply ADE velocity corrections from density poles using gradient of J.

        For density dispersion, applies the correction:
          vx[i] += -dt/ρ_inf * (J[i+1] - J[i]) / dx
          vy[j] += -dt/ρ_inf * (J[j+1] - J[j]) / dy
          vz[k] += -dt/ρ_inf * (J[k+1] - J[k]) / dz

        Only applies at faces where BOTH adjacent cells are in the material.

        Args:
            vx: X-velocity field (modified in place)
            vy: Y-velocity field (modified in place)
            vz: Z-velocity field (modified in place)
            J_debye: Debye auxiliary fields
            J_lorentz: Lorentz auxiliary fields
            material_id: Per-cell material ID array
            material_data: Pre-computed ADEMaterialData
            dt: Timestep
            inv_dx: Inverse grid spacing (1/dx)
        )doc",
        py::arg("vx"), py::arg("vy"), py::arg("vz"),
        py::arg("J_debye"), py::arg("J_lorentz"),
        py::arg("material_id"), py::arg("material_data"), py::arg("dt"),
        py::arg("inv_dx")
    );

    m.def("apply_ade_pressure_correction",
        [](py::array_t<float, py::array::c_style>& p,
           const py::array_t<float, py::array::c_style>& J_debye,
           const py::array_t<float, py::array::c_style>& J_lorentz,
           const py::array_t<uint8_t, py::array::c_style>& material_id,
           const fdtd::ADEMaterialData& material_data,
           float dt) {
            auto shape = get_shape(p);
            float* p_ptr = get_float_ptr(p, "p");
            const float* J_debye_ptr = static_cast<const float*>(J_debye.request().ptr);
            const float* J_lorentz_ptr = static_cast<const float*>(J_lorentz.request().ptr);
            const uint8_t* mat_ptr = static_cast<const uint8_t*>(material_id.request().ptr);

            fdtd::apply_ade_pressure_correction(
                p_ptr, J_debye_ptr, J_lorentz_ptr, mat_ptr, material_data, shape, dt
            );
        },
        R"doc(
        Apply ADE pressure corrections from modulus poles.

        Args:
            p: Pressure field (modified in place)
            J_debye: Debye auxiliary fields
            J_lorentz: Lorentz auxiliary fields
            material_id: Per-cell material ID array
            material_data: Pre-computed ADEMaterialData
            dt: Timestep
        )doc",
        py::arg("p"), py::arg("J_debye"), py::arg("J_lorentz"),
        py::arg("material_id"), py::arg("material_data"), py::arg("dt")
    );

    m.def("compute_divergence",
        [](py::array_t<float, py::array::c_style>& divergence,
           const py::array_t<float, py::array::c_style>& vx,
           const py::array_t<float, py::array::c_style>& vy,
           const py::array_t<float, py::array::c_style>& vz,
           float inv_dx) {
            auto shape = get_shape(divergence);
            float* div_ptr = get_float_ptr(divergence, "divergence");
            const float* vx_ptr = get_float_ptr_const(vx, "vx");
            const float* vy_ptr = get_float_ptr_const(vy, "vy");
            const float* vz_ptr = get_float_ptr_const(vz, "vz");

            fdtd::compute_divergence(div_ptr, vx_ptr, vy_ptr, vz_ptr, shape, inv_dx);
        },
        R"doc(
        Compute velocity divergence field.

        Args:
            divergence: Output divergence field (modified in place)
            vx: X-velocity field
            vy: Y-velocity field
            vz: Z-velocity field
            inv_dx: Inverse grid spacing (1/dx)
        )doc",
        py::arg("divergence"), py::arg("vx"), py::arg("vy"), py::arg("vz"),
        py::arg("inv_dx")
    );

    m.def("compute_divergence_nonuniform",
        [](py::array_t<float, py::array::c_style>& divergence,
           const py::array_t<float, py::array::c_style>& vx,
           const py::array_t<float, py::array::c_style>& vy,
           const py::array_t<float, py::array::c_style>& vz,
           const fdtd::NonuniformGridData& grid_data) {
            auto shape = get_shape(divergence);

            if (!grid_data.is_valid(shape)) {
                throw std::runtime_error(
                    "NonuniformGridData is not valid for the given array shape"
                );
            }

            float* div_ptr = get_float_ptr(divergence, "divergence");
            const float* vx_ptr = get_float_ptr_const(vx, "vx");
            const float* vy_ptr = get_float_ptr_const(vy, "vy");
            const float* vz_ptr = get_float_ptr_const(vz, "vz");

            fdtd::compute_divergence_nonuniform(
                div_ptr, vx_ptr, vy_ptr, vz_ptr, shape, grid_data
            );
        },
        R"doc(
        Compute velocity divergence field for nonuniform grids.

        Args:
            divergence: Output divergence field (modified in place)
            vx: X-velocity field
            vy: Y-velocity field
            vz: Z-velocity field
            grid_data: Pre-computed NonuniformGridData
        )doc",
        py::arg("divergence"), py::arg("vx"), py::arg("vy"), py::arg("vz"),
        py::arg("grid_data")
    );
}

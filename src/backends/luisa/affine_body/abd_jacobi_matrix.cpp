#include <affine_body/abd_jacobi_matrix.h>
/********************************************************************
 * @file affine_body/abd_jacobi_matrix.cpp
 * @brief T6.1a — the host-facing (Eigen) surface of ABDJacobi /
 *        ABDJacobiDyadicMass.
 *
 * The reference's abd_jacobi_matrix.cu is `UIPC_GENERIC`, so on the CUDA side the
 * same text serves host and device.  On the LC side the device bodies must live in
 * a header (a DSL call needs the body at the call site, cookbook §7.16 rule 1):
 * they are in details/abd_jacobi_matrix.inl.  This translation unit is the
 * reference's ".cu host half": the Eigen-typed convenience functions that the
 * ported host code (and the unit test) call.  None of them re-implements the
 * arithmetic -- each forwards to the factored routine with `S = Float`, so a host
 * result and a device result can only differ through the hardware:
 *
 *   reference                               port
 *   --------------------------------------  ---------------------------------------------
 *   j * q / j.point_x(q)                    abd_jacobi_point_x_eigen(j, q)
 *   j.to_mat()                              abd_jacobi_to_mat_eigen(j)
 *   ABDJacobiDyadicMass::from_dyadic_mass   make_abd_dyadic_mass(m, mx, mxx)
 *   M.to_mat()                              abd_dyadic_mass_to_mat_eigen(M)
 *   M.mass()                                abd_dyadic_mass_mass(M)
 *   M.center_of_mass()                      abd_dyadic_mass_center_of_mass_eigen(M)
 *   M.inertia_tensor_cm()                   abd_dyadic_mass_inertia_tensor_cm_eigen(M)
 *********************************************************************/
namespace uipc::backend::luisa
{
namespace
{
[[nodiscard]] Matrix3x12 to_eigen_mat3x12(const LMatrix3x12& m) noexcept
{
    Matrix3x12 r;
    for(int c = 0; c < 12; ++c)
        for(int i = 0; i < 3; ++i)
            r(i, c) = m(i, c);
    return r;
}
}  // namespace

ABDJacobiDyadicMass make_abd_dyadic_mass(Float            m,
                                         const Vector3&   m_x_bar,
                                         const Matrix3x3& m_x_bar_x_bar) noexcept
{
    return to_abd_dyadic_mass(m, m_x_bar, m_x_bar_x_bar);
}

Vector3 abd_jacobi_point_x_eigen(const ABDJacobi& j, const Vector12& q) noexcept
{
    return to_eigen(abd_jacobi_point<Float>(to_generic(j), to_luisa(q)));
}

Matrix3x12 abd_jacobi_to_mat_eigen(const ABDJacobi& j) noexcept
{
    return to_eigen_mat3x12(abd_jacobi_to_mat<Float>(to_generic(j)));
}

Matrix12x12 abd_dyadic_mass_to_mat_eigen(const ABDJacobiDyadicMass& m) noexcept
{
    return to_eigen(abd_dyadic_mass_to_mat<Float>(to_generic(m)));
}

Float abd_dyadic_mass_mass(const ABDJacobiDyadicMass& m) noexcept
{
    return m.m_mass;
}

Vector3 abd_dyadic_mass_center_of_mass_eigen(const ABDJacobiDyadicMass& m) noexcept
{
    return to_eigen(abd_dyadic_mass_center_of_mass<Float>(to_generic(m)));
}

Matrix3x3 abd_dyadic_mass_inertia_tensor_cm_eigen(const ABDJacobiDyadicMass& m) noexcept
{
    return to_eigen(abd_dyadic_mass_inertia_tensor_cm<Float>(to_generic(m)));
}
}  // namespace uipc::backend::luisa

#include <affine_body/abd_energy.h>
/********************************************************************
 * @file affine_body/abd_energy.cpp
 * @brief T6.1a — the host-facing (Eigen) half of affine_body/abd_energy.h.
 *
 * The reference's abd_energy.cu is UIPC_GENERIC: one text for host and device.
 * On the LC side the device bodies must be visible at the call site, so they live
 * in the header (see its scope note).  This translation unit provides the
 * reference's *Eigen-typed* signatures as forwards to the template with
 * `S = Float`, so
 *
 *   - a host caller reads exactly like the reference
 *     (`shape_energy_hessian(q, ddVdda1, ...)` with `Vector12`/`Matrix3x3`), and
 *   - the host and the device values come from the same arithmetic body, which is
 *     what the T6.1a test's closed-form / finite-difference oracle and its
 *     device-vs-host case compare against.
 *********************************************************************/
namespace uipc::backend::luisa
{
namespace
{
/// LMat<Float,3,3> -> Eigen (identical bytes, explicit element-wise conversion).
[[nodiscard]] Matrix3x3 to_eigen_mat3(const LMatrix3x3& m) noexcept
{
    Matrix3x3 r;
    for(int c = 0; c < 3; ++c)
        for(int i = 0; i < 3; ++i)
            r(i, c) = m(i, c);
    return r;
}

[[nodiscard]] LMatrix3x3 to_luisa_mat3(const Matrix3x3& m) noexcept
{
    LMatrix3x3 r;
    for(int c = 0; c < 3; ++c)
        for(int i = 0; i < 3; ++i)
            r(i, c) = m(i, c);
    return r;
}

[[nodiscard]] Matrix9x9 to_eigen_mat9(const lps::LMat<Float, 9, 9>& m) noexcept
{
    Matrix9x9 r;
    for(int c = 0; c < 9; ++c)
        for(int i = 0; i < 9; ++i)
            r(i, c) = m(i, c);
    return r;
}

[[nodiscard]] Vector9 to_eigen_vec9(const lps::LVec<Float, 9>& v) noexcept
{
    Vector9 r;
    for(int i = 0; i < 9; ++i)
        r(i) = v(i);
    return r;
}
}  // namespace

Float shape_energy(const Vector12& q) noexcept
{
    return shape_energy<Float>(to_luisa(q));
}

Vector9 shape_energy_gradient(const Vector12& q) noexcept
{
    return to_eigen_vec9(shape_energy_gradient<Float>(to_luisa(q)));
}

void shape_energy_hessian(const Vector12& q,
                          Matrix3x3&      ddVdda1,
                          Matrix3x3&      ddVdda2,
                          Matrix3x3&      ddVdda3,
                          Matrix3x3&      ddVda1da2,
                          Matrix3x3&      ddVda1da3,
                          Matrix3x3&      ddVda2da3) noexcept
{
    LMatrix3x3 h1, h2, h3, h12, h13, h23;
    shape_energy_hessian<Float>(
        to_luisa(q), h1, h2, h3, h12, h13, h23);
    ddVdda1    = to_eigen_mat3(h1);
    ddVdda2    = to_eigen_mat3(h2);
    ddVdda3    = to_eigen_mat3(h3);
    ddVda1da2  = to_eigen_mat3(h12);
    ddVda1da3  = to_eigen_mat3(h13);
    ddVda2da3  = to_eigen_mat3(h23);
}

Matrix9x9 shape_energy_hessian(const Vector12& q) noexcept
{
    return to_eigen_mat9(shape_energy_hessian<Float>(to_luisa(q)));
}
}  // namespace uipc::backend::luisa

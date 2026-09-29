#include <affine_body/utils.h>
/********************************************************************
 * @file affine_body/utils.cpp
 * @brief T6.1a — the host-only (Eigen) half of affine_body/utils.h.
 *
 * The reference's utils.cu is UIPC_GENERIC, so on the CUDA side the transform
 * conversions run on the host and on the device.  Nothing in this backend needs
 * them *in a kernel*: the host build-up of the ABD geometry
 * (affine_body_dynamics.cpp: `_build_geometry_on_host`/`write_scene`) and the
 * state accessor are host code, so the port keeps the Eigen-typed spelling and
 * leaves the DSL-side conversions (q_to_A, A_to_q, ...) in the header.  Reported
 * as a deviation in the T6.1a row.
 *********************************************************************/
namespace uipc::backend::luisa
{
// reference: UIPC_GENERIC Matrix4x4 q_to_transform(const Vector12& q)
Matrix4x4 q_to_transform(const Vector12& q) noexcept
{
    Matrix4x4 trans;
    // translation
    trans.block<3, 1>(0, 3) = q.segment<3>(0);
    // rotation
    trans.block<1, 3>(0, 0) = q.segment<3>(3).transpose();
    trans.block<1, 3>(1, 0) = q.segment<3>(6).transpose();
    trans.block<1, 3>(2, 0) = q.segment<3>(9).transpose();
    // last row
    trans.row(3) = Vector4{0, 0, 0, 1};
    return trans;
}

// reference: UIPC_GENERIC Vector12 transform_to_q(const Matrix4x4& trans)
Vector12 transform_to_q(const Matrix4x4& trans) noexcept
{
    Vector12 q;
    q.segment<3>(0) = trans.block<3, 1>(0, 3);
    q.segment<3>(3) = trans.block<1, 3>(0, 0).transpose();
    q.segment<3>(6) = trans.block<1, 3>(1, 0).transpose();
    q.segment<3>(9) = trans.block<1, 3>(2, 0).transpose();
    return q;
}

// reference: UIPC_GENERIC Matrix4x4 q_v_to_transform_v(const Vector12& q)
Matrix4x4 q_v_to_transform_v(const Vector12& q) noexcept
{
    Matrix4x4 trans;
    // translation
    trans.block<3, 1>(0, 3) = q.segment<3>(0);
    // rotation
    trans.block<1, 3>(0, 0) = q.segment<3>(3).transpose();
    trans.block<1, 3>(1, 0) = q.segment<3>(6).transpose();
    trans.block<1, 3>(2, 0) = q.segment<3>(9).transpose();
    // last row fill zero
    trans.row(3) = Vector4::Zero();
    return trans;
}

// reference: `transform_v_to_q_v` is literally `transform_to_q`
Vector12 transform_v_to_q_v(const Matrix4x4& transform_v) noexcept
{
    return transform_to_q(transform_v);
}
}  // namespace uipc::backend::luisa

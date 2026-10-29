#include <affine_body/affine_body_state_accessor_feature.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <joint_dof_system/global_joint_dof_manager.h>
#include <uipc/builtin/attribute_name.h>
#include <affine_body/utils.h>

/********************************************************************
 * T6.1a — AffineBodyStateAccessorFeatureOverrider (reference:
 * src/backends/cuda/affine_body/affine_body_state_accessor_feature.cu).
 *
 * Two deviations, both forced by the port's host/device split (the same call the
 * FEM sibling FiniteElementStateAccessorFeatureOverrider made in T5.1):
 *
 * 1. `do_copy_transform_to`/`do_copy_velocity_to` have no kernel.  The reference
 *    reinterprets `backend::BufferView::handle()` as a *device* pointer and writes
 *    it from a kernel; LC does not hand out device addresses, and the pybind
 *    frontend's BufferView is a host buffer, so the port reads `q` back to the
 *    host, converts with the host Eigen `q_to_transform`/`q_v_to_transform_v` and
 *    writes Matrix4x4 elements into the caller's host buffer.  The produced bytes
 *    are identical (Matrix4x4 is what the reference writes too), the destination
 *    is simply reached from the host.
 *
 * 2. `copy_from`/`copy_to` stage through explicit `LVector12` vectors instead of
 *    aliasing the `Vector12` array: Eigen's Vector12 and LVector12 are
 *    byte-identical but unrelated types, and `LBufferView::copy_from` is typed.
 *    Every element goes through `to_luisa`/`to_eigen` (affine_body/type_define.h).
 *
 * `std::ranges::transform(..., transform_to_q)` in the reference relies on the
 * function name resolving to one overload; the port has the host Eigen overload
 * and the DSL template `q_to_transform<S>`, so the call sites use an explicit
 * lambda (still a plain forwarder).
 *********************************************************************/
namespace uipc::backend::luisa
{
AffineBodyStateAccessorFeatureOverrider::AffineBodyStateAccessorFeatureOverrider(
    AffineBodyDynamics&     abd,
    AffineBodyVertexReporter& vertex_reporter,
    GlobalJointDofManager&  joint_dof_manager)
    : m_abd{abd}
    , m_vertex_reporter{vertex_reporter}
    , m_joint_dof_manager{joint_dof_manager}
{
}

SizeT AffineBodyStateAccessorFeatureOverrider::get_body_count()
{
    return m_abd.qs().size();
}

void AffineBodyStateAccessorFeatureOverrider::do_copy_from(const geometry::SimplicialComplex& state_geo)
{
    auto q_offset_attr = state_geo.meta().find<IndexT>(builtin::backend_abd_body_offset);
    UIPC_ASSERT(q_offset_attr, "Cannot find `backend_abd_body_offset` on State Geometry, why can it happen?");
    auto q_offset = q_offset_attr->view()[0];
    auto q_count  = state_geo.instances().size();

    // 1. Transform
    auto trans = state_geo.instances().find<Matrix4x4>(builtin::transform);
    m_buffer.resize(q_count);
    if(trans)
    {
        auto trans_view = trans->view();
        std::ranges::transform(trans_view,
                               m_buffer.begin(),
                               [](const Matrix4x4& m) { return transform_to_q(m); });
        // stage the Eigen->device conversion (byte-identical element types)
        std::vector<LVector12> staging(q_count);
        std::ranges::transform(m_buffer, staging.begin(), [](const Vector12& q) { return to_luisa(q); });
        auto q_subview = m_abd.m_impl.body_id_to_q.view(q_offset, q_count);
        q_subview.copy_from(span<const LVector12>{staging});
    }

    // 2. Velocity
    auto vel = state_geo.instances().find<Matrix4x4>(builtin::velocity);
    if(vel)
    {
        auto vel_view = vel->view();
        std::ranges::transform(vel_view,
                               m_buffer.begin(),
                               [](const Matrix4x4& v) { return transform_v_to_q_v(v); });
        std::vector<LVector12> staging(q_count);
        std::ranges::transform(m_buffer, staging.begin(), [](const Vector12& q) { return to_luisa(q); });
        auto q_v_subview = m_abd.m_impl.body_id_to_q_v.view(q_offset, q_count);
        q_v_subview.copy_from(span<const LVector12>{staging});
    }

    // request the vertex reporter to update attributes
    m_vertex_reporter.request_attribute_update();
    // re-sync persistent per-joint DOF state (e.g. revolute current_angles)
    // synchronously so it stays consistent with the freshly written body q.
    m_joint_dof_manager.update_dof_attributes();
}

void AffineBodyStateAccessorFeatureOverrider::do_copy_to(geometry::SimplicialComplex& state_geo)
{
    auto q_offset_attr = state_geo.meta().find<IndexT>(builtin::backend_abd_body_offset);
    UIPC_ASSERT(q_offset_attr, "Cannot find `backend_abd_body_offset` on State Geometry, why can it happen?");
    auto q_offset = q_offset_attr->view()[0];
    auto q_count  = state_geo.instances().size();

    // 1. Transform
    auto trans = state_geo.instances().find<Matrix4x4>(builtin::transform);
    m_buffer.resize(q_count);
    if(trans)
    {
        auto trans_view = view(*trans);
        auto q_subview  = m_abd.qs().subview(q_offset, q_count);
        std::vector<LVector12> staging(q_count);
        q_subview.copy_to(span<LVector12>{staging});
        lud::LStream::current().synchronize();
        std::ranges::transform(staging, m_buffer.begin(), [](const LVector12& q) { return to_eigen(q); });
        std::ranges::transform(m_buffer,
                               trans_view.begin(),
                               [](const Vector12& q) { return q_to_transform(q); });
    }

    // 2. Velocity
    auto vel = state_geo.instances().find<Matrix4x4>(builtin::velocity);
    if(vel)
    {
        auto vel_view    = view(*vel);
        auto q_v_subview = m_abd.q_vs().subview(q_offset, q_count);
        std::vector<LVector12> staging(q_count);
        q_v_subview.copy_to(span<LVector12>{staging});
        lud::LStream::current().synchronize();
        std::ranges::transform(staging, m_buffer.begin(), [](const LVector12& q) { return to_eigen(q); });
        std::ranges::transform(m_buffer,
                               vel_view.begin(),
                               [](const Vector12& q) { return q_v_to_transform_v(q); });
    }
}

void AffineBodyStateAccessorFeatureOverrider::do_copy_transform_to(
    backend::BufferView buffer_view,
    IndexT              body_offset,
    SizeT               body_count)
{
    if(body_count == 0u)
    {
        return;
    }

    auto q_view    = m_abd.qs();
    auto q_subview = q_view.subview(body_offset, body_count);

    std::vector<LVector12> staging(body_count);
    q_subview.copy_to(span<LVector12>{staging});
    lud::LStream::current().synchronize();

    // the caller's buffer is a host buffer (see the file header)
    auto* dst_ptr =
        reinterpret_cast<Matrix4x4*>(buffer_view.handle()) + buffer_view.offset();

    for(SizeT i = 0; i < body_count; ++i)
    {
        dst_ptr[i] = q_to_transform(to_eigen(staging[i]));
    }
}

void AffineBodyStateAccessorFeatureOverrider::do_copy_velocity_to(
    backend::BufferView buffer_view,
    IndexT              body_offset,
    SizeT               body_count)
{
    if(body_count == 0u)
    {
        return;
    }

    auto q_v_view    = m_abd.q_vs();
    auto q_v_subview = q_v_view.subview(body_offset, body_count);

    std::vector<LVector12> staging(body_count);
    q_v_subview.copy_to(span<LVector12>{staging});
    lud::LStream::current().synchronize();

    // the caller's buffer is a host buffer (see the file header)
    auto* dst_ptr =
        reinterpret_cast<Matrix4x4*>(buffer_view.handle()) + buffer_view.offset();

    for(SizeT i = 0; i < body_count; ++i)
    {
        dst_ptr[i] = q_v_to_transform_v(to_eigen(staging[i]));
    }
}
}  // namespace uipc::backend::luisa

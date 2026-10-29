#include <finite_element/finite_element_state_accessor_feature.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <uipc/builtin/attribute_name.h>

namespace uipc::backend::luisa
{
namespace
{
/// The reference's `copy_to_buffer_view`:
///   auto* dst_ptr = reinterpret_cast<Vector3*>(buffer_view.handle()) + buffer_view.offset();
///   cuda_tool::BufferView<Vector3> dst_muda_view{dst_ptr, vertex_count};
///   dst_muda_view.copy_from(src_subview);
///
/// i.e. `backend::BufferView::handle()` is a *raw pointer* that the backend
/// performs a device-to-device copy into. LuisaCompute does not expose a raw
/// pointer to a device buffer (its allocation is private to the backend), so the
/// portable interpretation is the one the pybind frontend already uses: the
/// handle is a **host** address owned by the caller (e.g. a numpy array), and the
/// port does the read-back explicitly (device -> host copy + wait). LVector3 and
/// Vector3 are byte-identical (static_assert in type_define.h), so writing
/// LVector3 elements into that host buffer yields exactly the Vector3 bytes the
/// reference produces. Reported as a T5.1 deviation.
void copy_to_buffer_view(lud::LCBufferView<LVector3> src_view,
                         backend::BufferView         buffer_view,
                         IndexT                      vertex_offset,
                         SizeT                       vertex_count)
{
    if(vertex_count == 0u)
    {
        return;
    }
    auto  src_subview = src_view.subview(static_cast<SizeT>(vertex_offset), vertex_count);
    auto* dst_ptr =
        reinterpret_cast<Vector3*>(buffer_view.handle()) + buffer_view.offset();
    src_subview.copy_to(
        span<LVector3>{reinterpret_cast<LVector3*>(dst_ptr), vertex_count});
    lud::LStream::current().synchronize();
}

/// Eigen (frontend attribute view) -> device buffer, element-wise.
std::vector<LVector3> to_device_positions(span<const Vector3> src)
{
    std::vector<LVector3> dst(src.size());
    for(size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = to_luisa(src[i]);
    }
    return dst;
}
}  // namespace

FiniteElementStateAccessorFeatureOverrider::FiniteElementStateAccessorFeatureOverrider(
    FiniteElementMethod& fem, FiniteElementVertexReporter& vertex_reporter)
    : m_fem{fem}
    , m_vertex_reporter{vertex_reporter}
{
}

SizeT FiniteElementStateAccessorFeatureOverrider::get_vertex_count()
{
    return m_fem.xs().size();
}

void FiniteElementStateAccessorFeatureOverrider::do_copy_from(const geometry::SimplicialComplex& state_geo)
{
    auto v_offset_attr = state_geo.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
    UIPC_ASSERT(v_offset_attr, "Cannot find `backend_fem_vertex_offset` on State Geometry, why can it happen?");
    auto v_offset = v_offset_attr->view()[0];
    auto v_count  = state_geo.vertices().size();


    // 1. Position
    auto pos = state_geo.vertices().find<Vector3>(builtin::position);
    if(pos)
    {
        auto pos_view  = pos->view();
        auto x_subview = m_fem.m_impl.xs.view(v_offset, v_count);
        auto tmp       = to_device_positions(pos_view);
        x_subview.copy_from(span<const LVector3>{tmp});
    }

    // 2. Velocity
    auto vel = state_geo.vertices().find<Vector3>(builtin::velocity);
    if(vel)
    {
        auto vel_view  = vel->view();
        auto v_subview = m_fem.m_impl.vs.view(v_offset, v_count);
        auto tmp       = to_device_positions(vel_view);
        v_subview.copy_from(span<const LVector3>{tmp});
    }

    // request the vertex reporter to update attributes
    m_vertex_reporter.request_attribute_update();
}

void FiniteElementStateAccessorFeatureOverrider::do_copy_to(geometry::SimplicialComplex& state_geo)
{
    auto v_offset_attr = state_geo.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
    UIPC_ASSERT(v_offset_attr, "Cannot find `backend_fem_vertex_offset` on State Geometry, why can it happen?");
    auto v_offset = v_offset_attr->view()[0];
    auto v_count  = state_geo.vertices().size();

    // 1. Position
    auto pos = state_geo.vertices().find<Vector3>(builtin::position);
    if(pos)
    {
        auto pos_view  = view(*pos);
        auto x_subview = m_fem.m_impl.xs.view(v_offset, v_count);
        std::vector<LVector3> tmp(v_count);
        x_subview.copy_to(span<LVector3>{tmp});
        lud::LStream::current().synchronize();
        for(size_t i = 0; i < tmp.size(); ++i)
        {
            pos_view[i] = to_eigen(tmp[i]);
        }
    }

    // 2. Velocity
    auto vel = state_geo.vertices().find<Vector3>(builtin::velocity);
    if(vel)
    {
        auto vel_view  = view(*vel);
        auto v_subview = m_fem.m_impl.vs.view(v_offset, v_count);
        std::vector<LVector3> tmp(v_count);
        v_subview.copy_to(span<LVector3>{tmp});
        lud::LStream::current().synchronize();
        for(size_t i = 0; i < tmp.size(); ++i)
        {
            vel_view[i] = to_eigen(tmp[i]);
        }
    }
}

void FiniteElementStateAccessorFeatureOverrider::do_copy_position_to(
    backend::BufferView buffer_view, IndexT vertex_offset, SizeT vertex_count)
{
    copy_to_buffer_view(m_fem.xs(), buffer_view, vertex_offset, vertex_count);
}

void FiniteElementStateAccessorFeatureOverrider::do_copy_velocity_to(
    backend::BufferView buffer_view, IndexT vertex_offset, SizeT vertex_count)
{
    copy_to_buffer_view(m_fem.vs(), buffer_view, vertex_offset, vertex_count);
}
}  // namespace uipc::backend::luisa

#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_body_reporter.h>
#include <uipc/builtin/attribute_name.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>

namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

namespace
{
/// The reference's `HalfPlaneVertexReporter_report_attributes_kernel`: one vertex
/// per half-plane instance, so
///   coindices(i)          = i                  (local index inside the slice)
///   positions(i)          = src_positions(i)   (a byte copy of the flattened P)
///   vertex_body_ids(i)    = body_offset + i    (each vertex *is* a body)
auto& half_plane_vertex_attributes_kernel()
{
    // Compiled once and deliberately leaked (cookbook §7.10 trap 1).
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<IndexT>,
                                                lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<IndexT>,
                                                IndexT,
                                                uint32_t>(
        [](lcrt::BufferVar<IndexT> coindices,
           lcrt::BufferVar<LVector3> dst_pos,
           lcrt::BufferVar<LVector3> src_pos,
           lcrt::BufferVar<IndexT>   dst_vertex_body_ids,
           lcrt::Var<IndexT>         body_offset,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto local = lcrt::cast<IndexT>(i);
                coindices->write(i, local);
                dst_pos->write(i, src_pos->read(i));
                dst_vertex_body_ids->write(i, body_offset + local);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(HalfPlaneVertexReporter);

constexpr U64 HalfPlaneVertexReporterUID = 2;

void HalfPlaneVertexReporter::do_build(BuildInfo& info)
{
    m_impl.half_plane    = &require<HalfPlane>();
    m_impl.body_reporter = &require<HalfPlaneBodyReporter>();
}

void HalfPlaneVertexReporter::Impl::report_count(GlobalVertexManager::VertexCountInfo& info)
{
    info.count(half_plane->m_impl.h_positions.size());
}

void HalfPlaneVertexReporter::Impl::report_attributes(GlobalVertexManager::VertexAttributeInfo& info)
{
    // fill the coindices for later use
    auto n = static_cast<uint32_t>(info.coindices().size());

    if(n > 0)
    {
        auto* k = &half_plane_vertex_attributes_kernel();
        frame_stream().record((*k)(info.coindices(),
                                   info.positions(),
                                   half_plane->m_impl.positions.cview(),
                                   info.body_ids(),
                                   body_reporter->body_offset(),
                                   n)
                                  .dispatch(n));
    }

    info.contact_element_ids().copy_from(span<const IndexT>{half_plane->m_impl.h_contact_ids},
                                         &frame_stream());
    info.subscene_element_ids().copy_from(span<const IndexT>{half_plane->m_impl.h_subscene_ids},
                                          &frame_stream());
}

void HalfPlaneVertexReporter::Impl::report_displacements(GlobalVertexManager::VertexDisplacementInfo& info)
{
    // Now, we only support fixed half plane
    info.displacements().fill(LVector3{}, &frame_stream());
}

void HalfPlaneVertexReporter::do_report_count(GlobalVertexManager::VertexCountInfo& info)
{
    m_impl.report_count(info);
}

void HalfPlaneVertexReporter::do_report_attributes(GlobalVertexManager::VertexAttributeInfo& info)
{
    m_impl.report_attributes(info);


    auto global_offset = info.coindices().offset();

    auto geo_slots = world().scene().geometries();

    // add global vertex offset attribute
    m_impl.half_plane->for_each(  //
        geo_slots,
        [&](const HalfPlane::ForEachInfo& I, geometry::ImplicitGeometry& ig)
        {
            auto gvo = ig.meta().find<IndexT>(builtin::global_vertex_offset);
            if(!gvo)
            {
                gvo = ig.meta().create<IndexT>(builtin::global_vertex_offset);
            }

            // [global-vertex-offset] = [vertex-offset-in-halfplane-system] + [halfplane-system-vertex-offset]
            view(*gvo)[0] = I.geo_info().vertex_offset + global_offset;
        });
}

void HalfPlaneVertexReporter::do_report_displacements(GlobalVertexManager::VertexDisplacementInfo& info)
{
    m_impl.report_displacements(info);
}

U64 HalfPlaneVertexReporter::get_uid() const noexcept
{
    return HalfPlaneVertexReporterUID;
}
}  // namespace uipc::backend::luisa

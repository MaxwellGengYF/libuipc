#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_body_reporter.h>
#include <uipc/builtin/attribute_name.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
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
    using namespace luisa::compute;

    // fill the coindices for later use
    auto N = info.coindices().size();

    auto& device = half_plane->engine().device();
    auto& stream = half_plane->engine().stream();

    Kernel1D report_kernel = [&](BufferVar<IndexT> coindices,
                                  BufferVar<Vector3> dst_pos,
                                  BufferVar<Vector3> src_pos,
                                  BufferVar<IndexT> dst_vertex_body_ids,
                                  Var<IndexT> body_offset) noexcept
    {
        auto i = dispatch_id().x;
        $if(i < N)
        {
            coindices.write(i, cast<IndexT>(i));
            dst_pos.write(i, src_pos.read(i));
            // each vertex corresponds to a body
            // so we can use the body offset + i to get global body id
            dst_vertex_body_ids.write(i, body_offset + i);
        };
    };

    auto shader = device.compile(report_kernel);
    stream << shader(info.coindices(),
                     info.positions(),
                     half_plane->m_impl.positions.view(),
                     info.body_ids(),
                     body_reporter->body_offset()).dispatch(N);

    stream << info.contact_element_ids().copy_from(half_plane->m_impl.h_contact_ids.data())
           << info.subscene_element_ids().copy_from(half_plane->m_impl.h_subscene_ids.data());
}

void HalfPlaneVertexReporter::Impl::report_displacements(GlobalVertexManager::VertexDisplacementInfo& info)
{
    // Now, we only support fixed half plane
    auto& stream = half_plane->engine().stream();
    stream << info.displacements().fill(Vector3::Zero());
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
    m_impl.half_plane->for_each(
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

luisa::ulong HalfPlaneVertexReporter::get_uid() const noexcept
{
    return HalfPlaneVertexReporterUID;
}
}  // namespace uipc::backend::luisa

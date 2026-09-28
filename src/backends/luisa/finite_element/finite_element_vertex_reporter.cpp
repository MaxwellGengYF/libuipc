#include <finite_element/finite_element_vertex_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <kernel_cout.h>
#include <finite_element/finite_element_body_reporter.h>
#include <uipc/builtin/attribute_name.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
namespace
{
namespace lla_ = luisa_device::linalg;

/// The reference's `FiniteElementVertexReporter_init_attributes_kernel`.
/// `dst_body_ids(i) += body_offset` (a device read-modify-write on the element the
/// vertex manager pre-filled with the *frame-local* body index) becomes an
/// explicit read + write; one thread owns each index, so no atomic is needed.
auto& init_attributes_kernel()
{
    static auto& k = lla_::detail::cached_kernel<lcrt::Buffer<IndexT>,
                                                 lcrt::Buffer<LVector3>,
                                                 lcrt::Buffer<LVector3>,
                                                 lcrt::Buffer<LVector3>,
                                                 int32_t,
                                                 lcrt::Buffer<IndexT>,
                                                 lcrt::Buffer<LVector3>,
                                                 uint32_t>(
        [](lcrt::BufferVar<IndexT>   coindices,
           lcrt::BufferVar<LVector3> src_pos,
           lcrt::BufferVar<LVector3> dst_pos,
           lcrt::BufferVar<LVector3> src_rest_pos,
           lcrt::Var<int32_t>        body_offset,
           lcrt::BufferVar<IndexT>   dst_body_ids,
           lcrt::BufferVar<LVector3> dst_rest_pos,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                coindices->write(i, lcrt::cast<IndexT>(i));
                dst_pos->write(i, src_pos->read(i));
                dst_rest_pos->write(i, src_rest_pos->read(i));
                // dst_body_ids(i) += body_offset, offset by the global body offset
                dst_body_ids->write(i, dst_body_ids->read(i) + body_offset);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FiniteElementVertexReporter);

constexpr static U64 FiniteElementVertexReporterUID = 1;

void FiniteElementVertexReporter::do_build(BuildInfo& info)
{
    m_impl.finite_element_method = &require<FiniteElementMethod>();
    m_impl.body_reporter        = &require<FiniteElementBodyReporter>();
}

void FiniteElementVertexReporter::request_attribute_update() noexcept
{
    m_impl.require_update_attributes = true;
}

void FiniteElementVertexReporter::Impl::report_count(VertexCountInfo& info)
{
    info.count(fem().xs.size());
}

void FiniteElementVertexReporter::Impl::init_attributes(VertexAttributeInfo& info)
{
    info.contact_element_ids().copy_from(span<const IndexT>{fem().h_vertex_contact_element_ids});
    info.subscene_element_ids().copy_from(
        span<const IndexT>{fem().h_vertex_subscene_contact_element_ids});

    info.dimensions().copy_from(span<const IndexT>{fem().h_dimensions});
    info.thicknesses().copy_from(span<const Float>{fem().h_thicknesses});

    info.body_ids().copy_from(span<const IndexT>{fem().h_vertex_body_id});
    info.d_hats().copy_from(span<const Float>{fem().h_vertex_d_hat});

    // fill the coindices for later use
    auto& k = init_attributes_kernel();
    auto  n = static_cast<uint32_t>(info.coindices().size());
    if(n > 0)
    {
        frame_stream().record(k(info.coindices(),
                                fem().xs.cview(),
                                info.positions(),
                                fem().x_bars.cview(),
                                static_cast<int32_t>(body_reporter->body_offset()),
                                info.body_ids(),
                                info.rest_positions(),
                                n)
                                  .dispatch(n));
    }
}

void FiniteElementVertexReporter::Impl::update_attributes(VertexAttributeInfo& info)
{
    info.positions().copy_from(fem().xs.cview());

    // This update will ruin the friction force computed in previous step, so we need to discard it.
    // ref: https://github.com/spiriMirror/libuipc/issues/303
    info.require_discard_friction();
}

void FiniteElementVertexReporter::Impl::report_displacements(VertexDisplacementInfo& info)
{
    info.displacements().copy_from(fem().dxs.cview());
}

void FiniteElementVertexReporter::do_report_count(VertexCountInfo& info)
{
    m_impl.report_count(info);
}

void FiniteElementVertexReporter::do_report_attributes(VertexAttributeInfo& info)
{
    if(info.frame() == 0)
    {
        auto global_offset = info.coindices().offset();

        auto geo_slots = world().scene().geometries();

        // add global vertex offset attribute
        m_impl.finite_element_method->for_each(  //
            geo_slots,
            [&](const FiniteElementMethod::ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                auto gvo = sc.meta().find<IndexT>(builtin::global_vertex_offset);
                if(!gvo)
                {
                    gvo = sc.meta().create<IndexT>(builtin::global_vertex_offset);
                }

                // [global-vertex-offset] = [vertex-offset-in-fem-system] + [fem-system-vertex-offset]
                view(*gvo)[0] = I.geo_info().vertex_offset + global_offset;
            });

        m_impl.init_attributes(info);
    }
    else
    {
        if(m_impl.require_update_attributes)
        {
            m_impl.update_attributes(info);
            m_impl.require_update_attributes = false;
        }
    }
}

void FiniteElementVertexReporter::do_report_displacements(VertexDisplacementInfo& info)
{
    m_impl.report_displacements(info);
}

U64 FiniteElementVertexReporter::get_uid() const noexcept
{
    return FiniteElementVertexReporterUID;
}
}  // namespace uipc::backend::luisa

#include <finite_element/finite_element_vertex_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <kernel_cout.h>
#include <finite_element/finite_element_body_reporter.h>
#include <uipc/builtin/attribute_name.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(FiniteElementVertexReporter);

constexpr static U64 FiniteElementVertexReporterUID = 1;

void FiniteElementVertexReporter::do_build(BuildInfo& info)
{
    m_impl.finite_element_method = &require<FiniteElementMethod>();
    m_impl.body_reporter         = &require<FiniteElementBodyReporter>();
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
    using namespace luisa;
    using namespace luisa::compute;

    auto& stream = fem().m_impl.engine().stream();

    stream << info.contact_element_ids().copy_from(fem().h_vertex_contact_element_ids.data());
    stream << info.subscene_element_ids().copy_from(
        fem().h_vertex_subscene_contact_element_ids.data());

    stream << info.dimensions().copy_from(fem().h_dimensions.data());
    stream << info.thicknesses().copy_from(fem().thicknesses.view());

    stream << info.body_ids().copy_from(fem().h_vertex_body_id.data());
    stream << info.d_hats().copy_from(fem().h_vertex_d_hat.data());

    // fill the coindices for later use
    auto N = info.coindices().size();
    
    auto& device = fem().m_impl.engine().device();
    
    Kernel1D init_attrs_kernel = [&](BufferView<IndexT> coindices,
                                      BufferView<const Vector3> src_pos,
                                      BufferView<Vector3> dst_pos,
                                      BufferView<const Vector3> src_rest_pos,
                                      IndexT body_offset,
                                      BufferView<IndexT> dst_body_ids,
                                      BufferView<Vector3> dst_rest_pos) {
        set_block_size(256);
        Var i = dispatch_x();
        coindices.write(i, i);
        Var p = src_pos.read(i);
        dst_pos.write(i, p);
        Var rp = src_rest_pos.read(i);
        dst_rest_pos.write(i, rp);
        Var bid = dst_body_ids.read(i);
        dst_body_ids.write(i, bid + body_offset);
    };

    auto shader = device.compile(init_attrs_kernel);
    stream << shader(info.coindices(),
                     fem().xs.view(),
                     info.positions(),
                     fem().x_bars.view(),
                     body_reporter->body_offset(),
                     info.body_ids(),
                     info.rest_positions())
                  .dispatch(N);
}

void FiniteElementVertexReporter::Impl::update_attributes(VertexAttributeInfo& info)
{
    auto& stream = fem().m_impl.engine().stream();
    stream << info.positions().copy_from(fem().xs.view());

    // This update will ruin the friction force computed in previous step, so we need to discard it.
    // ref: https://github.com/spiriMirror/libuipc/issues/303
    info.require_discard_friction();
}

void FiniteElementVertexReporter::Impl::report_displacements(VertexDisplacementInfo& info)
{
    auto& stream = fem().m_impl.engine().stream();
    stream << info.displacements().copy_from(fem().dxs.view());
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

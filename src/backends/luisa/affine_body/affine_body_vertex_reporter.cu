#include <affine_body/affine_body_vertex_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <affine_body/affine_body_body_reporter.h>
#include <uipc/builtin/attribute_name.h>
#include <luisa/dsl/syntax.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(AffineBodyVertexReporter);

constexpr static U64 AffineBodyVertexReporterUID = 0;

void AffineBodyVertexReporter::do_build(BuildInfo& info)
{
    m_impl.affine_body_dynamics = &require<AffineBodyDynamics>();
    m_impl.body_reporter        = &require<AffineBodyBodyReporter>();
}

void AffineBodyVertexReporter::request_attribute_update() noexcept
{
    m_impl.require_update_attributes = true;
}

void AffineBodyVertexReporter::Impl::report_count(VertexCountInfo& info)
{
    info.count(abd().h_vertex_id_to_J.size());
}

void AffineBodyVertexReporter::Impl::init_attributes(VertexAttributeInfo& info)
{
    auto N = info.positions().size();

    UIPC_ASSERT(body_reporter->body_offset() >= 0,
                "AffineBodyBodyReporter is not ready, body_offset={}, lifecycle issue?",
                body_reporter->body_offset());

    auto& device = affine_body_dynamics->engine().luisa_device();
    auto& stream = affine_body_dynamics->engine().compute_stream();

    Kernel1D init_kernel = [&](BufferVar<IndexT> coindices,
                                BufferVar<ABDJacobi> src_pos,
                                BufferVar<Vector3> dst_pos,
                                BufferVar<IndexT> v2b,
                                Var<IndexT> body_offset,
                                BufferVar<IndexT> dst_v2b,
                                BufferVar<Vector12> qs,
                                BufferVar<Vector3> dst_rest_pos) noexcept
    {
        auto i = dispatch_id().x;
        if_(i < N, [&] {
            coindices.write(i, i);

            auto body_id = v2b.read(i);
            auto q = qs.read(body_id);
            auto jacobi = src_pos.read(i);
            dst_pos.write(i, jacobi.point_x(q));
            dst_rest_pos.write(i, jacobi.x_bar());
            dst_v2b.write(i, body_id + body_offset);  // offset by the global body offset
        });
    };

    auto shader = device.compile(init_kernel);
    stream << shader(info.coindices(),
                     abd().vertex_id_to_J,
                     info.positions(),
                     abd().vertex_id_to_body_id,
                     body_reporter->body_offset(),
                     info.body_ids(),
                     abd().body_id_to_q,
                     info.rest_positions()).dispatch(N);

    // Copy host data to device
    stream << info.contact_element_ids().copy_from(abd().h_vertex_id_to_contact_element_id.data())
           << info.subscene_element_ids().copy_from(abd().h_vertex_id_to_subscene_contact_element_id.data())
           << info.d_hats().copy_from(abd().h_vertex_id_to_d_hat.data())
           << luisa::compute::synchronize();
}

void AffineBodyVertexReporter::Impl::update_attributes(VertexAttributeInfo& info)
{
    auto N = info.positions().size();

    auto& device = affine_body_dynamics->engine().luisa_device();
    auto& stream = affine_body_dynamics->engine().compute_stream();

    // only update positions
    Kernel1D update_kernel = [&](BufferVar<IndexT> coindices,
                                  BufferVar<ABDJacobi> src_pos,
                                  BufferVar<Vector3> dst_pos,
                                  BufferVar<IndexT> v2b,
                                  BufferVar<Vector12> qs) noexcept
    {
        auto i = dispatch_id().x;
        if_(i < N, [&] {
            coindices.write(i, i);
            auto body_id = v2b.read(i);
            auto q = qs.read(body_id);
            auto jacobi = src_pos.read(i);
            dst_pos.write(i, jacobi.point_x(q));
        });
    };

    auto shader = device.compile(update_kernel);
    stream << shader(info.coindices(),
                     abd().vertex_id_to_J,
                     info.positions(),
                     abd().vertex_id_to_body_id,
                     abd().body_id_to_q).dispatch(N);

    // This update will ruin the friction force computed in previous step, so we need to discard it.
    // ref: https://github.com/spiriMirror/libuipc/issues/303
    info.require_discard_friction();
}

void AffineBodyVertexReporter::Impl::report_displacements(VertexDisplacementInfo& info)
{
    auto N = info.coindices().size();

    auto& device = affine_body_dynamics->engine().luisa_device();
    auto& stream = affine_body_dynamics->engine().compute_stream();

    Kernel1D displacement_kernel = [&](BufferVar<IndexT> coindices,
                                        BufferVar<Vector3> displacements,
                                        BufferVar<IndexT> v2b,
                                        BufferVar<Vector12> dqs,
                                        BufferVar<ABDJacobi> Js) noexcept
    {
        auto vI = dispatch_id().x;
        if_(vI < N, [&] {
            auto body_id = v2b.read(vI);
            auto dq = dqs.read(body_id);
            auto J = Js.read(vI);
            displacements.write(vI, J * dq);
        });
    };

    auto shader = device.compile(displacement_kernel);
    stream << shader(info.coindices(),
                     info.displacements(),
                     abd().vertex_id_to_body_id,
                     abd().body_id_to_dq,
                     abd().vertex_id_to_J).dispatch(N);
}

void AffineBodyVertexReporter::do_report_count(VertexCountInfo& info)
{
    m_impl.report_count(info);
}

void AffineBodyVertexReporter::do_report_attributes(VertexAttributeInfo& info)
{
    if(info.frame() == 0)
    {
        auto global_offset = info.coindices().offset();

        auto geo_slots = world().scene().geometries();

        // add global vertex offset attribute
        m_impl.affine_body_dynamics->for_each(  //
            geo_slots,
            [&](const AffineBodyDynamics::ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                auto gvo = sc.meta().find<IndexT>(builtin::global_vertex_offset);
                if(!gvo)
                {
                    gvo = sc.meta().create<IndexT>(builtin::global_vertex_offset);
                }

                // [global-vertex-offset] = [vertex-offset-in-abd-system] + [abd-system-vertex-offset]
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

void AffineBodyVertexReporter::do_report_displacements(VertexDisplacementInfo& info)
{
    m_impl.report_displacements(info);
}

U64 AffineBodyVertexReporter::get_uid() const noexcept
{
    return AffineBodyVertexReporterUID;
}
}  // namespace uipc::backend::luisa

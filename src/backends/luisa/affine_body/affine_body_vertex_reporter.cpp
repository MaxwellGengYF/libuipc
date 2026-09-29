#include <affine_body/affine_body_vertex_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <affine_body/affine_body_body_reporter.h>
#include <uipc/builtin/attribute_name.h>
#include <luisa_device/linear/linear.h>
#include <frame_stream.h>

/********************************************************************
 * T6.1a — AffineBodyVertexReporter (reference:
 * src/backends/cuda/affine_body/affine_body_vertex_reporter.cu).
 *
 * Three kernels, all 1 thread per vertex:
 *   init_attributes     : coindices(i)=i;  pos = J(i) * q(v2b(i));
 *                         rest_pos = J(i).x_bar;  body_ids(i) = v2b(i) + body_offset
 *   update_attributes   : coindices(i)=i;  pos = J(i) * q(v2b(i))
 *   report_displacements: dx = J(i) * dq(v2b(i))
 *
 * `ABDJacobi::point_x(q)` is `abd_jacobi_point<lcrt::Var<double>>` and
 * `ABDJacobi::x_bar()` is the aggregate member `x_bar` (both in
 * affine_body/abd_jacobi_matrix.h / affine_body/type_define.h).  A buffer element
 * is written as a `lcrt::Var<...>` (a bare LVector3 is not an `Expr`, T4.2's
 * note), hence the explicit load/store helpers.
 *
 * The reporter's uid is 0 (the reference's `AffineBodyVertexReporterUID`); the FEM
 * sibling uses 1, and `GlobalVertexManager` sorts its reporters by uid only.
 *********************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla_ = luisa_device::linalg;

/// `AffineBodyVertexReporter_init_attributes_kernel`
auto& init_attributes_kernel()
{
    static auto& k = lla_::detail::cached_kernel<lcrt::Buffer<IndexT>,      // coindices
                                                 lcrt::Buffer<ABDJacobi>,   // src (J)
                                                 lcrt::Buffer<LVector3>,    // dst_pos
                                                 lcrt::Buffer<IndexT>,      // v2b
                                                 int32_t,                   // body_offset
                                                 lcrt::Buffer<IndexT>,      // dst body ids
                                                 lcrt::Buffer<LVector12>,   // qs
                                                 lcrt::Buffer<LVector3>,    // dst_rest_pos
                                                 uint32_t>(                 // n
        [](lcrt::BufferVar<IndexT>    coindices,
           lcrt::BufferVar<ABDJacobi> src_pos,
           lcrt::BufferVar<LVector3>  dst_pos,
           lcrt::BufferVar<IndexT>    v2b,
           lcrt::Var<int32_t>         body_offset,
           lcrt::BufferVar<IndexT>    dst_v2b,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector3>  dst_rest_pos,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                coindices->write(i, lcrt::cast<IndexT>(i));

                auto body_id = v2b->read(i);
                auto J       = load_jacobi(src_pos->read(i));
                auto q       = load_vec12(qs->read(body_id));

                lcrt::Var<LVector3> pos;
                store_vec3(pos, abd_jacobi_point(J, q));
                dst_pos->write(i, pos);

                lcrt::Var<LVector3> rest_pos;
                store_vec3(rest_pos, J.x_bar);
                dst_rest_pos->write(i, rest_pos);

                // offset by the global body offset
                dst_v2b->write(i, body_id + body_offset);
            };
        });
    return k;
}

/// `AffineBodyVertexReporter_update_attributes_kernel`
auto& update_attributes_kernel()
{
    static auto& k = lla_::detail::cached_kernel<lcrt::Buffer<IndexT>,     // coindices
                                                 lcrt::Buffer<ABDJacobi>,  // src (J)
                                                 lcrt::Buffer<LVector3>,   // dst_pos
                                                 lcrt::Buffer<IndexT>,     // v2b
                                                 lcrt::Buffer<LVector12>,  // qs
                                                 uint32_t>(                // n
        [](lcrt::BufferVar<IndexT>    coindices,
           lcrt::BufferVar<ABDJacobi> src_pos,
           lcrt::BufferVar<LVector3>  dst_pos,
           lcrt::BufferVar<IndexT>    v2b,
           lcrt::BufferVar<LVector12> qs,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                coindices->write(i, lcrt::cast<IndexT>(i));

                auto body_id = v2b->read(i);
                auto J       = load_jacobi(src_pos->read(i));
                auto q       = load_vec12(qs->read(body_id));

                lcrt::Var<LVector3> pos;
                store_vec3(pos, abd_jacobi_point(J, q));
                dst_pos->write(i, pos);
            };
        });
    return k;
}

/// `AffineBodyVertexReporter_report_displacements_kernel`
auto& report_displacements_kernel()
{
    static auto& k = lla_::detail::cached_kernel<lcrt::Buffer<LVector3>,   // displacements
                                                 lcrt::Buffer<IndexT>,     // v2b
                                                 lcrt::Buffer<LVector12>,  // dqs
                                                 lcrt::Buffer<ABDJacobi>,  // Js
                                                 uint32_t>(                // n
        [](lcrt::BufferVar<LVector3>  displacements,
           lcrt::BufferVar<IndexT>    v2b,
           lcrt::BufferVar<LVector12> dqs,
           lcrt::BufferVar<ABDJacobi> Js,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto body_id = v2b->read(i);
                auto dq      = load_vec12(dqs->read(body_id));
                auto J       = load_jacobi(Js->read(i));

                lcrt::Var<LVector3> dx;
                store_vec3(dx, abd_jacobi_point(J, dq));
                displacements->write(i, dx);
            };
        });
    return k;
}
}  // namespace

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

    auto n = static_cast<uint32_t>(N);
    if(n > 0)
    {
        auto& k = init_attributes_kernel();
        frame_stream().record(k(info.coindices(),
                                abd().vertex_id_to_J.cview(),
                                info.positions(),
                                abd().vertex_id_to_body_id.cview(),
                                static_cast<int32_t>(body_reporter->body_offset()),
                                info.body_ids(),
                                abd().body_id_to_q.cview(),
                                info.rest_positions(),
                                n)
                              .dispatch(n));
    }

    info.contact_element_ids().copy_from(
        span<const IndexT>{abd().h_vertex_id_to_contact_element_id});
    info.subscene_element_ids().copy_from(
        span<const IndexT>{abd().h_vertex_id_to_subscene_contact_element_id});
    info.d_hats().copy_from(span<const Float>{abd().h_vertex_id_to_d_hat});
}

void AffineBodyVertexReporter::Impl::update_attributes(VertexAttributeInfo& info)
{
    auto N = info.positions().size();

    // only update positions
    auto n = static_cast<uint32_t>(N);
    if(n > 0)
    {
        auto& k = update_attributes_kernel();
        frame_stream().record(k(info.coindices(),
                                abd().vertex_id_to_J.cview(),
                                info.positions(),
                                abd().vertex_id_to_body_id.cview(),
                                abd().body_id_to_q.cview(),
                                n)
                              .dispatch(n));
    }
    // This update will ruin the friction force computed in previous step, so we need to discard it.
    // ref: https://github.com/spiriMirror/libuipc/issues/303
    info.require_discard_friction();
}

void AffineBodyVertexReporter::Impl::report_displacements(VertexDisplacementInfo& info)
{
    auto N = info.coindices().size();

    auto n = static_cast<uint32_t>(N);
    if(n > 0)
    {
        auto& k = report_displacements_kernel();
        frame_stream().record(k(info.displacements(),
                                abd().vertex_id_to_body_id.cview(),
                                abd().body_id_to_dq.cview(),
                                abd().vertex_id_to_J.cview(),
                                n)
                              .dispatch(n));
    }
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
        auto geo_slots     = world().scene().geometries();
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

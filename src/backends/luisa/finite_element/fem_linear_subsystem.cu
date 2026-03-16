#include <sim_engine.h>
#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/fem_linear_subsystem_reporter.h>
#include <finite_element/finite_element_kinetic.h>
#include <kernel_cout.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/fem_dytopo_effect_receiver.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/flag.h>
#include <utils/report_extent_check.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(FEMLinearSubsystem);

// ref: https://github.com/spiriMirror/libuipc/issues/271
constexpr U64 FEMLinearSubsystemUID = 1ull;

U64 FEMLinearSubsystem::get_uid() const noexcept
{
    return FEMLinearSubsystemUID;
}

void FEMLinearSubsystem::do_build(DiagLinearSubsystem::BuildInfo&)
{
    m_impl.finite_element_method = require<FiniteElementMethod>();
    m_impl.finite_element_vertex_reporter = require<FiniteElementVertexReporter>();
    m_impl.sim_engine = &engine();
    auto dt_attr      = world().scene().config().find<Float>("dt");
    m_impl.dt         = dt_attr->view()[0];

    m_impl.dytopo_effect_receiver = find<FEMDyTopoEffectReceiver>();
}

void FEMLinearSubsystem::do_init(DiagLinearSubsystem::InitInfo& info)
{
    m_impl.init();
}

void FEMLinearSubsystem::Impl::init()
{
    auto reporter_view = reporters.view();
    for(auto&& [i, r] : enumerate(reporter_view))
        r->m_index = i;
    for(auto& r : reporter_view)
        r->init();

    reporter_gradient_offsets_counts.resize(reporter_view.size());
    reporter_hessian_offsets_counts.resize(reporter_view.size());
}

void FEMLinearSubsystem::Impl::report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    info.extent(fem().xs.size() * 3);
}

void FEMLinearSubsystem::Impl::receive_init_dof_info(WorldVisitor& w,
                                                     GlobalLinearSystem::InitDofInfo& info)
{
    auto& geo_infos = fem().geo_infos;
    auto  geo_slots = w.scene().geometries();

    IndexT offset = info.dof_offset();

    finite_element_method->for_each(
        geo_slots,
        [&](const FiniteElementMethod::ForEachInfo& foreach_info, geometry::SimplicialComplex& sc)
        {
            auto I          = foreach_info.global_index();
            auto dof_offset = sc.meta().find<IndexT>(builtin::dof_offset);
            UIPC_ASSERT(dof_offset, "dof_offset not found on FEM mesh why can it happen?");
            auto dof_count = sc.meta().find<IndexT>(builtin::dof_count);
            UIPC_ASSERT(dof_count, "dof_count not found on FEM mesh why can it happen?");

            IndexT this_dof_count = 3 * sc.vertices().size();
            view(*dof_offset)[0]  = offset;
            view(*dof_count)[0]   = this_dof_count;

            offset += this_dof_count;
        });

    UIPC_ASSERT(offset == info.dof_offset() + info.dof_count(), "dof size mismatch");
}

void FEMLinearSubsystem::Impl::report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    bool gradient_only = info.gradient_only();

    bool has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    // 1) Hessian Count
    IndexT grad_offset = 0;
    IndexT hess_offset = 0;

    // We assume reporters won't produce contact
    if(has_complement)
    {
        // Kinetic
        auto kinetic_grad_count = fem().xs.size();
        grad_offset += kinetic_grad_count;
        if(!gradient_only)
            hess_offset += fem().xs.size();

        // Reporters
        auto grad_counts = reporter_gradient_offsets_counts.counts();
        auto hess_counts = reporter_hessian_offsets_counts.counts();

        for(auto& reporter : reporters.view())
        {
            ReportExtentInfo this_info;
            this_info.m_gradient_only = gradient_only;
            reporter->report_extent(this_info);
            grad_counts[reporter->m_index] = this_info.m_gradient_count;
            hess_counts[reporter->m_index] = this_info.m_hessian_count;

            UIPC_ASSERT(!(gradient_only && !this_info.m_hessian_count == 0),
                        "When gradient_only is true, hessian_offset must be 0, yours {}.\n"
                        "Ref: https://github.com/spiriMirror/libuipc/issues/295",
                        this_info.m_hessian_count);
        }

        // [KineticG ... | OtherG ... ]
        // [KineticH ... | OtherH ... ]

        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();

        grad_offset += reporter_gradient_offsets_counts.total_count();
        hess_offset += reporter_hessian_offsets_counts.total_count();
    }

    if(dytopo_effect_receiver)  // if dytopo_effect enabled
    {
        grad_offset += dytopo_effect_receiver->gradients().doublet_count();
        hess_offset += dytopo_effect_receiver->hessians().triplet_count();

        UIPC_ASSERT(!(gradient_only
                      && !dytopo_effect_receiver->hessians().triplet_count() == 0),
                    "When gradient_only is true, hessian_offset must be 0, yours {}.\n"
                    "Ref: https://github.com/spiriMirror/libuipc/issues/295",
                    dytopo_effect_receiver->hessians().triplet_count());
    }

    // 2) Gradient Count
    auto dof_count = fem().xs.size() * 3;

    UIPC_ASSERT(!(gradient_only && !hess_offset == 0),
                "When gradient_only is true, hessian_offset must be 0, yours {}.\n"
                "Ref: https://github.com/spiriMirror/libuipc/issues/295",
                hess_offset);

    info.extent(hess_offset, dof_count);
}

void FEMLinearSubsystem::Impl::assemble(GlobalLinearSystem::DiagInfo& info)
{
    using namespace luisa::compute;

    // 0) record dof info
    auto frame = sim_engine->frame();
    fem().set_dof_info(frame, info.gradients().offset(), info.gradients().size());

    // 1) Prepare Gradient Buffer
    kinetic_gradients.resize_doublets(fem().xs.size());
    kinetic_gradients.reshape(fem().xs.size());
    loose_resize_entries(reporter_gradients, reporter_gradient_offsets_counts.total_count());
    reporter_gradients.reshape(fem().xs.size());

    info.gradients().buffer_view().fill(0);

    // 2) Assemble Gradient and Hessian
    bool has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    IndexT hess_offset = 0;
    if(has_complement)
    {
        _assemble_kinetic(hess_offset, info);
        _assemble_reporters(hess_offset, info);
    }

    if(dytopo_effect_receiver)  // if dytopo_effect enabled
    {
        // DyTopo System will decide the `component_flags` itself
        _assemble_dytopo_effect(hess_offset, info);
    }

    UIPC_ASSERT(hess_offset == info.hessians().triplet_count(),
                "Hessian offset mismatch expected {}, got {}",
                info.hessians().triplet_count(),
                hess_offset);


    // 3) Clear Fixed Vertex gradient (double check)
    {
        auto is_fixed = fem().is_fixed.view();
        auto gradients = info.gradients().viewer();
        auto vertex_count = fem().xs.size();

        Kernel1D clear_fixed_grad_kernel = [&](BufferVar<IndexT> is_fixed_buffer,
                                               BufferVar<Float> grad_buffer) noexcept
        {
            auto i = dispatch_x();
            $if(is_fixed_buffer.read(i) == 1)
            {
                auto idx = i * 3;
                grad_buffer.write(idx, 0.0f);
                grad_buffer.write(idx + 1, 0.0f);
                grad_buffer.write(idx + 2, 0.0f);
            };
        };

        auto shader = sim_engine->device().compile(clear_fixed_grad_kernel);
        sim_engine->stream() << shader(is_fixed, info.gradients().buffer_view()).dispatch(vertex_count);
    }

    if(info.gradient_only())
        return;

    // 4) Clear Fixed Vertex hessian
    {
        auto is_fixed = fem().is_fixed.view();
        auto hessians = info.hessians().viewer();
        auto hess_count = info.hessians().triplet_count();

        Kernel1D clear_fixed_hess_kernel = [&](BufferVar<IndexT> is_fixed_buffer,
                                               BufferVar<Matrix3x3> hess_buffer) noexcept
        {
            auto I = dispatch_x();
            auto triplet = hess_buffer.read(I);
            auto i = triplet[0];  // row index
            auto j = triplet[1];  // col index
            auto H3 = triplet[2]; // matrix value

            $if((is_fixed_buffer.read(i) == 1) || (is_fixed_buffer.read(j) == 1))
            {
                $if(i != j)
                {
                    hess_buffer.write(I, make_float3x3(0.0f));
                }
                $else
                {
                    hess_buffer.write(I, make_float3x3(1.0f, 0.0f, 0.0f,
                                                        0.0f, 1.0f, 0.0f,
                                                        0.0f, 0.0f, 1.0f));
                };
            };
        };

        auto shader = sim_engine->device().compile(clear_fixed_hess_kernel);
        sim_engine->stream() << shader(is_fixed, info.hessians().buffer_view()).dispatch(hess_count);
    }
}


void FEMLinearSubsystem::Impl::_assemble_kinetic(IndexT& hess_offset,
                                                 GlobalLinearSystem::DiagInfo& info)
{
    using namespace luisa::compute;

    IndexT hess_count = info.gradient_only() ? 0 : fem().xs.size();
    IndexT grad_count = fem().xs.size();

    auto gradient_view = kinetic_gradients.view();
    auto hessian_view  = info.hessians().subview(hess_offset, hess_count);

    FEMLinearSubsystem::ComputeGradientHessianInfo kinetic_info{
        info.gradient_only(), gradient_view, hessian_view, dt};
    kinetic->compute_gradient_hessian(kinetic_info);

    {
        auto dst = info.gradients().viewer();
        auto src = kinetic_gradients.view();
        auto is_fixed = fem().is_fixed.view();
        auto doublet_count = kinetic_gradients.doublet_count();

        Kernel1D assemble_kinetic_grad_kernel = [&](BufferVar<Float> dst_buffer,
                                                    BufferVar<Float> src_buffer,
                                                    BufferVar<IndexT> is_fixed_buffer,
                                                    Var<IndexT> src_offset,
                                                    Var<IndexT> src_count) noexcept
        {
            auto I = dispatch_x();
            $if(I < src_count)
            {
                auto src_idx = src_offset + I;
                // Read doublet: [index, g0, g1, g2]
                auto idx = cast<IndexT>(src_buffer.read(src_idx * 4));
                auto G0 = src_buffer.read(src_idx * 4 + 1);
                auto G1 = src_buffer.read(src_idx * 4 + 2);
                auto G2 = src_buffer.read(src_idx * 4 + 3);

                $if(is_fixed_buffer.read(idx) == 0)
                {
                    auto base = idx * 3;
                    // Atomic add to gradient
                    dst_buffer.atomic(base).fetch_add(G0);
                    dst_buffer.atomic(base + 1).fetch_add(G1);
                    dst_buffer.atomic(base + 2).fetch_add(G2);
                };
            };
        };

        auto shader = sim_engine->device().compile(assemble_kinetic_grad_kernel);
        sim_engine->stream() << shader(dst.buffer_view(), src.buffer_view(), is_fixed,
                                       src.offset(), src.count()).dispatch(doublet_count);
    }

    hess_offset += hess_count;
}

void FEMLinearSubsystem::Impl::_assemble_reporters(IndexT& hess_offset,
                                                   GlobalLinearSystem::DiagInfo& info)
{
    using namespace luisa::compute;
    auto grad_count = reporter_gradient_offsets_counts.total_count();
    auto hess_count =
        info.gradient_only() ? 0 : reporter_hessian_offsets_counts.total_count();

    // Let reporters assemble their gradient and hessian
    auto reporter_gradient_view = reporter_gradients.view();
    auto reporter_hessian_view = info.hessians().subview(hess_offset, hess_count);
    for(auto& R : reporters.view())
    {
        AssembleInfo assemble_info{this, R->m_index, reporter_hessian_view, info.gradient_only()};
        R->assemble(assemble_info);
    }

    {
        auto dst = info.gradients().viewer();
        auto src = reporter_gradients.view();
        auto is_fixed = fem().is_fixed.view();
        auto doublet_count = reporter_gradients.doublet_count();

        Kernel1D assemble_reporter_grad_kernel = [&](BufferVar<Float> dst_buffer,
                                                     BufferVar<Float> src_buffer,
                                                     BufferVar<IndexT> is_fixed_buffer,
                                                     Var<IndexT> src_offset,
                                                     Var<IndexT> src_count) noexcept
        {
            auto I = dispatch_x();
            $if(I < src_count)
            {
                auto src_idx = src_offset + I;
                // Read doublet: [index, g0, g1, g2]
                auto idx = cast<IndexT>(src_buffer.read(src_idx * 4));
                auto G0 = src_buffer.read(src_idx * 4 + 1);
                auto G1 = src_buffer.read(src_idx * 4 + 2);
                auto G2 = src_buffer.read(src_idx * 4 + 3);

                $if(is_fixed_buffer.read(idx) == 0)
                {
                    auto base = idx * 3;
                    // Atomic add to gradient
                    dst_buffer.atomic(base).fetch_add(G0);
                    dst_buffer.atomic(base + 1).fetch_add(G1);
                    dst_buffer.atomic(base + 2).fetch_add(G2);
                };
            };
        };

        auto shader = sim_engine->device().compile(assemble_reporter_grad_kernel);
        sim_engine->stream() << shader(dst.buffer_view(), src.buffer_view(), is_fixed,
                                       src.offset(), src.count()).dispatch(doublet_count);
    }

    // offset update
    hess_offset += hess_count;
}

void FEMLinearSubsystem::Impl::_assemble_dytopo_effect(IndexT& hess_offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    using namespace luisa::compute;

    // No need to add to grad_offset, the grad buffer is not from reporter_gradients
    auto grad_count = dytopo_effect_receiver->gradients().doublet_count();

    // 1) Assemble DyTopoEffect Gradient to Gradient
    if(grad_count)
    {
        auto dytopo_grad = dytopo_effect_receiver->gradients().view();
        auto gradients = info.gradients().viewer();
        auto vertex_offset = finite_element_vertex_reporter->vertex_offset();
        auto is_fixed = fem().is_fixed.view();

        Kernel1D assemble_dytopo_grad_kernel = [&](BufferVar<Float> dytopo_grad_buffer,
                                                   BufferVar<Float> grad_buffer,
                                                   BufferVar<IndexT> is_fixed_buffer,
                                                   Var<IndexT> dytopo_offset,
                                                   Var<IndexT> dytopo_count,
                                                   Var<IndexT> v_offset) noexcept
        {
            auto I = dispatch_x();
            $if(I < dytopo_count)
            {
                auto src_idx = dytopo_offset + I;
                // Read doublet: [global_index, g0, g1, g2]
                auto g_i = cast<IndexT>(dytopo_grad_buffer.read(src_idx * 4));
                auto G0 = dytopo_grad_buffer.read(src_idx * 4 + 1);
                auto G1 = dytopo_grad_buffer.read(src_idx * 4 + 2);
                auto G2 = dytopo_grad_buffer.read(src_idx * 4 + 3);

                auto i = g_i - v_offset;  // from global to local

                $if(is_fixed_buffer.read(i) == 0)
                {
                    auto base = i * 3;
                    // Atomic add to gradient
                    grad_buffer.atomic(base).fetch_add(G0);
                    grad_buffer.atomic(base + 1).fetch_add(G1);
                    grad_buffer.atomic(base + 2).fetch_add(G2);
                };
            };
        };

        auto shader = sim_engine->device().compile(assemble_dytopo_grad_kernel);
        sim_engine->stream() << shader(dytopo_grad.buffer_view(), info.gradients().buffer_view(),
                                       is_fixed, dytopo_grad.offset(), grad_count, vertex_offset)
                             .dispatch(grad_count);
    }

    if(info.gradient_only())
        return;

    // Need to update hess_offset, we are assembling to the global hessian buffer
    auto hess_count = dytopo_effect_receiver->hessians().triplet_count();

    // 2) Assemble DyTopoEffect Hessian to Hessian
    if(hess_count)
    {
        auto dst_H3x3s = info.hessians().subview(hess_offset, hess_count);
        auto dytopo_hess = dytopo_effect_receiver->hessians().view();
        auto vertex_offset = finite_element_vertex_reporter->vertex_offset();

        // NOTE: We don't consider fixed vertex here,
        // because in final phase we will clear the fixed vertex hessian anyway.
        Kernel1D assemble_dytopo_hess_kernel = [&](BufferVar<Matrix3x3> dytopo_hess_buffer,
                                                   BufferVar<Matrix3x3> hess_buffer,
                                                   Var<IndexT> dytopo_offset,
                                                   Var<IndexT> dytopo_count,
                                                   Var<IndexT> v_offset) noexcept
        {
            auto I = dispatch_x();
            $if(I < dytopo_count)
            {
                auto src_idx = dytopo_offset + I;
                auto triplet = dytopo_hess_buffer.read(src_idx);
                auto g_i = cast<IndexT>(triplet[0]);
                auto g_j = cast<IndexT>(triplet[1]);
                auto H3 = triplet[2];

                auto i = g_i - v_offset;
                auto j = g_j - v_offset;

                // Write triplet
                hess_buffer.write(I, make_float3(g_i, g_j, 0.0f), H3);
            };
        };

        auto shader = sim_engine->device().compile(assemble_dytopo_hess_kernel);
        sim_engine->stream() << shader(dytopo_hess.buffer_view(), dst_H3x3s.buffer_view(),
                                       dytopo_hess.offset(), hess_count, vertex_offset)
                             .dispatch(hess_count);
    }

    hess_offset += hess_count;
}

void FEMLinearSubsystem::Impl::accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    info.satisfied(true);
}

void FEMLinearSubsystem::Impl::retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    using namespace luisa::compute;

    auto dxs = fem().dxs.view();
    auto result = info.solution().viewer();
    auto vertex_count = fem().xs.size();

    Kernel1D retrieve_solution_kernel = [&](BufferVar<Vector3> dxs_buffer,
                                            BufferVar<Float> result_buffer) noexcept
    {
        auto i = dispatch_x();
        auto idx = i * 3;
        auto r0 = result_buffer.read(idx);
        auto r1 = result_buffer.read(idx + 1);
        auto r2 = result_buffer.read(idx + 2);
        dxs_buffer.write(i, make_float3(-r0, -r1, -r2));
    };

    auto shader = sim_engine->device().compile(retrieve_solution_kernel);
    sim_engine->stream() << shader(dxs.buffer_view(), result.buffer_view()).dispatch(vertex_count);
}

void FEMLinearSubsystem::Impl::loose_resize_entries(luisa::compute::DeviceDoubletVector<Float, 3>& v,
                                                    SizeT size)
{
    if(size > v.doublet_capacity())
    {
        v.reserve_doublets(size * reserve_ratio);
    }
    v.resize_doublets(size);
}

void FEMLinearSubsystem::do_report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    m_impl.report_extent(info);
}

void FEMLinearSubsystem::do_assemble(GlobalLinearSystem::DiagInfo& info)
{
    m_impl.assemble(info);
}

void FEMLinearSubsystem::do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    m_impl.accuracy_check(info);
}

void FEMLinearSubsystem::do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    m_impl.retrieve_solution(info);
}

void FEMLinearSubsystem::do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    m_impl.report_init_extent(info);
}

void FEMLinearSubsystem::do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info)
{
    m_impl.receive_init_dof_info(world(), info);
}

FEMLinearSubsystem::MutableDoubletVector3 FEMLinearSubsystem::AssembleInfo::gradients() const
{
    auto [offset, count] = m_impl->reporter_gradient_offsets_counts[m_index];
    return m_impl->reporter_gradients.view().subview(offset, count);
}

FEMLinearSubsystem::MutableTripletMatrix3x3 FEMLinearSubsystem::AssembleInfo::hessians() const
{
    auto [offset, count] = m_impl->reporter_hessian_offsets_counts[m_index];
    return m_hessians.subview(offset, count);
}

Float FEMLinearSubsystem::AssembleInfo::dt() const noexcept
{
    return m_impl->dt;
}

bool FEMLinearSubsystem::AssembleInfo::gradient_only() const noexcept
{
    return m_gradient_only;
}

void FEMLinearSubsystem::ReportExtentInfo::gradient_count(SizeT size)
{
    m_gradient_count = size;
}

void FEMLinearSubsystem::ReportExtentInfo::hessian_count(SizeT size)
{
    m_hessian_count = size;
}

void FEMLinearSubsystem::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_count, name);
}

void FEMLinearSubsystem::add_reporter(FEMLinearSubsystemReporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter cannot be null");
    check_state(SimEngineState::BuildSystems, "add_reporter");
    m_impl.reporters.register_sim_system(*reporter);
}

void FEMLinearSubsystem::add_kinetic(FiniteElementKinetic* kinetic)
{
    UIPC_ASSERT(kinetic, "kinetic cannot be null");
    check_state(SimEngineState::BuildSystems, "add_kinetic");
    m_impl.kinetic.register_sim_system(*kinetic);
}
}  // namespace uipc::backend::luisa

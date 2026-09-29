#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/fem_linear_subsystem_reporter.h>
#include <finite_element/fem_linear_subsystem_function.h>
#include <finite_element/fem_dytopo_effect_receiver.h>
#include <finite_element/finite_element_kinetic.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <sim_engine.h>
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/flag.h>
#include <utils/report_extent_check.h>
#include <algorithm>

/*************************************************************************************************
* T5.4a — FEMLinearSubsystem (port of
* src/backends/cuda/finite_element/fem_linear_subsystem.cu).
*
* Read the scope note in fem_linear_subsystem.h first: it names the now/defer
* split, the deviations and which parts a test can reach.
*
* Kernel layout (the reference's named kernels, in file order):
*   k1   clear_fixed_gradient      -> fem_linear_subsystem_detail::clear_fixed_gradient_row
*   k2   fix_hessian               -> ...::fix_hessian_row
*   scatter_gradient (kinetic)     -> ...::scatter_gradient_row
*   scatter_gradient (reporters)   -> ...::scatter_gradient_row     (identical body)
*   dytopo_k1 (gradient)           -> ...::scatter_dytopo_gradient_row
*   dytopo_k2 (hessian)            -> ...::copy_dytopo_hessian_row
*   retrieve_solution              -> ...::retrieve_solution_row
*   diag_norm                      -> ...::diag_block_norm_row (+ a gated store)
*   mass_norm                      -> ...::mass_block_norm_row
*
* Every kernel body lives in fem_linear_subsystem_function.h (the T5.4a brief's §2:
* the assembly is otherwise only reachable from T8.3's solve()), so the shipped
* lambda and the unit test run the same code (cookbook §7.4).
*
* NOTE (the kinetic/reporter scatter): the reference has two *identical* named
* kernels (`..._assemble_kinetic_kernel` and `..._assemble_reporters_kernel`
* differ only in their name).  The port keeps one shader and calls it from both
* sites -- one fewer JIT, identical arithmetic.
*************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
using DenseDeviceView  = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
using CDenseDeviceView = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using CTripletDeviceView = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using DoubletDeviceView  = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using CDoubletDeviceView = lla::LCDoubletVectorDeviceView<lcrt::Var<double>, 3>;

/// `FEMLinearSubsystem_assemble_k1_kernel`: clear the fixed vertices' gradient
/// segments (`gradients.segment<3>(i * 3) = 0`).
auto& clear_fixed_gradient_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,   // gradients (this subsystem's window)
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t,               // vertex count
                                                uint32_t>(              // dof count
        [](lcrt::BufferVar<double>  gradients,
           lcrt::BufferVar<int32_t> is_fixed,
           lcrt::Var<uint32_t>      n,
           lcrt::Var<uint32_t>      dof_count) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto g  = DenseDeviceView::from_handles(gradients, dof_count);
                auto ii = lcrt::cast<std::int32_t>(i);
                fem_linear_subsystem_detail::clear_fixed_gradient_row(
                    g, ii, [&](auto&& idx) noexcept { return is_fixed->read(idx); });
            };
        });
    return k;
}

/// `FEMLinearSubsystem_assemble_k2_kernel`: replace the blocks of a fixed vertex
/// by `Zero()` (off-diagonal) / `Identity()` (diagonal).
auto& fix_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<double>,
                                                int32_t,  // submatrix offset x
                                                int32_t,  // submatrix offset y
                                                int32_t,  // submatrix extent x
                                                int32_t,  // submatrix extent y
                                                int32_t,  // total block rows
                                                int32_t,  // total block cols
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t,               // triplet count
                                                uint32_t>(              // vertex count
        [](lcrt::BufferVar<int32_t> rows,
           lcrt::BufferVar<int32_t> cols,
           lcrt::BufferVar<double>  vals,
           lcrt::Var<int32_t> sm_offset_x,
           lcrt::Var<int32_t> sm_offset_y,
           lcrt::Var<int32_t> sm_extent_x,
           lcrt::Var<int32_t> sm_extent_y,
           lcrt::Var<int32_t> total_block_rows,
           lcrt::Var<int32_t> total_block_cols,
           lcrt::BufferVar<int32_t> is_fixed,
           lcrt::Var<uint32_t> n,
           lcrt::Var<uint32_t> /*vertex_count*/) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto H = TripletDeviceView::from_handles_full(rows,
                                                              cols,
                                                              vals,
                                                              total_block_rows,
                                                              total_block_cols,
                                                              std::int32_t{0},
                                                              n,
                                                              n,
                                                              sm_offset_x,
                                                              sm_offset_y,
                                                              sm_extent_x,
                                                              sm_extent_y);
                auto ii = lcrt::cast<std::int32_t>(i);
                fem_linear_subsystem_detail::fix_hessian_row(
                    H, ii, [&](auto&& idx) noexcept { return is_fixed->read(idx); });
            };
        });
    return k;
}

/// `FEMLinearSubsystem_assemble_{kinetic,reporters}_kernel`: scatter the shared
/// doublet buffer into the global gradient window (`+G3` at `3 * i`, skipped for
/// a fixed vertex).
auto& scatter_gradient_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,   // dst (this subsystem's window)
                                                lcrt::Buffer<int32_t>,  // src doublet indices
                                                lcrt::Buffer<double>,   // src doublet values
                                                int32_t,                // src subvector offset
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t,               // src doublet count
                                                uint32_t,               // vertex count
                                                uint32_t>(              // dst dof count
        [](lcrt::BufferVar<double>  dst,
           lcrt::BufferVar<int32_t> src_indices,
           lcrt::BufferVar<double>  src_values,
           lcrt::Var<int32_t>       src_subvector_offset,
           lcrt::BufferVar<int32_t> is_fixed,
           lcrt::Var<uint32_t>      src_count,
           lcrt::Var<uint32_t>      n,
           lcrt::Var<uint32_t>      dst_dof_count) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < src_count)
            {
                auto src = CDoubletDeviceView::from_handles_full(src_indices,
                                                                 src_values,
                                                                 n,  // total segments
                                                                 std::int32_t{0},
                                                                 src_count,
                                                                 src_count,
                                                                 src_subvector_offset,
                                                                 n);  // subvector extent
                auto dst_view = DenseDeviceView::from_handles(dst, dst_dof_count);
                auto ii       = lcrt::cast<std::int32_t>(i);
                fem_linear_subsystem_detail::scatter_gradient_row(
                    dst_view, src, ii, [&](auto&& idx) noexcept { return is_fixed->read(idx); });
            };
        });
    return k;
}

/// `FEMLinearSubsystem_assemble_dytopo_effect_k1_kernel`: the dytopo effect's
/// gradient is indexed by *global* vertex id -> scatter into the subsystem's
/// window at `g_i - vertex_offset`.
auto& scatter_dytopo_gradient_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,  // dytopo doublet indices
                                                lcrt::Buffer<double>,   // dytopo doublet values
                                                lcrt::Buffer<double>,   // dst (this subsystem's window)
                                                int32_t,                // vertex offset
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t,               // dytopo doublet count
                                                uint32_t,               // total segments
                                                uint32_t,               // vertex count
                                                uint32_t>(              // dst dof count
        [](lcrt::BufferVar<int32_t> dytopo_indices,
           lcrt::BufferVar<double>  dytopo_values,
           lcrt::BufferVar<double>  dst,
           lcrt::Var<int32_t>       vertex_offset,
           lcrt::BufferVar<int32_t> is_fixed,
           lcrt::Var<uint32_t>      src_count,
           lcrt::Var<uint32_t>      total_segments,
           lcrt::Var<uint32_t>      n,
           lcrt::Var<uint32_t>      dst_dof_count) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < src_count)
            {
                auto src = CDoubletDeviceView::from_handles(dytopo_indices,
                                                            dytopo_values,
                                                            total_segments,
                                                            src_count);
                auto dst_view = DenseDeviceView::from_handles(dst, dst_dof_count);
                auto ii       = lcrt::cast<std::int32_t>(i);
                fem_linear_subsystem_detail::scatter_dytopo_gradient_row(
                    dst_view,
                    src,
                    ii,
                    vertex_offset,
                    [&](auto&& idx) noexcept { return is_fixed->read(idx); });
            };
        });
    return k;
}

/// `FEMLinearSubsystem_assemble_dytopo_effect_k2_kernel`: copy the dytopo
/// effect's Hessian triplets into the subsystem's window at the *local* indices.
auto& copy_dytopo_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,  // src rows
                                                lcrt::Buffer<int32_t>,  // src cols
                                                lcrt::Buffer<double>,   // src values
                                                int32_t,                // src submatrix offset x
                                                int32_t,                // src submatrix offset y
                                                int32_t,                // src submatrix extent x
                                                int32_t,                // src submatrix extent y
                                                int32_t,                // src total rows
                                                int32_t,                // src total cols
                                                lcrt::Buffer<int32_t>,  // dst rows
                                                lcrt::Buffer<int32_t>,  // dst cols
                                                lcrt::Buffer<double>,   // dst values
                                                int32_t,                // dst triplet offset
                                                int32_t,                // dst submatrix offset x
                                                int32_t,                // dst submatrix offset y
                                                int32_t,                // dst submatrix extent x
                                                int32_t,                // dst submatrix extent y
                                                int32_t,                // dst total rows
                                                int32_t,                // dst total cols
                                                int32_t,                // vertex offset
                                                uint32_t,               // src count
                                                uint32_t>(              // dst count
        [](lcrt::BufferVar<int32_t> src_rows,
           lcrt::BufferVar<int32_t> src_cols,
           lcrt::BufferVar<double>  src_vals,
           lcrt::Var<int32_t>       src_sm_offset_x,
           lcrt::Var<int32_t>       src_sm_offset_y,
           lcrt::Var<int32_t>       src_sm_extent_x,
           lcrt::Var<int32_t>       src_sm_extent_y,
           lcrt::Var<int32_t>       src_total_rows,
           lcrt::Var<int32_t>       src_total_cols,
           lcrt::BufferVar<int32_t> dst_rows,
           lcrt::BufferVar<int32_t> dst_cols,
           lcrt::BufferVar<double>  dst_vals,
           lcrt::Var<int32_t>       dst_triplet_offset,
           lcrt::Var<int32_t>       dst_sm_offset_x,
           lcrt::Var<int32_t>       dst_sm_offset_y,
           lcrt::Var<int32_t>       dst_sm_extent_x,
           lcrt::Var<int32_t>       dst_sm_extent_y,
           lcrt::Var<int32_t>       dst_total_rows,
           lcrt::Var<int32_t>       dst_total_cols,
           lcrt::Var<int32_t>       vertex_offset,
           lcrt::Var<uint32_t>      src_count,
           lcrt::Var<uint32_t>      dst_count) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < src_count)
            {
                auto src = CTripletDeviceView::from_handles_full(src_rows,
                                                                 src_cols,
                                                                 src_vals,
                                                                 src_total_rows,
                                                                 src_total_cols,
                                                                 std::int32_t{0},
                                                                 src_count,
                                                                 src_count,
                                                                 src_sm_offset_x,
                                                                 src_sm_offset_y,
                                                                 src_sm_extent_x,
                                                                 src_sm_extent_y);
                auto dst = TripletDeviceView::from_handles_full(dst_rows,
                                                                dst_cols,
                                                                dst_vals,
                                                                dst_total_rows,
                                                                dst_total_cols,
                                                                dst_triplet_offset,
                                                                dst_count,
                                                                dst_count,
                                                                dst_sm_offset_x,
                                                                dst_sm_offset_y,
                                                                dst_sm_extent_x,
                                                                dst_sm_extent_y);
                auto ii = lcrt::cast<std::int32_t>(i);
                fem_linear_subsystem_detail::copy_dytopo_hessian_row(dst, src, ii, vertex_offset);
            };
        });
    return k;
}

/// `FEMLinearSubsystem_retrieve_solution_kernel`: `dxs(i) = -result.segment<3>(3 i)`.
auto& retrieve_solution_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,    // solution
                                                lcrt::Buffer<LVector3>,  // dxs
                                                uint32_t,                // vertex count
                                                uint32_t>(               // dof count
        [](lcrt::BufferVar<double>   solution,
           lcrt::BufferVar<LVector3> dxs,
           lcrt::Var<uint32_t>       n,
           lcrt::Var<uint32_t>       dof_count) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto result = CDenseDeviceView::from_handles(solution, dof_count);
                auto ii     = lcrt::cast<std::int32_t>(i);
                fem_linear_subsystem_detail::retrieve_solution_row(
                    result,
                    ii,
                    [&](auto&& v) noexcept
                    {
                        lcrt::Var<LVector3> out;
                        store_vec3(out, v);
                        dxs->write(ii, out);
                    });
            };
        });
    return k;
}

/// `FEMLinearSubsystem_diag_norm_kernel`.
auto& diag_norm_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<double>,
                                                int32_t,  // total block rows
                                                int32_t,  // total block cols
                                                lcrt::Buffer<double>,   // diag_blocks_norm
                                                int32_t,                // fem segment offset
                                                int32_t,                // fem segment count
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t,               // triplet count
                                                uint32_t>(              // vertex count
        [](lcrt::BufferVar<int32_t> rows,
           lcrt::BufferVar<int32_t> cols,
           lcrt::BufferVar<double>  vals,
           lcrt::Var<int32_t>       total_block_rows,
           lcrt::Var<int32_t>       total_block_cols,
           lcrt::BufferVar<double>  diag_blocks_norm,
           lcrt::Var<int32_t>       fem_segment_offset,
           lcrt::Var<int32_t>       fem_segment_count,
           lcrt::BufferVar<int32_t> is_fixed,
           lcrt::Var<uint32_t>      n,
           lcrt::Var<uint32_t>      /*vertex_count*/) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto A = CTripletDeviceView::from_handles_full(rows,
                                                               cols,
                                                               vals,
                                                               total_block_rows,
                                                               total_block_cols,
                                                               std::int32_t{0},
                                                               n,
                                                               n,
                                                               std::int32_t{0},
                                                               std::int32_t{0},
                                                               total_block_rows,
                                                               total_block_cols);
                auto ii = lcrt::cast<std::int32_t>(i);
                auto r  = fem_linear_subsystem_detail::diag_block_norm_row(
                    A,
                    ii,
                    fem_segment_offset,
                    fem_segment_count,
                    [&](auto&& idx) noexcept { return is_fixed->read(idx); });
                // the reference's two device guards (`return` on an out-of-window
                // triplet, write only for the diagonal block) -- the routine
                // returns them as predicates and the *store* is gated here.
                $if(scalar_detail::pred_and(!r.out_of_window, r.is_diagonal))
                {
                    diag_blocks_norm->write(r.index, r.value);
                };
            };
        });
    return k;
}

/// `FEMLinearSubsystem_mass_norm_kernel`.
auto& mass_norm_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,   // masses
                                                lcrt::Buffer<double>,   // diag_blocks_norm
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t>(              // vertex count
        [](lcrt::BufferVar<double>  mass,
           lcrt::BufferVar<double>  diag_blocks_norm,
           lcrt::BufferVar<int32_t> is_fixed,
           lcrt::Var<uint32_t>      n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto norm = DenseDeviceView::from_handles(diag_blocks_norm, n);
                auto m    = CDenseDeviceView::from_handles(mass, n);
                auto ii   = lcrt::cast<std::int32_t>(i);
                fem_linear_subsystem_detail::mass_block_norm_row(
                    m, norm, ii, [&](auto&& idx) noexcept { return is_fixed->read(idx); });
            };
        });
    return k;
}

/// The gradient window's zero fill.  LC has no fill *command*, and
/// `LBufferView::fill` submits on the main stream (which would leave a
/// `GlobalLinearSystem::solve()` frame's already-recorded commands un-ordered --
/// cookbook §7.11 rule 7), so the fill is a local kernel recorded on the caller's
/// frame stream like every other launch of this file.
auto& zero_gradient_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>, uint32_t>(
        [](lcrt::BufferVar<double> data, lcrt::Var<uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                data->write(i, lcrt::Var<double>{0.0});
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FEMLinearSubsystem);

// ref: https://github.com/spiriMirror/libuipc/issues/271
constexpr U64 FEMLinearSubsystemUID = 1ull;

U64 FEMLinearSubsystem::get_uid() const noexcept
{
    return FEMLinearSubsystemUID;
}

void FEMLinearSubsystem::do_build(DiagLinearSubsystem::BuildInfo& info)
{
    (void)info;
    m_impl.finite_element_method         = require<FiniteElementMethod>();
    m_impl.finite_element_vertex_reporter = require<FiniteElementVertexReporter>();
    m_impl.sim_engine                    = &engine();
    m_impl.dt_attr = world().scene().config().find<Float>("dt");
    UIPC_ASSERT(m_impl.dt_attr, "Scene config must have a 'dt' attribute.");

    m_impl.dytopo_effect_receiver = find<FEMDyTopoEffectReceiver>();
}

void FEMLinearSubsystem::do_init(DiagLinearSubsystem::InitInfo& info)
{
    (void)info;
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

            IndexT this_dof_count = 3 * static_cast<IndexT>(sc.vertices().size());
            view(*dof_offset)[0]  = offset;
            view(*dof_count)[0]   = this_dof_count;

            offset += this_dof_count;
            (void)I;
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

    SizeT dytopo_grad_count = 0;
    SizeT dytopo_hess_count = 0;
    if(dytopo_effect_receiver)  // if dytopo_effect enabled
    {
        dytopo_grad_count = dytopo_effect_receiver->gradients().doublet_count();
        dytopo_hess_count = dytopo_effect_receiver->hessians().triplet_count();

        grad_offset += static_cast<IndexT>(dytopo_grad_count);
        hess_offset += static_cast<IndexT>(dytopo_hess_count);

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

    // The arithmetic itself is the *factored* routine (the brief's §2 testability
    // requirement): same counts, same order, drivable by a unit test with
    // hand-derived values.
    auto budget = fem_linear_subsystem_detail::plan_extent(
        fem().xs.size(),
        gradient_only,
        has_complement,
        reporter_gradient_offsets_counts.counts(),
        reporter_hessian_offsets_counts.counts(),
        dytopo_grad_count,
        dytopo_hess_count);

    UIPC_ASSERT(budget.hessian_offset == static_cast<SizeT>(hess_offset),
                "hessian offset mismatch: factored {} vs reference order {}",
                budget.hessian_offset,
                hess_offset);
    UIPC_ASSERT(budget.gradient_offset == static_cast<SizeT>(grad_offset),
                "gradient offset mismatch: factored {} vs reference order {}",
                budget.gradient_offset,
                grad_offset);
    UIPC_ASSERT(budget.check_gradient_only(gradient_only),
                "gradient_only with a non-empty Hessian budget");

    info.extent(budget.hessian_offset, budget.dof_count);
}

void FEMLinearSubsystem::Impl::assemble(GlobalLinearSystem::DiagInfo& info)
{
    // 0) record dof info
    auto frame = sim_engine->frame();
    fem().set_dof_info(frame,
                       static_cast<IndexT>(info.gradients().offset()),
                       static_cast<IndexT>(info.gradients().size()));

    // 1) Prepare Gradient Buffer
    kinetic_gradients.resize_doublets_discard(fem().xs.size());
    kinetic_gradients.reshape(static_cast<int>(fem().xs.size()));
    loose_resize_entries(reporter_gradients, reporter_gradient_offsets_counts.total_count());
    reporter_gradients.reshape(static_cast<int>(fem().xs.size()));

    auto gradient_dof_count = static_cast<uint32_t>(info.gradients().size());
    if(gradient_dof_count > 0)
    {
        auto& k = zero_gradient_kernel();
        frame_stream().record(k(info.gradients().buffer_view(), gradient_dof_count)
                                  .dispatch(gradient_dof_count));
    }

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

    UIPC_ASSERT(hess_offset == static_cast<IndexT>(info.hessians().triplet_count()),
                "Hessian offset mismatch expected {}, got {}",
                info.hessians().triplet_count(),
                hess_offset);


    // 3) Clear Fixed Vertex gradient (double check)
    {
        auto n = static_cast<uint32_t>(fem().xs.size());
        if(n > 0)
        {
            auto& k = clear_fixed_gradient_kernel();
            frame_stream().record(k(info.gradients().buffer_view(),
                                    fem().is_fixed.cview(),
                                    n,
                                    gradient_dof_count)
                                      .dispatch(n));
        }
    }

    if(info.gradient_only())
        return;

    // 4) Clear Fixed Vertex hessian
    {
        const auto hessians = info.hessians();
        auto       n        = static_cast<uint32_t>(hessians.triplet_count());
        if(n > 0)
        {
            auto& k = fix_hessian_kernel();
            frame_stream().record(
                k(hessians.row_indices(),
                  hessians.col_indices(),
                  hessians.values(),
                  static_cast<std::int32_t>(hessians.submatrix_offset().x),
                  static_cast<std::int32_t>(hessians.submatrix_offset().y),
                  static_cast<std::int32_t>(hessians.extent().x),
                  static_cast<std::int32_t>(hessians.extent().y),
                  static_cast<std::int32_t>(hessians.total_rows()),
                  static_cast<std::int32_t>(hessians.total_cols()),
                  fem().is_fixed.cview(),
                  n,
                  static_cast<uint32_t>(fem().xs.size()))
                    .dispatch(n));
        }
    }
}


void FEMLinearSubsystem::Impl::_assemble_kinetic(IndexT& hess_offset,
                                                GlobalLinearSystem::DiagInfo& info)
{
    IndexT hess_count = info.gradient_only() ? 0 : static_cast<IndexT>(fem().xs.size());
    IndexT grad_count = static_cast<IndexT>(fem().xs.size());

    auto gradient_view = kinetic_gradients.view();
    auto hessian_view = info.hessians().subview(hess_offset, hess_count);

    FEMLinearSubsystem::ComputeGradientHessianInfo kinetic_info{
        info.gradient_only(), gradient_view, hessian_view, dt_attr->view()[0]};
    kinetic->compute_gradient_hessian(kinetic_info);

    {
        auto n = static_cast<uint32_t>(kinetic_gradients.doublet_count());
        if(n > 0)
        {
            // the kinetic doublet buffer is written at *local* vertex indices, so
            // its subvector offset is 0 and the index space is the whole buffer
            auto& k = scatter_gradient_kernel();
            frame_stream().record(k(info.gradients().buffer_view(),
                                    kinetic_gradients.indices(),
                                    kinetic_gradients.values(),
                                    static_cast<std::int32_t>(gradient_view.subvector_offset()),
                                    fem().is_fixed.cview(),
                                    n,
                                    static_cast<uint32_t>(fem().xs.size()),
                                    static_cast<uint32_t>(info.gradients().size()))
                                      .dispatch(n));
        }
    }

    hess_offset += hess_count;
}

void FEMLinearSubsystem::Impl::_assemble_reporters(IndexT& hess_offset,
                                                  GlobalLinearSystem::DiagInfo& info)
{
    auto grad_count = reporter_gradient_offsets_counts.total_count();
    auto hess_count = info.gradient_only()
                          ? 0
                          : reporter_hessian_offsets_counts.total_count();

    // Let reporters assemble their gradient and hessian
    auto reporter_gradient_view = reporter_gradients.view();
    auto reporter_hessian_view  = info.hessians().subview(hess_offset, hess_count);
    for(auto& R : reporters.view())
    {
        AssembleInfo assemble_info{this, R->m_index, reporter_hessian_view, info.gradient_only()};
        R->assemble(assemble_info);
    }

    {
        auto n = static_cast<uint32_t>(reporter_gradients.doublet_count());
        if(n > 0)
        {
            auto& k = scatter_gradient_kernel();
            frame_stream().record(k(info.gradients().buffer_view(),
                                    reporter_gradients.indices(),
                                    reporter_gradients.values(),
                                    static_cast<std::int32_t>(reporter_gradient_view.subvector_offset()),
                                    fem().is_fixed.cview(),
                                    n,
                                    static_cast<uint32_t>(fem().xs.size()),
                                    static_cast<uint32_t>(info.gradients().size()))
                                      .dispatch(n));
        }
    }

    // offset update
    hess_offset += static_cast<IndexT>(hess_count);
}

void FEMLinearSubsystem::Impl::_assemble_dytopo_effect(IndexT& hess_offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    // No need to add to grad_offset, the grad buffer is not from reporter_gradients
    auto grad_count = dytopo_effect_receiver->gradients().doublet_count();

    // 1) Assemble DyTopoEffect Gradient to Gradient
    if(grad_count)
    {
        auto n   = static_cast<uint32_t>(grad_count);
        auto& k  = scatter_dytopo_gradient_kernel();
        auto src = dytopo_effect_receiver->gradients();
        frame_stream().record(
            k(src.indices(),
              src.values(),
              info.gradients().buffer_view(),
              static_cast<std::int32_t>(finite_element_vertex_reporter->vertex_offset()),
              fem().is_fixed.cview(),
              n,
              static_cast<uint32_t>(src.total_extent()),
              static_cast<uint32_t>(fem().xs.size()),
              static_cast<uint32_t>(info.gradients().size()))
                .dispatch(n));
    }

    if(info.gradient_only())
        return;

    // Need to update hess_offset, we are assembling to the global hessian buffer
    auto hess_count = dytopo_effect_receiver->hessians().triplet_count();

    // 2) Assemble DyTopoEffect Hessian to Hessian
    if(hess_count)
    {
        auto  src       = dytopo_effect_receiver->hessians();
        auto  dst       = info.hessians().subview(hess_offset, hess_count);
        auto  n         = static_cast<uint32_t>(hess_count);

        // NOTE: We don't consider fixed vertex here,
        // because in final phase we will clear the fixed vertex hessian anyway.
        auto& k = copy_dytopo_hessian_kernel();
        frame_stream().record(
            k(src.row_indices(),
              src.col_indices(),
              src.values(),
              static_cast<std::int32_t>(src.submatrix_offset().x),
              static_cast<std::int32_t>(src.submatrix_offset().y),
              static_cast<std::int32_t>(src.extent().x),
              static_cast<std::int32_t>(src.extent().y),
              static_cast<std::int32_t>(src.total_rows()),
              static_cast<std::int32_t>(src.total_cols()),
              dst.row_indices(),
              dst.col_indices(),
              dst.values(),
              static_cast<std::int32_t>(dst.triplet_index_offset()),
              static_cast<std::int32_t>(dst.submatrix_offset().x),
              static_cast<std::int32_t>(dst.submatrix_offset().y),
              static_cast<std::int32_t>(dst.extent().x),
              static_cast<std::int32_t>(dst.extent().y),
              static_cast<std::int32_t>(dst.total_rows()),
              static_cast<std::int32_t>(dst.total_cols()),
              static_cast<std::int32_t>(finite_element_vertex_reporter->vertex_offset()),
              n,
              static_cast<uint32_t>(dst.triplet_count()))
                .dispatch(n));
    }

    hess_offset += static_cast<IndexT>(hess_count);
}

void FEMLinearSubsystem::Impl::accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    (void)info;
    info.satisfied(true);
}

void FEMLinearSubsystem::Impl::retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    auto dxs = fem().dxs.view();

    auto n = static_cast<uint32_t>(fem().xs.size());
    if(n > 0)
    {
        auto& k = retrieve_solution_kernel();
        frame_stream().record(k(info.solution().buffer_view(),
                                dxs,
                                n,
                                static_cast<uint32_t>(info.solution().size()))
                                  .dispatch(n));
    }
}

Float FEMLinearSubsystem::Impl::diag_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    diag_blocks_norm.resize(finite_element_method->xs().size());

    auto n = static_cast<uint32_t>(info.A().triplet_count());
    if(n > 0)
    {
        auto A  = info.A();
        auto& k = diag_norm_kernel();
        frame_stream().record(
            k(A.row_indices(),
              A.col_indices(),
              A.values(),
              static_cast<std::int32_t>(A.total_rows()),
              static_cast<std::int32_t>(A.total_cols()),
              diag_blocks_norm.view(),
              static_cast<std::int32_t>(info.dof_offset() / 3),
              static_cast<std::int32_t>(info.dof_count() / 3),
              fem().is_fixed.cview(),
              n,
              static_cast<uint32_t>(fem().xs.size()))
                .dispatch(n));
    }

    lla::device_reduce_max(frame_stream(),
                           diag_blocks_norm.cview(),
                           reduced_diag_norm.view(),
                           diag_blocks_norm.size());

    return reduced_diag_norm;
}

Float FEMLinearSubsystem::Impl::mass_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    (void)info;
    diag_blocks_norm.resize(fem().xs.size());
    UIPC_ASSERT(fem().xs.size() == fem().masses.size(), "size not matched");

    auto n = static_cast<uint32_t>(fem().xs.size());
    if(n > 0)
    {
        auto& k = mass_norm_kernel();
        frame_stream().record(k(fem().masses.cview(),
                                diag_blocks_norm.view(),
                                fem().is_fixed.cview(),
                                n)
                                  .dispatch(n));
    }

    lla::device_reduce_max(frame_stream(),
                           diag_blocks_norm.cview(),
                           reduced_diag_norm.view(),
                           diag_blocks_norm.size());

    return reduced_diag_norm;
}

void FEMLinearSubsystem::Impl::loose_resize_entries(lla::LDoubletVector<Float, 3>& v,
                                                    SizeT size)
{
    if(size > v.doublet_capacity())
    {
        v.reserve_doublets_discard(size * reserve_ratio);
    }
    v.resize_doublets_discard(size);
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

Float FEMLinearSubsystem::do_diag_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    return m_impl.diag_norm(info);
}

Float FEMLinearSubsystem::do_mass_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    return m_impl.mass_norm(info);
}

// ---------------------------------------------------------------------------
// The four members T5.1 defined in fem_linear_subsystem_reporter.cpp are defined
// here, next to the Impl state they read (T5.1's scope note said exactly this:
// "the four trivial members ... T5.4 moves those definitions into its own
// fem_linear_subsystem.cpp").
// ---------------------------------------------------------------------------
lla::LDoubletVectorView<Float, 3> FEMLinearSubsystem::AssembleInfo::gradients() const
{
    auto [offset, count] = fem_linear_subsystem_detail::slice_of(
        m_impl->reporter_gradient_offsets_counts, m_index);
    return m_impl->reporter_gradients.view().subview(offset, count);
}

lla::LTripletMatrixView<Float, 3, 3> FEMLinearSubsystem::AssembleInfo::hessians() const
{
    auto [offset, count] = fem_linear_subsystem_detail::slice_of(
        m_impl->reporter_hessian_offsets_counts, m_index);
    return m_hessians.subview(offset, count);
}

Float FEMLinearSubsystem::AssembleInfo::dt() const noexcept
{
    return m_impl->dt_attr->view()[0];
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
}  // namespace uipc::backend::luisa

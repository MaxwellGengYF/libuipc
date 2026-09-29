#include <finite_element/fem_diag_preconditioner.h>
#include <finite_element/fem_diag_preconditioner_function.h>
#include <linear_system/global_linear_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <frame_stream.h>
#include <uipc/geometry/simplicial_complex.h>

/*************************************************************************************************
* T5.4b — FEMDiagPreconditioner (port of
* src/backends/cuda/finite_element/fem_diag_preconditioner.cu).
*
* Read the scope note in fem_diag_preconditioner.h first.  Kernel mapping:
*   FEMDiagPreconditioner_do_assemble_kernel -> fem_diag_preconditioner_detail::diag_inv_row
*                                               (+ a gated store)
*   FEMDiagPreconditioner_do_apply_kernel    -> fem_diag_preconditioner_detail::diag_apply_row
*
* Both bodies live in fem_diag_preconditioner_function.h, so the shipped shader and
* the unit test run the same code (cookbook §7.4).
*
* Reference-fidelity notes:
*  * `info.A()` is the *assembled* symmetric BCOO matrix (`Impl::bcoo_A`), so
*    `do_assemble` only sees the upper-packed blocks the assembly produced; the
*    reference's `i == j` test therefore picks exactly the diagonal blocks.
*  * `diag_inv.resize(finite_element_method->xs().size())` is the reference's own
*    sizing (one block per FEM vertex); zero-filled by `LDeviceBuffer::resize`, so
*    a block with no assembled diagonal triplet keeps the reference's behaviour
*    (the reference's `DeviceBuffer::resize` value-initialises the same way).
*  * the dispatch grid is `info.A().triplet_count()` (NOT the vertex count), and
*    the store is gated on `!(i >= count || j >= count)`; the reference's early
*    `return` is that guard (RULE M2 — the predicate is device data).
*************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
using TripletDeviceView = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using CDenseDeviceView  = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
using DenseDeviceView   = lla::LDenseVectorDeviceView<lcrt::Var<double>>;

/// `FEMDiagPreconditioner_do_assemble_kernel`.
auto& fem_diag_preconditioner_assemble_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,   // A rows
                                                lcrt::Buffer<std::int32_t>,   // A cols
                                                lcrt::Buffer<double>,         // A values
                                                lcrt::Buffer<LMatrix3x3>,     // diag_inv
                                                std::int32_t,                 // total block rows
                                                std::int32_t,                 // total block cols
                                                std::int32_t,                 // fem segment offset
                                                std::int32_t,                 // fem segment count
                                                uint32_t>(                    // triplet count
        [](lcrt::BufferVar<std::int32_t> rows,
           lcrt::BufferVar<std::int32_t> cols,
           lcrt::BufferVar<double>       vals,
           lcrt::BufferVar<LMatrix3x3>   diag_inv,
           lcrt::Var<std::int32_t>       total_block_rows,
           lcrt::Var<std::int32_t>       total_block_cols,
           lcrt::Var<std::int32_t>       fem_segment_offset,
           lcrt::Var<std::int32_t>       fem_segment_count,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto A = TripletDeviceView::from_handles(
                    rows, cols, vals, total_block_rows, total_block_cols, n);

                auto row = fem_diag_preconditioner_detail::diag_inv_row(
                    A, lcrt::cast<std::int32_t>(I), fem_segment_offset, fem_segment_count);

                // the reference's two early returns, as the store's two guards
                $if(row.in_window)
                {
                    $if(row.is_diagonal)
                    {
                        lcrt::Var<LMatrix3x3> out;
                        store_mat3x3(out, row.value);
                        diag_inv->write(row.index, out);
                    };
                };
            };
        });
    return k;
}

/// `FEMDiagPreconditioner_do_apply_kernel`.
auto& fem_diag_preconditioner_apply_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,       // r (this subsystem's window)
                                                lcrt::Buffer<double>,       // z (this subsystem's window)
                                                lcrt::Buffer<std::int32_t>,  // converged
                                                lcrt::Buffer<LMatrix3x3>,    // diag_inv
                                                uint32_t>(                   // block count
        [](lcrt::BufferVar<double>       r,
           lcrt::BufferVar<double>       z,
           lcrt::BufferVar<std::int32_t> converged,
           lcrt::BufferVar<LMatrix3x3>   diag_inv,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // the reference's `if(*converged != 0) return;` -- one scalar read
                // for the whole dispatch, so it wraps the body
                $if(converged->read(0) == 0)
                {
                    auto R = CDenseDeviceView::from_handles(r, n * 3u);
                    auto Z = DenseDeviceView::from_handles(z, n * 3u);
                    auto D = fem_diag_preconditioner_detail::DiagInvDeviceView::from_handles(
                        diag_inv, n);
                    fem_diag_preconditioner_detail::diag_apply_row(
                        Z, R, D, lcrt::cast<std::int32_t>(i));
                };
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FEMDiagPreconditioner);

void FEMDiagPreconditioner::do_build(BuildInfo& info)
{
    finite_element_method = &require<FiniteElementMethod>();
    global_linear_system  = &require<GlobalLinearSystem>();
    fem_linear_subsystem  = &require<FEMLinearSubsystem>();
    // the reference's fourth dependency edge: it is `require`d for the build order
    // (the preconditioner reads the method's buffers), never dereferenced.
    (void)require<GlobalVertexManager>();

    // MAS is selected via config; defer to FEMMASPreconditioner when on.
    // (The port keeps the reference's *shutdown* idiom verbatim: throwing
    // SimSystemException invalidates this system and everything that depends on it.)
    auto precond = world().scene().config().find<std::string>("linear_system/fem_preconditioner");
    if(precond && precond->view()[0] == "mas")
    {
        throw SimSystemException(
            "FEMDiagPreconditioner: linear_system/fem_preconditioner == \"mas\", "
            "deferring to FEMMASPreconditioner.");
    }

    // This FEMDiagPreconditioner depends on FEMLinearSubsystem
    info.connect(fem_linear_subsystem);
}

void FEMDiagPreconditioner::do_init(InitInfo& info)
{
    (void)info;
}

void FEMDiagPreconditioner::do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info)
{
    diag_inv.resize(finite_element_method->xs().size());

    // 1) collect diagonal blocks
    auto n = static_cast<uint32_t>(info.A().triplet_count());
    if(n > 0)
    {
        auto  A = info.A();
        auto& k = fem_diag_preconditioner_assemble_kernel();
        frame_stream()
            .record(k(A.row_indices(),
                      A.col_indices(),
                      A.values(),
                      diag_inv.view(),
                      static_cast<std::int32_t>(A.total_rows()),
                      static_cast<std::int32_t>(A.total_cols()),
                      static_cast<std::int32_t>(info.dof_offset() / 3),
                      static_cast<std::int32_t>(info.dof_count() / 3),
                      n)
                        .dispatch(n));
    }
}

void FEMDiagPreconditioner::do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info)
{
    auto converged = info.converged();

    auto n = static_cast<uint32_t>(diag_inv.size());
    if(n > 0)
    {
        auto& k = fem_diag_preconditioner_apply_kernel();
        frame_stream().record(k(info.r().buffer_view(),
                                info.z().buffer_view(),
                                converged,
                                diag_inv.cview(),
                                n)
                                  .dispatch(n));
    }
}
}  // namespace uipc::backend::luisa

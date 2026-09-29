#include <affine_body/abd_diag_preconditioner.h>
#include <affine_body/abd_diag_preconditioner_function.h>
#include <linear_system/global_linear_system.h>
#include <frame_stream.h>

/*************************************************************************************************
* T6.4a — ABDDiagPreconditioner (port of
* src/backends/cuda/affine_body/abd_diag_preconditioner.cu).
*
* Read the scope note in abd_diag_preconditioner.h first.  Kernel mapping:
*   ABDDiagPreconditioner_do_assemble_kernel -> ..._detail::assemble_row
*   ABDDiagPreconditioner_do_apply_kernel    -> ..._detail::apply_row
*
* Both bodies live in abd_diag_preconditioner_function.h, so the shipped shader and
* the unit test run the same code (cookbook §7.4).
*
* Reference-fidelity notes:
*  * `diag_hessian` is the subsystem's *dense per-body* 12x12 buffer (one element
*    per body, no assembled-BCOO scanning and no `i == j` test -- the ABD sibling's
*    `do_assemble` differs from the FEM one exactly there);
*  * `diag_inv.resize(diag_hessian.size())` is the reference's own sizing, zero
*    filled by `LDeviceBuffer::resize`, so a body with no assembled diagonal keeps
*    the reference's behaviour (`DeviceBuffer::resize` value-initialises the same
*    way);
*  * the dispatch is `n = diag_inv.size()` (the body count) and the reference's
*    `if(i >= n) return;` is the kernel's `$if` guard (RULE M2 -- the predicate is
*    device data);
*  * `info.r()` / `info.z()` are already the subsystem's *window* (the global
*    system slices by `diag_dof_offsets_counts`), so the per-body `i * 12` offsets
*    of the reference kernel are window-relative, exactly as in the FEM sibling.
*************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
using DenseDeviceView  = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
using CDenseDeviceView = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
using DiagInvDeviceView = abd_diag_preconditioner_detail::DiagInvDeviceView;

/// `ABDDiagPreconditioner_do_assemble_kernel`.
auto& abd_diag_preconditioner_assemble_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LMatrix12x12>,  // diag_hessian
                                                lcrt::Buffer<LMatrix12x12>,  // diag_inv
                                                uint32_t>(                   // body count
        [](lcrt::BufferVar<LMatrix12x12> diag_hessian,
           lcrt::BufferVar<LMatrix12x12> diag_inv,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto D = DiagInvDeviceView::from_handles(diag_hessian, n);
                auto H = abd_diag_preconditioner_detail::assemble_row(
                    D, lcrt::cast<std::int32_t>(i));

                lcrt::Var<LMatrix12x12> out;
                store_mat12x12(out, H);
                diag_inv->write(i, out);
            };
        });
    return k;
}

/// `ABDDiagPreconditioner_do_apply_kernel`.
auto& abd_diag_preconditioner_apply_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,        // r (this subsystem's window)
                                                lcrt::Buffer<double>,        // z (this subsystem's window)
                                                lcrt::Buffer<std::int32_t>,  // converged
                                                lcrt::Buffer<LMatrix12x12>,  // diag_inv
                                                uint32_t>(                   // block count
        [](lcrt::BufferVar<double>        r,
           lcrt::BufferVar<double>        z,
           lcrt::BufferVar<std::int32_t>  converged,
           lcrt::BufferVar<LMatrix12x12>  diag_inv,
           lcrt::Var<uint32_t>            n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // the reference's `if(*converged != 0) return;` -- one scalar read
                // for the whole dispatch, so it wraps the body
                $if(converged->read(0) == 0)
                {
                    constexpr auto kDof = abd_linear_subsystem_detail::kDofPerBody;
                    auto R = CDenseDeviceView::from_handles(r, n * kDof);
                    auto Z = DenseDeviceView::from_handles(z, n * kDof);
                    auto D = DiagInvDeviceView::from_handles(diag_inv, n);
                    abd_diag_preconditioner_detail::apply_row(
                        Z, R, D, lcrt::cast<std::int32_t>(i));
                };
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);

void ABDDiagPreconditioner::do_build(BuildInfo& info)
{
    // the reference's first `require` is only a dependency edge (the base class
    // requires the same system in its own `do_build`), so it stays unused.
    auto& global_linear_system = require<GlobalLinearSystem>();
    (void)global_linear_system;

    abd_linear_subsystem = &require<ABDLinearSubsystem>();

    info.connect(abd_linear_subsystem);
}

void ABDDiagPreconditioner::do_init(InitInfo& info)
{
    (void)info;
}

void ABDDiagPreconditioner::do_assemble(
    GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info)
{
    (void)info;
    auto diag_hessian = abd_linear_subsystem->diag_hessian();
    diag_inv.resize(diag_hessian.size());

    auto n = static_cast<uint32_t>(diag_inv.size());
    if(n > 0)
    {
        auto& k = abd_diag_preconditioner_assemble_kernel();
        frame_stream().record(k(diag_hessian, diag_inv.view(), n).dispatch(n));
    }
}

void ABDDiagPreconditioner::do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info)
{
    auto converged = info.converged();

    auto n = static_cast<uint32_t>(diag_inv.size());
    if(n > 0)
    {
        auto& k = abd_diag_preconditioner_apply_kernel();
        frame_stream().record(k(info.r().buffer_view(),
                                info.z().buffer_view(),
                                converged,
                                diag_inv.cview(),
                                n)
                                  .dispatch(n));
    }
}
}  // namespace uipc::backend::luisa

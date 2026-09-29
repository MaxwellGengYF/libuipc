#include <affine_body/abd_tolerance_checker.h>
#include <affine_body/abd_tolerance_checker_function.h>
#include <frame_stream.h>
#include <fmt/format.h>

/*************************************************************************************************
* T6.4a — ABDToleranceChecker (port of
* src/backends/cuda/affine_body/abd_tolerance_checker.cu).
*
* Read the scope note in abd_tolerance_checker.h first (it carries the two pinned
* reference quirks).  Kernel mapping:
*   abd_tolerance_checker_do_check_kernel -> ..._detail::failed_row (+ the caller's
*   atomic exchange and the `success` flag store).
*
* Reference-fidelity notes:
*  * `BufferLaunch().fill(success.view(), 1)` (a *command-stream* fill in the
*    reference, on the legacy default stream, the same one its kernel is launched
*    on) becomes `success.copy_from(IndexT{1}, &frame_stream())` here: the port's
*    kernels are recorded on the frame stream and a fill on the main stream would
*    break the ordering (cookbook §7.22 rule 7 -- the T5.4a/T5.4b finding).  Both
*    the seed and the read-back are therefore on the frame stream.
*  * `cuda_tool::atomic_exch(success.data(), 0)` has no direct LC spelling in the
*    DSL's *view* API; the port uses LC's own `Buffer<T>::atomic(i).exchange(v)`
*    through the `luisa_device::atomic_exch` shim (luisa_device/atomic.h, the file
*    that exists exactly for these DSL gaps) -- same semantics: an
*    `ATOMIC_EXCHANGE` returning the *previous* value, which the reference discards.
*  * the reference's `if(i >= n) return;` and the `success_value == 0` early return
*    are the kernel's two `$if` guards (RULE M2: both predicates are device data).
*************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
using DqDeviceView     = abd_tolerance_checker_detail::DqDeviceView;
using SuccessDeviceVar = lcrt::BufferVar<std::int32_t>;

/// `abd_tolerance_checker_do_check_kernel`.
auto& abd_tolerance_checker_do_check_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector12>,  // dqs
                                                lcrt::Buffer<std::int32_t>,  // success
                                                double,                      // abs_tol
                                                uint32_t>(                   // body count
        [](lcrt::BufferVar<LVector12>   dqs,
           SuccessDeviceVar             success,
           lcrt::Var<double>            abs_tol,
           lcrt::Var<uint32_t>          n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto D = DqDeviceView::from_handles(dqs, n);

                // `IndexT success_value = *success;` -- a plain device read
                auto success_value = success->read(0);

                // `if(success_value == 0) return;` -- already failed stays failed
                $if(abd_tolerance_checker_detail::failed_row(
                        D, lcrt::cast<std::int32_t>(I), abs_tol, success_value))
                {
                    (void)luisa_device::atomic_exch(success->atomic(0), std::int32_t{0});
                };
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(ABDToleranceChecker);

void ABDToleranceChecker::do_build(BuildInfo& info)
{
    (void)info;
    affine_body_dynamics = require<AffineBodyDynamics>();
    auto& config         = world().scene().config();
    dt_attr              = config.find<Float>("dt");
    UIPC_ASSERT(dt_attr, "Scene config must have a 'dt' attribute.");
    auto transrate_tol_attr = config.find<Float>("newton/transrate_tol");
    transrate_tol           = transrate_tol_attr->view()[0];
}

void ABDToleranceChecker::do_check(CheckResultInfo& info)
{
    abs_tol  = transrate_tol * dt_attr->view()[0];
    auto dqs = affine_body_dynamics->dqs();
    success.copy_from(IndexT{1}, &frame_stream());  // reset success flag

    auto n = static_cast<uint32_t>(dqs.size());
    if(n > 0)
    {
        auto& k = abd_tolerance_checker_do_check_kernel();
        frame_stream().record(k(dqs, success.view(), abs_tol, n).dispatch(n));
    }

    // copy from device to host (the reference's implicit `bool h_success = success;`;
    // note the LOCAL shadows the member -- see the header's pinned quirk 1)
    bool h_success = success.copy_to(&frame_stream()) != 0;
    info.converged(h_success);
}

std::string ABDToleranceChecker::do_report()
{
    return fmt::format("Tol: {}{}", (h_success ? "< " : "> "), abs_tol);
}
}  // namespace uipc::backend::luisa

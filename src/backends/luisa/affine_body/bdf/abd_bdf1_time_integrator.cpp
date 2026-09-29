#include <affine_body/bdf/abd_bdf1_time_integrator.h>
#include <frame_stream.h>

/***************************************************************
 * T6.2 — ABDBDF1Integrator: the two kernels, their cached shaders and the system
 * registration (reference: src/backends/cuda/affine_body/bdf/abd_bdf1_time_integrator.cu).
 *
 * Kernel mapping (cookbook §3.1/§7.10):
 *   `__global__ void k(views..., Float dt, int n)` with
 *   `blockIdx * blockDim + threadIdx` bounds check
 *   -> `lla::detail::cached_kernel<buffers..., double, uint32_t>(lambda)` with
 *   `dispatch_x()` and `$if(i < n)`, dispatched over
 *   `frame_stream().record(k(...).dispatch(n))`.
 *   `cuda_tool::CBufferView<Vector12>`/`BufferView<Vector12>` ->
 *   `lud::LBufferView<LVector12>` (the `LBufferView` converts implicitly to
 *   `lcrt::BufferView<T>`, so the views go to the invocation directly, T5.3's
 *   note); a 12-vector element is loaded with `load_vec12()` and written with
 *   `store_vec12(Var<LVector12>&, ...)` (affine_body/type_define.h) -- the ABD
 *   analogue of the FEM kernels' `load_vec3`/`store_vec3`.
 * The per-body bodies are the generic `bdf1_detail` routines, i.e. the same code the
 * unit test instantiates (cookbook §7.4).  `BDF1Flag` is required, so only the scene
 * config that selects BDF1 builds this system.
 ***************************************************************/
namespace uipc::backend::luisa
{
namespace
{
/// `abd_bdf1_integrator_predict_dof_kernel`
auto& bdf1_predict_dof_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,    // is_fixed
                                                lcrt::Buffer<int32_t>,    // is_dynamic
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_vs
                                                lcrt::Buffer<LVector12>,  // q_tildes
                                                lcrt::Buffer<LVector12>,  // gravities
                                                lcrt::Buffer<LVector12>,  // external_force_accs
                                                double,                   // dt
                                                uint32_t>(                // n
        [](lcrt::BufferVar<int32_t>   is_fixed,
           lcrt::BufferVar<int32_t>   is_dynamic,
           lcrt::BufferVar<LVector12> q_prevs,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_vs,
           lcrt::BufferVar<LVector12> q_tildes,
           lcrt::BufferVar<LVector12> gravities,
           lcrt::BufferVar<LVector12> external_force_accs,
           lcrt::Var<double>          dt,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // record previous q (unconditional in the reference)
                q_prevs->write(i, qs->read(i));

                auto q_tilde = bdf1_detail::predict_dof(load_vec12(qs->read(i)),
                                                       load_vec12(q_vs->read(i)),
                                                       load_vec12(gravities->read(i)),
                                                       load_vec12(external_force_accs->read(i)),
                                                       dt,
                                                       is_fixed->read(i),
                                                       is_dynamic->read(i));

                lcrt::Var<LVector12> out;
                store_vec12(out, q_tilde);
                q_tildes->write(i, out);
            };
        });
    return k;
}

/// `abd_bdf1_integrator_update_state_kernel`
auto& bdf1_update_state_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_vs
                                                lcrt::Buffer<LVector12>,  // q_prevs
                                                double,                   // dt
                                                uint32_t>(                // n
        [](lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_vs,
           lcrt::BufferVar<LVector12> q_prevs,
           lcrt::Var<double>          dt,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto q_v = bdf1_detail::update_state(
                    load_vec12(qs->read(i)), load_vec12(q_prevs->read(i)), dt);

                lcrt::Var<LVector12> out;
                store_vec12(out, q_v);
                q_vs->write(i, out);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(ABDBDF1Integrator);

void ABDBDF1Integrator::do_build(BuildInfo& info)
{
    (void)info;
    // require the BDF1 flag
    require<BDF1Flag>();
}

void ABDBDF1Integrator::do_init(InitInfo& info)
{
    (void)info;
}

void ABDBDF1Integrator::do_predict_dof(PredictDofInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n > 0)
    {
        auto& k = bdf1_predict_dof_kernel();
        frame_stream().record(k(
                                  // NOTE: the *mutable* `q_prevs()` of PredictDofInfo;
                                  // the reference writes `q_n := qs` through it
                                  info.is_fixed(),
                                  info.is_dynamic(),
                                  info.q_prevs(),
                                  info.qs(),
                                  info.q_vs(),
                                  info.q_tildes(),
                                  info.gravities(),
                                  info.external_force_accs(),
                                  info.dt(),
                                  n)
                                  .dispatch(n));
    }
}

void ABDBDF1Integrator::do_update_state(UpdateVelocityInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n > 0)
    {
        auto& k = bdf1_update_state_kernel();
        frame_stream().record(k(info.qs(), info.q_vs(), info.q_prevs(), info.dt(), n).dispatch(n));
    }
}
}  // namespace uipc::backend::luisa

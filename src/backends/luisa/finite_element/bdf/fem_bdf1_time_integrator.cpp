#include <finite_element/bdf/fem_bdf1_time_integrator.h>
#include <frame_stream.h>

/***************************************************************
 * T5.3 — FEMBDF1Integrator: the two kernels, their cached shaders and the
 * system registration (reference:
 * src/backends/cuda/finite_element/bdf/fem_bdf1_time_integrator.cu).
 *
 * Kernel mapping (cookbook §3.1/§7.10):
 *   `__global__ void k(views..., Float dt, int n)` with
 *   `blockIdx * blockDim + threadIdx` bounds check
 *   -> `lla::detail::cached_kernel<buffers..., double, uint32_t>(lambda)` with
 *   `dispatch_x()` and `$if(i < n)`, dispatched over
 *   `frame_stream().record(k(...).dispatch(n))`.
 *   `cuda_tool::CBufferView<T>`/`BufferView<T>` -> `lud::LBufferView<T>` (the
 *   `LBufferView` has an implicit conversion to `lcrt::BufferView<T>`, so the
 *   views are passed to the invocation directly, like the T5.2 constitutions
 *   do); a vector element is loaded with `load_vec3()` and written with
 *   `store_vec3(Var<LVector3>&, ...)` (type_define.h).
 * The per-vertex bodies are the generic `bdf1_detail` routines, i.e. the same
 * code the unit test instantiates (cookbook §7.4).  `BDF1Flag` is required, so
 * only the scene config that selects BDF1 builds this system.
 ***************************************************************/
namespace uipc::backend::luisa
{
namespace
{
/// `FEMBDF1Integrator_do_predict_dof_kernel`
auto& bdf1_predict_dof_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,   // is_fixed
                                                lcrt::Buffer<int32_t>,   // is_dynamic
                                                lcrt::Buffer<LVector3>,  // x_prevs
                                                lcrt::Buffer<LVector3>,  // xs
                                                lcrt::Buffer<LVector3>,  // vs
                                                lcrt::Buffer<LVector3>,  // x_tildes
                                                lcrt::Buffer<LVector3>,  // gravities
                                                lcrt::Buffer<LVector3>,  // external_force_accs
                                                double,                  // dt
                                                uint32_t>(               // n
        [](lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::BufferVar<int32_t>  is_dynamic,
           lcrt::BufferVar<LVector3> x_prevs,
           lcrt::BufferVar<LVector3> xs,
           lcrt::BufferVar<LVector3> vs,
           lcrt::BufferVar<LVector3> x_tildes,
           lcrt::BufferVar<LVector3> gravities,
           lcrt::BufferVar<LVector3> external_force_accs,
           lcrt::Var<double>         dt,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // record previous position (unconditional in the reference)
                x_prevs->write(i, xs->read(i));

                auto x_tilde = bdf1_detail::predict_dof(load_vec3(xs->read(i)),
                                                        load_vec3(vs->read(i)),
                                                        load_vec3(gravities->read(i)),
                                                        load_vec3(external_force_accs->read(i)),
                                                        dt,
                                                        is_fixed->read(i),
                                                        is_dynamic->read(i));

                lcrt::Var<LVector3> out;
                store_vec3(out, x_tilde);
                x_tildes->write(i, out);
            };
        });
    return k;
}

/// `FEMBDF1Integrator_do_update_state_kernel`
auto& bdf1_update_state_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3>,  // xs
                                                lcrt::Buffer<LVector3>,  // vs
                                                lcrt::Buffer<LVector3>,  // x_prevs
                                                double,                  // dt
                                                uint32_t>(               // n
        [](lcrt::BufferVar<LVector3> xs,
           lcrt::BufferVar<LVector3> vs,
           lcrt::BufferVar<LVector3> x_prevs,
           lcrt::Var<double>         dt,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto v = bdf1_detail::update_state(
                    load_vec3(xs->read(i)), load_vec3(x_prevs->read(i)), dt);

                lcrt::Var<LVector3> out;
                store_vec3(out, v);
                vs->write(i, out);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FEMBDF1Integrator);

void FEMBDF1Integrator::do_build(BuildInfo& info)
{
    (void)info;
    // require the BDF1 flag
    require<BDF1Flag>();
}

void FEMBDF1Integrator::do_init(InitInfo& info)
{
    (void)info;
}

void FEMBDF1Integrator::do_predict_dof(PredictDofInfo& info)
{
    auto n = static_cast<uint32_t>(info.xs().size());
    if(n > 0)
    {
        auto& k = bdf1_predict_dof_kernel();
        frame_stream().record(k(
                                  // NOTE: the *mutable* `x_prevs()` of PredictDofInfo;
                                  // the reference writes `x_n := xs` through it
                                  info.is_fixed(),
                                  info.is_dynamic(),
                                  info.x_prevs(),
                                  info.xs(),
                                  info.vs(),
                                  info.x_tildes(),
                                  info.gravities(),
                                  info.external_force_accs(),
                                  info.dt(),
                                  n)
                                  .dispatch(n));
    }
}

void FEMBDF1Integrator::do_update_state(UpdateVelocityInfo& info)
{
    auto n = static_cast<uint32_t>(info.xs().size());
    if(n > 0)
    {
        auto& k = bdf1_update_state_kernel();
        frame_stream().record(k(info.xs(), info.vs(), info.x_prevs(), info.dt(), n).dispatch(n));
    }
}
}  // namespace uipc::backend::luisa

#include <finite_element/bdf/fem_bdf2_time_integrator.h>
#include <frame_stream.h>

/***************************************************************
 * T5.3 — FEMBDF2Integrator: the two kernels, their cached shaders, the
 * history bootstrap and the system registration (reference:
 * src/backends/cuda/finite_element/bdf/fem_bdf2_time_integrator.cu).
 *
 * `cuda_tool::DeviceReduce`-free and branch-free apart from the bounds check:
 * the reference's two `if` chains inside the predict kernel are `select`s in
 * `bdf2_detail::predict_dof` (see fem_bdf2_time_integrator.h), the update kernel
 * writes its three history values in the reference's order (the right-hand sides
 * are all read *before* any write, so the order of the three writes is only
 * load-bearing for `x_n_1s`, whose value is read by nothing later in the kernel).
 *
 * Bootstrap: the reference resizes the state's history buffers to the method's
 * vertex count and copies `x_prevs`/`vs` into them on the first predict_dof --
 * `copy_device` is required here because an `LDeviceBuffer` assignment *shares*
 * the allocation instead of deep-copying it (cookbook §7.16 rule 4b).
 ***************************************************************/
namespace uipc::backend::luisa
{
namespace
{
/// The reference's `state->x_n_1s().copy_from(info.x_prevs())` bootstrap: a
/// *deep* device copy (cuda_tool's `DeviceBuffer::copy_from`), recorded on the
/// shared frame stream like every other command of the frame.  An
/// `LDeviceBuffer` assignment would only share the allocation instead
/// (cookbook §7.16 rule 4b).
void copy_history(lud::LBufferView<LVector3> dst, lud::LCBufferView<LVector3> src)
{
    if(dst.size() != 0u)
    {
        // `.native()` returns the `lcrt::BufferView` whose `copy_from` yields a
        // Command (the `LBufferView` wrapper's `copy_from` would record it on the
        // main stream instead of the frame).
        frame_stream().record(dst.native().copy_from(src.native()));
    }
}

/// `FEMBDF2Integrator_do_predict_dof_kernel`
auto& bdf2_predict_dof_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,   // is_fixed
                                                lcrt::Buffer<int32_t>,   // is_dynamic
                                                lcrt::Buffer<LVector3>,  // xs
                                                lcrt::Buffer<LVector3>,  // x_ns (x_prevs)
                                                lcrt::Buffer<LVector3>,  // x_n_1s
                                                lcrt::Buffer<LVector3>,  // v_ns (vs)
                                                lcrt::Buffer<LVector3>,  // v_n_1s
                                                lcrt::Buffer<LVector3>,  // x_tildes
                                                lcrt::Buffer<LVector3>,  // gravity
                                                lcrt::Buffer<LVector3>,  // external_force_accs
                                                double,                  // dt
                                                uint32_t>(               // n
        [](lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::BufferVar<int32_t>  is_dynamic,
           lcrt::BufferVar<LVector3> xs,
           lcrt::BufferVar<LVector3> x_ns,
           lcrt::BufferVar<LVector3> x_n_1s,
           lcrt::BufferVar<LVector3> v_ns,
           lcrt::BufferVar<LVector3> v_n_1s,
           lcrt::BufferVar<LVector3> x_tildes,
           lcrt::BufferVar<LVector3> gravity,
           lcrt::BufferVar<LVector3> external_force_accs,
           lcrt::Var<double>         dt,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // x_n is tracked in x_prevs for current step.
                x_ns->write(i, xs->read(i));

                auto x_n = load_vec3(xs->read(i));

                auto x_tilde = bdf2_detail::predict_dof(x_n,
                                                        load_vec3(x_n_1s->read(i)),
                                                        load_vec3(v_ns->read(i)),
                                                        load_vec3(v_n_1s->read(i)),
                                                        load_vec3(gravity->read(i)),
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

/// `FEMBDF2Integrator_do_update_state_kernel`
auto& bdf2_update_state_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3>,  // xs
                                                lcrt::Buffer<LVector3>,  // x_ns (x_prevs)
                                                lcrt::Buffer<LVector3>,  // x_n_1s
                                                lcrt::Buffer<LVector3>,  // v_ns (vs)
                                                lcrt::Buffer<LVector3>,  // v_n_1s
                                                double,                  // dt
                                                uint32_t>(               // n
        [](lcrt::BufferVar<LVector3> xs,
           lcrt::BufferVar<LVector3> x_ns,
           lcrt::BufferVar<LVector3> x_n_1s,
           lcrt::BufferVar<LVector3> v_ns,
           lcrt::BufferVar<LVector3> v_n_1s,
           lcrt::Var<double>         dt,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto x   = load_vec3(xs->read(i));
                auto x_n = load_vec3(x_ns->read(i));

                lps::LVec<lcrt::Var<double>, 3> v;
                sym::fem_bdf2::compute_v(v, x, x_n, load_vec3(x_n_1s->read(i)), dt);

                // Update velocity history.
                v_n_1s->write(i, v_ns->read(i));
                lcrt::Var<LVector3> v_out;
                store_vec3(v_out, v);
                v_ns->write(i, v_out);

                // Update position history.
                x_n_1s->write(i, x_ns->read(i));
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FEMBDF2Integrator);

void FEMBDF2Integrator::do_build(BuildInfo& info)
{
    (void)info;
    // require the BDF2 flag
    require<BDF2Flag>();

    state = require<FEMBDF2State>();
}

void FEMBDF2Integrator::do_init(InitInfo& info)
{
    (void)info;
}

void FEMBDF2Integrator::do_predict_dof(PredictDofInfo& info)
{
    if(state->x_n_1s().size() != info.xs().size())
    {
        state->resize(info.xs().size());
        // Bootstrap history from current x_n and v_n.
        copy_history(state->x_n_1s(), info.x_prevs());
        copy_history(state->v_n_1s(), info.vs());
    }

    auto n = static_cast<uint32_t>(info.xs().size());
    if(n > 0)
    {
        auto& k = bdf2_predict_dof_kernel();
        frame_stream().record(k(info.is_fixed(),
                                info.is_dynamic(),
                                info.xs(),
                                info.x_prevs(),
                                state->x_n_1s(),
                                info.vs(),
                                state->v_n_1s(),
                                info.x_tildes(),
                                info.gravities(),
                                info.external_force_accs(),
                                info.dt(),
                                n)
                                .dispatch(n));
    }
}

void FEMBDF2Integrator::do_update_state(UpdateVelocityInfo& info)
{
    auto n = static_cast<uint32_t>(info.xs().size());
    if(n > 0)
    {
        auto& k = bdf2_update_state_kernel();
        frame_stream().record(k(info.xs(),
                                info.x_prevs(),
                                state->x_n_1s(),
                                info.vs(),
                                state->v_n_1s(),
                                info.dt(),
                                n)
                                .dispatch(n));
    }
}
}  // namespace uipc::backend::luisa

#include <affine_body/bdf/abd_bdf2_time_integrator.h>
#include <frame_stream.h>

/***************************************************************
 * T6.2 — ABDBDF2Integrator: the two kernels, their cached shaders, the history
 * bootstrap and the system registration (reference:
 * src/backends/cuda/affine_body/bdf/abd_bdf2_time_integrator.cu).
 *
 * Branch-free apart from the bounds check: the reference's two `if` chains inside
 * the predict kernel are `select`s in `bdf2_detail::predict_dof` (see
 * abd_bdf2_time_integrator.h), and the update kernel writes its three history values
 * in the reference's order (all three right-hand sides are read *before* any write,
 * so only `q_n_1s`'s write, which nothing later reads, is order-sensitive).
 *
 * Bootstrap: the reference resizes the state's history buffers to the dynamics'
 * body count and copies `q_prevs`/`q_vs` into them on the first predict_dof --
 * `copy_device` is required because an `LDeviceBuffer` assignment *shares* the
 * allocation instead of deep-copying it (cookbook §7.16 rule 4b).
 ***************************************************************/
namespace uipc::backend::luisa
{
namespace
{
/// The reference's `state->q_n_1s().copy_from(info.q_prevs())` bootstrap: a *deep*
/// device copy (cuda_tool's `DeviceBuffer::copy_from`), recorded on the shared frame
/// stream like every other command of the frame.  An `LDeviceBuffer` assignment
/// would only share the allocation instead (cookbook §7.16 rule 4b).
void copy_history(lud::LBufferView<LVector12> dst, lud::LCBufferView<LVector12> src)
{
    if(dst.size() != 0u)
    {
        // `.native()` returns the `lcrt::BufferView` whose `copy_from` yields a
        // Command (the `LBufferView` wrapper's `copy_from` would record it on the
        // main stream instead of the frame).
        frame_stream().record(dst.native().copy_from(src.native()));
    }
}

/// `abd_bdf2_integrator_predict_dof_kernel`
auto& bdf2_predict_dof_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,    // is_fixed
                                                lcrt::Buffer<int32_t>,    // is_dynamic
                                                lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_ns (q_prevs)
                                                lcrt::Buffer<LVector12>,  // q_n_1s
                                                lcrt::Buffer<LVector12>,  // q_v_ns (q_vs)
                                                lcrt::Buffer<LVector12>,  // q_v_n_1s
                                                lcrt::Buffer<LVector12>,  // q_tildes
                                                lcrt::Buffer<LVector12>,  // gravity
                                                double,                   // dt
                                                uint32_t>(                // n
        [](lcrt::BufferVar<int32_t>   is_fixed,
           lcrt::BufferVar<int32_t>   is_dynamic,
           lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_ns,
           lcrt::BufferVar<LVector12> q_n_1s,
           lcrt::BufferVar<LVector12> q_v_ns,
           lcrt::BufferVar<LVector12> q_v_n_1s,
           lcrt::BufferVar<LVector12> q_tildes,
           lcrt::BufferVar<LVector12> gravity,
           lcrt::Var<double>          dt,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // q_n is tracked in q_prevs for the current step.
                q_ns->write(i, qs->read(i));

                auto q_n = load_vec12(qs->read(i));

                auto q_tilde = bdf2_detail::predict_dof(q_n,
                                                        load_vec12(q_n_1s->read(i)),
                                                        load_vec12(q_v_ns->read(i)),
                                                        load_vec12(q_v_n_1s->read(i)),
                                                        // NOTE: gravity *alone* -- the
                                                        // reference's ABD BDF2 kernel has
                                                        // no external-force argument
                                                        // (see the header)
                                                        load_vec12(gravity->read(i)),
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

/// `abd_bdf2_integrator_update_state_kernel`
auto& bdf2_update_state_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector12>,  // qs
                                                lcrt::Buffer<LVector12>,  // q_ns (q_prevs)
                                                lcrt::Buffer<LVector12>,  // q_n_1s
                                                lcrt::Buffer<LVector12>,  // q_v_ns (q_vs)
                                                lcrt::Buffer<LVector12>,  // q_v_n_1s
                                                double,                   // dt
                                                uint32_t>(                // n
        [](lcrt::BufferVar<LVector12> qs,
           lcrt::BufferVar<LVector12> q_ns,
           lcrt::BufferVar<LVector12> q_n_1s,
           lcrt::BufferVar<LVector12> q_v_ns,
           lcrt::BufferVar<LVector12> q_v_n_1s,
           lcrt::Var<double>          dt,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto q   = load_vec12(qs->read(i));
                auto q_n = load_vec12(q_ns->read(i));

                lps::LVec<lcrt::Var<double>, 12> qv;
                sym::abd_bdf2::compute_qv(qv, q, q_n, load_vec12(q_n_1s->read(i)), dt);

                // Update velocity history.
                q_v_n_1s->write(i, q_v_ns->read(i));
                lcrt::Var<LVector12> qv_out;
                store_vec12(qv_out, qv);
                q_v_ns->write(i, qv_out);

                // Update position history.
                q_n_1s->write(i, q_ns->read(i));
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(ABDBDF2Integrator);

void ABDBDF2Integrator::do_build(BuildInfo& info)
{
    (void)info;
    // require the BDF2 flag
    require<BDF2Flag>();

    state = require<ABDBDF2State>();
}

void ABDBDF2Integrator::do_init(InitInfo& info)
{
    (void)info;
}

void ABDBDF2Integrator::do_predict_dof(PredictDofInfo& info)
{
    if(state->q_n_1s().size() != info.qs().size())
    {
        state->resize(info.qs().size());
        // Bootstrap history from current q_n and q_v.
        copy_history(state->q_n_1s(), info.q_prevs());
        copy_history(state->q_v_n_1s(), info.q_vs());
    }

    auto n = static_cast<uint32_t>(info.qs().size());
    if(n > 0)
    {
        auto& k = bdf2_predict_dof_kernel();
        frame_stream().record(k(info.is_fixed(),
                                info.is_dynamic(),
                                info.qs(),
                                info.q_prevs(),
                                state->q_n_1s(),
                                info.q_vs(),
                                state->q_v_n_1s(),
                                info.q_tildes(),
                                info.gravities(),
                                info.dt(),
                                n)
                                .dispatch(n));
    }
}

void ABDBDF2Integrator::do_update_state(UpdateVelocityInfo& info)
{
    auto n = static_cast<uint32_t>(info.qs().size());
    if(n > 0)
    {
        auto& k = bdf2_update_state_kernel();
        frame_stream().record(k(info.qs(),
                                info.q_prevs(),
                                state->q_n_1s(),
                                info.q_vs(),
                                state->q_v_n_1s(),
                                info.dt(),
                                n)
                                .dispatch(n));
    }
}
}  // namespace uipc::backend::luisa

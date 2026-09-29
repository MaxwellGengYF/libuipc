#include <finite_element/finite_element_external_force_manager.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_external_force_reporter.h>
#include <frame_stream.h>
#include <luisa_device/linear/linear.h>

/***************************************************************
 * T5.3 — FEMExternalForceManager (reference:
 * src/backends/cuda/finite_element/finite_element_external_force_manager.cu).
 *
 * The two kernels:
 *   clear: forces(i) = 0
 *   step : `a = F / m` if `m > 0`, else `a = 0`.
 * The `m > 0` test is *device* data, so it cannot be a host `if` (RULE M2); the
 * reference's two branches are evaluated and `select`ed, which is bit-identical
 * (the unselected candidate is computed from the same operands and discarded).
 * A massless vertex therefore gets exactly `(0, 0, 0)`, and a vertex with a
 * negative mass (which the reference also sends to the `setZero()` branch, because
 * `m > 0.0` is false) gets zero as well.
 *
 * `vertex_external_forces`/`vertex_external_force_accs` are the method's buffers;
 * the manager reaches them through `finite_element_method->m_impl` (the reference
 * declares `friend class FEMExternalForceManager;` in the method).
 ***************************************************************/
namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

namespace
{
/// `FEMExternalForceManager_clear_kernel`
auto& fem_external_force_clear_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3>,  // forces
                                                uint32_t>(               // n
        [](lcrt::BufferVar<LVector3> forces, lcrt::Var<uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // a buffer element is written as a `Var<LVector3>`: a bare
                // `LVector3` is not an `Expr<LVector3>` (T4.2's note)
                lcrt::Var<LVector3> zero;
                store_vec3(zero,
                           lps::make_vec<lcrt::Var<double>, 3>(
                               [](int) noexcept { return lcrt::Var<double>{0.0}; }));
                forces->write(i, zero);
            };
        });
    return k;
}

/// `FEMExternalForceManager_step_kernel`
auto& fem_external_force_step_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3>,  // forces
                                                lcrt::Buffer<LVector3>,  // force_accs
                                                lcrt::Buffer<double>,    // masses
                                                uint32_t>(               // n
        [](lcrt::BufferVar<LVector3> forces,
           lcrt::BufferVar<LVector3> force_accs,
           lcrt::BufferVar<double>   masses,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto F = load_vec3(forces->read(i));
                auto m = masses->read(i);

                // a = F / m (avoid division by zero for massless vertices).
                // The `m > 0.0` test is device data, so the reference's two
                // branches become an element-wise select of the *same* operands
                // (the discarded candidate is exactly what the reference discards).
                auto a  = F / m;
                auto ok = (m > 0.0);
                auto out = lps::make_vec<lcrt::Var<double>, 3>(
                    [&](int k) noexcept
                    { return lcrt::select(lcrt::Var<double>{0.0}, a(k), ok); });

                lcrt::Var<LVector3> acc;
                store_vec3(acc, out);
                force_accs->write(i, acc);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FEMExternalForceManager);

void FEMExternalForceManager::do_build(BuildInfo& info)
{
    (void)info;
    m_impl.finite_element_method = &require<FiniteElementMethod>();
}

void FEMExternalForceManager::register_reporter(FiniteElementExternalForceReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "register_reporter");
    m_impl.m_reporters.register_sim_system(*reporter);
}

void FEMExternalForceManager::Impl::clear()
{
    auto external_forces = finite_element_method->m_impl.vertex_external_forces.view();

    auto n = static_cast<uint32_t>(external_forces.size());
    if(n > 0)
    {
        auto& k = fem_external_force_clear_kernel();
        frame_stream().record(k(external_forces, n).dispatch(n));
    }
}

void FEMExternalForceManager::Impl::step()
{
    ExternalForceInfo info{this};
    for(auto reporter : m_reporters.view())
    {
        reporter->step(info);
    }

    auto& fem = finite_element_method->m_impl;

    auto force_accs = fem.vertex_external_force_accs.view();
    auto forces     = fem.vertex_external_forces.view();
    auto masses     = finite_element_method->masses();

    auto n = static_cast<uint32_t>(forces.size());
    if(n > 0)
    {
        auto& k = fem_external_force_step_kernel();
        frame_stream().record(k(forces, force_accs, masses, n).dispatch(n));
    }
}

void FEMExternalForceManager::do_init()
{
    for(auto reporter : m_impl.m_reporters.view())
    {
        reporter->init();
    }
}

void FEMExternalForceManager::do_clear()
{
    m_impl.clear();
}

void FEMExternalForceManager::do_step()
{
    m_impl.step();
}

lud::LBufferView<LVector3> FEMExternalForceManager::ExternalForceInfo::external_forces() noexcept
{
    return m_impl->finite_element_method->m_impl.vertex_external_forces.view();
}
}  // namespace uipc::backend::luisa

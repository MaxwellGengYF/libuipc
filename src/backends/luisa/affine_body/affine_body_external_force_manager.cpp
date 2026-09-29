#include <affine_body/affine_body_external_force_manager.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_external_force_reporter.h>
#include <frame_stream.h>
#include <luisa_device/linear/linear.h>

/********************************************************************
 * T6.1a — AffineBodyExternalForceManager (reference:
 * src/backends/cuda/affine_body/affine_body_external_force_manager.cu).
 *
 * The two kernels:
 *   clear: forces(i) = 0
 *   step : force_accs(i) = M_inv(i) * F(i)   (LMat * LVec, exactly the
 *          reference's `M_inv * F`; both operands are loaded from reachable
 *          buffers, so no selection is needed here -- unlike the FEM sibling,
 *          this kernel has no device-side branch at all)
 *
 * `body_id_to_external_force` / `body_id_to_external_force_acc` are the dynamics'
 * buffers; the manager reaches them through `affine_body_dynamics->m_impl` (the
 * reference declares `friend class AffineBodyExternalForceManager;` in
 * AffineBodyDynamics).
 *********************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla_ = luisa_device::linalg;

/// `AffineBodyExternalForceManager_clear_kernel`
auto& abd_external_force_clear_kernel()
{
    static auto& k = lla_::detail::cached_kernel<lcrt::Buffer<LVector12>,  // forces
                                                 uint32_t>(                // n
        [](lcrt::BufferVar<LVector12> forces, lcrt::Var<uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // a buffer element is written as a `lcrt::Var<...>`: a bare
                // `LVector12` is not an `Expr` (T4.2's note)
                lcrt::Var<LVector12> zero;
                store_vec12(zero,
                            lps::make_vec<lcrt::Var<double>, 12>(
                                [](int) noexcept { return lcrt::Var<double>{0.0}; }));
                forces->write(i, zero);
            };
        });
    return k;
}

/// `AffineBodyExternalForceManager_step_kernel`
auto& abd_external_force_step_kernel()
{
    static auto& k = lla_::detail::cached_kernel<lcrt::Buffer<LVector12>,     // forces
                                                 lcrt::Buffer<LVector12>,     // force_accs
                                                 lcrt::Buffer<LMatrix12x12>,  // masses_inv
                                                 uint32_t>(                   // n
        [](lcrt::BufferVar<LVector12>    forces,
           lcrt::BufferVar<LVector12>    force_accs,
           lcrt::BufferVar<LMatrix12x12> masses_inv,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto F     = load_vec12(forces->read(i));
                auto M_inv = load_mat12x12(masses_inv->read(i));

                // Compute acceleration: a = M^{-1} * F (like gravity)
                lcrt::Var<LVector12> acc;
                store_vec12(acc, M_inv * F);
                force_accs->write(i, acc);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyExternalForceManager);

void AffineBodyExternalForceManager::do_build(BuildInfo& info)
{
    (void)info;
    m_impl.affine_body_dynamics = &require<AffineBodyDynamics>();
}

void AffineBodyExternalForceManager::register_reporter(AffineBodyExternalForceReporter* reporter)
{
    check_state(SimEngineState::BuildSystems, "register_reporter");
    m_impl.m_reporters.register_sim_system(*reporter);
}

void AffineBodyExternalForceManager::Impl::clear()
{
    // Clear external force buffer BEFORE constraints write to it
    auto external_forces = affine_body_dynamics->m_impl.body_id_to_external_force.view();

    auto n = static_cast<uint32_t>(external_forces.size());
    if(n > 0)
    {
        auto& k = abd_external_force_clear_kernel();
        frame_stream().record(k(external_forces, n).dispatch(n));
    }
}

void AffineBodyExternalForceManager::Impl::step()
{
    // Step all sub-reporters
    ExternalForceInfo info{this};
    for(auto reporter : m_reporters.view())
    {
        reporter->step(info);
    }

    // At this point, constraints have already written to external_force buffer
    // Now compute accelerations from external forces
    auto& abd = affine_body_dynamics->m_impl;

    auto force_accs = abd.body_id_to_external_force_acc.view();

    auto forces     = affine_body_dynamics->body_external_forces();
    auto masses_inv = affine_body_dynamics->body_mass_invs();

    auto n = static_cast<uint32_t>(forces.size());
    if(n > 0)
    {
        auto& k = abd_external_force_step_kernel();
        frame_stream().record(k(forces, force_accs, masses_inv, n).dispatch(n));
    }
}

void AffineBodyExternalForceManager::do_init()
{
    // Initialize all sub-reporters
    for(auto reporter : m_impl.m_reporters.view())
    {
        reporter->init();
    }
}

void AffineBodyExternalForceManager::do_clear()
{
    m_impl.clear();
}

void AffineBodyExternalForceManager::do_step()
{
    m_impl.step();
}

lud::LBufferView<LVector12> AffineBodyExternalForceManager::ExternalForceInfo::external_forces() noexcept
{
    return m_impl->affine_body_dynamics->m_impl.body_id_to_external_force.view();
}
}  // namespace uipc::backend::luisa

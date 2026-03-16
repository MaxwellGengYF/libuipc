#include <affine_body/affine_body_external_force_manager.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_external_force_reporter.h>
#include <luisa/dsl/syntax.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(AffineBodyExternalForceManager);

void AffineBodyExternalForceManager::do_build(BuildInfo& info)
{
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
    // Read Write BufferView
    auto external_forces =
        affine_body_dynamics->m_impl.body_id_to_external_force.view();

    auto& device = affine_body_dynamics->engine().luisa_device();
    auto& stream = affine_body_dynamics->engine().compute_stream();

    Kernel1D clear_forces_kernel = [&](BufferVar<Vector12> forces) noexcept
    {
        auto i = dispatch_id().x;
        forces.write(i, Vector12::Zero());
    };

    auto shader = device.compile(clear_forces_kernel);
    stream << shader(external_forces).dispatch(external_forces.size());
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
    
    // Read Write BufferView
    auto force_accs = abd.body_id_to_external_force_acc.view();

    // Read Only BufferViews
    auto forces     = affine_body_dynamics->body_external_forces();
    auto masses_inv = affine_body_dynamics->body_mass_invs();

    SizeT body_count = forces.size();

    auto& device = affine_body_dynamics->engine().luisa_device();
    auto& stream = affine_body_dynamics->engine().compute_stream();

    Kernel1D compute_acceleration_kernel = [&](BufferVar<Vector12> force_accs,
                                                BufferVar<Vector12> forces,
                                                BufferVar<Matrix12x12> masses_inv) noexcept
    {
        auto i = dispatch_id().x;
        if_(i < body_count, [&] {
            auto F = forces.read(i);
            auto M_inv = masses_inv.read(i);

            // Compute acceleration: a = M^{-1} * F (like gravity)
            force_accs.write(i, M_inv * F);
        });
    };

    auto shader = device.compile(compute_acceleration_kernel);
    stream << shader(force_accs, forces, masses_inv).dispatch(body_count);
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

luisa::compute::BufferView<Vector12> AffineBodyExternalForceManager::ExternalForceInfo::external_forces() noexcept
{
    return m_impl->affine_body_dynamics->m_impl.body_id_to_external_force.view();
}
}  // namespace uipc::backend::luisa

#include <finite_element/finite_element_kinetic.h>
#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/fem_line_search_reporter.h>

/***************************************************************
 * T5.3 — FiniteElementKinetic's wiring (reference:
 * src/backends/cuda/finite_element/finite_element_kinetic.cu).
 *
 * do_build() is the reference's three-step graph edge: the method, then both
 * consumers register this system (`FEMLineSearchReporter::add_kinetic`,
 * `FEMLinearSubsystem::add_kinetic`) -- both check
 * `SimEngineState::BuildSystems`, exactly like the reference's `check_state`.
 * The two forwarding overrides wrap the outer info into this class's nested
 * `Compute*Info`, whose base accessors are the method's buffers.
 *
 * `FEMLinearSubsystem::add_kinetic` is defined here (next to its only caller), as
 * the T5.1 scope note in fem_linear_subsystem.h says T5.4 would; it keeps T5.1's
 * four definitions in fem_linear_subsystem_reporter.cpp untouched.
 ***************************************************************/
namespace uipc::backend::luisa
{
void FiniteElementKinetic::do_build()
{
    // provide fem data for solving
    m_impl.finite_element_method = require<FiniteElementMethod>();

    auto& linear_system = require<FEMLinearSubsystem>();
    auto& line_searcher = require<FEMLineSearchReporter>();

    BuildInfo this_info;
    do_build(this_info);

    line_searcher.add_kinetic(this);
    linear_system.add_kinetic(this);
}

void FiniteElementKinetic::compute_energy(FEMLineSearchReporter::ComputeEnergyInfo& info)
{
    ComputeEnergyInfo this_info{&m_impl, &info};
    do_compute_energy(this_info);
}

void FiniteElementKinetic::compute_gradient_hessian(FEMLinearSubsystem::ComputeGradientHessianInfo& info)
{
    ComputeGradientHessianInfo this_info{&m_impl, &info};
    do_compute_gradient_hessian(this_info);
}

void FEMLinearSubsystem::add_kinetic(FiniteElementKinetic* kinetic)
{
    UIPC_ASSERT(kinetic, "kinetic is null");
    check_state(SimEngineState::BuildSystems, "add_kinetic()");
    m_impl.kinetic.register_sim_system(*kinetic);
}
}  // namespace uipc::backend::luisa

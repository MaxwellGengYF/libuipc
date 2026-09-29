#include <finite_element/fem_time_integrator.h>

/***************************************************************
 * T5.3 — the four forwarding overrides of FEMTimeIntegrator.
 * Identical to the reference's fem_time_integrator.cu; the only change is the
 * namespace.  `require<FiniteElementMethod>()` lives in do_build(), so a scene
 * without a FiniteElement constitution (and without the `Constraint` type that
 * would create the animator) never builds a concrete BDF integrator.
 ***************************************************************/
namespace uipc::backend::luisa
{
void FEMTimeIntegrator::do_build(TimeIntegrator::BuildInfo& info)
{
    m_impl.finite_element_method = require<FiniteElementMethod>();

    BuildInfo this_info;
    do_build(this_info);
}

void FEMTimeIntegrator::do_init(TimeIntegrator::InitInfo& info)
{
    InitInfo this_info;
    do_init(this_info);
}

void FEMTimeIntegrator::do_predict_dof(TimeIntegrator::PredictDofInfo& info)
{
    PredictDofInfo this_info{&m_impl, &info};
    do_predict_dof(this_info);
}

void FEMTimeIntegrator::do_update_state(TimeIntegrator::UpdateVelocityInfo& info)
{
    UpdateVelocityInfo this_info{&m_impl, &info};
    do_update_state(this_info);
}
}  // namespace uipc::backend::luisa

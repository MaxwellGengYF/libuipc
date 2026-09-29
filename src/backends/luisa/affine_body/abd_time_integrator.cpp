#include <affine_body/abd_time_integrator.h>
/********************************************************************
 * @file affine_body/abd_time_integrator.cpp
 * @brief T6.1a — ABDTimeIntegrator's plumbing (port of the reference's
 * abd_time_integrator.cu).  The four forwards plus the `require<AffineBodyDynamics>`
 * in `do_build`; the concrete BDF integrators are T6.2's.
 *********************************************************************/
namespace uipc::backend::luisa
{
void ABDTimeIntegrator::do_build(TimeIntegrator::BuildInfo& info)
{
    (void)info;
    m_impl.affine_body_dynamics = require<AffineBodyDynamics>();

    BuildInfo this_info;
    do_build(this_info);
}

void ABDTimeIntegrator::do_init(TimeIntegrator::InitInfo& info)
{
    (void)info;
    InitInfo this_info;
    do_init(this_info);
}

void ABDTimeIntegrator::do_predict_dof(TimeIntegrator::PredictDofInfo& info)
{
    PredictDofInfo this_info{&m_impl, &info};
    do_predict_dof(this_info);
}

void ABDTimeIntegrator::do_update_state(TimeIntegrator::UpdateVelocityInfo& info)
{
    UpdateVelocityInfo this_info{&m_impl, &info};
    do_update_state(this_info);
}
}  // namespace uipc::backend::luisa

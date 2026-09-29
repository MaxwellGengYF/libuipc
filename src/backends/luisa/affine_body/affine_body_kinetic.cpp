#include <affine_body/affine_body_kinetic.h>
/********************************************************************
 * @file affine_body/affine_body_kinetic.cpp
 * @brief T6.1a — AffineBodyKinetic's plumbing (port of the reference's
 * affine_body_kinetic.cu; the reference's unused
 * <finite_element/finite_element_diff_dof_reporter.h> include is dropped, T5.3).
 *
 * The concrete kinetics (BDF1/BDF2) arrive in T6.2; this file only has to make the
 * base class link and to register the *one* kinetic system with the dynamics
 * (`AffineBodyDynamics::add_kinetic`), which is what `Impl::kinetic` is for.
 *********************************************************************/
namespace uipc::backend::luisa
{
void AffineBodyKinetic::do_build()
{
    m_impl.affine_body_dynamics = require<AffineBodyDynamics>();

    BuildInfo this_info;
    do_build(this_info);

    m_impl.affine_body_dynamics->add_kinetic(this);
}

void AffineBodyKinetic::compute_energy(ABDLineSearchReporter::ComputeEnergyInfo& info)
{
    ComputeEnergyInfo this_info{&m_impl, &info};
    do_compute_energy(this_info);
}

void AffineBodyKinetic::compute_gradient_hessian(
    ABDLinearSubsystem::ComputeGradientHessianInfo& info)
{
    ComputeGradientHessianInfo this_info{&m_impl, &info};
    do_compute_gradient_hessian(this_info);
}
}  // namespace uipc::backend::luisa

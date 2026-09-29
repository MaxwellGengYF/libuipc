#include <affine_body/affine_body_constitution.h>
/********************************************************************
 * @file affine_body/affine_body_constitution.cpp
 * @brief T6.1a — AffineBodyConstitution's plumbing (port of the reference's
 * affine_body_constitution.cu).
 *
 * Includes the reference's three ui/d-based hooks and the `BaseInfo` window
 * accessors.  `do_build()` keeps the reference's
 * `throw SimSystemException(fmt::format("Requires Constitution UID={}", uid()))`
 * verbatim: a test-local constitution must report exactly the uid the *frontend*
 * wrote into the geometry's `constitution_uid` meta attribute, which is what makes
 * the uid contract observable in the T6.1a scene case.
 *********************************************************************/
namespace uipc::backend::luisa
{
U64 AffineBodyConstitution::uid() const
{
    return get_uid();
}

void AffineBodyConstitution::do_build()
{
    m_impl.affine_body_dynamics = require<AffineBodyDynamics>();

    auto scene = world().scene();
    // Check if we have the Affine Body Constitution
    auto uids = scene.constitution_tabular().uids();
    if(!std::binary_search(uids.begin(), uids.end(), uid()))
    {
        throw SimSystemException(fmt::format("Requires Constitution UID={}", uid()));
    }

    BuildInfo info;
    do_build(info);

    m_impl.affine_body_dynamics->add_constitution(this);
}

void AffineBodyConstitution::init(AffineBodyDynamics::FilteredInfo& info)
{
    return do_init(info);
}

void AffineBodyConstitution::compute_energy(ABDLineSearchReporter::ComputeEnergyInfo& info)
{
    ComputeEnergyInfo this_info{&m_impl, m_index, &info};
    return do_compute_energy(this_info);
}

void AffineBodyConstitution::compute_gradient_hessian(
    ABDLinearSubsystem::ComputeGradientHessianInfo& info)
{
    ComputeGradientHessianInfo this_info{&m_impl, m_index, &info};
    return do_compute_gradient_hessian(this_info);
}

lud::LCBufferView<LVector12> AffineBodyConstitution::BaseInfo::qs() const noexcept
{
    auto& abd = m_impl->abd();
    return abd.subview(abd.body_id_to_q, m_index);
}

lud::LCBufferView<LVector12> AffineBodyConstitution::BaseInfo::q_prevs() const noexcept
{
    auto& abd = m_impl->abd();
    return abd.subview(abd.body_id_to_q_prev, m_index);
}

lud::LCBufferView<Float> AffineBodyConstitution::BaseInfo::volumes() const noexcept
{
    auto& abd = m_impl->abd();
    return abd.subview(abd.body_id_to_volume, m_index);
}
}  // namespace uipc::backend::luisa

#include <affine_body/affine_body_animator.h>
/********************************************************************
 * @file affine_body/affine_body_animator.cpp
 * @brief SCOPE NOTE (T6.1b) — the three AffineBodyAnimator members T6.1a must
 * define so that the in-scope interfaces link (see the header's scope note).
 * Everything else the reference defines in affine_body_animator.cu stays T6.1b's,
 * undefined (never stubbed).
 *
 * `ReportExtentInfo::check` is here because `AffineBodyConstraint::report_extent`
 * (T6.1a) calls it and that call is reachable — leaving it undefined would be a
 * link error, not a not-implemented signal.  The body is the reference's one
 * line, forwarded to the shared `check_report_extent` helper (cookbook §7.17).
 *********************************************************************/
#include <affine_body/affine_body_constraint.h>
#include <utils/report_extent_check.h>

namespace uipc::backend::luisa
{
void AffineBodyAnimator::add_constraint(AffineBodyConstraint* constraint)
{
    m_impl.constraints.register_sim_system(*constraint);
}

// The reference declares BaseInfo::dt() and never defines it (the same finding
// T5.3 recorded for FiniteElementAnimator::BaseInfo::dt()); `m_dt` is the only
// sensible reading of the member it stores.
Float AffineBodyAnimator::BaseInfo::dt() const noexcept
{
    return m_dt;
}

void AffineBodyAnimator::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_block_count, name);
}
}  // namespace uipc::backend::luisa

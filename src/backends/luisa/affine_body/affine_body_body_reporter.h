#pragma once
/********************************************************************
 * @file affine_body_body_reporter.h
 * @brief T6.1a — ABD body reporter (port of
 * src/backends/cuda/affine_body/affine_body_body_reporter.h).
 *
 * Reports one body per ABD instance (its self-collision flag) to the
 * GlobalBodyManager.  Public surface identical to the reference.
 *
 * The body-count source is `m_impl.body_count()` (= `abd_body_count`, the number
 * of *instances*), while the reference's FEM sibling reports
 * `h_body_self_collision.size()`; both are the same number for ABD (one entry per
 * body), and `report_attributes` asserts the equality against the manager's own
 * buffer size below.
 *********************************************************************/
#include <global_geometry/body_reporter.h>
#include <affine_body/affine_body_dynamics.h>

namespace uipc::backend::luisa
{
class AffineBodyBodyReporter final : public BodyReporter
{
  public:
    using BodyReporter::BodyReporter;

    class Impl
    {
      public:
        void report_count(BodyCountInfo& info);
        void report_attributes(BodyAttributeInfo& info);

        AffineBodyDynamics* affine_body_dynamics = nullptr;
    };

  private:
    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(InitInfo& info) override;
    virtual void do_report_count(BodyCountInfo& info) override;
    virtual void do_report_attributes(BodyAttributeInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::luisa

#pragma once
/********************************************************************
 * @file finite_element_body_reporter.h
 * @brief T5.1 — FEM body reporter (port of
 * src/backends/cuda/finite_element/finite_element_body_reporter.h).
 *
 * Reports one body per FEM geometry (its self-collision flag) to the
 * GlobalBodyManager. Public surface identical to the reference.
 *********************************************************************/
#include <global_geometry/body_reporter.h>
#include <finite_element/finite_element_method.h>

namespace uipc::backend::luisa
{
class FiniteElementBodyReporter final : public BodyReporter
{
  public:
    using BodyReporter::BodyReporter;

    class Impl
    {
      public:
        void report_count(BodyCountInfo& info);
        void report_attributes(BodyAttributeInfo& info);

        FiniteElementMethod* finite_element_method = nullptr;
    };

  private:
    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(InitInfo& info) override;
    virtual void do_report_count(BodyCountInfo& info) override;
    virtual void do_report_attributes(BodyAttributeInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::luisa

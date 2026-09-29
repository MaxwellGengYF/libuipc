/********************************************************************
 * @file empty_0d.cpp
 * @brief T5.2 (batch c) — Empty0D, the "no constitution" placeholder for 0d
 *        geometries (uid 0).
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/empty_0d.cu.
 *
 * Why it exists: the frontend writes `builtin::constitution_uid == 0` into every
 * geometry that no constitution was applied to, and the FEM method only keeps the
 * geometries whose uid is among the *registered* constitutions' uids.  Without this
 * system a uid-0 point cloud would silently drop out of the FEM method; with it, the
 * (dim 0, uid 0) group is owned by a constitution that reports zero energy.  The uid is
 * therefore load-bearing, not decorative.
 *
 * A direct port: same class name, same uid, same `REGISTER_SIM_SYSTEM`, same two
 * overrides.  `do_report_extent` is not overridden (the base's is `final`, all-zero),
 * and, exactly like the 3d/2d/1d siblings, the reference's stray includes
 * (hookean_spring_1d_function.h, kernel_cout.h, cuda_tool, Eigen, codim_thickness) are
 * dropped — nothing in the body uses them.
 ********************************************************************/
#include <finite_element/codim_0d_constitution.h>

namespace uipc::backend::luisa
{
class Empty0D final : public Codim0DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 0ull;

    using Codim0DConstitution::Codim0DConstitution;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override
    {
        (void)info;
        // Do nothing
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        (void)info;
        // Do nothing
    }
};

REGISTER_SIM_SYSTEM(Empty0D);
}  // namespace uipc::backend::luisa

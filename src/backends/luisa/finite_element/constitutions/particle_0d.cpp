/********************************************************************
 * @file particle_0d.cpp
 * @brief T5.2 (batch c) — Particle0D, the 0d particle constitution (uid 13).
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/particle_0d.cu.
 *
 * A direct port: same class name, same uid, same `REGISTER_SIM_SYSTEM`, same two
 * overrides (`do_build`/`do_init` are no-ops — a particle carries no energy, so the
 * whole constitution exists only to own the (dim 0, uid 13) geometry group).
 *
 * DEVIATION (reported in the T5.2 batch-c report): the class stays in the .cpp, as in
 * the reference — it has no public surface a unit test needs (there is nothing to
 * compute), so the header split batch a/b introduced for their per-element buffers does
 * not apply here.  `do_report_extent` is *not* overridden: `Codim0DConstitution`'s own
 * override is `final` in both the reference and the port, and reports all-zero extents.
 ********************************************************************/
#include <finite_element/codim_0d_constitution.h>

namespace uipc::backend::luisa
{
class Particle0D final : public Codim0DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64 ConstitutionUID = 13;

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

REGISTER_SIM_SYSTEM(Particle0D);
}  // namespace uipc::backend::luisa

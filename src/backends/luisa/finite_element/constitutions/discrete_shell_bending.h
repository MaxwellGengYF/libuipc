#pragma once
/********************************************************************
 * @file discrete_shell_bending.h
 * @brief T5.2 (batch b) — DiscreteShellBending, the elastic discrete-shells
 *        bending constitution (uid 17).
 *
 * REFERENCE: src/backends/cuda/finite_element/constitutions/
 * discrete_shell_bending.cu (class + kernels + `REGISTER_SIM_SYSTEM` in one
 * translation unit; `FiniteElementExtraConstitution` base).
 *
 * DEVIATIONS (reported in the T5.2 batch-b report):
 *  1. the *class declaration* lives here and the kernels + out-of-line member
 *     definitions + the registration stay in `discrete_shell_bending.cpp` — the same
 *     split T5.2 batch a introduced (cookbook §7.17 rule 11), so the unit test can
 *     reach the reference's *public* surface (its buffers and the five hooks) and
 *     drive the real constitution end to end;
 *  2. the reference specializes `std::hash<uipc::Vector2i>` in this translation unit
 *     for its `unordered_map<Vector2i, InitInfo>`; the port uses a local
 *     `Vector2iHash` functor with the *identical* hash body (`front << 32 | end`),
 *     exactly like the reference's own plastic-shell siblings already do — the
 *     hash values (and therefore the map's iteration order) are unchanged, but no
 *     `std::hash` specialization leaks out of a header.
 ********************************************************************/
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/constitutions/discrete_shell_bending_function.h>

namespace uipc::backend::luisa
{
class DiscreteShellBending final : public FiniteElementExtraConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 17;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;
    using Base                             = FiniteElementExtraConstitution;

    using Base::Base;

    virtual U64 get_uid() const noexcept override;

    class InitInfo
    {
      public:
        bool        valid_bending() const { return oppo_verts.size() == 2; }
        IndexT      edge_index = -1;
        set<IndexT> oppo_verts;
        Float       stiffness = 0.0;
    };

    /// X0, X1, X2, X3; (X1, X2) is middle edge
    vector<Vector4i> h_stencils;
    vector<Float>    h_bending_stiffness;
    vector<Float>    h_rest_volumes;
    vector<Float>    h_rest_lengths;
    vector<Float>    h_h_bars;
    vector<Float>    h_theta_bars;
    vector<Float>    h_V_bars;

    /// X0, X1, X2, X3; (X1, X2) is middle edge
    lud::LDeviceBuffer<LVector4i> stencils;
    lud::LDeviceBuffer<Float>     bending_stiffnesses;
    lud::LDeviceBuffer<Float>     rest_lengths;
    lud::LDeviceBuffer<Float>     h_bars;
    lud::LDeviceBuffer<Float>     theta_bars;
    lud::LDeviceBuffer<Float>     V_bars;

    virtual void do_build(BuildInfo& info) override;

    virtual void do_init(FilteredInfo& info) override;

    virtual void do_report_extent(ReportExtentInfo& info) override;

    virtual void do_compute_energy(ComputeEnergyInfo& info) override;

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override;
};
}  // namespace uipc::backend::luisa

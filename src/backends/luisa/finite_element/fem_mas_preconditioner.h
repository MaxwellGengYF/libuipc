#pragma once
/********************************************************************
 * @file fem_mas_preconditioner.h
 * @brief T6.4c — the FEM MultiLevel Additive Schwarz preconditioner
 *        (port of src/backends/cuda/finite_element/fem_mas_preconditioner.cu, 473 lines).
 *
 * WHAT IT IS
 * ----------
 * The `LocalPreconditioner` that drives `MASPreconditionerEngine` for the FEM
 * subsystem.  It is activated by the scene config
 * `linear_system/fem_preconditioner == "mas"` (the reference's own reader) and
 * *defers* otherwise by throwing `SimSystemException`; its sibling
 * `FEMDiagPreconditioner` throws on the same config value, so exactly one of the two
 * is ever valid in a scene (the port's established "explicitly not in play" idiom).
 *
 * `do_init` builds the vertex adjacency from the element connectivity (tets /
 * triangles / edges), partitions every non-Empty FEM geometry with the fixed
 * cluster size BANKSIZE = 16 (a pre-existing `mesh_part` vertex attribute is
 * respected as-is) and hands the engine the CSR adjacency + the partition
 * mappings.  `do_assemble` scatters the assembled BCOO Hessian into the cluster
 * matrices and inverts them; `do_apply` runs restrict -> local solve -> prolongate,
 * plus a *per-vertex diagonal* fallback for the vertices that stayed unpartitioned.
 *
 * The reference declares the class *and* its two fallback kernels inside the `.cu`.
 * The port keeps the reference's surface (the three `require<>`d systems, the engine,
 * the two fallback buffers, the four hooks) in this header, the kernels +
 * `REGISTER_SIM_SYSTEM` in the `.cpp`, and the fallback *bodies* in
 * `fem_mas_preconditioner_function.h` — exactly like T5.4b's diagonal pair, so the
 * unit test can register the *real* preconditioner into a standalone
 * `GlobalLinearSystem::Impl` and drive `assemble`/`apply` directly.
 *
 * `m_has_partition` / `m_has_unpartitioned` are public for the same reason (the
 * test asserts the "nothing partitionable" identity fallback), and so are the two
 * buffers.
 *********************************************************************/
#include <linear_system/local_preconditioner.h>
#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/mas_preconditioner_engine.h>
#include <luisa_device/linear/linear.h>

namespace uipc::backend::luisa
{
class FEMMASPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    static constexpr int BANKSIZE = MASPreconditionerEngine::BANKSIZE;

    FiniteElementMethod* finite_element_method = nullptr;
    GlobalLinearSystem*  global_linear_system  = nullptr;
    FEMLinearSubsystem*  fem_linear_subsystem  = nullptr;

    MASPreconditionerEngine engine;
    bool                    m_has_partition     = false;
    bool                    m_has_unpartitioned = false;

    /// The diagonal fallback for unpartitioned vertices (one inverse block each).
    lud::LDeviceBuffer<LMatrix3x3> diag_inv;
    /// 1 = unpartitioned (the diagonal fallback owns it), 0 = partitioned.
    lud::LDeviceBuffer<IndexT> unpartitioned_flags;

    void do_build(BuildInfo& info) override;
    void do_init(InitInfo& info) override;
    void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override;
    void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override;
};
}  // namespace uipc::backend::luisa

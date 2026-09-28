#pragma once
/********************************************************************
 * @file stable_neo_hookean_3d_function.h
 * @brief T5.2 (batch a) — the stable Neo-Hookean 3D energies (port of
 *        src/backends/cuda/finite_element/constitutions/
 *        stable_neo_hookean_3d_function.h).
 *
 * The reference header is a thin include shell around two
 * "no-includes-allowed" `.inl` files:
 *   `sym::stable_neo_hookean_3d` — the SymEigen-family closed forms
 *                                  (`detail/stable_neo_hookean_3d.inl`);
 *   `snk1`                       — Stiff-GIPC's SNK1 energy/gradient plus the
 *                                  analytically SPD-projected Hessian
 *                                  (`detail/stable_neo_hookean_3d_snk1.inl`).
 * `StableNeoHookean3D` uses `snk1` (`namespace SNH = snk1;` in the reference's
 * .cu) — the first namespace is dead code in the reference as well; the port
 * compiles and unit-tests it instead of dropping it.
 *
 * Reference include map (the .inl bodies can use these unqualified names):
 *   <cuda_tool/cuda_tool.h>     -> <type_define.h> + <luisa_device/math/math.h>
 *   <cuda_tool/eigen/evd.h>     -> <luisa_device/math/evd.h>   (`lps::evd`)
 *   <algorithm/qr_svd.hpp>      -> <luisa_device/math/svd.h>   (`lps::qr_svd`)
 *
 * DEVIATION (reported): the reference does not include
 * `sym/simplified_stable_neo_hookean_3d.inl` anywhere (T5.2's brief lists it for
 * this batch).  It is included here — in its own namespace, exactly as the
 * reference's other `sym/*.inl` files are included — so the transcription is
 * compiled and unit-testable instead of silently rotting.
 *
 * `snh_3d_detail` factors the reference's two kernels into generic per-element
 * routines, so the kernel and the unit test instantiate *the same* code
 * (cookbook §7.4); see the `tet_gradient`/`tet_hessian` comments for the two
 * documented restructurings.
 ********************************************************************/
#include <type_define.h>
#include <finite_element/matrix_utils.h>
#include <finite_element/fem_utils.h>
#include <luisa_device/math/evd.h>
#include <luisa_device/math/svd.h>

namespace uipc::backend::luisa
{
namespace sym::stable_neo_hookean_3d
{
#include "detail/stable_neo_hookean_3d.inl"
}

// Stiff-GIPC's SNK1: energy + gradient + analytically SPD-projected Hessian
// (replaces the SymEigen-generated derivatives + generic make_spd EVD)
namespace snk1
{
#include "detail/stable_neo_hookean_3d_snk1.inl"
}

// See the file comment: the reference leaves this sibling unreferenced.
namespace sym::simplified_stable_neo_hookean_3d
{
#include "sym/simplified_stable_neo_hookean_3d.inl"
}

namespace snh_3d_detail
{
/**
 * @brief The reference's `StableNeoHookean3D_do_compute_energy_kernel` body.
 *
 * The constitution uses the `snk1` variant (`namespace SNH = snk1;`), so this is
 * the energy the kernel computes; the test instantiates the same routine for
 * `S = double` and `S = lcrt::Var<Float>` (cookbook §7.4).
 */
template <typename S>
[[nodiscard]] inline S tet_energy(const S&                   mu,
                                  const S&                   lambda,
                                  const S&                   dt,
                                  const S&                   volume,
                                  const lps::LVec<S, 3>&     x0,
                                  const lps::LVec<S, 3>&     x1,
                                  const lps::LVec<S, 3>&     x2,
                                  const lps::LVec<S, 3>&     x3,
                                  const lps::LMat<S, 3, 3>&  Dm_inv)
{
    auto F = fem::F(x0, x1, x2, x3, Dm_inv);

    S E;
    snk1::E(E, mu, lambda, F);
    // reference: `E *= dt * dt * volumes(I);`
    E = lps::detail::mul(E, lps::detail::mul(lps::detail::mul(dt, dt), volume));
    return E;
}

/**
 * @brief The local 4-stencil gradient `(4 x 3)`, flattened in the reference's
 *        stencil order (`segment<3>(3 * i)` = stencil `i`).
 *
 * The reference's gradient/Hessian kernel writes
 * `DVA(I * StencilSize + i).write(tet(i), G)` for `i = 0..3`; that is exactly
 * `DoubletVectorAssembler::segment<4>(I * 4).write(tet, G12)`, so the port
 * returns the 12-vector and keeps the reference's write loop in the assembler.
 */
template <typename S>
[[nodiscard]] inline lps::LVec<S, 12> tet_gradient(const S&                  mu,
                                                   const S&                  lambda,
                                                   const S&                  dt,
                                                   const S&                  volume,
                                                   const lps::LVec<S, 3>&    x0,
                                                   const lps::LVec<S, 3>&    x1,
                                                   const lps::LVec<S, 3>&    x2,
                                                   const lps::LVec<S, 3>&    x3,
                                                   const lps::LMat<S, 3, 3>& Dm_inv)
{
    auto F = fem::F(x0, x1, x2, x3, Dm_inv);

    // reference: `auto Vdt2 = volumes(I) * dt * dt;`
    auto Vdt2 = lps::detail::mul(lps::detail::mul(volume, dt), dt);

    lps::LMat<S, 3, 3> dEdF;
    snk1::dEdF(dEdF, mu, lambda, F);
    auto VecdEdF = flatten(dEdF);
    VecdEdF *= Vdt2;

    auto shape_gradients = fem::tetrahedron_shape_gradients(Dm_inv);

    lps::LVec<S, 12> G12;
    for(int i = 0; i < 4; ++i)
        G12.segment<3>(3 * i) = fem::project_F_gradient(shape_gradients.col(i), VecdEdF);
    return G12;
}

/**
 * @brief The local 12x12 Hessian of the SNK1 energy.
 *
 * RESTRUCTURE (reported): the reference's kernel computes, for every `i <= j`,
 * only the *global-upper* block -- it swaps `(i, j)` when `tet(i) > tet(j)` and
 * then calls `project_F_hessian_block(col(left), ddEddF, col(right))`.  The swap
 * depends on *device* indices, so a ported loop cannot re-order the call
 * arguments (RULE M2).  The port therefore fills the whole 12x12 matrix
 * (`block(3i, 3j) = project_F_hessian_block(col(i), ddEddF, col(j))`, both
 * triangles) and lets `TripletMatrixAssembler::half_block` select the block the
 * reference would have written: with `(left, right)` sorted, that is
 * `block(3 left, 3 right)` -- the identical formula on the identical inputs, so
 * every written 3x3 block is bit-identical to the reference's.  Cost: 16 instead
 * of 10 3x3 projections per element (the two decompositions dominate).
 */
template <typename S>
[[nodiscard]] inline lps::LMat<S, 12, 12> tet_hessian(const S&                  mu,
                                                      const S&                  lambda,
                                                      const S&                  dt,
                                                      const S&                  volume,
                                                      const lps::LVec<S, 3>&    x0,
                                                      const lps::LVec<S, 3>&    x1,
                                                      const lps::LVec<S, 3>&    x2,
                                                      const lps::LVec<S, 3>&    x3,
                                                      const lps::LMat<S, 3, 3>& Dm_inv)
{
    auto F = fem::F(x0, x1, x2, x3, Dm_inv);

    auto Vdt2 = lps::detail::mul(lps::detail::mul(volume, dt), dt);

    // analytically SPD-projected 9x9 energy Hessian (Stiff SNK1);
    // scaling by the positive Vdt2 commutes with the projection
    lps::LMat<S, 9, 9> ddEddF;
    snk1::ddEddF_spd(ddEddF, mu, lambda, F);
    ddEddF *= Vdt2;

    auto shape_gradients = fem::tetrahedron_shape_gradients(Dm_inv);

    lps::LMat<S, 12, 12> H12;
    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            H12.block<3, 3>(3 * i, 3 * j) =
                fem::project_F_hessian_block(shape_gradients.col(i), ddEddF,
                                             shape_gradients.col(j));
    return H12;
}
}  // namespace snh_3d_detail
}  // namespace uipc::backend::luisa

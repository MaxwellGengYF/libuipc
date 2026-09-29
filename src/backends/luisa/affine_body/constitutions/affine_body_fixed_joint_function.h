#pragma once
/********************************************************************
 * @file affine_body/constitutions/affine_body_fixed_joint_function.h
 * @brief T6.3b1 — the ABD fixed joint's arithmetic (port of
 *        src/backends/cuda/affine_body/constitutions/affine_body_fixed_joint_function.h).
 *
 * The reference header is a thin include shell around the SymEigen.py-generated
 * `sym/affine_body_fixed_joint.inl`; the port keeps that shape and adds the
 * *factored* per-joint routines the two kernels and the unit test share (the T6.3a
 * `abd_arap_detail` precedent, cookbook §7.4/§7.24): one implementation over the
 * scalar type `S` (`double` on the host, `lcrt::Var<double>` in the DSL), never a
 * re-spelling of the generated statements.
 *
 * Reference include map:
 *   <cuda_tool/cuda_tool.h>  -> <affine_body/type_define.h> + <luisa_device/math/math.h>
 *   the generated `std::pow` -> `lps::pow` (spelled by the transcriber, not here)
 *   <utils/make_spd.h>       -> `lps::make_spd_inplace` (matrix_ops.h, via math.h)
 *
 * THE JOINT (what the two kernels compute, verbatim from the reference's `.cu`):
 *   kappa = strength_ratio(I) * (body_masses(bids(0)).mass() + body_masses(bids(1)).mass())
 *   E  = Er + Et          (the two generated quadratic laws, see the sym/*.inl comment)
 *   G  = JrT_Gr + JtT_Gt  (the 24-vector `[body_i ; body_j]`)
 *   H  = JrT_Hr_Jr + JtT_Ht_Jt
 * with the *pair* dof vector `q = [qi ; qj]` (12 dofs each).  Both bracket terms are
 * the reference's own fan-outs `J^T G`, `J^T H J` of the two constraint Jacobians.
 *
 * THE REFERENCE CLAMPS: `make_spd(ddEdFr_val)` (9x9) and `make_spd(ddEdFt_val)`
 * (3x3) are called *before* the two `J^T H J` fan-outs — in that order, on the
 * *element* Hessians, exactly like `ortho_potential.cu` (§7.24 rule 4).  The clamp is
 * a no-op for `kappa > 0` (both element Hessians are `kappa I`) and it is what keeps
 * the assembled joint SPD if a user passes a negative `strength_ratio`; the port
 * keeps it (`kMakeSpdSweeps`, the T6.3a §7.17 rule 31 plateau) rather than dropping
 * a reference statement.  `utils/make_spd.h` is a *tolerance-driven* device loop, so
 * a device port must pick a fixed, measured count (RULE M2).
 *
 * THE TWO KERNEL WRITE PATTERNS (the reference's `DoubletVectorAssembler` /
 * `TripletMatrixAssembler`, cookbook §7.22 rule 2): the gradient is written as the
 * 2-doublet run `DVA.segment<2>(2 * I)` at the (global) body indices
 * `{bids(0), bids(1)}`; the Hessian as the *upper-triangular* 2x2 block run
 * `TMA.half_block<2>(HalfHessianSize * I)` (`HalfHessianSize = 3`), where the single
 * off-diagonal triplet is written at `(bids(L), bids(R))` with
 * `(L, R) = (0, 1)` if `bids(0) < bids(1)` and `(1, 0)` otherwise (`upper_LR`) — a
 * *device* predicate, so the port selects the block and the two indices
 * (`scalar_detail::select_scalar` / `select_container`, RULE M2) instead of
 * branching.  The two write helpers below are that pattern, generic over the device
 * view the assembly hands over.
 ********************************************************************/
#include <affine_body/type_define.h>
#include <luisa_device/math/math.h>
// `lps::make_spd_inplace` (the reference's `utils/make_spd.h`) lives in matrix_ops.h;
// math.h alone does not declare it (measured C2039 in T6.3a).
#include <luisa_device/math/matrix_ops.h>
// `scalar_detail::{select_scalar, select_container}` — the device-predicate selects
// the reference's `upper_LR` needs (the FEM shells' shared helper, T5.2 batch b).
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
namespace sym::affine_body_fixed_joint
{
#include "sym/affine_body_fixed_joint.inl"
}  // namespace sym::affine_body_fixed_joint

namespace abd_fixed_joint_detail
{
/// The reference's `make_spd` is a tolerance-driven Jacobi loop (a *device* loop
/// count, RULE M2), so the port picks a fixed one: 8 is §7.17 rule 31's measured
/// plateau for N = 6/9/12 in this layer, and both element Hessians here are tiny
/// (9x9 and 3x3).  The test re-measures it against the layer default (20).
inline constexpr int kMakeSpdSweeps = 8;

/**
 * @brief One joint's rest data, unpacked from the reference's four `Vector6` buffers.
 *
 * The reference's kernel keeps the rows as `Vector6` and reads them with
 * `.segment<3>(0)` / `.segment<3>(3)`; the port passes the two no-includes-allowed
 * `sym/...inl` functions exactly those segments, so the struct stores the same
 * `[first | second]` layout and `body_*` slices it on the way in.
 */
template <typename S>
struct FixedJointRestT
{
    // clang-format off
    lps::LVec<S, 6> rest_c;  // [ci_bar | cj_bar]  (the two attachment points, body-local)
    lps::LVec<S, 6> rest_t;  // [ti_bar | tj_bar]  (the frames' x directions, body-local)
    lps::LVec<S, 6> rest_n;  // [ni_bar | nj_bar]  (y)
    lps::LVec<S, 6> rest_b;  // [bi_bar | bj_bar]  (z)
    // clang-format on
};

/// The reference's `Es(I) = Er_val + Et_val;` (`Fr`/`Er` then `Ft`/`Et`, in that
/// order — the generated statements are evaluated in the reference's order).
template <typename S>
[[nodiscard]] inline S body_energy(const S&                    kappa,
                                   const FixedJointRestT<S>&   r,
                                   const lps::LVec<S, 12>&     qi,
                                   const lps::LVec<S, 12>&     qj)
{
    lps::LVec<S, 9> Fr_val;
    sym::affine_body_fixed_joint::Fr<S>(Fr_val,
                                        r.rest_t.template segment<3>(0),
                                        r.rest_n.template segment<3>(0),
                                        r.rest_b.template segment<3>(0),
                                        qi,
                                        r.rest_t.template segment<3>(3),
                                        r.rest_n.template segment<3>(3),
                                        r.rest_b.template segment<3>(3),
                                        qj);
    S Er_val;
    sym::affine_body_fixed_joint::Er<S>(Er_val, kappa, Fr_val);

    lps::LVec<S, 3> Ft_val;
    sym::affine_body_fixed_joint::Ft<S>(Ft_val,
                                        r.rest_c.template segment<3>(0),
                                        qi,
                                        r.rest_c.template segment<3>(3),
                                        qj);
    S Et_val;
    sym::affine_body_fixed_joint::Et<S>(Et_val, kappa, Ft_val);

    return lps::detail::add(Er_val, Et_val);
}

/// The reference's `G = JrT_Gr_val + JtT_Gt_val;` (the 24-vector `[body_i ; body_j]`).
template <typename S>
[[nodiscard]] inline lps::LVec<S, 24> body_gradient(const S&                  kappa,
                                                    const FixedJointRestT<S>& r,
                                                    const lps::LVec<S, 12>&   qi,
                                                    const lps::LVec<S, 12>&   qj)
{
    lps::LVec<S, 9> Fr_val;
    sym::affine_body_fixed_joint::Fr<S>(Fr_val,
                                        r.rest_t.template segment<3>(0),
                                        r.rest_n.template segment<3>(0),
                                        r.rest_b.template segment<3>(0),
                                        qi,
                                        r.rest_t.template segment<3>(3),
                                        r.rest_n.template segment<3>(3),
                                        r.rest_b.template segment<3>(3),
                                        qj);

    lps::LVec<S, 9> dEdFr_val;
    sym::affine_body_fixed_joint::dEdFr<S>(dEdFr_val, kappa, Fr_val);

    lps::LVec<S, 24> JrT_Gr_val;
    sym::affine_body_fixed_joint::JrT_Gr<S>(JrT_Gr_val,
                                            dEdFr_val,
                                            r.rest_t.template segment<3>(0),
                                            r.rest_n.template segment<3>(0),
                                            r.rest_b.template segment<3>(0),
                                            qi,
                                            r.rest_t.template segment<3>(3),
                                            r.rest_n.template segment<3>(3),
                                            r.rest_b.template segment<3>(3),
                                            qj);

    lps::LVec<S, 3> Ft_val;
    sym::affine_body_fixed_joint::Ft<S>(Ft_val,
                                        r.rest_c.template segment<3>(0),
                                        qi,
                                        r.rest_c.template segment<3>(3),
                                        qj);

    lps::LVec<S, 3> dEdFt_val;
    sym::affine_body_fixed_joint::dEdFt<S>(dEdFt_val, kappa, Ft_val);

    lps::LVec<S, 24> JtT_Gt_val;
    sym::affine_body_fixed_joint::JtT_Gt<S>(JtT_Gt_val,
                                            dEdFt_val,
                                            r.rest_c.template segment<3>(0),
                                            r.rest_c.template segment<3>(3));

    return JrT_Gr_val + JtT_Gt_val;
}

/// The reference's `H = JrT_Hr_Jr_val + JtT_Ht_Jt_val;`, with the reference's two
/// `make_spd` clamps in its own order (9x9 first, then 3x3 — both *before* their
/// fan-out).
template <typename S>
[[nodiscard]] inline lps::LMat<S, 24, 24> body_hessian(const S&                  kappa,
                                                       const FixedJointRestT<S>& r,
                                                       const lps::LVec<S, 12>&   qi,
                                                       const lps::LVec<S, 12>&   qj)
{
    lps::LVec<S, 9> Fr_val;
    sym::affine_body_fixed_joint::Fr<S>(Fr_val,
                                        r.rest_t.template segment<3>(0),
                                        r.rest_n.template segment<3>(0),
                                        r.rest_b.template segment<3>(0),
                                        qi,
                                        r.rest_t.template segment<3>(3),
                                        r.rest_n.template segment<3>(3),
                                        r.rest_b.template segment<3>(3),
                                        qj);

    lps::LMat<S, 9, 9> ddEdFr_val;
    sym::affine_body_fixed_joint::ddEdFr<S>(ddEdFr_val, kappa, Fr_val);
    lps::make_spd_inplace(ddEdFr_val, kMakeSpdSweeps);

    lps::LMat<S, 24, 24> JrT_Hr_Jr_val;
    sym::affine_body_fixed_joint::JrT_Hr_Jr<S>(JrT_Hr_Jr_val,
                                               ddEdFr_val,
                                               r.rest_t.template segment<3>(0),
                                               r.rest_n.template segment<3>(0),
                                               r.rest_b.template segment<3>(0),
                                               qi,
                                               r.rest_t.template segment<3>(3),
                                               r.rest_n.template segment<3>(3),
                                               r.rest_b.template segment<3>(3),
                                               qj);

    lps::LVec<S, 3> Ft_val;
    sym::affine_body_fixed_joint::Ft<S>(Ft_val,
                                        r.rest_c.template segment<3>(0),
                                        qi,
                                        r.rest_c.template segment<3>(3),
                                        qj);

    lps::LMat<S, 3, 3> ddEdFt_val;
    sym::affine_body_fixed_joint::ddEdFt<S>(ddEdFt_val, kappa, Ft_val);
    lps::make_spd_inplace(ddEdFt_val, kMakeSpdSweeps);

    lps::LMat<S, 24, 24> JtT_Ht_Jt_val;
    sym::affine_body_fixed_joint::JtT_Ht_Jt<S>(JtT_Ht_Jt_val,
                                               ddEdFt_val,
                                               r.rest_c.template segment<3>(0),
                                               r.rest_c.template segment<3>(3));

    return JrT_Hr_Jr_val + JtT_Ht_Jt_val;
}

/**
 * @brief The reference's `DVA.segment<2>(2 * I).write(indices, G)`.
 *
 * Two consecutive doublets: slot `base` gets body `idx0`'s 12-vector `G[0:12]`, slot
 * `base + 1` gets body `idx1`'s `G[12:24]` (the reference's `ProxyRange<2>::write`
 * loop).  The `DoubletView` is the *device* flavour the kernel builds from the
 * assembled buffer's handles.
 */
template <typename DoubletView, typename S, typename Base, typename Idx>
inline void write_pair_gradient(const DoubletView&      doublets,
                                const lps::LVec<S, 24>& G,
                                const Base&             base,
                                const lps::LVec<Idx, 2>& indices) noexcept
{
    doublets(base).write(indices(0), lps::LVec<S, 12>(G.template segment<12>(0)));
    doublets(base + 1).write(indices(1), lps::LVec<S, 12>(G.template segment<12>(12)));
}

/**
 * @brief The reference's `TMA.half_block<2>(HalfHessianSize * I).write(indices, H)`.
 *
 * The upper triangle of the 2x2 *block* matrix in the reference's own `upper_LR`
 * order: `(0,0)` and `(1,1)` are fixed, and the off-diagonal block is written at
 * `(indices(L), indices(R))` with `(L, R) = (0, 1)` iff `indices(0) < indices(1)`,
 * otherwise `(1, 0)` — with the *same* rule applied to the block's content
 * (`H.block(L, R)`).  Both are device predicates, so they become selects (RULE M2).
 * The device triplet view adds its own `submatrix_offset` (zero for a joint's
 * slice, exactly like the reference's assembler).
 */
template <typename TripletView, typename S, typename Base, typename Idx>
inline void write_pair_half_hessian(const TripletView&          triplets,
                                    const lps::LMat<S, 24, 24>& H,
                                    const Base&                 base,
                                    const lps::LVec<Idx, 2>&    indices) noexcept
{
    /// The 12x12 off-diagonal block at *block* position `(L, R)` (`L, R` in {0, 1});
    /// `H` must be const here so the numerics layer's `block<>` returns a copy.
    const auto block = [&](int L, int R)
    { return lps::LMat<S, 12, 12>(H.template block<12, 12>(L * 12, R * 12)); };

    const auto ascending = indices(0) < indices(1);  // the reference's upper_LR test
    const auto off_row   = scalar_detail::select_scalar(indices(1), indices(0), ascending);
    const auto off_col   = scalar_detail::select_scalar(indices(0), indices(1), ascending);
    // `ascending ? H.block(0, 1) : H.block(1, 0)` (`select_container(f, t, p)` is
    // `p ? t : f`)
    const auto off_block = scalar_detail::select_container(block(1, 0), block(0, 1), ascending);

    triplets(base).write(indices(0), indices(0), block(0, 0));
    triplets(base + 1).write(off_row, off_col, off_block);
    triplets(base + 2).write(indices(1), indices(1), block(1, 1));
}
}  // namespace abd_fixed_joint_detail
}  // namespace uipc::backend::luisa

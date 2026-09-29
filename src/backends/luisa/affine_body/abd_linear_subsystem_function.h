#pragma once
/********************************************************************
 * @file affine_body/abd_linear_subsystem_function.h
 * @brief T6.1b — the *generic*, host- and device-callable routines of
 *        `ABDLinearSubsystem`'s assembly: the extent arithmetic and every named
 *        kernel body of src/backends/cuda/affine_body/abd_linear_subsystem.cu.
 *
 * WHY THIS HEADER EXISTS (the FEM twin is fem_linear_subsystem_function.h, T5.4a)
 * ------------------------------------------------------------------------------
 * The ABD assembly only ever runs from `GlobalLinearSystem::solve()` — the T8.3
 * solve path — so a unit test cannot reach it through the engine.  Everything
 * below is therefore factored out of the kernels as a plain generic routine over
 *
 *   * the *view flavour* (`storage::device` inside a kernel, `storage::memory`
 *     for host evaluation) and
 *   * the *scalar type* (`double` on the host, `lcrt::Var<double>` on the
 *     device, cookbook §7.16 rule 1),
 *
 * so the shipped kernel lambda and a unit test run the *same* code (cookbook
 * §7.4).  `abd_linear_subsystem.cpp`'s kernels call these; the T6.1b test drives
 * them against hand-derived indices and values, plus `double` vs
 * `lcrt::Var<double>` bitwise (the `abd_linear_device_vs_host` case).
 *
 * THE ABD INDEX CONVENTIONS (what the routines here pin down)
 * -----------------------------------------------------------
 *  * One *body* owns 12 dofs (`q = [t; a1; a2; a3]`), and the global linear
 *    system's block size is 3 (`DiagLinearSubsystem`/`GlobalLinearSystem`
 *    `DoFBlockSize`), so a body's 12x12 block is a **4x4 block of 3x3 blocks**
 *    at blocked row/col `body * 4 + {0..3}`.  `GlobalLinearSystem`'s triplet
 *    buffer is a 3x3-block COO; the ABD subsystem's *own* reporters work in
 *    12x12 blocks (`reporter_hessians`) and each such block is *unpacked* into
 *    16 consecutive 3x3 triplets on the way into the global buffer
 *    (`unpack_body_block`, the reference's `TripletMatrixAssembler::block<4,4>`
 *    + `write(row_begin, col_begin, H12x12)`, whose triplet order is
 *    `counter = a * 4 + b` with the sub-block `H12.block<3,3>(3a, 3b)`).
 *  * `H` on a diagonal block is stored **upper-triangular in 3x3 blocks**
 *    (`zero_out_lower`, the reference's robustness step); for `body_i < body_j`
 *    all 16 sub-blocks are written.
 *  * The gradient is a *dense window* (the subsystem's slice of the global `b`),
 *    indexed `body * 12 + c`, and the reference's per-row scatters are
 *    `atomic_add`s (several reporters can address the same body).
 *  * The dytopo effect arrives in the *vertex* index space and is mapped with the
 *    per-vertex jacobian: `G12 = J_v^T G3` at `v2b(v)` and, for a vertex pair,
 *    `J_i^T H J_j` with the reference's `(L, R)` swap (`L = min(body_i, body_j)`)
 *    — both halves of the "Jacobi triple".
 *
 * BRANCHLESS WHERE THE PREDICATE IS DEVICE DATA (RULE M2)
 * ------------------------------------------------------
 * The reference branches on `is_fixed(i)` / `is_external_kinetic(i)` / `i == j`,
 * which are device predicates; each becomes a `select`, which is exact because a
 * select passes its chosen operand through untouched:
 *   * "skip the scatter" becomes "scatter exactly 0" (the fixed rows are cleared
 *     by the reference's own k1/k2 anyway);
 *   * the diagonal-Hessian accumulation is gated by adding a *selected* block
 *     (the reference's `if(body_i == body_j && !has_fixed)`), so no branch is
 *     needed on device data.
 *
 * The per-body source buffers (`shape`/`kinetic` gradient and Hessian, the mass,
 * `is_fixed`) are passed as *callables* (`shape_gradient_at(i)`), because the host
 * flavour reads a host array and the device flavour reads a buffer — both
 * `[&](auto&& i) { return ...; }` at the call site.  This is the T5.4a
 * `is_fixed_at` convention, extended to every source.
 *********************************************************************/
#include <type_define.h>
#include <affine_body/abd_jacobi_matrix.h>
#include <utils/matrix_assembler.h>
#include <utils/offset_count_collection.h>
#include <utils/scalar_util.h>

#include <algorithm>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

namespace uipc::backend::luisa
{
// T3.1's linear-algebra layer (cookbook §7.15 rule 6).
namespace lla = luisa_device::linalg;

namespace abd_linear_subsystem_detail
{
using scalar_detail::pred_and;
using scalar_detail::pred_or;
using scalar_detail::select_container;
using scalar_detail::select_scalar;

// ===========================================================================
// 0) shapes of the ABD assembly (named, so no magic 4 / 12 / 16 in the code)
// ===========================================================================
/// dofs per ABD body.
constexpr int kDofPerBody = 12;
/// 3x3 sub-blocks per body block, per axis (`12 / 3`).
constexpr int kBlocksPerBody = kDofPerBody / 3;
/// 3x3 triplets per 12x12 body block.
constexpr int kTripletsPerBodyBlock = kBlocksPerBody * kBlocksPerBody;

// ===========================================================================
// 1) the fan-out slice: which [offset, count) of a shared assembly buffer
//    belongs to reporter/constitution `index`
// ===========================================================================
/// The reference's `auto [offset, count] = offsets_counts[m_index];` step,
/// factored so a test can drive it against hand-derived offsets
/// (`OffsetCountCollection` is host-only).
[[nodiscard]] inline std::pair<IndexT, IndexT> slice_of(
    const OffsetCountCollection<IndexT>& offsets_counts, SizeT index)
{
    auto [offset, count] = offsets_counts[index];
    return std::pair<IndexT, IndexT>{offset, count};
}

// ===========================================================================
// 2) `Impl::report_extent`'s extent arithmetic
// ===========================================================================
/// The counts `ABDLinearSubsystem::Impl::report_extent` arrives at, in the
/// reference's own order: the ABD gradient buffer is the *global* one, so only the
/// Hessian needs an offset budget:
/// `[KineticShapeH | OtherH ... | DyTopoH]` with `16` global 3x3 triplets per
/// 12x12 body block.
struct ExtentBudget
{
    /// 12x12 blocks contributed by the body term (kinetic + shape), 1 per body,
    /// only when the Hessian is assembled at all.
    SizeT body_hessian_block_count = 0;
    /// the reporters' 12x12 gradient doublets (the reporters' slice sizes; the
    /// *gradient* extent reported to the linear system is the dof count).
    SizeT reporter_gradient_count = 0;
    /// the reporters' 12x12 Hessian blocks.
    SizeT reporter_hessian_block_count = 0;
    /// the dytopo effect's gradient doublets (diagnostic: they index *vertices*).
    SizeT dytopo_gradient_count = 0;
    /// the dytopo effect's 3x3 Hessian triplets (the ABD Jacobi triple expands
    /// each of them into one 12x12 block).
    SizeT dytopo_hessian_count = 0;
    /// the reference's `H12x12_count`: the number of 12x12 blocks before the x16.
    SizeT h12x12_count = 0;
    /// the reference's `H3x3_count` == the first argument of
    /// `DiagExtentInfo::extent`.
    SizeT hessian_block_count = 0;
    /// the reference's `dof_count` == `body_count * 12`.
    SizeT dof_count = 0;

    /// The reference's `UIPC_ASSERT(H3x3_count == 0, ...)` when `gradient_only`:
    /// in gradient-only mode nothing may contribute a Hessian block.
    [[nodiscard]] bool check_gradient_only(bool gradient_only) const noexcept
    {
        return !(gradient_only && hessian_block_count != 0);
    }
};

/// The reference's `Impl::report_extent` arithmetic.  `reporter_*_counts` are the
/// per-reporter counts the reporters just reported (which `Impl::report_extent`
/// stores into its offset/count collections first); the dytopo counts are the
/// receiver's own (`0` when there is no receiver).
template <typename GradCountRange, typename HessCountRange>
[[nodiscard]] inline ExtentBudget plan_extent(SizeT body_count,
                                              bool  gradient_only,
                                              bool  has_complement,
                                              const GradCountRange& reporter_gradient_counts,
                                              const HessCountRange& reporter_hessian_counts,
                                              SizeT dytopo_gradient_count,
                                              SizeT dytopo_hessian_count)
{
    ExtentBudget budget;
    SizeT        h12x12_count = 0;

    auto sum = [](const auto& range) noexcept
    {
        return std::accumulate(range.begin(),
                               range.end(),
                               SizeT{0},
                               [](SizeT acc, const auto& c) noexcept
                               { return acc + static_cast<SizeT>(c); });
    };

    if(has_complement)
    {
        // 1) the body term: kinetic + shape, one 12x12 block per body
        if(!gradient_only)
        {
            budget.body_hessian_block_count = body_count;
            h12x12_count += body_count;
        }

        // 2) the reporters (the reference scans both collections unconditionally,
        //    in gradient-only mode too, because the reporters' *gradient* slices
        //    are still needed)
        budget.reporter_gradient_count    = sum(reporter_gradient_counts);
        budget.reporter_hessian_block_count = gradient_only ? SizeT{0} : sum(reporter_hessian_counts);
        if(!gradient_only)
            h12x12_count += budget.reporter_hessian_block_count;
    }

    // 3) the dytopo effect (the DyTopo system decides `component_flags` itself);
    //    the reference counts its Hessian only when assembling one
    budget.dytopo_gradient_count = dytopo_gradient_count;
    budget.dytopo_hessian_count  = dytopo_hessian_count;
    if(!gradient_only)
        h12x12_count += dytopo_hessian_count;

    budget.h12x12_count        = h12x12_count;
    budget.hessian_block_count = h12x12_count * static_cast<SizeT>(kTripletsPerBodyBlock);
    budget.dof_count           = body_count * static_cast<SizeT>(kDofPerBody);
    return budget;
}

// ===========================================================================
// 3) the 12x12 body block: the upper-triangle zeroing and the 4x4 unpacking
// ===========================================================================
/// The reference's `zero_out_lower(Matrix12x12&)`: clear the strictly-lower 3x3
/// blocks (their mirrors are regenerated by the symmetric block solver).  Only
/// compile-time indices are involved, so the loop body is identical on host and
/// device.
template <typename HMat>
inline void zero_out_lower(HMat& h) noexcept
{
    using S = lps::mat_element_t<HMat>;
    auto zero = lps::make_mat<S, 3, 3>([](int, int) noexcept { return lps::zero_of<S>(); });
    for(int jj = 0; jj < kBlocksPerBody; ++jj)
    {
        for(int ii = jj + 1; ii < kBlocksPerBody; ++ii)
        {
            h.template block<3, 3>(ii * 3, jj * 3) = zero;
        }
    }
}

/// The upper-triangle-only flavour of `zero_out_lower` (a *value*, which is what a
/// `select` needs).
template <typename HMat>
[[nodiscard]] inline HMat upper_only(const HMat& h) noexcept
{
    HMat out = h;
    zero_out_lower(out);
    return out;
}

/// Write one 12x12 *body* block into a 3x3-block global triplet view: the
/// reference's `TripletMatrixUnpacker::block<4,4>(I * 16).write(row_begin,
/// col_begin, H12x12)` with `row_begin = body_i * 4`, `col_begin = body_j * 4`.
/// Triplet order is `counter = a * 4 + b`, sub-block `h.block<3,3>(3a, 3b)` — the
/// order the reference's unpacker generates.
template <typename TripletView, typename OffT, typename I, typename HMat>
inline void unpack_body_block(TripletView    triplets,
                              const OffT&    triplet_offset,
                              const I&       body_i,
                              const I&       body_j,
                              const HMat&    h) noexcept
{
    using S  = lps::mat_element_t<HMat>;
    using H3 = lps::LMat<S, 3, 3>;
    for(int a = 0; a < kBlocksPerBody; ++a)
    {
        for(int b = 0; b < kBlocksPerBody; ++b)
        {
            H3 blk = h.template block<3, 3>(a * 3, b * 3);
            triplets(triplet_offset + (a * kBlocksPerBody + b))
                .write(body_i * kBlocksPerBody + a, body_j * kBlocksPerBody + b, blk);
        }
    }
}

/// The reporters' own buffer flavour: one triplet per **12x12 body block** (the
/// reference's `reporter_hessians.reshape(N, N)`), so `body_i`/`body_j` are body
/// indices here, not 3x3 block indices.
template <typename TripletView, typename I, typename HMat>
inline void write_body_block(TripletView blocks12,
                             const I&    index,
                             const I&    body_i,
                             const I&    body_j,
                             const HMat& h) noexcept
{
    blocks12(index).write(body_i, body_j, h);
}

/// Read one doublet/triplet element of a view: the non-const flavour hands out a
/// write *proxy* (whose `read()` is rvalue-qualified), the const flavour's
/// `operator()` already returns the value (the T5.4a helper).
template <typename View, typename I>
[[nodiscard]] inline auto read_element(View v, I&& i) noexcept
{
    if constexpr(requires { v(std::forward<I>(i)).read(); })
    {
        return v(std::forward<I>(i)).read();
    }
    else
    {
        return v(std::forward<I>(i));
    }
}

// ===========================================================================
// 4) the diagonal-Hessian record used by the diag-inv preconditioner
//    (`diag_hessian`, one flat 12x12 per body)
// ===========================================================================
/// Flat element index of `(r, c)` of body `body` in the 12x12-per-body buffer
/// (column-major, matching `LMatrix12x12`'s own storage).
[[nodiscard]] constexpr std::size_t diag_hessian_index(std::int32_t body, int r, int c) noexcept
{
    return static_cast<std::size_t>(body) * (kDofPerBody * kDofPerBody)
           + static_cast<std::size_t>(c * kDofPerBody + r);
}

/// `diag_hessian(body) = h` (the reference's k2 assignment).
///
/// NOTE: the flat index is built with *DSL-safe* arithmetic (`body * 144 + ...`,
/// never a `static_cast<int32_t>` of a device value) so the same body compiles for
/// the host and the device flavour (cookbook §7.6 trap 2).  `(kDofPerBody *
/// kDofPerBody)` and `(c * kDofPerBody + r)` are compile-time constants.
template <typename FlatView, typename I, typename HMat>
inline void store_diag_block(FlatView flat, const I& body, const HMat& h) noexcept
{
    for(int c = 0; c < kDofPerBody; ++c)
        for(int r = 0; r < kDofPerBody; ++r)
            flat.write(body * (kDofPerBody * kDofPerBody) + (c * kDofPerBody + r), h(r, c));
}

/// `diag_hessian(body) += h` (the reference's `eigen::atomic_add`): several
/// triplets can address the same body, so this must be an *atomic* accumulate,
/// which is why the buffer is passed as its flat scalar view (cookbook §7.20
/// rule 1: the scalar alias of an AoS buffer).
template <typename FlatView, typename I, typename HMat>
inline void add_diag_block(FlatView flat, const I& body, const HMat& h) noexcept
{
    for(int c = 0; c < kDofPerBody; ++c)
        for(int r = 0; r < kDofPerBody; ++r)
            (void)flat.atomic_add(body * (kDofPerBody * kDofPerBody) + (c * kDofPerBody + r),
                                  h(r, c));
}

// ===========================================================================
// 5) the assembly kernel bodies (one implementation for host and device)
// ===========================================================================
/// `abd_linear_subsystem_assemble_kinetic_shape_k1_kernel`:
/// `src = is_fixed(i) ? 0 : (shape_gradient(i) + (is_external_kinetic(i) ? 0 :
/// kinetic_gradient(i))); gradients.segment<12>(i * 12) = src;`
template <typename GradView, typename I, typename ShapeGradAt, typename KineticGradAt, typename FixedAt, typename ExternalKineticAt>
inline void body_gradient_row(GradView gradients,
                              const I&  i,
                              ShapeGradAt&&   shape_gradient_at,
                              KineticGradAt&& kinetic_gradient_at,
                              FixedAt&&       is_fixed_at,
                              ExternalKineticAt&& is_external_kinetic_at) noexcept
{
    using GVec = std::remove_cvref_t<decltype(shape_gradient_at(i))>;
    using S    = lps::vec_element_t<GVec>;

    auto shape   = shape_gradient_at(i);
    auto kinetic = kinetic_gradient_at(i);
    auto fixed   = is_fixed_at(i) != 0;
    auto ext     = is_external_kinetic_at(i) != 0;

    // `if(!is_external_kinetic(i)) src += kinetic_gradient(i);` -- a select
    auto with_kinetic = select_container(shape + kinetic, shape, ext);
    // `if(is_fixed(i)) src.setZero();`
    auto src = select_container(with_kinetic,
                                lps::make_vec<S, kDofPerBody>(
                                    [](int) noexcept { return lps::zero_of<S>(); }),
                                fixed);
    gradients.template segment<kDofPerBody>(i * kDofPerBody) = src;
}

/// `abd_linear_subsystem_assemble_kinetic_shape_k2_kernel`:
/// `H12 = is_fixed(i) ? Identity() : (shape_hessian(i) + (is_external_kinetic(i) ?
/// 0 : kinetic_hessian(i))); diag_hessian(i) = H12; zero_out_lower(H12);
/// write(i * 4, i * 4, H12);`
template <typename TripletView, typename OffT, typename FlatView, typename I, typename ShapeHessAt, typename KineticHessAt, typename FixedAt, typename ExternalKineticAt>
inline void body_hessian_row(TripletView    triplets,
                             const OffT&    triplet_offset,
                             FlatView       flat_diag,
                             const I&       i,
                             ShapeHessAt&&    shape_hessian_at,
                             KineticHessAt&&  kinetic_hessian_at,
                             FixedAt&&        is_fixed_at,
                             ExternalKineticAt&& is_external_kinetic_at) noexcept
{
    using HMat = std::remove_cvref_t<decltype(shape_hessian_at(i))>;
    using S    = lps::mat_element_t<HMat>;

    auto shape   = shape_hessian_at(i);
    auto kinetic = kinetic_hessian_at(i);
    auto fixed   = is_fixed_at(i) != 0;
    auto ext     = is_external_kinetic_at(i) != 0;

    auto with_kinetic = select_container(shape + kinetic, shape, ext);
    // Fill the kinetic Hessian with Identity() for a fixed body (the reference's
    // anti-singularity step)
    auto h = select_container(with_kinetic, HMat::Identity(), fixed);

    store_diag_block(flat_diag, i, h);

    // `zero_out_lower(H12x12)` -- the reference's robustness step
    zero_out_lower(h);

    unpack_body_block(triplets, triplet_offset, i, i, h);
}

/// `abd_linear_subsystem_assemble_reporters_k1_kernel`:
/// `if(is_fixed(body_i)) {} else { gradients.segment<12>(body_i * 12).atomic_add(G12); }`
/// Branchless: "skip" becomes "add exactly 0" (k1 zeroes those rows anyway).
template <typename GradView, typename SrcView, typename I, typename FixedAt>
inline void reporter_gradient_row(GradView gradients,
                                  SrcView  src,
                                  const I& I_index,
                                  FixedAt&& is_fixed_at)
{
    auto d      = read_element(src, I_index);
    using GVec  = std::remove_cvref_t<decltype(d.value)>;
    using S     = lps::vec_element_t<GVec>;
    auto zero   = lps::make_vec<S, kDofPerBody>([](int) noexcept { return lps::zero_of<S>(); });
    auto g      = select_container(d.value, zero, is_fixed_at(d.index) != 0);
    gradients.template segment<kDofPerBody>(d.index * kDofPerBody).atomic_add(g);
}

/// `abd_linear_subsystem_assemble_reporters_k2_kernel`:
/// `if(body_i == body_j && !has_fixed) diag_hessian(body_i) += H12x12;`
/// `has_fixed ? 0 : (body_i == body_j ? upper(H12x12) : (body_i > body_j ? 0 : H12x12))`
/// `write(body_i * 4, body_j * 4, H12x12);`
///
/// (the reporters report the *upper* triangle, so `body_i <= body_j`; the
/// `body_i > body_j` branch is the reference's defensive zeroing).
template <typename TripletView, typename OffT, typename FlatView, typename SrcView, typename I, typename FixedAt>
inline void reporter_hessian_row(TripletView triplets,
                                 const OffT& triplet_offset,
                                 FlatView    flat_diag,
                                 SrcView     src,
                                 const I&    I_index,
                                 FixedAt&&   is_fixed_at)
{
    auto t = read_element(src, I_index);
    using HMat = std::remove_cvref_t<decltype(t.value)>;
    using S    = lps::mat_element_t<HMat>;

    auto body_i = t.row_index;
    auto body_j = t.col_index;

    auto has_fixed = pred_or(is_fixed_at(body_i) != 0, is_fixed_at(body_j) != 0);
    auto same      = body_i == body_j;
    auto gt        = body_i > body_j;

    // Fill the diagonal Hessian for the diag-inv preconditioner: gate by adding a
    // selected block (the reference's `if(body_i == body_j && !has_fixed)`)
    auto diag_contrib = select_container(t.value,
                                         lps::make_mat<S, kDofPerBody, kDofPerBody>(
                                             [](int, int) noexcept { return lps::zero_of<S>(); }),
                                         pred_and(same, !has_fixed));
    add_diag_block(flat_diag, body_i, diag_contrib);

    auto zero_mat = lps::make_mat<S, kDofPerBody, kDofPerBody>(
        [](int, int) noexcept { return lps::zero_of<S>(); });

    // `if(body_i == body_j) zero_out_lower(H12x12); else if(body_i > body_j)
    //  H12x12.setZero();`
    auto lower_case = select_container(zero_mat, upper_only(t.value), same);
    auto kept       = select_container(t.value, lower_case, pred_or(same, gt));
    auto h          = select_container(kept, zero_mat, has_fixed);

    unpack_body_block(triplets, triplet_offset, body_i, body_j, h);
}

/// `abd_linear_subsystem_assemble_dytopo_effect_k1_kernel`:
/// `i = g_i - vertex_offset; body_i = v2b(i); G12 = Js(i).T() * G3;`
/// `if(is_fixed(body_i)) {} else { gradients.segment<12>(body_i * 12).atomic_add(G12); }`
///
/// The ABD-specific half of the "Jacobi triple": the force of a *vertex* is
/// mapped to the body dofs by `J^T`.
template <typename GradView, typename SrcView, typename I, typename OffsetT, typename V2BAt, typename JAt, typename FixedAt>
inline void dytopo_gradient_row(GradView gradients,
                                SrcView  src,
                                const I& I_index,
                                const OffsetT& vertex_offset,
                                V2BAt&&  body_of_vertex_at,
                                JAt&&    jacobi_at,
                                FixedAt&& is_fixed_at)
{
    auto d      = read_element(src, I_index);
    auto i      = d.index - vertex_offset;  // from global to local vertex id
    auto body_i = body_of_vertex_at(i);
    using GVec  = std::remove_cvref_t<decltype(d.value)>;
    using S     = lps::vec_element_t<GVec>;

    auto g12  = abd_jacobi_T_mul(jacobi_at(i), d.value);
    auto zero = lps::make_vec<S, kDofPerBody>([](int) noexcept { return lps::zero_of<S>(); });
    auto g    = select_container(g12, zero, is_fixed_at(body_i) != 0);
    gradients.template segment<kDofPerBody>(body_i * kDofPerBody).atomic_add(g);
}

/// `abd_linear_subsystem_assemble_dytopo_effect_k2_kernel`:
/// `i = g_i - vertex_offset; j = g_j - vertex_offset; body_i = v2b(i); body_j = v2b(j);`
/// `L = min(body_i, body_j); R = max(body_i, body_j);`
/// `H12 = is_fixed(body_i) || is_fixed(body_j) ? 0 :`
/// `      (body_i < body_j ? J_i^T H J_j :`
/// `       body_i > body_j ? J_j^T H^T J_i :`
/// `       (i != j ? J_i^T H J_j + J_j^T H^T J_i : J_i^T H J_j));`
/// `if(body_i == body_j) { diag_hessian(body_i) += H12; zero_out_lower(H12); }`
/// `write(L * 4, R * 4, H12);`
///
/// NOTE: "we know the half contact Hessian has `i <= j`, but we don't know the
/// body order, so test and swap if necessary" (the reference) — the *global*
/// block always lands in the upper triangle at `(L, R)`.
template <typename TripletView, typename OffT, typename FlatView, typename SrcView, typename I, typename OffsetT, typename V2BAt, typename JAt, typename FixedAt>
inline void dytopo_hessian_row(TripletView triplets,
                               const OffT& triplet_offset,
                               FlatView    flat_diag,
                               SrcView     src,
                               const I&    I_index,
                               const OffsetT& vertex_offset,
                               V2BAt&&   body_of_vertex_at,
                               JAt&&     jacobi_at,
                               FixedAt&& is_fixed_at)
{
    auto t = read_element(src, I_index);
    using HMat = std::remove_cvref_t<decltype(t.value)>;
    using S    = lps::mat_element_t<HMat>;

    auto i      = t.row_index - vertex_offset;
    auto j      = t.col_index - vertex_offset;
    auto body_i = body_of_vertex_at(i);
    auto body_j = body_of_vertex_at(j);

    auto J_i = jacobi_at(i);
    auto J_j = jacobi_at(j);

    // `L = body_i, R = body_j; if(body_i > body_j) swap;`
    auto swap   = body_i > body_j;
    auto L      = select_scalar(body_j, body_i, swap);
    auto R      = select_scalar(body_i, body_j, swap);

    auto h_lt      = abd_jacobi_JT_H_J(J_i, t.value, J_j);
    auto H_T       = t.value.transpose();
    auto h_gt      = abd_jacobi_JT_H_J(J_j, H_T, J_i);
    auto h_pair    = h_lt + abd_jacobi_JT_H_J(J_j, H_T, J_i);
    auto same_body = body_i == body_j;
    auto i_ne_j    = i != j;

    // body_i == body_j: `i != j` sums the two halves, `i == j` keeps one
    auto h_same = select_container(h_lt, h_pair, i_ne_j);
    // body_i != body_j
    auto h_diff = select_container(h_gt, h_lt, body_i < body_j);

    auto h_full = select_container(h_diff, h_same, same_body);
    auto has_fixed = pred_or(is_fixed_at(body_i) != 0, is_fixed_at(body_j) != 0);

    auto zero_mat = lps::make_mat<S, kDofPerBody, kDofPerBody>(
        [](int, int) noexcept { return lps::zero_of<S>(); });

    // diag-inv preconditioner: only a same-body pair contributes, and it
    // contributes the *un-zeroed* H12 (the reference's order)
    add_diag_block(flat_diag, body_i, select_container(zero_mat, h_full, same_body));

    // `zero_out_lower(H12x12)` for the same-body case
    auto h_body = select_container(upper_only(h_full), h_full, same_body);
    auto h      = select_container(h_body, zero_mat, has_fixed);

    unpack_body_block(triplets, triplet_offset, L, R, h);
}

/// `abd_linear_subsystem_retrieve_solution_kernel`:
/// `dq(i) = -x.segment<12>(i * 12).as_eigen();`
///
/// `put` receives the negated 12-vector (the caller stores it into its own element
/// type: the reference's `BufferView<Vector12>` is `lud::LBufferView<LVector12>`).
template <typename ResultView, typename I, typename Put>
inline void retrieve_solution_row(ResultView result, const I& i, Put&& put) noexcept
{
    put(-result.template segment<kDofPerBody>(i * kDofPerBody).template to_vec<kDofPerBody>());
}

/// `abd_linear_subsystem_diag_norm_kernel`:
/// `for(i = 0..11) diag_blocks_norm(idx * 12 + i) = is_fixed(idx) ? 0 :
///  abs(diag_hess(idx)(i, i));`
///
/// One writer per segment (12 segments per body), so this is a plain
/// select-backed store.
template <typename FlatView, typename NormView, typename I, typename FixedAt>
inline void diag_norm_row(FlatView  flat_diag,
                          NormView  diag_blocks_norm,
                          const I&  idx,
                          FixedAt&& is_fixed_at) noexcept
{
    auto value0 = flat_diag.read(idx * (kDofPerBody * kDofPerBody));
    using S     = std::remove_cvref_t<decltype(value0)>;
    auto zero   = lps::zero_of<S>();
    auto fixed  = is_fixed_at(idx) != 0;
    for(int i = 0; i < kDofPerBody; ++i)
    {
        auto diag =
            flat_diag.read(idx * (kDofPerBody * kDofPerBody) + (i * kDofPerBody + i));
        diag_blocks_norm(idx * kDofPerBody + i) = select_scalar(lps::abs(diag), zero, fixed);
    }
}

/// `abd_linear_subsystem_mass_norm_kernel`:
/// `block_norm(idx) = is_fixed(idx) ? 0 : mass(idx).mass();`
///
/// The mass is a *callable*, not a view: an `ABDJacobiDyadicMass` buffer is AoS
/// with a 104 B element, so it has no `double`-stride scalar alias (cookbook
/// §7.20 rule 1's precondition) -- the caller reads its own element type.
template <typename MassAt, typename NormView, typename I, typename FixedAt>
inline void mass_norm_row(MassAt&& mass_at,
                          NormView block_norm,
                          const I& idx,
                          FixedAt&& is_fixed_at) noexcept
{
    auto value = mass_at(idx).m_mass;
    using S    = std::remove_cvref_t<decltype(value)>;
    auto zero  = lps::zero_of<S>();
    block_norm(idx) = select_scalar(value, zero, is_fixed_at(idx) != 0);
}
}  // namespace abd_linear_subsystem_detail
}  // namespace uipc::backend::luisa

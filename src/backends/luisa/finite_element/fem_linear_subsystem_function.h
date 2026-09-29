#pragma once
/********************************************************************
 * @file fem_linear_subsystem_function.h
 * @brief T5.4a — the *generic*, host- and device-callable routines of
 *        `FEMLinearSubsystem`'s assembly: the fan-out/slice arithmetic of
 *        `Impl::{report_extent,assemble}` and every named kernel body of
 *        src/backends/cuda/finite_element/fem_linear_subsystem.cu.
 *
 * WHY THIS HEADER EXISTS (the T5.4a brief's §2)
 * --------------------------------------------
 * The assembly only ever runs inside `GlobalLinearSystem::solve()`, which is the
 * T8.3 solve path, and `SimEngine` is not wired in T5.4a.  Everything below is
 * therefore factored out of the kernels as a plain generic routine over
 *
 *   * the *view flavour* — `storage::device` inside a kernel,
 *     `storage::memory` for host evaluation — and
 *   * the *scalar type* — `double` on the host, `lcrt::Var<double>` on the
 *     device, exactly like T5.2's `*_function.h` / T5.3's `*_detail` routines —
 *
 * so the shipped kernel lambda and a unit test run the *same* code (cookbook
 * §7.4).  `fem_linear_subsystem.cpp`'s kernels call these; the T5.4a test drives
 * them against hand-derived indices and values, plus `double` vs
 * `lcrt::Var<double>` bitwise (the `fem_linear_device_vs_host` case).
 *
 * BRANCHLESS WHERE THE PREDICATE IS DEVICE DATA (RULE M2)
 * ------------------------------------------------------
 * The reference kernels branch on `is_fixed(i)` and on `i == j`; a device
 * predicate cannot drive host control flow, so those branches become `select`s
 * (the T5.2 assembler's `upper_LR` formulation).  A select passes its chosen
 * operand through untouched, so:
 *   * "skip the scatter for a fixed vertex" becomes "scatter exactly 0" — the
 *     fixed rows are zeroed afterwards by the reference's own k1 kernel anyway;
 *   * "write Zero() on the diagonal / Identity() off it" is one element-wise
 *     select of the two candidates.
 *
 * The index/`is_fixed` accessors are *callables* (`is_fixed_at(i)`), because the
 * host flavour indexes a host array and the device flavour reads a buffer —
 * both `[&](auto&& i) { return ...; }` at the call site.
 *********************************************************************/
#include <type_define.h>
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

namespace fem_linear_subsystem_detail
{
using scalar_detail::pred_or;
using scalar_detail::select_container;
using scalar_detail::select_scalar;

// ===========================================================================
// 1) the fan-out slice: which [offset, count) of a shared assembly buffer
//    belongs to subsystem/reporter/constitution `index`
// ===========================================================================
/// The reference's `auto [offset, count] =
/// offsets_counts[reporter->m_index];` step, factored so a test can drive it
/// against hand-derived offsets (`OffsetCountCollection` is host-only).
[[nodiscard]] inline std::pair<IndexT, IndexT> slice_of(
    const OffsetCountCollection<IndexT>& offsets_counts, SizeT index)
{
    auto [offset, count] = offsets_counts[index];
    return std::pair<IndexT, IndexT>{offset, count};
}

// ===========================================================================
// 2) `Impl::report_extent`'s extent arithmetic
// ===========================================================================
/// The counts `FEMLinearSubsystem::Impl::report_extent` arrives at, in the
/// reference's own order:
/// `[KineticG | OtherG ... | DyTopoG]` / `[KineticH | OtherH ... | DyTopoH]`.
struct ExtentBudget
{
    SizeT kinetic_gradient_count  = 0;
    SizeT kinetic_hessian_count   = 0;
    SizeT reporter_gradient_count = 0;
    SizeT reporter_hessian_count  = 0;
    SizeT dytopo_gradient_count   = 0;
    SizeT dytopo_hessian_count    = 0;
    /// the reference's `grad_offset` (diagnostic: the *gradient* buffer is the
    /// global one, only the Hessian needs an offset budget)
    SizeT gradient_offset = 0;
    /// the reference's `hess_offset` == the block count reported to
    /// `DiagExtentInfo::extent`
    SizeT hessian_offset = 0;
    /// `fem().xs.size() * 3`
    SizeT dof_count = 0;

    /// The reference's `UIPC_ASSERT(!(gradient_only && !hess_offset == 0), ...)`
    /// (issue #295): in gradient-only mode nothing may report Hessian blocks.
    [[nodiscard]] bool check_gradient_only(bool gradient_only) const noexcept
    {
        return !(gradient_only && hessian_offset != 0);
    }
};

/// The reference's `Impl::report_extent` arithmetic.  `reporter_*_counts` are the
/// per-reporter counts the reporters just reported (and which
/// `Impl::report_extent` stores into its offset/count collections); the dytopo
/// counts are the receiver's own (`0` when there is no receiver).
template <typename GradCountRange, typename HessCountRange>
[[nodiscard]] inline ExtentBudget plan_extent(SizeT vertex_count,
                                              bool  gradient_only,
                                              bool  has_complement,
                                              const GradCountRange& reporter_gradient_counts,
                                              const HessCountRange& reporter_hessian_counts,
                                              SizeT dytopo_gradient_count,
                                              SizeT dytopo_hessian_count)
{
    ExtentBudget budget;
    SizeT        grad_offset = 0;
    SizeT        hess_offset = 0;

    // 1) Hessian count
    // We assume reporters won't produce contact
    if(has_complement)
    {
        // Kinetic
        auto kinetic_grad_count       = vertex_count;
        budget.kinetic_gradient_count = kinetic_grad_count;
        grad_offset += kinetic_grad_count;
        if(!gradient_only)
        {
            budget.kinetic_hessian_count = vertex_count;
            hess_offset += vertex_count;
        }

        // Reporters
        budget.reporter_gradient_count =
            std::accumulate(reporter_gradient_counts.begin(),
                            reporter_gradient_counts.end(),
                            SizeT{0},
                            [](SizeT acc, const auto& c) noexcept
                            { return acc + static_cast<SizeT>(c); });
        // a reporter must not report Hessian blocks in gradient-only mode (the
        // reference asserts it per reporter, see FemLinearSubsystemReporter)
        budget.reporter_hessian_count =
            gradient_only
                ? SizeT{0}
                : std::accumulate(reporter_hessian_counts.begin(),
                                  reporter_hessian_counts.end(),
                                  SizeT{0},
                                  [](SizeT acc, const auto& c) noexcept
                                  { return acc + static_cast<SizeT>(c); });

        grad_offset += budget.reporter_gradient_count;
        hess_offset += budget.reporter_hessian_count;
    }

    // 2) Dytopo effect (the DyTopo system decides `component_flags` itself)
    budget.dytopo_gradient_count = dytopo_gradient_count;
    budget.dytopo_hessian_count  = dytopo_hessian_count;
    grad_offset += dytopo_gradient_count;
    hess_offset += dytopo_hessian_count;

    budget.gradient_offset = grad_offset;
    budget.hessian_offset  = hess_offset;
    budget.dof_count       = vertex_count * 3;
    return budget;
}

// ===========================================================================
// 3) the element fan-out write: N global vertex ids + the element's local
//    gradient/Hessian -> the shared buffers at the element's slice, with the
//    reference's `upper_LR` swap (the T5.2 assembler case)
// ===========================================================================
/// The gradient half: `N` doublets of 3 components at `doublet_offset`,
/// carrying the element's `N` *global* vertex indices.
template <typename DoubletView, typename OffsetT, typename IVec, typename GVec>
void write_element_gradient(DoubletView    doublets,
                            const OffsetT& doublet_offset,
                            const IVec&    global_indices,
                            const GVec&    element_gradient)
{
    constexpr int N = static_cast<int>(lps::vec_dim_v<IVec>);
    DoubletVectorAssembler{doublets}.segment<N>(doublet_offset).write(global_indices,
                                                                     element_gradient);
}

/// The Hessian half: the `N (N + 1) / 2` upper-triangle 3x3 blocks at
/// `triplet_offset`, at global indices, with the `upper_LR` swap
/// (`TripletMatrixAssembler::half_block`, T5.2).
template <typename TripletView, typename OffsetT, typename IVec, typename HMat>
void write_element_hessian(TripletView    triplets,
                           const OffsetT& triplet_offset,
                           const IVec&    global_indices,
                           const HMat&    element_hessian)
{
    constexpr int N = static_cast<int>(lps::vec_dim_v<IVec>);
    TripletMatrixAssembler{triplets}.half_block<N>(triplet_offset).write(global_indices,
                                                                        element_hessian);
}

// ===========================================================================
// 4) the assembly kernel bodies (one implementation for host and device)
// ===========================================================================
/// Read one doublet/triplet element of a view: the non-const flavour hands out a
/// write *proxy* (whose `read()` is rvalue-qualified), the const flavour's
/// `operator()` already returns the value.
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

/// `FEMLinearSubsystem_assemble_k1_kernel`:
/// `if(is_fixed(i)) gradients.segment<3>(i * 3) = 0;`
///
/// Branchless: the segment is written as `select(current, 0, is_fixed)` — the
/// caller has zero-filled the buffer, so for a non-fixed vertex the written
/// value is exactly the value already there.
template <typename GradView, typename I, typename FixedAt>
void clear_fixed_gradient_row(GradView gradients, const I& i, FixedAt&& is_fixed_at) noexcept
{
    auto seg = gradients.template segment<3>(i * 3);
    using S  = lps::vec_element_t<decltype(seg.template to_vec<3>())>;
    auto zero = lps::make_vec<S, 3>([](int) noexcept { return lps::zero_of<S>(); });
    auto cur  = seg.template to_vec<3>();
    seg       = select_container(cur, zero, is_fixed_at(i) != 0);
}

/// `FEMLinearSubsystem_assemble_k2_kernel`:
/// `if(is_fixed(i) || is_fixed(j)) { i != j ? H = Zero() : H = Identity(); }`
///
/// Branchless: both candidates are built and selected element-wise.
template <typename TripletView, typename I, typename FixedAt>
void fix_hessian_row(TripletView hessians, const I& I_index, FixedAt&& is_fixed_at) noexcept
{
    auto t     = read_element(hessians, I_index);
    auto fixed = pred_or(is_fixed_at(t.row_index) != 0, is_fixed_at(t.col_index) != 0);
    auto diagonal = t.row_index == t.col_index;

    using HMat = std::remove_cvref_t<decltype(t.value)>;
    using S    = lps::mat_element_t<HMat>;

    auto zeros = lps::make_mat<S, 3, 3>(
        [](int, int) noexcept { return lps::zero_of<S>(); });
    auto identity    = lps::LMat<S, 3, 3>::Identity();
    auto replacement = select_container(zeros, identity, diagonal);
    auto out         = select_container(t.value, replacement, fixed);
    hessians(I_index).write(t.row_index, t.col_index, out);
}

/// `FEMLinearSubsystem_assemble_kinetic_kernel` /
/// `..._assemble_reporters_kernel` (identical bodies):
/// `if(is_fixed(i)) return; dst.segment<3>(i * 3).atomic_add(G3);`
///
/// Branchless: "skip" becomes "add exactly 0" (k1 zeroes those rows anyway).
template <typename DstView, typename SrcView, typename I, typename FixedAt>
void scatter_gradient_row(DstView dst, SrcView src, const I& I_index, FixedAt&& is_fixed_at)
{
    auto d      = read_element(src, I_index);
    using GVec  = std::remove_cvref_t<decltype(d.value)>;
    using S     = lps::vec_element_t<GVec>;
    auto zero   = lps::make_vec<S, 3>([](int) noexcept { return lps::zero_of<S>(); });
    auto g      = select_container(d.value, zero, is_fixed_at(d.index) != 0);
    dst.segment<3>(d.index * 3).atomic_add(g);
}

/// `FEMLinearSubsystem_assemble_dytopo_effect_k1_kernel`:
/// `auto i = g_i - vertex_offset; if(is_fixed(i)) return;
///  gradients.segment<3>(i * 3).atomic_add(G3);`
template <typename DstView, typename SrcView, typename I, typename OffsetT, typename FixedAt>
void scatter_dytopo_gradient_row(DstView     gradients,
                                 SrcView     src,
                                 const I&    I_index,
                                 const OffsetT& vertex_offset,
                                 FixedAt&&   is_fixed_at)
{
    auto d     = read_element(src, I_index);
    auto i     = d.index - vertex_offset;  // from global to local
    using GVec = std::remove_cvref_t<decltype(d.value)>;
    using S    = lps::vec_element_t<GVec>;
    auto zero  = lps::make_vec<S, 3>([](int) noexcept { return lps::zero_of<S>(); });
    auto g     = select_container(d.value, zero, is_fixed_at(i) != 0);
    gradients.segment<3>(i * 3).atomic_add(g);
}

/// `FEMLinearSubsystem_assemble_dytopo_effect_k2_kernel`:
/// `auto [g_i, g_j, H3] = dytopo_effect_hessian(I);
///  hessians(I).write(g_i - vertex_offset, g_j - vertex_offset, H3);`
///
/// NOTE: the reference deliberately ignores the fixed mask here ("in final phase
/// we will clear the fixed vertex hessian anyway").
template <typename DstView, typename SrcView, typename I, typename OffsetT>
void copy_dytopo_hessian_row(DstView        hessians,
                             SrcView        src,
                             const I&       I_index,
                             const OffsetT& vertex_offset)
{
    auto t = read_element(src, I_index);
    hessians(I_index).write(t.row_index - vertex_offset, t.col_index - vertex_offset, t.value);
}

/// `FEMLinearSubsystem_retrieve_solution_kernel`:
/// `dxs(i) = -result.segment<3>(i * 3).as_eigen();`
///
/// `put` receives the negated 3-vector (the caller stores it into its own
/// element type: the reference's `BufferView<Vector3>` is
/// `lud::LBufferView<LVector3>` in the port).
template <typename ResultView, typename I, typename Put>
void retrieve_solution_row(ResultView result, const I& i, Put&& put) noexcept
{
    put(-result.template segment<3>(i * 3).template to_vec<3>());
}

/// The per-triplet result of `FEMLinearSubsystem_diag_norm_kernel`'s body.
template <typename I, typename V, typename P>
struct DiagBlockNormRow
{
    I index;        ///< the segment index to store into, clamped into the window
    V value;        ///< `is_fixed(i) ? 0 : max(|H00|, |H11|, |H22|)`
    P out_of_window;  ///< `i >= segment_count || j >= segment_count` (the ref returns)
    P is_diagonal;    ///< `i == j` (only a diagonal block defines the segment norm)
};

/// `FEMLinearSubsystem_diag_norm_kernel`'s body:
/// `auto [g_i, g_j, H3x3] = triplet(I);
///  i = g_i - fem_segment_offset; j = g_j - fem_segment_offset;
///  if(i >= count || j >= count) return;
///  if(i == j) diag_blocks_norm(i) = is_fixed(i) ? 0 : max(|H00|,|H11|,|H22|);`
///
/// The two guards are *device* predicates, so the value is computed
/// unconditionally and **the caller gates the store** (the reference's early
/// `return` and its `if(i == j)` are the two predicates returned here).  A
/// branchless store is not an option: several triplets of one row address the
/// same segment, and a non-diagonal thread writing back the value it read would
/// race with the diagonal thread's store.
///
/// The returned index is *clamped* into `[0, segment_count)`: the reference's
/// grid is sized by the triplet count while the segment window is the subsystem's
/// dof window, so a lane can genuinely address a segment outside it, and the
/// device buffer only covers `segment_count` entries (cookbook §7.12 rule 2).
template <typename TripletView, typename I, typename OffsetT, typename FixedAt>
[[nodiscard]] inline auto diag_block_norm_row(TripletView    triplet,
                                              const I&       I_index,
                                              const OffsetT& fem_segment_offset,
                                              const OffsetT& fem_segment_count,
                                              FixedAt&&      is_fixed_at) noexcept
{
    auto t = read_element(triplet, I_index);
    auto i = t.row_index - fem_segment_offset;
    auto j = t.col_index - fem_segment_offset;

    auto out_of_window = pred_or(i >= fem_segment_count, j >= fem_segment_count);
    auto is_diagonal   = i == j;

    using HMat = std::remove_cvref_t<decltype(t.value)>;
    using S    = lps::mat_element_t<HMat>;
    auto a = lps::abs(t.value(0, 0));
    auto b = lps::abs(t.value(1, 1));
    auto c = lps::abs(t.value(2, 2));

    auto block_norm = lps::max(lps::max(a, b), c);
    auto zero       = lps::zero_of<S>();
    // is_fixed(i) ? 0 : block_norm
    auto value = select_scalar(block_norm, zero, is_fixed_at(i) != 0);

    auto clamped = lps::min(lps::max(i, OffsetT{0}), OffsetT{fem_segment_count - 1});
    return DiagBlockNormRow{clamped, value, out_of_window, is_diagonal};
}

/// `FEMLinearSubsystem_mass_norm_kernel`:
/// `diag_blocks_norm(I) = is_fixed(I) ? 0 : mass(I);`
///
/// One writer per segment, so this one *is* a plain select-backed store.
template <typename MassView, typename NormView, typename I, typename FixedAt>
void mass_block_norm_row(MassView mass, NormView diag_blocks_norm, const I& i, FixedAt&& is_fixed_at)
{
    // `read(i)` (not `operator()`): the non-const flavour's `operator()` hands out
    // an assignable proxy, and `zero_of<proxy>` is not a thing.
    auto value = mass.read(i);
    using V    = std::remove_cvref_t<decltype(value)>;
    auto zero  = lps::zero_of<V>();
    diag_blocks_norm(i) = select_scalar(value, zero, is_fixed_at(i) != 0);
}
}  // namespace fem_linear_subsystem_detail
}  // namespace uipc::backend::luisa

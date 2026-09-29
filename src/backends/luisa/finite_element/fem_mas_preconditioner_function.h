#pragma once
/********************************************************************
 * @file fem_mas_preconditioner_function.h
 * @brief T6.4c — the FEM MAS preconditioner's two *fallback* kernel bodies, factored
 *        so the shipped shaders and the unit test run the same code.
 *
 * Same recipe as T5.4b's `fem_diag_preconditioner_function.h` (cookbook §7.4 /
 * §7.20 rule 1).  The reference
 * (`src/backends/cuda/finite_element/fem_mas_preconditioner.cu`) has two `__global__`
 * kernels besides the engine's, both handling the vertices that the partition mapper
 * left unpartitioned (`part_ids[v] < 0`, e.g. an Empty-constitution geometry):
 *
 *   assemble_diag_inv_for_unpartitioned:
 *       auto&& [g_i, g_j, H3x3] = triplet(I);
 *       int i = g_i - fem_offset, j = g_j - fem_offset;
 *       if(i < 0 || i >= fem_count || j < 0 || j >= fem_count) return;
 *       if(i == j && unpart(i) == 1) diag(i) = eigen::inverse(H3x3);
 *   apply_diag_inv_for_unpartitioned:
 *       if(i >= n) return;
 *       if(*converged != 0) return;
 *       if(unpart(i) == 1) z.segment<3>(i*3) = diag(i) * r.segment<3>(i*3);
 *
 * Both early `return`s are *device* predicates (RULE M2), so — exactly like
 * `fem_diag_preconditioner_function.h`'s `diag_inv_row` — the assemble routine
 * returns the guards and its caller gates the store (`$if` in the shipped kernel, an
 * `if` on the host).  The `converged` early-out of the apply kernel is one scalar read
 * for the whole dispatch, so it stays in the caller.
 *
 * NOTE (the reference's own guard, transcribed): unlike the diagonal preconditioner,
 * this assemble row *does* test `i < 0`, so the reference's signed-index hole is
 * closed here.  The port keeps the difference.
 *********************************************************************/
#include <type_define.h>
#include <finite_element/fem_linear_subsystem_function.h>
#include <finite_element/fem_diag_preconditioner_function.h>
#include <luisa_device/linear/linear.h>
#include <luisa_device/math/matrix_ops.h>  // lps::inverse (T2.2)
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
namespace fem_mas_preconditioner_detail
{
/// A dense vector whose element is a 3x3 block (`diag_inv` in the reference is
/// `DeviceBuffer<Matrix3x3>`, so `diag(i)` is the view's element access).
using DiagInvDeviceView     = lla::LDenseVectorDeviceView<lcrt::Var<LMatrix3x3>>;
using DiagInvHostMemoryView = lla::LDenseVectorHostMemoryView<LMatrix3x3>;

/// The reference's *unpartitioned* flag buffer (`DeviceBuffer<int>`, 1 = unpartitioned).
using UnpartDeviceView     = lla::LCDenseVectorDeviceView<lcrt::Var<IndexT>>;
using UnpartHostMemoryView = lla::LCDenseVectorHostMemoryView<IndexT>;

/// `assemble_diag_inv_for_unpartitioned_kernel`'s row body.  Fields are generic over
/// the flavour: on the device `index`/`in_window`/`is_unpart` are DSL expressions, on
/// the host `int32_t`/`bool`.
template <typename I, typename M, typename P>
struct UnpartDiagRow
{
    I index{};       // the reference's `i` (the *local* block index)
    M value{};       // eigen::inverse(H3x3)
    P in_window{};   // !(i < 0 || i >= count || j < 0 || j >= count)
    P is_unpart{};   // (i == j) && unpart(i) == 1
};

/// `assemble_diag_inv_for_unpartitioned_kernel`'s row body (see the file header).
template <typename TripletView, typename UnpartView, typename I, typename OffsetT>
[[nodiscard]] inline auto unpart_diag_inv_row(TripletView     triplet,
                                              UnpartView      unpart,
                                              const I&        triplet_index,
                                              const OffsetT&  fem_block_offset,
                                              const OffsetT&  fem_block_count) noexcept
{
    auto t = fem_linear_subsystem_detail::read_element(triplet, triplet_index);
    auto i = t.row_index - fem_block_offset;
    auto j = t.col_index - fem_block_offset;

    using I_ = std::remove_cvref_t<decltype(i)>;
    using M_ = std::remove_cvref_t<decltype(lps::inverse(t.value))>;
    using P_ = std::remove_cvref_t<decltype(scalar_detail::pred_and(
        i < fem_block_count, j < fem_block_count))>;

    return UnpartDiagRow<I_, M_, P_>{
        i,
        lps::inverse(t.value),
        // the reference's single guard, as one predicate over all four comparisons
        scalar_detail::pred_and(scalar_detail::pred_and(i >= 0, j >= 0),
                                scalar_detail::pred_and(i < fem_block_count,
                                                        j < fem_block_count)),
        i == j};
}

/// The reference's `unpart(i) == 1` read, generic over the view flavour.
template <typename View, typename I>
[[nodiscard]] inline auto unpart_flag(View unpart, I&& i) noexcept
{
    return unpart.read(std::forward<I>(i)) == IndexT{1};
}

/// `apply_diag_inv_for_unpartitioned_kernel`'s row body:
/// `if(unpart(i) == 1) z.segment<3>(i*3) = diag(i) * r.segment<3>(i*3);`
/// The `i >= n` and `converged != 0` early outs stay in the caller (see the header).
template <typename ZView, typename RView, typename DiagInvView, typename UnpartView, typename I>
void unpart_diag_apply_row(ZView        z,
                           RView        r,
                           DiagInvView  diag_inv,
                           UnpartView   unpart,
                           const I&     i) noexcept
{
    auto r3 = r.template segment<3>(i * 3).template to_vec<3>();
    auto H  = fem_diag_preconditioner_detail::load_mat3x3_block(diag_inv, i);
    z.template segment<3>(i * 3) = H * r3;
    (void)unpart;
}
}  // namespace fem_mas_preconditioner_detail
}  // namespace uipc::backend::luisa

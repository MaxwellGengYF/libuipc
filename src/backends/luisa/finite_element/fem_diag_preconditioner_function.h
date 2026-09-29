#pragma once
/********************************************************************
 * @file fem_diag_preconditioner_function.h
 * @brief T5.4b — the FEM diagonal preconditioner's two kernel bodies, factored so
 *        the shipped shaders and the unit test run the *same* code.
 *
 * Same recipe as T5.2's `utils/matrix_assembler.h`, T5.3's `*_detail` namespaces
 * and T5.4a's `fem_linear_subsystem_function.h` (cookbook §7.4 / §7.19 rule 3):
 * the kernel lambda builds the device views and calls one routine here, which the
 * unit test also instantiates with `storage::memory` views over host arrays.
 *
 * The reference (`src/backends/cuda/finite_element/fem_diag_preconditioner.cu`)
 * is a `.cu`-only class with two `__global__` kernels:
 *
 *   do_assemble: `auto&& [g_i, g_j, H3x3] = triplet(I);`
 *                `IndexT i = g_i - fem_segment_offset; j = g_j - offset;`
 *                `if(i >= fem_segment_count || j >= fem_segment_count) return;`
 *                `if(i == j) diag_inv(i) = eigen::inverse(H3x3);`
 *   do_apply:    `if(*converged != 0) return;`
 *                `z.segment<3>(i * 3) = diag_inv(i) * r.segment<3>(i * 3);`
 *
 * The two early `return`s of `do_assemble` are *device* predicates (RULE M2), so
 * — exactly like T5.4a's `diag_block_norm_row` (cookbook §7.19 rule 4) — the
 * routine *returns* the guards and its caller gates the store (`$if` in the
 * shipped kernel, an `if` on the host).  The `converged` early-out of `do_apply`
 * is *not* folded into the row routine: it is one scalar read for the whole
 * dispatch, so the kernel wraps its body in `$if(converged->read(0) == 0)` and the
 * host driver wraps its call in the same test.
 *********************************************************************/
#include <type_define.h>
#include <finite_element/fem_linear_subsystem_function.h>
#include <luisa_device/linear/linear.h>
#include <luisa_device/math/matrix_ops.h>  // lps::inverse (T2.2)
#include <utils/scalar_util.h>

namespace uipc::backend::luisa
{
namespace fem_diag_preconditioner_detail
{
/// A dense vector whose *element* is a 3x3 block: `diag_inv` in the reference is
/// `DeviceBuffer<Matrix3x3>`, so `diag_inv(i)` is the view's element access.  The
/// two flavours are the same pair the routines below are instantiated with.
using DiagInvDeviceView     = lla::LDenseVectorDeviceView<lcrt::Var<LMatrix3x3>>;
using DiagInvHostMemoryView = lla::LDenseVectorHostMemoryView<LMatrix3x3>;

/// The reference's `eigen::inverse(H3x3)` plus the two guards that decide whether
/// the caller may store it.  Every field is generic over the flavour: on the
/// device `index`/`in_window`/`is_diagonal` are DSL expressions, on the host they
/// are `int32_t`/`bool`.
template <typename I, typename M, typename P>
struct DiagInvRow
{
    I index{};        // the reference's `i` (the *local* block index)
    M value{};        // eigen::inverse(H3x3)
    P in_window{};    // !(i >= fem_segment_count || j >= fem_segment_count)
    P is_diagonal{};  // i == j
};

/// `FEMDiagPreconditioner_do_assemble_kernel`'s row body (see the file header).
template <typename TripletView, typename I, typename OffsetT>
[[nodiscard]] inline auto diag_inv_row(TripletView     triplet,
                                       const I&        triplet_index,
                                       const OffsetT&  fem_segment_offset,
                                       const OffsetT&  fem_segment_count) noexcept
{
    auto t = fem_linear_subsystem_detail::read_element(triplet, triplet_index);
    auto i = t.row_index - fem_segment_offset;
    auto j = t.col_index - fem_segment_offset;

    using I_ = std::remove_cvref_t<decltype(i)>;
    using M_ = std::remove_cvref_t<decltype(lps::inverse(t.value))>;
    using P_ = std::remove_cvref_t<decltype(scalar_detail::pred_and(
        i < fem_segment_count, j < fem_segment_count))>;

    return DiagInvRow<I_, M_, P_>{
        i,
        lps::inverse(t.value),
        scalar_detail::pred_and(i < fem_segment_count, j < fem_segment_count),
        i == j};
}

/// Read a 3x3 block out of a view: the `storage::memory` flavour already hands out
/// the value, the device flavour's `Expr<LMatrix3x3>` has to go through
/// `load_mat3x3` (type_define.h).
template <typename View, typename I>
[[nodiscard]] inline auto load_mat3x3_block(View v, I&& i) noexcept
{
    if constexpr(View::storage_type::is_device)
    {
        return load_mat3x3(v.read(std::forward<I>(i)));
    }
    else
    {
        return v.read(std::forward<I>(i));
    }
}

/// `FEMDiagPreconditioner_do_apply_kernel`'s row body:
/// `z.segment<3>(i * 3) = diag_inv(i) * r.segment<3>(i * 3);`
template <typename ZView, typename RView, typename DiagInvView, typename I>
void diag_apply_row(ZView z, RView r, DiagInvView diag_inv, const I& i) noexcept
{
    auto r3 = r.template segment<3>(i * 3).template to_vec<3>();
    auto H  = load_mat3x3_block(diag_inv, i);
    z.template segment<3>(i * 3) = H * r3;
}
}  // namespace fem_diag_preconditioner_detail
}  // namespace uipc::backend::luisa

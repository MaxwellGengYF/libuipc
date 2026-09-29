#include <affine_body/abd_linear_subsystem.h>
#include <affine_body/abd_linear_subsystem_reporter.h>
#include <affine_body/abd_linear_subsystem_function.h>
#include <affine_body/abd_dytopo_effect_receiver.h>
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/affine_body_constitution.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/flag.h>
#include <utils/report_extent_check.h>
#include <algorithm>

/*************************************************************************************************
* T6.1b — ABDLinearSubsystem (port of
* src/backends/cuda/affine_body/abd_linear_subsystem.cu).
*
* Read the scope note in abd_linear_subsystem.h first: it names the now/defer
* split and the ABD index conventions.  This file completes the declaration
* surface T6.1a left (`ReportExtentInfo`'s three members, `AssembleInfo`'s ctor +
* accessors, every `Impl` method and every `do_*` override, `REGISTER_SIM_SYSTEM`
* + `get_uid()`).
*
* Kernel layout (the reference's named kernels, in file order):
*   k1  kinetic+shape gradient  -> ...::body_gradient_row
*   k2  kinetic+shape hessian   -> ...::body_hessian_row
*   reporters k1 (gradient)     -> ...::reporter_gradient_row
*   reporters k2 (hessian)      -> ...::reporter_hessian_row
*   dytopo k1 (gradient)        -> ...::dytopo_gradient_row
*   dytopo k2 (hessian)         -> ...::dytopo_hessian_row
*   retrieve_solution           -> ...::retrieve_solution_row
*   diag_norm                   -> ...::diag_norm_row
*   mass_norm                   -> ...::mass_norm_row
* plus `zero_gradient_kernel` for the reference's `gradients.buffer_view().fill(0)`
* (see the note at its definition).
*
* Every kernel body lives in abd_linear_subsystem_function.h (the brief's §2
* testability requirement: the assembly is otherwise only reachable from T8.3's
* solve()), so the shipped lambda and the unit test run the same code (cookbook
* §7.4).
*
* DEVIATIONS (all reported in the T6.1b row / cookbook §7.22):
*  1. `diag_norm()`/`mass_norm()` need a device max-reduction: the reference's
*     `cuda_tool::DeviceReduce().Max` becomes T3.2's
*     `lla::device_reduce_max(frame_stream(), in, out, n)` (the T5.4a/T5.4b
*     precedent).
*  2. `eigen::atomic_add(diag_hessian(body), H12x12)` (a CUDA atomic on a
*     *struct* in global memory) has no direct LC spelling; it becomes the flat
*     scalar alias of the 12x12-per-body buffer (`native().as<double>()`) plus
*     144 scalar `fetch_add`s -- cookbook §7.20 rule 1's idiom, one size up from
*     `LVector3`.  The stride contract is asserted at the call site.
*  3. `info.gradients().buffer_view().fill(0)` (the contact-only branch) is a
*     *command stream* fill in the reference; the port records a local kernel on
*     the caller's frame stream instead (LC has no fill command, and a
*     `LBufferView::fill` would submit on the main stream and break the ordering
*     of an already-recorded frame -- cookbook §7.11 rule 7, the T5.4a finding).
*  4. `Zero()/Identity()` for the fixed-row guards are `select`s (RULE M2); the
*     diagonal-Hessian accumulation is gated by adding a *selected* block, which
*     makes the reference's `if(body_i == body_j && !has_fixed)` branchless.
*  5. The three 12x12 <-> 3x3 block unpackings go through the ported
*     `TripletMatrixAssembler`'s block form spelled explicitly (16 consecutive
*     triplets, `counter = a * 4 + b`, `H12.block<3,3>(3a, 3b)`); it is the same
*     arithmetic the reference's `TripletMatrixUnpacker::block<4,4>().write(...)`
*     performs, with the index vectors removed.
*************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
using DenseDeviceView      = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
using CDenseDeviceView     = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
using TripletDeviceView    = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using CTripletDeviceView   = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using BodyTripletDeviceView  = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 12, 12>;
using CBodyTripletDeviceView = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 12, 12>;
using BodyDoubletDeviceView  = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 12>;
using CBodyDoubletDeviceView = lla::LCDoubletVectorDeviceView<lcrt::Var<double>, 12>;
using CDoublet3DeviceView    = lla::LCDoubletVectorDeviceView<lcrt::Var<double>, 3>;

// ===========================================================================
// host-side helpers
// ===========================================================================
/// `resize`/`reserve` with the reference's growth policy: the ABD `Impl` keeps a
/// `reserve_ratio` and the container is only ever *loosely* resized (the same
/// helper shape the FEM subsystem uses, T5.4a).
void loose_resize_doublets(lla::LDoubletVector<Float, 12>& v,
                           SizeT                            size,
                           Float                            reserve_ratio)
{
    if(size > v.doublet_capacity())
    {
        v.reserve_doublets_discard(static_cast<SizeT>(static_cast<Float>(size) * reserve_ratio));
    }
    v.resize_doublets_discard(size);
}

void loose_resize_triplets(lla::LTripletMatrix<Float, 12, 12>& m,
                           SizeT                                  size,
                           Float                                  reserve_ratio)
{
    if(size > m.triplet_capacity())
    {
        m.reserve_triplets_discard(static_cast<SizeT>(static_cast<Float>(size) * reserve_ratio));
    }
    m.resize_triplets_discard(size);
}

// ===========================================================================
// `abd_linear_subsystem_assemble_kinetic_shape_k1_kernel`
// ===========================================================================
auto& kinetic_shape_gradient_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,  // is_fixed
                                                lcrt::Buffer<int32_t>,  // is_external_kinetic
                                                lcrt::Buffer<LVector12>,  // shape gradient
                                                lcrt::Buffer<LVector12>,  // kinetic gradient
                                                lcrt::Buffer<double>,     // gradients (this subsystem's window)
                                                uint32_t,                 // body count
                                                uint32_t>(                // dof count
        [](lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::BufferVar<int32_t>  is_external_kinetic,
           lcrt::BufferVar<LVector12> shape_gradient,
           lcrt::BufferVar<LVector12> kinetic_gradient,
           lcrt::BufferVar<double>   gradients,
           lcrt::Var<uint32_t>       n,
           lcrt::Var<uint32_t>       dof_count) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto g  = DenseDeviceView::from_handles(gradients, dof_count);
                auto ii = lcrt::cast<std::int32_t>(i);
                abd_linear_subsystem_detail::body_gradient_row(
                    g,
                    ii,
                    [&](auto&& j) noexcept { return load_vec12(shape_gradient->read(j)); },
                    [&](auto&& j) noexcept { return load_vec12(kinetic_gradient->read(j)); },
                    [&](auto&& j) noexcept { return is_fixed->read(j); },
                    [&](auto&& j) noexcept { return is_external_kinetic->read(j); });
            };
        });
    return k;
}

// ===========================================================================
// `abd_linear_subsystem_assemble_kinetic_shape_k2_kernel`
// ===========================================================================
/// The `dst` triplet view is the *body* section of the shared Hessian buffer, so
/// the subview's own metadata travels with it (the FEM port's shape).
auto& kinetic_shape_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,  // is_fixed
                                                lcrt::Buffer<int32_t>,  // is_external_kinetic
                                                lcrt::Buffer<LMatrix12x12>,  // shape hessian
                                                lcrt::Buffer<LMatrix12x12>,  // kinetic hessian
                                                lcrt::Buffer<int32_t>,  // dst rows
                                                lcrt::Buffer<int32_t>,  // dst cols
                                                lcrt::Buffer<double>,   // dst values
                                                int32_t,  // dst total block rows
                                                int32_t,  // dst total block cols
                                                int32_t,  // dst triplet index offset
                                                int32_t,  // dst triplet count
                                                int32_t,  // dst total triplet count
                                                int32_t,  // dst submatrix offset x
                                                int32_t,  // dst submatrix offset y
                                                int32_t,  // dst submatrix extent x
                                                int32_t,  // dst submatrix extent y
                                                lcrt::Buffer<double>,  // diag_hessian (flat scalar alias)
                                                uint32_t,              // body count
                                                uint32_t>(             // diag hessian scalar count
        [](lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::BufferVar<int32_t>  is_external_kinetic,
           lcrt::BufferVar<LMatrix12x12> shape_hessian,
           lcrt::BufferVar<LMatrix12x12> kinetic_hessian,
           lcrt::BufferVar<int32_t>  dst_rows,
           lcrt::BufferVar<int32_t>  dst_cols,
           lcrt::BufferVar<double>   dst_values,
           lcrt::Var<int32_t>        dst_total_rows,
           lcrt::Var<int32_t>        dst_total_cols,
           lcrt::Var<int32_t>        dst_triplet_offset,
           lcrt::Var<int32_t>        dst_triplet_count,
           lcrt::Var<int32_t>        dst_total_triplet_count,
           lcrt::Var<int32_t>        dst_sm_offset_x,
           lcrt::Var<int32_t>        dst_sm_offset_y,
           lcrt::Var<int32_t>        dst_sm_extent_x,
           lcrt::Var<int32_t>        dst_sm_extent_y,
           lcrt::BufferVar<double>   diag_hessian,
           lcrt::Var<uint32_t>       n,
           lcrt::Var<uint32_t>       diag_count) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto dst = TripletDeviceView::from_handles_full(dst_rows,
                                                                dst_cols,
                                                                dst_values,
                                                                dst_total_rows,
                                                                dst_total_cols,
                                                                dst_triplet_offset,
                                                                dst_triplet_count,
                                                                dst_total_triplet_count,
                                                                dst_sm_offset_x,
                                                                dst_sm_offset_y,
                                                                dst_sm_extent_x,
                                                                dst_sm_extent_y);
                auto flat = DenseDeviceView::from_handles(diag_hessian, diag_count);
                auto ii   = lcrt::cast<std::int32_t>(I);
                abd_linear_subsystem_detail::body_hessian_row(
                    dst,
                    // the reference's `MA.block<4,4>(I * 4 * 4)`: thread I owns the
                    // I-th 12x12 block (16 consecutive 3x3 triplets)
                    ii * std::int32_t{abd_linear_subsystem_detail::kTripletsPerBodyBlock},
                    flat,
                    ii,
                    [&](auto&& j) noexcept { return load_mat12x12(shape_hessian->read(j)); },
                    [&](auto&& j) noexcept { return load_mat12x12(kinetic_hessian->read(j)); },
                    [&](auto&& j) noexcept { return is_fixed->read(j); },
                    [&](auto&& j) noexcept { return is_external_kinetic->read(j); });
            };
        });
    return k;
}

// ===========================================================================
// `abd_linear_subsystem_assemble_reporters_k1_kernel`
// ===========================================================================
auto& reporter_gradient_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,   // dst gradient window
                                                lcrt::Buffer<int32_t>,  // src doublet indices
                                                lcrt::Buffer<double>,   // src doublet values (12)
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t,               // src doublet count
                                                uint32_t,               // total segments
                                                uint32_t,               // dst dof count
                                                uint32_t>(              // body count
        [](lcrt::BufferVar<double>   dst,
           lcrt::BufferVar<int32_t>  src_indices,
           lcrt::BufferVar<double>   src_values,
           lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::Var<uint32_t>       src_count,
           lcrt::Var<uint32_t>       total_segments,
           lcrt::Var<uint32_t>       dst_dof_count,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < src_count)
            {
                // The reporter's own 12-component doublet buffer: one doublet per
                // body, indexed by the *body* id.
                auto src = CBodyDoubletDeviceView::from_handles(
                    src_indices, src_values, total_segments, src_count);
                auto dst_view = DenseDeviceView::from_handles(dst, dst_dof_count);
                auto ii       = lcrt::cast<std::int32_t>(I);
                abd_linear_subsystem_detail::reporter_gradient_row(
                    dst_view, src, ii, [&](auto&& j) noexcept { return is_fixed->read(j); });
                (void)n;
            };
        });
    return k;
}

// ===========================================================================
// `abd_linear_subsystem_assemble_reporters_k2_kernel`
// ===========================================================================
auto& reporter_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,  // dst rows
                                                lcrt::Buffer<int32_t>,  // dst cols
                                                lcrt::Buffer<double>,   // dst values
                                                int32_t,  // dst total block rows
                                                int32_t,  // dst total block cols
                                                int32_t,  // dst triplet index offset
                                                int32_t,  // dst triplet count
                                                int32_t,  // dst total triplet count
                                                int32_t,  // dst sm offset x
                                                int32_t,  // dst sm offset y
                                                int32_t,  // dst sm extent x
                                                int32_t,  // dst sm extent y
                                                lcrt::Buffer<double>,  // diag_hessian (flat)
                                                lcrt::Buffer<int32_t>,  // src 12x12 rows
                                                lcrt::Buffer<int32_t>,  // src 12x12 cols
                                                lcrt::Buffer<double>,   // src 12x12 values
                                                int32_t,  // src body count (== total rows/cols/extent)
                                                int32_t,  // src triplet count
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t,               // src triplet count (dispatch bound)
                                                uint32_t>(              // diag hessian scalar count
        [](lcrt::BufferVar<int32_t> dst_rows,
           lcrt::BufferVar<int32_t> dst_cols,
           lcrt::BufferVar<double>  dst_values,
           lcrt::Var<int32_t>       dst_total_rows,
           lcrt::Var<int32_t>       dst_total_cols,
           lcrt::Var<int32_t>       dst_triplet_offset,
           lcrt::Var<int32_t>       dst_triplet_count,
           lcrt::Var<int32_t>       dst_total_triplet_count,
           lcrt::Var<int32_t>       dst_sm_offset_x,
           lcrt::Var<int32_t>       dst_sm_offset_y,
           lcrt::Var<int32_t>       dst_sm_extent_x,
           lcrt::Var<int32_t>       dst_sm_extent_y,
           lcrt::BufferVar<double>  diag_hessian,
           lcrt::BufferVar<int32_t> src_rows,
           lcrt::BufferVar<int32_t> src_cols,
           lcrt::BufferVar<double>  src_values,
           lcrt::Var<int32_t>       src_body_count,
           lcrt::Var<int32_t>       src_triplet_count,
           lcrt::BufferVar<int32_t> is_fixed,
           lcrt::Var<uint32_t>      n,
           lcrt::Var<uint32_t>      diag_count) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto dst = TripletDeviceView::from_handles_full(dst_rows,
                                                                dst_cols,
                                                                dst_values,
                                                                dst_total_rows,
                                                                dst_total_cols,
                                                                dst_triplet_offset,
                                                                dst_triplet_count,
                                                                dst_total_triplet_count,
                                                                dst_sm_offset_x,
                                                                dst_sm_offset_y,
                                                                dst_sm_extent_x,
                                                                dst_sm_extent_y);
                auto src = CBodyTripletDeviceView::from_handles_full(src_rows,
                                                                     src_cols,
                                                                     src_values,
                                                                     src_body_count,
                                                                     src_body_count,
                                                                     std::int32_t{0},
                                                                     src_triplet_count,
                                                                     src_triplet_count,
                                                                     std::int32_t{0},
                                                                     std::int32_t{0},
                                                                     src_body_count,
                                                                     src_body_count);
                auto flat = DenseDeviceView::from_handles(diag_hessian, diag_count);
                auto ii   = lcrt::cast<std::int32_t>(I);
                abd_linear_subsystem_detail::reporter_hessian_row(
                    dst,
                    ii * std::int32_t{abd_linear_subsystem_detail::kTripletsPerBodyBlock},
                    flat,
                    src,
                    ii,
                    [&](auto&& j) noexcept { return is_fixed->read(j); });
            };
        });
    return k;
}

// ===========================================================================
// `abd_linear_subsystem_assemble_dytopo_effect_k1_kernel`
// ===========================================================================
auto& dytopo_gradient_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,  // dytopo doublet indices (vertex ids)
                                                lcrt::Buffer<double>,   // dytopo doublet values (3)
                                                lcrt::Buffer<double>,   // dst gradient window
                                                lcrt::Buffer<int32_t>,  // v2b
                                                lcrt::Buffer<ABDJacobi>,  // per-vertex jacobians
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                int32_t,                // vertex offset
                                                uint32_t,               // dytopo doublet count
                                                uint32_t,               // total segments
                                                uint32_t,               // dst dof count
                                                uint32_t>(              // body count
        [](lcrt::BufferVar<int32_t>  indices,
           lcrt::BufferVar<double>   values,
           lcrt::BufferVar<double>   dst,
           lcrt::BufferVar<int32_t>  v2b,
           lcrt::BufferVar<ABDJacobi> Js,
           lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::Var<int32_t>        vertex_offset,
           lcrt::Var<uint32_t>       src_count,
           lcrt::Var<uint32_t>       total_segments,
           lcrt::Var<uint32_t>       dst_dof_count,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < src_count)
            {
                auto src = CDoublet3DeviceView::from_handles(
                    indices, values, total_segments, src_count);
                auto dst_view = DenseDeviceView::from_handles(dst, dst_dof_count);
                auto ii       = lcrt::cast<std::int32_t>(I);
                abd_linear_subsystem_detail::dytopo_gradient_row(
                    dst_view,
                    src,
                    ii,
                    vertex_offset,
                    [&](auto&& j) noexcept { return v2b->read(j); },
                    [&](auto&& j) noexcept { return load_jacobi(Js->read(j)); },
                    [&](auto&& j) noexcept { return is_fixed->read(j); });
                (void)n;
            };
        });
    return k;
}

// ===========================================================================
// `abd_linear_subsystem_assemble_dytopo_effect_k2_kernel`
// ===========================================================================
auto& dytopo_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,  // dst rows
                                                lcrt::Buffer<int32_t>,  // dst cols
                                                lcrt::Buffer<double>,   // dst values
                                                int32_t,  // dst total block rows
                                                int32_t,  // dst total block cols
                                                int32_t,  // dst triplet index offset
                                                int32_t,  // dst triplet count
                                                int32_t,  // dst total triplet count
                                                int32_t,  // dst sm offset x
                                                int32_t,  // dst sm offset y
                                                int32_t,  // dst sm extent x
                                                int32_t,  // dst sm extent y
                                                lcrt::Buffer<double>,  // diag_hessian (flat)
                                                lcrt::Buffer<int32_t>,  // src 3x3 rows
                                                lcrt::Buffer<int32_t>,  // src 3x3 cols
                                                lcrt::Buffer<double>,   // src 3x3 values
                                                int32_t,  // src total block rows/cols
                                                int32_t,  // src triplet count
                                                lcrt::Buffer<int32_t>,  // v2b
                                                lcrt::Buffer<ABDJacobi>,  // per-vertex jacobians
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                int32_t,                // vertex offset
                                                uint32_t,               // src triplet count (dispatch bound)
                                                uint32_t>(              // diag hessian scalar count
        [](lcrt::BufferVar<int32_t> dst_rows,
           lcrt::BufferVar<int32_t> dst_cols,
           lcrt::BufferVar<double>  dst_values,
           lcrt::Var<int32_t>       dst_total_rows,
           lcrt::Var<int32_t>       dst_total_cols,
           lcrt::Var<int32_t>       dst_triplet_offset,
           lcrt::Var<int32_t>       dst_triplet_count,
           lcrt::Var<int32_t>       dst_total_triplet_count,
           lcrt::Var<int32_t>       dst_sm_offset_x,
           lcrt::Var<int32_t>       dst_sm_offset_y,
           lcrt::Var<int32_t>       dst_sm_extent_x,
           lcrt::Var<int32_t>       dst_sm_extent_y,
           lcrt::BufferVar<double>  diag_hessian,
           lcrt::BufferVar<int32_t> src_rows,
           lcrt::BufferVar<int32_t> src_cols,
           lcrt::BufferVar<double>  src_values,
           lcrt::Var<int32_t>       src_total,
           lcrt::Var<int32_t>       src_triplet_count,
           lcrt::BufferVar<int32_t> v2b,
           lcrt::BufferVar<ABDJacobi> Js,
           lcrt::BufferVar<int32_t> is_fixed,
           lcrt::Var<int32_t>       vertex_offset,
           lcrt::Var<uint32_t>      n,
           lcrt::Var<uint32_t>      diag_count) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto dst = TripletDeviceView::from_handles_full(dst_rows,
                                                                dst_cols,
                                                                dst_values,
                                                                dst_total_rows,
                                                                dst_total_cols,
                                                                dst_triplet_offset,
                                                                dst_triplet_count,
                                                                dst_total_triplet_count,
                                                                dst_sm_offset_x,
                                                                dst_sm_offset_y,
                                                                dst_sm_extent_x,
                                                                dst_sm_extent_y);
                auto src = CTripletDeviceView::from_handles_full(src_rows,
                                                                 src_cols,
                                                                 src_values,
                                                                 src_total,
                                                                 src_total,
                                                                 std::int32_t{0},
                                                                 src_triplet_count,
                                                                 src_triplet_count,
                                                                 std::int32_t{0},
                                                                 std::int32_t{0},
                                                                 src_total,
                                                                 src_total);
                auto flat = DenseDeviceView::from_handles(diag_hessian, diag_count);
                auto ii   = lcrt::cast<std::int32_t>(I);
                abd_linear_subsystem_detail::dytopo_hessian_row(
                    dst,
                    ii * std::int32_t{abd_linear_subsystem_detail::kTripletsPerBodyBlock},
                    flat,
                    src,
                    ii,
                    vertex_offset,
                    [&](auto&& j) noexcept { return v2b->read(j); },
                    [&](auto&& j) noexcept { return load_jacobi(Js->read(j)); },
                    [&](auto&& j) noexcept { return is_fixed->read(j); });
            };
        });
    return k;
}

// ===========================================================================
// `abd_linear_subsystem_retrieve_solution_kernel`
// ===========================================================================
auto& retrieve_solution_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,  // solution
                                                lcrt::Buffer<LVector12>,  // dq
                                                uint32_t,  // body count
                                                uint32_t>(  // dof count
        [](lcrt::BufferVar<double>   solution,
           lcrt::BufferVar<LVector12> dq,
           lcrt::Var<uint32_t>        n,
           lcrt::Var<uint32_t>        dof_count) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto result = CDenseDeviceView::from_handles(solution, dof_count);
                auto ii     = lcrt::cast<std::int32_t>(i);
                abd_linear_subsystem_detail::retrieve_solution_row(
                    result,
                    ii,
                    [&](const lps::LVec<lcrt::Var<double>, 12>& v) noexcept
                    {
                        lcrt::Var<LVector12> out;
                        store_vec12(out, v);
                        dq->write(ii, out);
                    });
            };
        });
    return k;
}

// ===========================================================================
// `abd_linear_subsystem_diag_norm_kernel`
// ===========================================================================
auto& diag_norm_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,   // diag_hessian (flat scalar alias)
                                                lcrt::Buffer<double>,   // diag_blocks_norm
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t,               // body count
                                                uint32_t>(              // diag hessian scalar count
        [](lcrt::BufferVar<double>  diag_hessian,
           lcrt::BufferVar<double>  diag_blocks_norm,
           lcrt::BufferVar<int32_t> is_fixed,
           lcrt::Var<uint32_t>      n,
           lcrt::Var<uint32_t>      diag_count) noexcept
        {
            auto idx = lcrt::dispatch_x();
            $if(idx < n)
            {
                auto flat = CDenseDeviceView::from_handles(diag_hessian, diag_count);
                auto norm = DenseDeviceView::from_handles(diag_blocks_norm, n * 12u);
                auto ii   = lcrt::cast<std::int32_t>(idx);
                abd_linear_subsystem_detail::diag_norm_row(
                    flat, norm, ii, [&](auto&& j) noexcept { return is_fixed->read(j); });
            };
        });
    return k;
}

// ===========================================================================
// `abd_linear_subsystem_mass_norm_kernel`
// ===========================================================================
auto& mass_norm_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<ABDJacobiDyadicMass>,  // masses
                                                lcrt::Buffer<double>,  // block_norm
                                                lcrt::Buffer<int32_t>,  // is_fixed
                                                uint32_t>(              // body count
        [](lcrt::BufferVar<ABDJacobiDyadicMass> mass,
           lcrt::BufferVar<double>              block_norm,
           lcrt::BufferVar<int32_t>             is_fixed,
           lcrt::Var<uint32_t>                  n) noexcept
        {
            auto idx = lcrt::dispatch_x();
            $if(idx < n)
            {
                auto norm = DenseDeviceView::from_handles(block_norm, n);
                auto ii   = lcrt::cast<std::int32_t>(idx);
                abd_linear_subsystem_detail::mass_norm_row(
                    [&](auto&& j) noexcept { return mass->read(j); },
                    norm,
                    ii,
                    [&](auto&& j) noexcept { return is_fixed->read(j); });
            };
        });
    return k;
}

/// The contact-only branch's `gradients.buffer_view().fill(0)`.  LC has no fill
/// *command*, and `LBufferView::fill` submits on the main stream (which would
/// leave an already-recorded frame un-ordered -- cookbook §7.11 rule 7), so the
/// fill is a local kernel recorded on the caller's frame stream.
auto& zero_gradient_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>, uint32_t>(
        [](lcrt::BufferVar<double> data, lcrt::Var<uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                data->write(i, lcrt::Var<double>{0.0});
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(ABDLinearSubsystem);

// ref: https://github.com/spiriMirror/libuipc/issues/271
constexpr U64 ABDLinearSubsystemUID = 0ull;

U64 ABDLinearSubsystem::get_uid() const noexcept
{
    return ABDLinearSubsystemUID;
}

AffineBodyDynamics::Impl& ABDLinearSubsystem::Impl::abd() const noexcept
{
    return affine_body_dynamics->m_impl;
}

void ABDLinearSubsystem::do_build(DiagLinearSubsystem::BuildInfo& info)
{
    (void)info;
    m_impl.affine_body_dynamics        = require<AffineBodyDynamics>();
    m_impl.affine_body_vertex_reporter = require<AffineBodyVertexReporter>();
    m_impl.dt_attr = world().scene().config().find<Float>("dt");
    UIPC_ASSERT(m_impl.dt_attr, "Scene config must have a 'dt' attribute.");

    m_impl.dytopo_effect_receiver = find<ABDDyTopoEffectReceiver>();
}

void ABDLinearSubsystem::do_init(InitInfo& info)
{
    (void)info;
    m_impl.init();
}

void ABDLinearSubsystem::Impl::init()
{
    auto reporter_view = reporters.view();
    for(auto&& [i, r] : enumerate(reporter_view))
        r->m_index = i;
    for(auto& r : reporter_view)
        r->init();

    reporter_gradient_offsets_counts.resize(reporter_view.size());
    reporter_hessian_offsets_counts.resize(reporter_view.size());

    SizeT body_count = abd().body_count();
    body_id_to_shape_hessian.resize(body_count);
    body_id_to_shape_gradient.resize(body_count);
    body_id_to_kinetic_hessian.resize(body_count);
    body_id_to_kinetic_gradient.resize(body_count);
    diag_hessian.resize(body_count);
}

void ABDLinearSubsystem::Impl::report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    info.extent(abd().body_count() * static_cast<SizeT>(abd_linear_subsystem_detail::kDofPerBody));
}

void ABDLinearSubsystem::Impl::receive_init_dof_info(WorldVisitor& w,
                                                     GlobalLinearSystem::InitDofInfo& info)
{
    auto  geo_slots = w.scene().geometries();

    IndexT offset = info.dof_offset();

    // fill the dof_offset and dof_count for each geometry
    affine_body_dynamics->for_each(
        geo_slots,
        [&](const AffineBodyDynamics::ForEachInfo& foreach_info, geometry::SimplicialComplex& sc)
        {
            auto I          = foreach_info.global_index();
            auto dof_offset = sc.meta().find<IndexT>(builtin::dof_offset);
            UIPC_ASSERT(dof_offset, "dof_offset not found on ABD mesh why can it happen?");
            auto dof_count = sc.meta().find<IndexT>(builtin::dof_count);
            UIPC_ASSERT(dof_count, "dof_count not found on ABD mesh why can it happen?");

            IndexT this_dof_count =
                static_cast<IndexT>(sc.instances().size())
                * static_cast<IndexT>(abd_linear_subsystem_detail::kDofPerBody);
            view(*dof_offset)[0] = offset;
            view(*dof_count)[0]  = this_dof_count;

            offset += this_dof_count;
            (void)I;
        });

    UIPC_ASSERT(offset == info.dof_offset() + info.dof_count(), "dof size mismatch");
}

void ABDLinearSubsystem::Impl::report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    // 1. Gradient Count
    constexpr SizeT G12_to_dof = abd_linear_subsystem_detail::kDofPerBody;
    SizeT           body_count = abd().body_count();
    auto            dof_count  = body_count * G12_to_dof;

    auto has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    // 2) the reporters report their own extents (and must not report Hessian
    //    blocks in gradient-only mode: their `check()` enforces it)
    auto reporter_view = reporters.view();
    auto grad_counts   = reporter_gradient_offsets_counts.counts();
    auto hess_counts   = reporter_hessian_offsets_counts.counts();

    if(has_complement)
    {
        for(auto&& R : reporter_view)
        {
            ReportExtentInfo extent_info;
            extent_info.m_gradient_only = info.gradient_only();
            R->report_extent(extent_info);

            grad_counts[R->m_index] = extent_info.m_gradient_count;
            hess_counts[R->m_index] = extent_info.m_hessian_count;
        }

        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();
    }

    SizeT dytopo_hessian_count = 0;
    if(dytopo_effect_receiver && !info.gradient_only())
    {
        dytopo_hessian_count = dytopo_effect_receiver->hessians().triplet_count();
    }

    // The arithmetic itself is the *factored* routine (the brief's §2 testability
    // requirement): same counts, same order, drivable by a unit test with
    // hand-derived values.
    auto budget = abd_linear_subsystem_detail::plan_extent(
        body_count,
        info.gradient_only(),
        has_complement,
        reporter_gradient_offsets_counts.counts(),
        reporter_hessian_offsets_counts.counts(),
        0,
        dytopo_hessian_count);

    UIPC_ASSERT(budget.check_gradient_only(info.gradient_only()),
                "When gradient_only is true, hessian_block_count must be 0, yours {}.\n"
                "Ref: https://github.com/spiriMirror/libuipc/issues/295",
                budget.hessian_block_count);
    UIPC_ASSERT(budget.dof_count == dof_count,
                "ABD dof count mismatch: factored {} vs reference {}",
                budget.dof_count,
                dof_count);

    info.extent(budget.hessian_block_count, dof_count);
}

void ABDLinearSubsystem::Impl::assemble(GlobalLinearSystem::DiagInfo& info)
{
    // 0) Prepare the reporters' buffers
    {
        auto N = abd().body_count();

        reporter_gradients.reshape(static_cast<int>(N));
        loose_resize_doublets(reporter_gradients,
                              static_cast<SizeT>(reporter_gradient_offsets_counts.total_count()),
                              reserve_ratio);

        reporter_hessians.reshape(static_cast<int>(N), static_cast<int>(N));
        loose_resize_triplets(reporter_hessians,
                              static_cast<SizeT>(reporter_hessian_offsets_counts.total_count()),
                              reserve_ratio);
    }

    bool has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    IndexT hess_offset = 0;

    // 1) Static Topology Effect: Kinetic + Shape + Other Reporters
    if(has_complement)
    {
        _assemble_kinetic_shape(hess_offset, info);
        _assemble_reporters(hess_offset, info);
    }
    else  // contact only
    {
        auto gradient_dof_count = static_cast<uint32_t>(info.gradients().size());
        if(gradient_dof_count > 0)
        {
            auto& k = zero_gradient_kernel();
            frame_stream().record(
                k(info.gradients().buffer_view(), gradient_dof_count).dispatch(gradient_dof_count));
        }
    }

    // 2) Dynamic Topology Effect
    _assemble_dytopo_effect(hess_offset, info);

    UIPC_ASSERT(hess_offset == static_cast<IndexT>(info.hessians().triplet_count()),
                "Hessian size mismatch: expected {}, got {}",
                info.hessians().triplet_count(),
                hess_offset);
}

void ABDLinearSubsystem::Impl::_assemble_kinetic_shape(IndexT& hess_offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    Float dt = dt_attr->view()[0];

    // Collect Kinetic
    ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
        info.gradient_only(), body_id_to_kinetic_gradient.view(), body_id_to_kinetic_hessian.view(), dt};
    abd().kinetic->compute_gradient_hessian(this_info);

    // Collect Shape
    for(auto&& [i, cst] : enumerate(abd().constitutions.view()))
    {
        ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
            info.gradient_only(),
            abd().subview(body_id_to_shape_gradient, cst->m_index),
            abd().subview(body_id_to_shape_hessian, cst->m_index),
            dt};

        cst->compute_gradient_hessian(this_info);
    }

    auto body_count = abd().body_count();
    {
        auto n = static_cast<uint32_t>(body_count);
        if(n > 0)
        {
            auto& k = kinetic_shape_gradient_kernel();
            frame_stream().record(
                k(abd().body_id_to_is_fixed.cview(),
                  abd().body_id_to_external_kinetic.cview(),
                  body_id_to_shape_gradient.cview(),
                  body_id_to_kinetic_gradient.cview(),
                  info.gradients().buffer_view(),
                  n,
                  static_cast<uint32_t>(info.gradients().size()))
                    .dispatch(n));
        }
    }

    if(info.gradient_only())
        return;

    auto H3x3_count = body_count * static_cast<SizeT>(abd_linear_subsystem_detail::kTripletsPerBodyBlock);
    if(H3x3_count == 0)
    {
        return;
    }

    auto diag_count = static_cast<uint32_t>(body_count
                                            * static_cast<SizeT>(abd_linear_subsystem_detail::kDofPerBody
                                                                 * abd_linear_subsystem_detail::kDofPerBody));
    // The scalar alias of the 12x12-per-body diag Hessian (cookbook §7.20 rule 1):
    // it is exact iff LC lays the registered struct out as sizeof(T)-strided.
    auto diag = diag_hessian.view().native();
    UIPC_ASSERT(diag.stride() == sizeof(LMatrix12x12),
                "ABDLinearSubsystem: the scalar alias of diag_hessian needs element_stride == "
                "sizeof(LMatrix12x12) == {} B, measured {} B",
                sizeof(LMatrix12x12),
                diag.stride());

    auto body_H3x3 = info.hessians().subview(hess_offset, H3x3_count);
    {
        auto n  = static_cast<uint32_t>(body_count);
        auto& k = kinetic_shape_hessian_kernel();
        frame_stream().record(
            k(abd().body_id_to_is_fixed.cview(),
              abd().body_id_to_external_kinetic.cview(),
              body_id_to_shape_hessian.cview(),
              body_id_to_kinetic_hessian.cview(),
              body_H3x3.row_indices(),
              body_H3x3.col_indices(),
              body_H3x3.values(),
              static_cast<std::int32_t>(body_H3x3.total_rows()),
              static_cast<std::int32_t>(body_H3x3.total_cols()),
              static_cast<std::int32_t>(body_H3x3.triplet_index_offset()),
              static_cast<std::int32_t>(body_H3x3.triplet_count()),
              static_cast<std::int32_t>(body_H3x3.total_triplet_count()),
              static_cast<std::int32_t>(body_H3x3.submatrix_offset().x),
              static_cast<std::int32_t>(body_H3x3.submatrix_offset().y),
              static_cast<std::int32_t>(body_H3x3.extent().x),
              static_cast<std::int32_t>(body_H3x3.extent().y),
              diag.as<double>(),
              n,
              diag_count)
                .dispatch(n));
    }

    hess_offset += static_cast<IndexT>(H3x3_count);
}

void ABDLinearSubsystem::Impl::_assemble_reporters(IndexT& hess_offset,
                                                   GlobalLinearSystem::DiagInfo& info)
{
    // Fill the TripletMatrix and the DoubletVector: each reporter writes its own
    // slice through the `AssembleInfo` (the reference's fan-out order).
    for(auto& R : reporters.view())
    {
        AssembleInfo assemble_info{this, R->m_index, info.gradient_only()};
        R->assemble(assemble_info);
    }

    auto reporter_gradient_count = static_cast<uint32_t>(reporter_gradients.doublet_count());
    if(reporter_gradient_count > 0)
    {
        auto& k = reporter_gradient_kernel();
        frame_stream().record(
            k(info.gradients().buffer_view(),
              reporter_gradients.indices(),
              reporter_gradients.values(),
              abd().body_id_to_is_fixed.cview(),
              reporter_gradient_count,
              static_cast<uint32_t>(reporter_gradients.count()),
              static_cast<uint32_t>(info.gradients().size()),
              static_cast<uint32_t>(abd().body_count()))
                .dispatch(reporter_gradient_count));
    }

    auto reporter_hessian_count = static_cast<uint32_t>(reporter_hessians.triplet_count());
    if(!info.gradient_only() && reporter_hessian_count > 0)
    {
        auto body_count = abd().body_count();
        auto diag_count = static_cast<uint32_t>(body_count
                                                * static_cast<SizeT>(abd_linear_subsystem_detail::kDofPerBody
                                                                     * abd_linear_subsystem_detail::kDofPerBody));
        auto diag = diag_hessian.view().native();
        UIPC_ASSERT(diag.stride() == sizeof(LMatrix12x12),
                    "ABDLinearSubsystem: the scalar alias of diag_hessian needs element_stride == "
                    "sizeof(LMatrix12x12) == {} B, measured {} B",
                    sizeof(LMatrix12x12),
                    diag.stride());

        // the reporters' 12x12 buffer reshaped to (N, N): its triplet i is the
        // (body_i, body_j) block of the reporter
        auto src = reporter_hessians.view();
        // the global buffer's slice the reporters' blocks go into
        auto H3x3s = info.hessians().subview(hess_offset);

        auto& k = reporter_hessian_kernel();
        frame_stream().record(k(H3x3s.row_indices(),
                                H3x3s.col_indices(),
                                H3x3s.values(),
                                static_cast<std::int32_t>(H3x3s.total_rows()),
                                static_cast<std::int32_t>(H3x3s.total_cols()),
                                static_cast<std::int32_t>(H3x3s.triplet_index_offset()),
                                static_cast<std::int32_t>(H3x3s.triplet_count()),
                                static_cast<std::int32_t>(H3x3s.total_triplet_count()),
                                static_cast<std::int32_t>(H3x3s.submatrix_offset().x),
                                static_cast<std::int32_t>(H3x3s.submatrix_offset().y),
                                static_cast<std::int32_t>(H3x3s.extent().x),
                                static_cast<std::int32_t>(H3x3s.extent().y),
                                diag.as<double>(),
                                src.row_indices(),
                                src.col_indices(),
                                src.values(),
                                static_cast<std::int32_t>(body_count),
                                static_cast<std::int32_t>(src.triplet_count()),
                                abd().body_id_to_is_fixed.cview(),
                                reporter_hessian_count,
                                diag_count)
                                  .dispatch(reporter_hessian_count));

        hess_offset +=
            static_cast<IndexT>(reporter_hessian_count)
            * static_cast<IndexT>(abd_linear_subsystem_detail::kTripletsPerBodyBlock);
    }
}

void ABDLinearSubsystem::Impl::_assemble_dytopo_effect(IndexT& hess_offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    auto  vertex_offset = affine_body_vertex_reporter->vertex_offset();
    SizeT dytopo_effect_gradient_count = 0;
    if(dytopo_effect_receiver)
    {
        dytopo_effect_gradient_count = dytopo_effect_receiver->gradients().doublet_count();
    }

    if(dytopo_effect_gradient_count)
    {
        auto n  = static_cast<uint32_t>(dytopo_effect_gradient_count);
        auto& k = dytopo_gradient_kernel();
        auto  src = dytopo_effect_receiver->gradients();
        frame_stream().record(
            k(src.indices(),
              src.values(),
              info.gradients().buffer_view(),
              abd().vertex_id_to_body_id.cview(),
              abd().vertex_id_to_J.cview(),
              abd().body_id_to_is_fixed.cview(),
              static_cast<std::int32_t>(vertex_offset),
              n,
              static_cast<uint32_t>(src.total_extent()),
              static_cast<uint32_t>(info.gradients().size()),
              static_cast<uint32_t>(abd().body_count()))
                .dispatch(n));
    }

    if(info.gradient_only())
        return;

    SizeT dytopo_effect_hessian_count = 0;
    if(dytopo_effect_receiver)
        dytopo_effect_hessian_count = dytopo_effect_receiver->hessians().triplet_count();

    auto H3x3_count = dytopo_effect_hessian_count
                      * static_cast<SizeT>(abd_linear_subsystem_detail::kTripletsPerBodyBlock);
    if(H3x3_count == 0)
    {
        return;
    }

    auto dytopo_effect_H3x3 = info.hessians().subview(hess_offset, H3x3_count);

    {
        // Half Contact Hessian
        // ref: https://github.com/spiriMirror/libuipc/issues/272
        auto body_count = abd().body_count();
        auto diag_count = static_cast<uint32_t>(body_count
                                                * static_cast<SizeT>(abd_linear_subsystem_detail::kDofPerBody
                                                                     * abd_linear_subsystem_detail::kDofPerBody));
        auto diag = diag_hessian.view().native();
        UIPC_ASSERT(diag.stride() == sizeof(LMatrix12x12),
                    "ABDLinearSubsystem: the scalar alias of diag_hessian needs element_stride == "
                    "sizeof(LMatrix12x12) == {} B, measured {} B",
                    sizeof(LMatrix12x12),
                    diag.stride());

        auto  n   = static_cast<uint32_t>(dytopo_effect_hessian_count);
        auto  src = dytopo_effect_receiver->hessians();
        auto& k   = dytopo_hessian_kernel();
        frame_stream().record(
            k(dytopo_effect_H3x3.row_indices(),
              dytopo_effect_H3x3.col_indices(),
              dytopo_effect_H3x3.values(),
              static_cast<std::int32_t>(dytopo_effect_H3x3.total_rows()),
              static_cast<std::int32_t>(dytopo_effect_H3x3.total_cols()),
              static_cast<std::int32_t>(dytopo_effect_H3x3.triplet_index_offset()),
              static_cast<std::int32_t>(dytopo_effect_H3x3.triplet_count()),
              static_cast<std::int32_t>(dytopo_effect_H3x3.total_triplet_count()),
              static_cast<std::int32_t>(dytopo_effect_H3x3.submatrix_offset().x),
              static_cast<std::int32_t>(dytopo_effect_H3x3.submatrix_offset().y),
              static_cast<std::int32_t>(dytopo_effect_H3x3.extent().x),
              static_cast<std::int32_t>(dytopo_effect_H3x3.extent().y),
              diag.as<double>(),
              src.row_indices(),
              src.col_indices(),
              src.values(),
              static_cast<std::int32_t>(src.total_rows()),
              static_cast<std::int32_t>(src.triplet_count()),
              abd().vertex_id_to_body_id.cview(),
              abd().vertex_id_to_J.cview(),
              abd().body_id_to_is_fixed.cview(),
              static_cast<std::int32_t>(vertex_offset),
              n,
              diag_count)
                .dispatch(n));
    }

    hess_offset += static_cast<IndexT>(H3x3_count);
}

void ABDLinearSubsystem::Impl::accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    info.satisfied(true);
}

void ABDLinearSubsystem::Impl::retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    auto dq = abd().body_id_to_dq.view();

    auto n = static_cast<uint32_t>(abd().body_count());
    if(n > 0)
    {
        auto& k = retrieve_solution_kernel();
        frame_stream().record(k(info.solution().buffer_view(),
                                dq,
                                n,
                                static_cast<uint32_t>(info.solution().size()))
                                  .dispatch(n));
    }
}

Float ABDLinearSubsystem::Impl::diag_norm()
{
    auto body_count = abd().body_count();
    block_norm.resize(body_count
                      * static_cast<SizeT>(abd_linear_subsystem_detail::kDofPerBody));

    auto n = static_cast<uint32_t>(body_count);
    if(n > 0)
    {
        auto diag = diag_hessian.view().native();
        UIPC_ASSERT(diag.stride() == sizeof(LMatrix12x12),
                    "ABDLinearSubsystem: the scalar alias of diag_hessian needs element_stride == "
                    "sizeof(LMatrix12x12) == {} B, measured {} B",
                    sizeof(LMatrix12x12),
                    diag.stride());
        auto diag_count = static_cast<uint32_t>(block_norm.size());
        auto& k         = diag_norm_kernel();
        frame_stream().record(k(diag.as<double>(),
                                block_norm.view(),
                                abd().body_id_to_is_fixed.cview(),
                                n,
                                diag_count)
                                  .dispatch(n));
    }

    lla::device_reduce_max(frame_stream(),
                           block_norm.cview(),
                           reduced_norm.view(),
                           block_norm.size());

    return reduced_norm;
}

Float ABDLinearSubsystem::Impl::mass_norm()
{
    auto body_count = abd().body_count();
    block_norm.resize(body_count);

    auto n = static_cast<uint32_t>(body_count);
    if(n > 0)
    {
        auto& k = mass_norm_kernel();
        frame_stream().record(k(abd().body_id_to_abd_mass.cview(),
                                block_norm.view(),
                                abd().body_id_to_is_fixed.cview(),
                                n)
                                  .dispatch(n));
    }

    lla::device_reduce_max(frame_stream(),
                           block_norm.cview(),
                           reduced_norm.view(),
                           block_norm.size());

    return reduced_norm;
}

void ABDLinearSubsystem::do_report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    m_impl.report_extent(info);
}

void ABDLinearSubsystem::do_assemble(GlobalLinearSystem::DiagInfo& info)
{
    m_impl.assemble(info);
}

void ABDLinearSubsystem::do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    m_impl.accuracy_check(info);
}

void ABDLinearSubsystem::do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    m_impl.retrieve_solution(info);
}

void ABDLinearSubsystem::do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    m_impl.report_init_extent(info);
}

void ABDLinearSubsystem::do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info)
{
    m_impl.receive_init_dof_info(world(), info);
}

Float ABDLinearSubsystem::do_diag_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    (void)info;
    return m_impl.diag_norm();
}

Float ABDLinearSubsystem::do_mass_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    (void)info;
    return m_impl.mass_norm();
}

void ABDLinearSubsystem::add_reporter(ABDLinearSubsystemReporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter cannot be null");
    check_state(SimEngineState::BuildSystems, "add_reporter");
    m_impl.reporters.register_sim_system(*reporter);
}

// ---------------------------------------------------------------------------
// the nested info classes' bodies (the reference defines them at the end of its
// .cu; T6.1a had to leave them to T6.1b because a stub would turn a
// missing-feature build error into a runtime no-op -- cookbook §7.16 rule 13)
// ---------------------------------------------------------------------------
ABDLinearSubsystem::AssembleInfo::AssembleInfo(Impl*  impl,
                                               IndexT index,
                                               bool   gradient_only) noexcept
    : m_impl(impl)
    , m_index(index)
    , m_gradient_only(gradient_only)
{
}

lla::LDoubletVectorView<Float, 12> ABDLinearSubsystem::AssembleInfo::gradients() const
{
    auto [offset, count] = abd_linear_subsystem_detail::slice_of(
        m_impl->reporter_gradient_offsets_counts, m_index);
    return m_impl->reporter_gradients.view().subview(offset, count);
}

lla::LTripletMatrixView<Float, 12, 12> ABDLinearSubsystem::AssembleInfo::hessians() const
{
    auto [offset, count] = abd_linear_subsystem_detail::slice_of(
        m_impl->reporter_hessian_offsets_counts, m_index);
    return m_impl->reporter_hessians.view().subview(offset, count);
}

bool ABDLinearSubsystem::AssembleInfo::gradient_only() const noexcept
{
    return m_gradient_only;
}

void ABDLinearSubsystem::ReportExtentInfo::gradient_count(SizeT size)
{
    m_gradient_count = size;
}

void ABDLinearSubsystem::ReportExtentInfo::hessian_count(SizeT size)
{
    m_hessian_count = size;
}

void ABDLinearSubsystem::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_count, name);
}
}  // namespace uipc::backend::luisa

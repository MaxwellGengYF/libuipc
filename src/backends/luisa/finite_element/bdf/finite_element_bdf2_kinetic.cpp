#include <finite_element/bdf/finite_element_bdf2_kinetic.h>
#include <frame_stream.h>

/***************************************************************
   * T5.3 — FiniteElementBDF2Kinetic: the two kernels, their cached shaders and the
   * system registration (reference:
   * src/backends/cuda/finite_element/bdf/finite_element_bdf2_kinetic.cu).
   *
   * `FiniteElementBDF2Kinetic::inv_beta` is passed as a *kernel argument*, exactly
   * as the reference's kernels take `Float inv_beta` (the reference's constant
   * `beta = 4.0/9.0`, `inv_beta = Float{1} / beta = 9.0/4`).
   *
   * The gradient writes a *doublet* block and the Hessian a *triplet* block of the
 * linear subsystem's shared assembly space.  Like the T5.2 constitutions, the
 * device doublet/triplet *views* cannot be kernel arguments (a captured view
 * becomes a compile-time-frozen shader argument, cookbook §7.9 rule 2), so their
 * flat buffers are passed pre-offset to this system's slice (the view's
 * `doublet_index_offset()` / `triplet_index_offset()`) and the device view is
 * rebuilt inside the kernel with `from_handles_full`; the view's own index offset
 * inside the handles is 0, and the `subvector_offset` / `submatrix_offset` (the
 * one piece of index arithmetic that cannot travel inside a buffer handle) is
 * passed as a scalar (cookbook §7.17 rule 2).  The device views' `write` takes the
 * numerics layer's `LVec`/`LMat` by value, so no `store_vec3`/`store_mat3x3`
 * conversion is needed -- and `G3s(i).write(i, G)` / `H3x3s(i).write(i, i, H)` are
 * literally the reference's statements.
 *
 * The reference's `if(gradient_only) return;` -- which skips the Hessian stage --
 * becomes `$if(gradient_only == 0u)` (RULE M2), and the *empty* Hessian view a
 * gradient-only pass hands over (null handles) has to be guarded on the host side:
 * `LBufferView::subview()` on a null handle is a hard crash in LC (cookbook §7.17
 * rule 6).  The port substitutes the (valid) doublet buffers as placeholders; the
 * kernel's Hessian stage is predicated and never touches them.
 ***************************************************************/
namespace uipc::backend::luisa
{
namespace
{
using DoubletDeviceView = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using TripletDeviceView = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;

/// `FiniteElementBDF2Kinetic_do_compute_energy_kernel`
auto& bdf2_kinetic_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,   // is_fixed
                                                lcrt::Buffer<LVector3>,  // xs
                                                lcrt::Buffer<LVector3>,  // x_tildes
                                                lcrt::Buffer<double>,    // masses
                                                lcrt::Buffer<double>,    // Ks
                                                double,                  // inv_beta
                                                uint32_t>(               // n
        [](lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::BufferVar<LVector3> xs,
           lcrt::BufferVar<LVector3> x_tildes,
           lcrt::BufferVar<double>   masses,
           lcrt::BufferVar<double>   Ks,
           lcrt::Var<double>         inv_beta,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto K = bdf2_kinetic_detail::energy(load_vec3(xs->read(i)),
                                                     load_vec3(x_tildes->read(i)),
                                                     masses->read(i),
                                                     inv_beta,
                                                     is_fixed->read(i));
                Ks->write(i, K);
            };
        });
    return k;
}

/// `FiniteElementBDF2Kinetic_do_compute_gradient_hessian_kernel`
auto& bdf2_kinetic_gradient_hessian_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<int32_t>,   // is_fixed
                                                lcrt::Buffer<LVector3>,  // xs
                                                lcrt::Buffer<LVector3>,  // x_tildes
                                                lcrt::Buffer<double>,    // masses
                                                // gradients (this system's doublet slice)
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<double>,
                                                int32_t,  // subvector_offset
                                                // hessians (this system's triplet slice)
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<int32_t>,
                                                lcrt::Buffer<double>,
                                                int32_t,  // submatrix_offset.x
                                                int32_t,  // submatrix_offset.y
                                                int32_t,  // submatrix_extent.x
                                                int32_t,  // submatrix_extent.y
                                                int32_t,  // total_block_rows
                                                int32_t,  // total_block_cols
                                                uint32_t,  // gradient_only
                                                double,    // inv_beta
                                                uint32_t>(  // n
        [](lcrt::BufferVar<int32_t>  is_fixed,
           lcrt::BufferVar<LVector3> xs,
           lcrt::BufferVar<LVector3> x_tildes,
           lcrt::BufferVar<double>   masses,
           lcrt::BufferVar<int32_t>  g_indices,
           lcrt::BufferVar<double>   g_values,
           lcrt::Var<int32_t>        g_subvector_offset,
           lcrt::BufferVar<int32_t>  h_row_indices,
           lcrt::BufferVar<int32_t>  h_col_indices,
           lcrt::BufferVar<double>   h_values,
           lcrt::Var<int32_t>        h_submatrix_offset_x,
           lcrt::Var<int32_t>        h_submatrix_offset_y,
           lcrt::Var<int32_t>        h_submatrix_extent_x,
           lcrt::Var<int32_t>        h_submatrix_extent_y,
           lcrt::Var<int32_t>        h_total_block_rows,
           lcrt::Var<int32_t>        h_total_block_cols,
           lcrt::Var<uint32_t>       gradient_only,
           lcrt::Var<double>         inv_beta,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto m = masses->read(i);

                // The linear-algebra views' indices are `int32_t` (the reference's
                // `int i`), while `dispatch_x()` is unsigned -- cast once, like the
                // dytopo effect manager's kernels do (`lcrt::cast<std::int32_t>`).
                auto ii = lcrt::cast<std::int32_t>(i);

                // the handles are pre-offset to this system's slice, so the device
                // view starts at its own index 0
                auto G3s = DoubletDeviceView::from_handles_full(g_indices,
                                                                g_values,
                                                                n,               // total_segments
                                                                int32_t{0},      // doublet_index_offset
                                                                n,               // doublet_count
                                                                n,               // total_doublet_count
                                                                g_subvector_offset,
                                                                n);              // subvector_extent

                auto G = bdf2_kinetic_detail::gradient(
                    load_vec3(xs->read(i)), load_vec3(x_tildes->read(i)), m, inv_beta, is_fixed->read(i));
                G3s(ii).write(ii, G);

                $if(gradient_only == 0u)
                {
                    auto H3x3s = TripletDeviceView::from_handles_full(
                        h_row_indices,
                        h_col_indices,
                        h_values,
                        h_total_block_rows,
                        h_total_block_cols,
                        int32_t{0},  // triplet_index_offset (lives in the handles)
                        n,           // triplet_count
                        n,           // total_triplet_count
                        h_submatrix_offset_x,
                        h_submatrix_offset_y,
                        h_submatrix_extent_x,
                        h_submatrix_extent_y);

                    H3x3s(ii).write(ii, ii, bdf2_kinetic_detail::hessian(m, inv_beta));
                };
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FiniteElementBDF2Kinetic);

void FiniteElementBDF2Kinetic::do_build(BuildInfo& info)
{
    (void)info;
    // require BDF2 integration flag
    require<BDF2Flag>();
}

void FiniteElementBDF2Kinetic::do_compute_energy(ComputeEnergyInfo& info)
{
    // Compute kinetic energy
    auto n = static_cast<uint32_t>(info.xs().size());
    if(n > 0)
    {
        auto& k = bdf2_kinetic_energy_kernel();
        frame_stream().record(k(info.is_fixed(),
                                info.xs(),
                                info.x_tildes(),
                                info.masses(),
                                info.energies(),
                                inv_beta,
                                n)
                                  .dispatch(n));
    }
}

void FiniteElementBDF2Kinetic::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    auto n = static_cast<uint32_t>(info.xs().size());
    if(n == 0)
        return;

    const auto gradients = info.gradients();
    const auto hessians  = info.hessians();

    const bool gradient_only = info.gradient_only();

    const auto g_offset = static_cast<size_t>(gradients.doublet_index_offset());
    const auto g_count  = static_cast<size_t>(gradients.doublet_count());
    auto g_indices = gradients.indices().subview(g_offset, g_count);
    auto g_values  = gradients.values().subview(g_offset * 3u, g_count * 3u);

    // A *gradient-only* pass hands over an empty Hessian view (null buffer
    // handles): `LBufferView::subview()` on it is a hard crash, so the (valid)
    // doublet buffers are used as placeholders -- the kernel's Hessian stage is
    // predicated on `gradient_only == 0u` and never touches them.
    lud::LCBufferView<int32_t> h_row_indices;
    lud::LCBufferView<int32_t> h_col_indices;
    lud::LCBufferView<double>  h_values;
    int32_t                    h_submatrix_offset_x = 0;
    int32_t                    h_submatrix_offset_y = 0;
    int32_t                    h_submatrix_extent_x = 0;
    int32_t                    h_submatrix_extent_y = 0;
    int32_t                    h_total_block_rows   = 0;
    int32_t                    h_total_block_cols   = 0;

    if(gradient_only || hessians.triplet_count() == 0)
    {
        h_row_indices        = g_indices;
        h_col_indices        = g_indices;
        h_values             = g_values;
        h_total_block_rows   = h_total_block_cols = static_cast<int32_t>(g_count);
    }
    else
    {
        const auto h_offset = static_cast<size_t>(hessians.triplet_index_offset());
        const auto h_count  = static_cast<size_t>(hessians.triplet_count());
        h_row_indices       = hessians.row_indices().subview(h_offset, h_count);
        h_col_indices       = hessians.col_indices().subview(h_offset, h_count);
        h_values            = hessians.values().subview(h_offset * 9u, h_count * 9u);
        h_submatrix_offset_x = static_cast<int32_t>(hessians.submatrix_offset().x);
        h_submatrix_offset_y = static_cast<int32_t>(hessians.submatrix_offset().y);
        h_submatrix_extent_x = static_cast<int32_t>(hessians.extent().x);
        h_submatrix_extent_y = static_cast<int32_t>(hessians.extent().y);
        h_total_block_rows   = static_cast<int32_t>(hessians.total_rows());
        h_total_block_cols   = static_cast<int32_t>(hessians.total_cols());
    }

    auto& k = bdf2_kinetic_gradient_hessian_kernel();
    frame_stream().record(k(info.is_fixed(),
                            info.xs(),
                            info.x_tildes(),
                            info.masses(),
                            g_indices,
                            g_values,
                            static_cast<int32_t>(gradients.subvector_offset()),
                            h_row_indices,
                            h_col_indices,
                            h_values,
                            h_submatrix_offset_x,
                            h_submatrix_offset_y,
                            h_submatrix_extent_x,
                            h_submatrix_extent_y,
                            h_total_block_rows,
                            h_total_block_cols,
                            gradient_only ? 1u : 0u,
                            inv_beta,
                            n)
                              .dispatch(n));
}
}  // namespace uipc::backend::luisa

/********************************************************************
 * @file spmv.cpp
 * @brief T3.4 — SpMV kernels (see spmv.h for the reference mapping, the exact
 *        symmetric/duplicate convention and the equivalence argument of every
 *        deviation).
 *
 * Kernel inventory
 *   sym_spmv           one thread per triplet; atomic mirrored expansion
 *   rbk_spmv           per-output-row warp segmented fold (general BCOO)
 *   rbk_sym_spmv       per-output-row warp segmented fold + mirrored atomic
 *   rbk_sym_spmv_dot   rbk_sym_spmv + the fused dot contribution (one atomic per
 *                      block) + the device-side triplet count
 *   cpu_sym_spmv       host fallback (no kernel)
 * Every kernel is a function-local static of `detail::cached_kernel` (compiled
 * once per (kernel, block size), R4) and every buffer is a launch argument
 * (cookbook §7.9).
 *********************************************************************/
#include <luisa_device/linear/spmv.h>
#include <luisa_device/tuning.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace uipc::backend::luisa::luisa_device::linalg
{
namespace
{
using lcrt::Buffer;
using lcrt::BufferVar;
using lcrt::Var;

namespace detail = linalg::detail;

constexpr std::uint32_t kBlock    = 256u;
constexpr std::uint32_t kWarpSize = 32u;
constexpr std::uint32_t kWarps    = kBlock / kWarpSize;

[[nodiscard]] inline std::uint32_t round_up(std::uint64_t n, std::uint32_t block = kBlock) noexcept
{
    return static_cast<std::uint32_t>((n + block - 1u) / block * block);
}

/// `p ? t : f` — the argument order is the reverse of how it reads
/// (cookbook §7.6 trap 1).
template <typename F, typename T, typename P>
[[nodiscard]] inline auto pick(F&& f, T&& t, P&& p) noexcept
{
    return lcrt::select(std::forward<F>(f), std::forward<T>(t), std::forward<P>(p));
}

// ---------------------------------------------------------------------------
//  device helpers
// ---------------------------------------------------------------------------
/// `Eigen::Matrix::transpose()` for a block: the 1x1 scalar block is its own
/// transpose (the reference's `block.transpose()` on a `Matrix<T,1,1>`).
template <int N, typename B>
[[nodiscard]] inline auto block_transpose(const B& b) noexcept
{
    if constexpr(N == 1)
        return b;
    else
        return b.transpose();
}

/// Component-wise `pred ? v : 0` — used to zero the contribution of the lanes
/// past the device-side triplet count without branching around the warp fold.
template <typename T, int N, typename P>
[[nodiscard]] inline LVec<Var<T>, N> mask_vec(const LVec<Var<T>, N>& v, const P& pred) noexcept
{
    return lmath::make_vec<Var<T>, N>([&](int c) { return pick(T{0}, v(c), pred); });
}

/// Warp-wide *segment label*: constant for every lane of one segment and strictly
/// increasing across segments — the inclusive count of `is_head` lanes (T3.2's
/// `warp_segment_label`, algorithms.cpp).
template <typename P>
[[nodiscard]] inline auto segment_label(const P& is_head) noexcept
{
    return lcrt::warp_prefix_count_bits(is_head) + pick(0u, 1u, is_head);
}

/// `cub::WarpReduce::HeadSegmentedReduce` transcription for an N-vector, keyed by
/// the segment labels: the head lane of every segment ends with the segment's
/// component-wise sum (the same downward Hillis-Steele fold as
/// algorithms.cpp's `warp_segmented_fold`).
template <typename T, int N, typename O>
[[nodiscard]] inline LVec<Var<T>, N> seg_fold_vec(const LVec<Var<T>, N>& source,
                                                  const O&                ord) noexcept
{
    // Copy element-wise: for N == 1 the LVec single-component constructor
    // (`is_convertible_v<U,T>` holds for anything, because Var's converting
    // constructor is unconstrained) would otherwise win over the copy constructor
    // for a *non-const* lvalue argument.
    LVec<Var<T>, N> v;
    for(int c = 0; c < N; ++c)
    {
        v(c) = source(c);
    }

    auto lane = lcrt::warp_lane_id();
    for(std::uint32_t d = 1u; d < kWarpSize; d <<= 1u)
    {
        auto src       = lcrt::min(lane + d, kWarpSize - 1u);
        auto other_ord = lcrt::warp_read_lane(ord, src);
        auto merge     = ((lane + d) < kWarpSize) & (other_ord == ord);
        for(int c = 0; c < N; ++c)
        {
            auto other  = lcrt::warp_read_lane(v(c), src);
            auto merged = v(c) + other;
            v(c)        = pick(v(c), merged, merge);
        }
    }
    return v;
}

// ---------------------------------------------------------------------------
//  sym_spmv — one thread per triplet, mirrored expansion
// ---------------------------------------------------------------------------
template <typename T, int N>
[[nodiscard]] inline auto& sym_spmv_shader()
{
    using CView  = LCTripletMatrixDeviceView<Var<T>, N, N>;
    using CDense = LCDenseVectorDeviceView<Var<T>>;
    using Dense  = LDenseVectorDeviceView<Var<T>>;
    using Vec    = LVec<Var<T>, N>;

    static auto& k = detail::cached_kernel<Buffer<std::int32_t>, Buffer<std::int32_t>, Buffer<T>,
                                           Buffer<T>, Buffer<T>, T, std::uint32_t>(
        [](BufferVar<std::int32_t> rows, BufferVar<std::int32_t> cols, BufferVar<T> vals,
           BufferVar<T> x, BufferVar<T> y, Var<T> a, Var<std::uint32_t> count) noexcept
        {
            // OP-C1: this kernel is block-size agnostic -- one thread per triplet, every
            // write an atomic add, no shared memory and no warp collectives -- so its block
            // size is a pure scheduling choice and is read from the tuning knob
            // (luisa_device/tuning.h; UIPC_LUISA_BLOCK_DIM_TABLE=spmv.sym=<n>).  The grid is
            // dispatched at round_up(n) (a multiple of the historical 256), so any block size
            // is sound here: a smaller block launches the same range in more blocks, a larger
            // one a few inactive threads in the last block, and the body is guarded by
            // `$if (i < count)`.
            lcrt::set_block_size(luisa_device::block_dim_for("spmv.sym"));
            auto A  = CView::from_handles(rows, cols, vals, count, count, count);
            auto xd = CDense::from_handles(x, count);
            auto yd = Dense::from_handles(y, count);
            auto i  = lcrt::dispatch_x();
            $if (i < count)
            {
                auto t   = A(i);
                auto ri  = t.row_index * static_cast<std::uint32_t>(N);
                auto cj  = t.col_index * static_cast<std::uint32_t>(N);
                Vec  xj  = xd.segment<N>(cj).to_vec<N>();
                Vec  bx  = t.value * xj;  // B * x_j
                Vec  res = a * bx;
                yd.segment<N>(ri).atomic_add(res);

                auto is_diagonal = t.row_index == t.col_index;
                $if (!is_diagonal)
                {
                    // mirrored, transposed contribution of the (j, i) block
                    Vec  xi   = xd.segment<N>(ri).to_vec<N>();
                    auto bt   = block_transpose<N>(t.value);
                    Vec  bx2  = bt * xi;
                    Vec  res2 = a * bx2;
                    yd.segment<N>(cj).atomic_add(res2);
                };
            };
        });
    return k;
}

// ---------------------------------------------------------------------------
//  rbk_spmv — per-output-row segmented fold over a general (row grouped) BCOO
// ---------------------------------------------------------------------------
template <typename T, int N>
[[nodiscard]] inline auto& rbk_spmv_shader()
{
    using CView  = LCTripletMatrixDeviceView<Var<T>, N, N>;
    using CDense = LCDenseVectorDeviceView<Var<T>>;
    using Dense  = LDenseVectorDeviceView<Var<T>>;
    using Vec    = LVec<Var<T>, N>;

    static auto& k = detail::cached_kernel<Buffer<std::int32_t>, Buffer<std::int32_t>, Buffer<T>,
                                           Buffer<T>, Buffer<T>, T, std::uint32_t>(
        [](BufferVar<std::int32_t> rows, BufferVar<std::int32_t> cols, BufferVar<T> vals,
           BufferVar<T> x, BufferVar<T> y, Var<T> a, Var<std::uint32_t> count) noexcept
        {
            auto A  = CView::from_handles(rows, cols, vals, count, count, count);
            auto xd = CDense::from_handles(x, count);
            auto yd = Dense::from_handles(y, count);

            auto raw   = lcrt::dispatch_x();
            auto valid = raw < count;
            auto idx   = lcrt::min(raw, count - 1u);
            auto row_r = rows->read(idx);
            auto col_r = cols->read(idx);
            // invalid lanes get row -1 so they can never join a valid segment's
            // write; the reference relies on the grid-stride loop guard instead
            auto row_s = pick(std::int32_t{-1}, row_r, valid);
            // prev row, -1 outside [0, count)
            auto prev_ok  = (raw > 0u) & (raw <= count);
            auto prev_idx = lcrt::min(raw - 1u, count - 1u);
            auto prev_row = pick(std::int32_t{-1}, rows->read(prev_idx), prev_ok);

            auto block = A(idx).value;
            Vec  xj    = xd.segment<N>(col_r * static_cast<std::uint32_t>(N)).to_vec<N>();
            Vec  bx    = block * xj;
            Vec  bz    = mask_vec<T, N>(bx, valid);

            auto lane    = lcrt::warp_lane_id();
            auto is_head = ((lane == 0u) | (prev_row != row_s)) & valid;
            auto label   = segment_label(is_head);
            Vec  folded  = seg_fold_vec<T, N>(bz, label);
            $if (is_head)
            {
                Vec res = a * folded;
                yd.segment<N>(row_r * static_cast<std::uint32_t>(N)).atomic_add(res);
            };
        });
    return k;
}

// ---------------------------------------------------------------------------
//  rbk_sym_spmv — per-output-row segmented fold + the mirrored atomic write
// ---------------------------------------------------------------------------
template <typename T, int N>
[[nodiscard]] inline auto& rbk_sym_spmv_shader()
{
    using CView  = LCTripletMatrixDeviceView<Var<T>, N, N>;
    using CDense = LCDenseVectorDeviceView<Var<T>>;
    using Dense  = LDenseVectorDeviceView<Var<T>>;
    using Vec    = LVec<Var<T>, N>;

    static auto& k = detail::cached_kernel<Buffer<std::int32_t>, Buffer<std::int32_t>, Buffer<T>,
                                           Buffer<T>, Buffer<T>, T, std::uint32_t>(
        [](BufferVar<std::int32_t> rows, BufferVar<std::int32_t> cols, BufferVar<T> vals,
           BufferVar<T> x, BufferVar<T> y, Var<T> a, Var<std::uint32_t> count) noexcept
        {
            auto A  = CView::from_handles(rows, cols, vals, count, count, count);
            auto xd = CDense::from_handles(x, count);
            auto yd = Dense::from_handles(y, count);

            auto raw   = lcrt::dispatch_x();
            auto valid = raw < count;
            auto idx   = lcrt::min(raw, count - 1u);
            auto row_r = rows->read(idx);
            auto col_r = cols->read(idx);
            auto row_s = pick(std::int32_t{-1}, row_r, valid);

            auto prev_ok  = (raw > 0u) & (raw <= count);
            auto prev_idx = lcrt::min(raw - 1u, count - 1u);
            auto prev_row = pick(std::int32_t{-1}, rows->read(prev_idx), prev_ok);

            auto block = A(idx).value;
            Vec  xj    = xd.segment<N>(col_r * static_cast<std::uint32_t>(N)).to_vec<N>();
            Vec  bx    = block * xj;

            // mirrored lower-triangle contribution: y_j += a * B^T * x_i
            $if (valid & (row_r != col_r))
            {
                Vec  xi   = xd.segment<N>(row_r * static_cast<std::uint32_t>(N)).to_vec<N>();
                auto bt   = block_transpose<N>(block);
                Vec  bx2  = bt * xi;
                Vec  res2 = a * bx2;
                yd.segment<N>(col_r * static_cast<std::uint32_t>(N)).atomic_add(res2);
            };

            Vec bz = mask_vec<T, N>(bx, valid);

            auto lane    = lcrt::warp_lane_id();
            auto is_head = ((lane == 0u) | (prev_row != row_s)) & valid;
            auto label   = segment_label(is_head);
            Vec  folded  = seg_fold_vec<T, N>(bz, label);
            $if (is_head)
            {
                Vec res = a * folded;
                yd.segment<N>(row_r * static_cast<std::uint32_t>(N)).atomic_add(res);
            };
        });
    return k;
}

// ---------------------------------------------------------------------------
//  rbk_sym_spmv_dot — the fused variant
// ---------------------------------------------------------------------------
template <typename T, int N>
[[nodiscard]] inline auto& rbk_sym_spmv_dot_shader()
{
    using CView  = LCTripletMatrixDeviceView<Var<T>, N, N>;
    using CDense = LCDenseVectorDeviceView<Var<T>>;
    using Dense  = LDenseVectorDeviceView<Var<T>>;
    using Vec    = LVec<Var<T>, N>;

    static auto& k =
        detail::cached_kernel<Buffer<std::int32_t>, Buffer<std::int32_t>, Buffer<T>, Buffer<T>,
                              Buffer<T>, Buffer<T>, Buffer<std::int32_t>, T, std::uint32_t,
                              std::uint32_t>(
            [](BufferVar<std::int32_t> rows, BufferVar<std::int32_t> cols, BufferVar<T> vals,
               BufferVar<T> x, BufferVar<T> y, BufferVar<T> dot_parts,
               BufferVar<std::int32_t> d_triplet_count, Var<T> a, Var<std::uint32_t> capacity,
               Var<std::uint32_t> dofs) noexcept
            {
                // the count lives on the device, the grid is sized by capacity
                auto count = lcrt::cast<std::uint32_t>(d_triplet_count->read(0u));
                auto A     = CView::from_handles(rows, cols, vals, capacity, capacity, capacity);
                auto xd    = CDense::from_handles(x, dofs);
                auto yd    = Dense::from_handles(y, dofs);

                // Every lane past `count` reads the reserved-but-unwritten tail of the
                // row/col arrays (the reference skips those lanes entirely).  The
                // *buffer* read is safe (the allocation covers the capacity), but the
                // indices it yields are garbage, so the segment offsets they would
                // produce are clamped into the vector — the contribution itself is
                // masked to zero and gated on `valid`.
                auto segments = lcrt::max(dofs / static_cast<std::uint32_t>(N), 1u);
                auto seg_max  = lcrt::cast<std::int32_t>(segments - 1u);

                auto raw   = lcrt::dispatch_x();
                auto valid = raw < count;
                auto idx   = lcrt::min(raw, capacity - 1u);
                auto row_r = lcrt::min(lcrt::max(rows->read(idx), std::int32_t{0}), seg_max);
                auto col_r = lcrt::min(lcrt::max(cols->read(idx), std::int32_t{0}), seg_max);
                auto row_s = pick(std::int32_t{-1}, row_r, valid);

                auto prev_ok  = (raw > 0u) & (raw <= count);
                auto prev_idx = lcrt::min(raw - 1u, lcrt::min(count - 1u, capacity - 1u));
                auto prev_row = pick(std::int32_t{-1}, rows->read(prev_idx), prev_ok);

                auto block = A(idx).value;
                Vec  xj    = xd.segment<N>(col_r * static_cast<std::uint32_t>(N)).to_vec<N>();
                Vec  bx    = block * xj;

                // dot contribution: a * x_i^T B x_i (diagonal) or
                // 2a * x_i^T B x_j (off-diagonal, mirrored pair counted twice)
                Var<T> dot_local = T{0};
                $if (valid & (row_r == col_r))
                {
                    dot_local = a * xj.dot(bx);
                };
                $if (valid & (row_r != col_r))
                {
                    Vec  xi    = xd.segment<N>(row_r * static_cast<std::uint32_t>(N)).to_vec<N>();
                    dot_local  = (T{2} * a) * xi.dot(bx);
                    auto bt    = block_transpose<N>(block);
                    Vec  bx2   = bt * xi;
                    Vec  res2  = a * bx2;
                    yd.segment<N>(col_r * static_cast<std::uint32_t>(N)).atomic_add(res2);
                };

                Vec bz = mask_vec<T, N>(bx, valid);

                auto lane    = lcrt::warp_lane_id();
                auto tid     = lcrt::thread_x();
                auto wid     = tid / kWarpSize;
                auto is_head = ((lane == 0u) | (prev_row != row_s)) & valid;
                auto label   = segment_label(is_head);
                Vec  folded  = seg_fold_vec<T, N>(bz, label);
                $if (is_head)
                {
                    Vec res = a * folded;
                    yd.segment<N>(row_r * static_cast<std::uint32_t>(N)).atomic_add(res);
                };

                // block-level fold of the dot contribution: one partial per block
                // (the reference's `s_dot_partials` two-level reduction) written to
                // a scratch array so the second pass is deterministic (R6)
                $shared<T> s_dot{kWarps};
                auto       warp_dot = lcrt::warp_active_sum(dot_local);
                $if (lane == 0u) { s_dot[wid] = warp_dot; };
                lcrt::sync_block();
                $if (tid == 0u)
                {
                    Var<T> acc = s_dot[0];
                    $for (j, 1u, kWarps) { acc = acc + s_dot[j]; };
                    dot_parts->write(lcrt::block_id().x, acc);
                };
            });
    return k;
}

/// One block folds the per-block dot partials in a fixed order (no atomics).
template <typename T>
[[nodiscard]] inline auto& dot_fold_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<T>, std::uint32_t>(
        [](BufferVar<T> parts, BufferVar<T> out, Var<std::uint32_t> count) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<T> warp_red{kWarps};
            auto tid  = lcrt::thread_x();
            auto lane = lcrt::warp_lane_id();
            auto wid  = tid / kWarpSize;
            Var<T>             acc = T{0};
            Var<std::uint32_t> j   = tid;
            $while (j < count)
            {
                acc = acc + parts->read(j);
                j   = j + kBlock;
            };
            auto w = lcrt::warp_active_sum(acc);
            $if (lane == 0u) { warp_red[wid] = w; };
            lcrt::sync_block();
            $if (tid == 0u)
            {
                Var<T> r = warp_red[0];
                $for (j2, 1u, kWarps) { r = r + warp_red[j2]; };
                out->write(0u, r);
            };
        });
    return k;
}
}// namespace

// ---------------------------------------------------------------------------
//  LSpmv — the reference Spmv surface
// ---------------------------------------------------------------------------
template <typename T, int N>
void LSpmv<T, N>::sym_spmv(T a, MatrixView A, VectorView x, T b, MutVectorView y, LStream& stream)
{
    scale_or_zero(b, y, stream);
    const auto n = static_cast<std::uint32_t>(A.triplet_count());
    if(n == 0u || y.size() == 0)
        return;
    stream.record(sym_spmv_shader<T, N>()(A.row_indices(), A.col_indices(), A.values(),
                                          x.buffer_view(), y.buffer_view(), a, n)
                      .dispatch(round_up(n)));
}

template <typename T, int N>
void LSpmv<T, N>::rbk_spmv(T a, MatrixView A, VectorView x, T b, MutVectorView y, LStream& stream)
{
    scale_or_zero(b, y, stream);
    const auto n = static_cast<std::uint32_t>(A.triplet_count());
    if(n == 0u || y.size() == 0)
        return;
    stream.record(rbk_spmv_shader<T, N>()(A.row_indices(), A.col_indices(), A.values(),
                                          x.buffer_view(), y.buffer_view(), a, n)
                      .dispatch(round_up(n)));
}

template <typename T, int N>
void LSpmv<T, N>::rbk_sym_spmv(T a, MatrixView A, VectorView x, T b, MutVectorView y,
                               LStream& stream)
{
    scale_or_zero(b, y, stream);
    const auto n = static_cast<std::uint32_t>(A.triplet_count());
    if(n == 0u || y.size() == 0)
        return;
    stream.record(rbk_sym_spmv_shader<T, N>()(A.row_indices(), A.col_indices(), A.values(),
                                              x.buffer_view(), y.buffer_view(), a, n)
                      .dispatch(round_up(n)));
}

template <typename T, int N>
void LSpmv<T, N>::rbk_sym_spmv_dot(T a, MatrixView A, VectorView x, T b, MutVectorView y,
                                   DotView d_dot, CountView d_triplet_count,
                                   std::size_t triplet_capacity, LStream& stream)
{
    if(triplet_capacity == 0u || x.size() == 0 || y.size() == 0)
        return;

    scale_or_zero(b, y, stream);

    // One dot partial per block, sized by the *reserved capacity* (the same shape
    // the launch uses), then a deterministic fold into d_dot — the reference
    // accumulates the same per-block partial with an atomicAdd, which would make
    // the last bits schedule-dependent (R6).
    const auto blocks = round_up(static_cast<std::uint64_t>(triplet_capacity)) / kBlock;
    auto       parts  = detail::carve(m_dot_partials, m_dot_retired, blocks);

    // grid sized by the capacity: the count is read on the device, so the launch
    // shape does not change when nnz varies within the capacity
    stream.record(
        rbk_sym_spmv_dot_shader<T, N>()(A.row_indices(), A.col_indices(), A.values(),
                                       x.buffer_view(), y.buffer_view(), parts, d_triplet_count, a,
                                       static_cast<std::uint32_t>(triplet_capacity),
                                       static_cast<std::uint32_t>(x.size()))
            .dispatch(round_up(static_cast<std::uint64_t>(triplet_capacity))));
    stream.record(
        dot_fold_shader<T>()(parts, d_dot, blocks).dispatch(kBlock));
}

/// `y = b*y` (or a zero fill when `b == 0`), shared by every variant — the
/// reference's `Spmv_scale_y_kernel` / `BufferLaunch().fill(0)`.
template <typename T, int N>
void LSpmv<T, N>::scale_or_zero(T b, MutVectorView y, LStream& stream)
{
    if(y.size() == 0)
        return;
    if(b != T{0})
    {
        scale(b, y.buffer_view(), stream);
    }
    else
    {
        fill(y.buffer_view(), T{0}, stream);
    }
}

template <typename T, int N>
void LSpmv<T, N>::cpu_sym_spmv(T a, MatrixView A, VectorView x, T b, MutVectorView y)
{
    const std::size_t n = A.triplet_count();
    std::vector<std::int32_t> rows(n, 0);
    std::vector<std::int32_t> cols(n, 0);
    std::vector<T>            values(n * static_cast<std::size_t>(N) * static_cast<std::size_t>(N),
                                     T{0});
    if(n != 0u)
    {
        A.row_indices().copy_to(lcrt::span<std::int32_t>{rows});
        A.col_indices().copy_to(lcrt::span<std::int32_t>{cols});
        A.values().copy_to(lcrt::span<T>{values});
    }

    const std::size_t          dofs = static_cast<std::size_t>(x.size());
    std::vector<T>             xh(dofs, T{0});
    std::vector<T>             yh(dofs, T{0});
    if(dofs != 0u)
    {
        x.buffer_view().copy_to(lcrt::span<T>{xh});
        y.buffer_view().copy_to(lcrt::span<T>{yh});
    }
    LStream::current().synchronize();

    // the symmetric (upper-packed) expansion: one entry per (i, j, block) plus the
    // transposed mirror when i != j; duplicates accumulate as they are visited
    std::vector<T> product(dofs, T{0});
    for(std::size_t t = 0; t < n; ++t)
    {
        const auto i      = rows[t];
        const auto j      = cols[t];
        const T*   block  = values.data() + t * static_cast<std::size_t>(N) * N;
        for(int r = 0; r < N; ++r)
        {
            for(int c = 0; c < N; ++c)
            {
                const std::size_t si = static_cast<std::size_t>(i * N + r);
                const std::size_t sj = static_cast<std::size_t>(j * N + c);
                product[si] += block[c * N + r] * xh[sj];
                if(i != j)
                {
                    product[static_cast<std::size_t>(j * N + r)] +=
                        block[r * N + c] * xh[static_cast<std::size_t>(i * N + c)];
                }
            }
        }
    }
    for(std::size_t i = 0; i < dofs; ++i)
    {
        yh[i] = a * product[i] + b * yh[i];
    }
    if(dofs != 0u)
    {
        y.buffer_view().copy_from(lcrt::span<const T>{yh});
        LStream::current().synchronize();
    }
}

// ---------------------------------------------------------------------------
//  explicit instantiation
// ---------------------------------------------------------------------------
template class LSpmv<float, 1>;
template class LSpmv<float, 2>;
template class LSpmv<float, 3>;
template class LSpmv<double, 1>;
template class LSpmv<double, 2>;
template class LSpmv<double, 3>;
}// namespace uipc::backend::luisa::luisa_device::linalg

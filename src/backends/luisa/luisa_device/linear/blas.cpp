/********************************************************************
 * @file blas.cpp
 * @brief T3.4 — dense BLAS kernels (see blas.h for the contract, the reference
 *        mapping and the equivalence argument of every deviation).
 *
 * Kernel inventory
 *   dot_partial    one 256-thread block -> one partial (warp sum + shared fold)
 *   dot_fold       one block folds the partials (fixed order, no atomics)
 *   norm_partial   one block -> one (scale, square_sum) partial
 *   norm_fold      one block folds the (scale, square_sum) partials
 *   scale          y = a*y
 *   axpy           y += a*x
 *   axpby          y = a*x + b*y
 *   update_xr      x += alpha*p ; r -= alpha*Ap
 *   update_p       p = z + beta*p
 * Every kernel is a function-local static of `detail::cached_kernel` (compiled
 * once per (kernel, scalar type), R4) and every buffer is a launch argument
 * (cookbook §7.9).  The reductions are two-level and atomic-free, so two runs on
 * the same input are bitwise identical (R6).
 *********************************************************************/
#include <luisa_device/linear/blas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace uipc::backend::luisa::luisa_device::linalg
{
namespace
{
using lcrt::Buffer;
using lcrt::BufferVar;
using lcrt::Var;

namespace detail = linalg::detail;

constexpr std::uint32_t kBlock    = detail::kReductionBlockSize;
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
//  ScaledSquareSum pair, branchless (see blas.h)
// ---------------------------------------------------------------------------
template <typename S>
struct ScaledPair
{
    Var<S> scale{};
    Var<S> square_sum{};
};

/// The reference's `ScaledSquareSumOp`, branchless so it works on device values
/// (RULE M2).  For finite inputs this is exactly the reference's three cases.
template <typename S>
[[nodiscard]] inline ScaledPair<S> combine_scaled(const ScaledPair<S>& a,
                                                  const ScaledPair<S>& b) noexcept
{
    auto lo = lcrt::min(a.scale, b.scale);
    auto hi = lcrt::max(a.scale, b.scale);
    // (hi == lo) -> 1: equal scales (including 0 vs 0) add their square sums
    // unscaled; the reference's third case.
    auto ratio = pick(lo / hi, S{1}, hi == lo);  // (hi == lo) ? 1 : lo/hi
    // the larger-scale operand keeps its square sum, the smaller one is rescaled
    auto a_is_hi = a.scale >= b.scale;
    auto hi_ss   = pick(b.square_sum, a.square_sum, a_is_hi);
    auto lo_ss   = pick(a.square_sum, b.square_sum, a_is_hi);
    return ScaledPair<S>{hi, hi_ss + lo_ss * ratio * ratio};
}

/// Warp-wide fold of a ScaledSquareSum pair; every lane ends with the total
/// (Hillis-Steele, the same pattern algorithms.cpp uses for the segmented fold).
template <typename S>
[[nodiscard]] inline ScaledPair<S> warp_fold_scaled(ScaledPair<S> p) noexcept
{
    auto lane = lcrt::warp_lane_id();
    for(std::uint32_t d = 1u; d < kWarpSize; d <<= 1u)
    {
        auto idx         = lcrt::min(lane + d, kWarpSize - 1u);
        auto other_scale = lcrt::warp_read_lane(p.scale, idx);
        auto other_ss    = lcrt::warp_read_lane(p.square_sum, idx);
        ScaledPair<S> other{other_scale, other_ss};
        auto          merged = combine_scaled(p, other);
        // the read is clamped, so the predicate is what discards the out-of-range
        // contribution
        auto merge = (lane + d) < kWarpSize;
        p = ScaledPair<S>{pick(p.scale, merged.scale, merge),
                          pick(p.square_sum, merged.square_sum, merge)};
    }
    return p;
}

// ---------------------------------------------------------------------------
//  dot
// ---------------------------------------------------------------------------
template <typename T>
[[nodiscard]] inline auto& dot_partial_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<T>, Buffer<T>, std::uint32_t>(
        [](BufferVar<T> x, BufferVar<T> y, BufferVar<T> parts, Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<T> warp_red{kWarps};
            auto tid   = lcrt::thread_x();
            auto gid   = lcrt::dispatch_x();
            auto bid   = lcrt::block_id().x;
            auto lane  = lcrt::warp_lane_id();
            auto wid   = tid / kWarpSize;
            auto valid = gid < n;
            auto idx   = lcrt::min(gid, n - 1u);
            auto prod  = x->read(idx) * y->read(idx);
            auto v     = pick(T{0}, prod, valid);   // valid ? prod : 0
            auto w     = lcrt::warp_active_sum(v);  // the total, on every lane
            $if (lane == 0u) { warp_red[wid] = w; };
            lcrt::sync_block();
            $if (tid == 0u)
            {
                Var<T> acc = warp_red[0];
                $for (j, 1u, kWarps) { acc = acc + warp_red[j]; };
                parts->write(bid, acc);
            };
        });
    return k;
}

/// One block folds the partials in a fixed stride order (deterministic).
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

// ---------------------------------------------------------------------------
//  norm (scaled square sum)
// ---------------------------------------------------------------------------
template <typename T>
[[nodiscard]] inline auto& norm_partial_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<T>, Buffer<T>, std::uint32_t>(
        [](BufferVar<T> x, BufferVar<T> scale_parts, BufferVar<T> ss_parts,
           Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            $shared<T> warp_scale{kWarps};
            $shared<T> warp_ss{kWarps};
            auto tid   = lcrt::thread_x();
            auto gid   = lcrt::dispatch_x();
            auto bid   = lcrt::block_id().x;
            auto lane  = lcrt::warp_lane_id();
            auto wid   = tid / kWarpSize;
            auto valid = gid < n;
            auto idx   = lcrt::min(gid, n - 1u);
            auto m     = lcrt::abs(x->read(idx));
            auto is_nz = (m != T{0}) & valid;
            // valid ? make_scaled_square_sum(x[idx]) : neutral {0,1}; the
            // square sum of a single term is always 1.
            ScaledPair<T> p{pick(T{0}, m, is_nz), T{1}};
            p = warp_fold_scaled<T>(p);
            $if (lane == 0u)
            {
                warp_scale[wid] = p.scale;
                warp_ss[wid]    = p.square_sum;
            };
            lcrt::sync_block();
            $if (tid == 0u)
            {
                ScaledPair<T> acc{warp_scale[0], warp_ss[0]};
                $for (j, 1u, kWarps)
                {
                    ScaledPair<T> other{warp_scale[j], warp_ss[j]};
                    acc = combine_scaled(acc, other);
                };
                scale_parts->write(bid, acc.scale);
                ss_parts->write(bid, acc.square_sum);
            };
        });
    return k;
}

template <typename T>
[[nodiscard]] inline auto& norm_fold_shader()
{
    static auto& k =
        detail::cached_kernel<Buffer<T>, Buffer<T>, Buffer<T>, Buffer<T>, std::uint32_t>(
            [](BufferVar<T> scale_parts, BufferVar<T> ss_parts, BufferVar<T> out_scale,
               BufferVar<T> out_ss, Var<std::uint32_t> count) noexcept
            {
                lcrt::set_block_size(kBlock);
                $shared<T> warp_scale{kWarps};
                $shared<T> warp_ss{kWarps};
                auto tid  = lcrt::thread_x();
                auto lane = lcrt::warp_lane_id();
                auto wid  = tid / kWarpSize;
                ScaledPair<T>      acc{T{0}, T{1}};
                Var<std::uint32_t> j = tid;
                $while (j < count)
                {
                    ScaledPair<T> other{scale_parts->read(j), ss_parts->read(j)};
                    acc = combine_scaled(acc, other);
                    j   = j + kBlock;
                };
                acc = warp_fold_scaled<T>(acc);
                $if (lane == 0u)
                {
                    warp_scale[wid] = acc.scale;
                    warp_ss[wid]    = acc.square_sum;
                };
                lcrt::sync_block();
                $if (tid == 0u)
                {
                    ScaledPair<T> r{warp_scale[0], warp_ss[0]};
                    $for (j2, 1u, kWarps)
                    {
                        ScaledPair<T> other{warp_scale[j2], warp_ss[j2]};
                        r = combine_scaled(r, other);
                    };
                    out_scale->write(0u, r.scale);
                    out_ss->write(0u, r.square_sum);
                };
            });
    return k;
}

// ---------------------------------------------------------------------------
//  vector kernels
// ---------------------------------------------------------------------------
template <typename T>
[[nodiscard]] inline auto& scale_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, T, std::uint32_t>(
        [](BufferVar<T> y, Var<T> a, Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n) { y->write(i, a * y->read(i)); };
        });
    return k;
}

template <typename T>
[[nodiscard]] inline auto& axpy_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<T>, T, std::uint32_t>(
        [](BufferVar<T> x, BufferVar<T> y, Var<T> a, Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n) { y->write(i, y->read(i) + a * x->read(i)); };
        });
    return k;
}

template <typename T>
[[nodiscard]] inline auto& axpby_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<T>, T, T, std::uint32_t>(
        [](BufferVar<T> x, BufferVar<T> y, Var<T> a, Var<T> b, Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n) { y->write(i, a * x->read(i) + b * y->read(i)); };
        });
    return k;
}

template <typename T>
[[nodiscard]] inline auto& update_xr_shader()
{
    static auto& k =
        detail::cached_kernel<Buffer<T>, Buffer<T>, Buffer<T>, Buffer<T>, T, std::uint32_t>(
            [](BufferVar<T> x, BufferVar<T> p, BufferVar<T> r, BufferVar<T> Ap, Var<T> alpha,
               Var<std::uint32_t> n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if (i < n)
                {
                    x->write(i, x->read(i) + alpha * p->read(i));
                    r->write(i, r->read(i) - alpha * Ap->read(i));
                };
            });
    return k;
}

template <typename T>
[[nodiscard]] inline auto& update_p_shader()
{
    static auto& k = detail::cached_kernel<Buffer<T>, Buffer<T>, T, std::uint32_t>(
        [](BufferVar<T> p, BufferVar<T> z, Var<T> beta, Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if (i < n) { p->write(i, z->read(i) + beta * p->read(i)); };
        });
    return k;
}

// ---------------------------------------------------------------------------
//  host-side helpers
// ---------------------------------------------------------------------------
/// Commit the caller's frame CommandList, wait, read the single scalar through the
/// caller's stream, then re-open the frame (the T3.3 read-back pattern,
/// cookbook §7.11 restructure 4).
template <typename T>
[[nodiscard]] T read_back_scalar(LStream& stream, const LDeviceVar<T>& v)
{
    const bool was_open = stream.frame_open();
    stream.synchronize();  // commits the recorded work, then waits
    T host{};
    stream.record(v.cview().native().copy_to(lcrt::span{&host, 1u}));
    stream.synchronize();
    if(was_open)
    {
        stream.begin();
    }
    return host;
}

/// Number of reduction blocks for an n-element buffer (reference
/// `linear_reduction_block_count`).
template <typename T>
[[nodiscard]] std::uint32_t block_count_of(const LCBufferView<T>& x) noexcept
{
    return round_up(static_cast<std::uint64_t>(x.size()));
}
}// namespace

// ---------------------------------------------------------------------------
//  default stream
// ---------------------------------------------------------------------------
LStream& default_blas_stream()
{
    // One process-wide stream, deliberately: each LStream owns its own CommandList,
    // so a second one would be a hidden batching split (its commands would jump ahead
    // of the frame's recorded work, recording nothing into the frame's list).
    return LStream::current();
}

namespace detail
{
// ---------------------------------------------------------------------------
//  dot / norm
// ---------------------------------------------------------------------------
double dot_impl(LStream& stream, const LCBufferView<double>& x, const LCBufferView<double>& y,
                LinearReductionStorage<double>& s)
{
    const auto blocks = block_count_of(x);
    auto       parts  = carve(s.dot_partials, s.retired, blocks);
    stream.record(dot_partial_shader<double>()(x, y, parts, static_cast<std::uint32_t>(x.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(x.size()))));
    stream.record(dot_fold_shader<double>()(parts, s.dot_result.view(), blocks).dispatch(kBlock));
    return read_back_scalar(stream, s.dot_result);
}

float dot_impl(LStream& stream, const LCBufferView<float>& x, const LCBufferView<float>& y,
               LinearReductionStorage<float>& s)
{
    const auto blocks = block_count_of(x);
    auto       parts  = carve(s.dot_partials, s.retired, blocks);
    stream.record(dot_partial_shader<float>()(x, y, parts, static_cast<std::uint32_t>(x.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(x.size()))));
    stream.record(dot_fold_shader<float>()(parts, s.dot_result.view(), blocks).dispatch(kBlock));
    return read_back_scalar(stream, s.dot_result);
}

double norm_impl(LStream& stream, const LCBufferView<double>& x, LinearReductionStorage<double>& s)
{
    const auto blocks      = block_count_of(x);
    auto       scale_parts = carve(s.norm_scale_partials, s.retired, blocks);
    auto       ss_parts    = carve(s.norm_square_partials, s.retired, blocks);
    stream.record(norm_partial_shader<double>()(x, scale_parts, ss_parts,
                                                static_cast<std::uint32_t>(x.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(x.size()))));
    stream.record(norm_fold_shader<double>()(scale_parts, ss_parts, s.norm_scale_result.view(),
                                             s.norm_square_result.view(), blocks)
                      .dispatch(kBlock));
    const auto scale = read_back_scalar(stream, s.norm_scale_result);
    const auto ss    = read_back_scalar(stream, s.norm_square_result);
    return scale * std::sqrt(ss);
}

float norm_impl(LStream& stream, const LCBufferView<float>& x, LinearReductionStorage<float>& s)
{
    const auto blocks      = block_count_of(x);
    auto       scale_parts = carve(s.norm_scale_partials, s.retired, blocks);
    auto       ss_parts    = carve(s.norm_square_partials, s.retired, blocks);
    stream.record(norm_partial_shader<float>()(x, scale_parts, ss_parts,
                                               static_cast<std::uint32_t>(x.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(x.size()))));
    stream.record(norm_fold_shader<float>()(scale_parts, ss_parts, s.norm_scale_result.view(),
                                            s.norm_square_result.view(), blocks)
                      .dispatch(kBlock));
    const auto scale = read_back_scalar(stream, s.norm_scale_result);
    const auto ss    = read_back_scalar(stream, s.norm_square_result);
    return scale * std::sqrt(ss);
}

// ---------------------------------------------------------------------------
//  vector kernels
// ---------------------------------------------------------------------------
void scale_impl(LStream& stream, double a, const LBufferView<double>& y)
{
    if(y.size() == 0u)
        return;
    stream.record(scale_shader<double>()(y, a, static_cast<std::uint32_t>(y.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(y.size()))));
}
void scale_impl(LStream& stream, float a, const LBufferView<float>& y)
{
    if(y.size() == 0u)
        return;
    stream.record(scale_shader<float>()(y, a, static_cast<std::uint32_t>(y.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(y.size()))));
}

void axpy_impl(LStream& stream, double a, const LCBufferView<double>& x,
               const LBufferView<double>& y)
{
    if(y.size() == 0u)
        return;
    stream.record(axpy_shader<double>()(x, y, a, static_cast<std::uint32_t>(y.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(y.size()))));
}
void axpy_impl(LStream& stream, float a, const LCBufferView<float>& x, const LBufferView<float>& y)
{
    if(y.size() == 0u)
        return;
    stream.record(axpy_shader<float>()(x, y, a, static_cast<std::uint32_t>(y.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(y.size()))));
}

void axpby_impl(LStream& stream, double a, const LCBufferView<double>& x, double b,
                const LBufferView<double>& y)
{
    if(y.size() == 0u)
        return;
    stream.record(axpby_shader<double>()(x, y, a, b, static_cast<std::uint32_t>(y.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(y.size()))));
}
void axpby_impl(LStream& stream, float a, const LCBufferView<float>& x, float b,
                const LBufferView<float>& y)
{
    if(y.size() == 0u)
        return;
    stream.record(axpby_shader<float>()(x, y, a, b, static_cast<std::uint32_t>(y.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(y.size()))));
}

void update_xr_impl(LStream& stream, double alpha, const LBufferView<double>& x,
                    const LCBufferView<double>& p, const LBufferView<double>& r,
                    const LCBufferView<double>& Ap)
{
    if(r.size() == 0u)
        return;
    stream.record(update_xr_shader<double>()(x, p, r, Ap, alpha, static_cast<std::uint32_t>(r.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(r.size()))));
}
void update_xr_impl(LStream& stream, float alpha, const LBufferView<float>& x,
                    const LCBufferView<float>& p, const LBufferView<float>& r,
                    const LCBufferView<float>& Ap)
{
    if(r.size() == 0u)
        return;
    stream.record(update_xr_shader<float>()(x, p, r, Ap, alpha, static_cast<std::uint32_t>(r.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(r.size()))));
}

void update_p_impl(LStream& stream, const LBufferView<double>& p, const LCBufferView<double>& z,
                   double beta)
{
    if(p.size() == 0u)
        return;
    stream.record(update_p_shader<double>()(p, z, beta, static_cast<std::uint32_t>(p.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(p.size()))));
}
void update_p_impl(LStream& stream, const LBufferView<float>& p, const LCBufferView<float>& z,
                   float beta)
{
    if(p.size() == 0u)
        return;
    stream.record(update_p_shader<float>()(p, z, beta, static_cast<std::uint32_t>(p.size()))
                      .dispatch(round_up(static_cast<std::uint64_t>(p.size()))));
}
}// namespace detail
}// namespace uipc::backend::luisa::luisa_device::linalg

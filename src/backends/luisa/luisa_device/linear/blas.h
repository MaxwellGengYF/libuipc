#pragma once
/********************************************************************
 * @file blas.h
 * @brief T3.4 — dense BLAS on device vectors: the LuisaCompute port of
 *        src/backends/cuda/cuda_tool/linear_reduction.h +
 *        `cuda_tool::LinearSystemContext` (cuda_tool/linear_system.h) and of the
 *        dense-vector / reduction ops the PCG and preconditioner consumers use.
 *
 * Reference surface
 * -----------------
 *  1. `cuda_tool::LinearSystemContext` — the persistent dot/norm reduction
 *     context:
 *         T dot (x, y)   // dense dot,    result = sum_i x[i]*y[i]
 *         T norm(x)      // dense 2-norm, LAPACK nrm2 semantics
 *     It owns one `details::LinearReductionStorage<T>` per scalar type (two here:
 *     float and double), so the partials and the result scalar are reused across
 *     calls instead of being allocated per reduction.  Both reductions are
 *     host-facing: like the reference they return a `T` by value, i.e. they
 *     synchronize.
 *
 *  2. `details::ScaledSquareSum<T>` — the (scale, square_sum) accumulator that
 *     makes the norm a *scaled* square sum: every term is divided by the running
 *     magnitude, so a vector spanning [1e-150, 1e150] neither overflows nor
 *     underflows (LAPACK `nrm2` / `lassq`).  PCG's `check_*_rz_nan_inf` consume
 *     this norm, so its value (not just finiteness) is part of the contract.
 *
 *  3. The dense-vector kernels the PCG consumers use, hoisted out of the
 *     reference's `linear_pcg.cu` / `linear_fused_pcg.cu` translation units:
 *         fill(y, v)                         y[i] = v      (BufferLaunch().fill)
 *         scale(a, y)                        y[i] = a*y[i]  (Spmv_scale_y_kernel)
 *         axpy(a, x, y)                      y[i] += a*x[i]
 *         axpby(a, x, b, y)                  y[i] = a*x[i] + b*y[i]
 *         copy(dst, src)                     dst[i] = src[i]
 *         update_xr(alpha, x, p, r, Ap)      x += alpha*p ; r -= alpha*Ap
 *         update_p(p, z, beta)               z + beta*p   -> p
 *     (`update_xr` / `update_p` are the reference's two PCG vector updates;
 *     `axpby` is their generic spelling.)
 *
 * Deviations (each with its equivalence argument)
 * ----------------------------------------------
 *  * `ScaledSquareSum<T>` is a *host-side* value description here; the device
 *    reduction carries the same two scalars in two separate buffers instead of a
 *    `DeviceVector<ScaledSquareSum<T>>`.  LC buffers need a registered element
 *    type, and T3.3 established the precedent of splitting a reference struct
 *    buffer into plain scalar arrays (matrix_converter.inl's pair).  The scalars
 *    and their combination order are identical.
 *  * The reference's `ScaledSquareSumOp` branches on the two scales; a branch on
 *    a device value cannot drive device control flow (RULE M2), so the port
 *    combines the two operands *branchlessly*:
 *        lo = min(a.scale, b.scale), hi = max(a.scale, b.scale),
 *        ratio = (hi == lo) ? 1 : lo/hi,
 *        result = { hi, big.square_sum + small.square_sum * ratio^2 }.
 *    For every finite input this is exactly the reference's three cases
 *    (a.scale < b.scale, b.scale < a.scale, equal — including the 0/0 case, where
 *    both scales are 0 and `ratio` is selected to 1).  The reference's fourth
 *    case (a NaN scale propagates as `a.scale + b.scale`) becomes a finite
 *    `min`/`max` operand; the norm then reports the finite operand instead of NaN.
 *    That path only ever feeds PCG's diagnostic hint (the assertion message), not
 *    a convergence decision — `dot` (which propagates NaN/Inf exactly) is what
 *    PCG tests.
 *  * Reductions are deterministic (R6): per-block partials then one fold block, in
 *    a fixed schedule, with **no atomics** — the reference's `DeviceReduce` second
 *    stage is replaced by an equivalent two-level fold (cookbook §7.10's
 *    `device_reduce` pattern, reused here for the pair accumulator).
 *
 * Buffer lifetime (cookbook §7.10)
 * -------------------------------
 * The partials are carved with `detail::carve`, which retires a displaced
 * allocation instead of freeing it: a caller may still have a recorded command
 * referencing the previous call's partials in its open frame CommandList.  The
 * result scalars are allocated once (size 1) and never resized.
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/buffer.h>
#include <luisa_device/export.h>
#include <luisa_device/linear/algorithms.h>
#include <luisa_device/linear/containers.h>
#include <luisa_device/linear/views.h>
#include <luisa_device/stream.h>
#include <luisa_device/view.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace uipc::backend::luisa::luisa_device::linalg
{
namespace lud   = ::uipc::backend::luisa::luisa_device;
namespace lmath = ::uipc::backend::luisa::luisa_device::math;

// ---------------------------------------------------------------------------
//  ScaledSquareSum — the reference's cuda_tool::details::ScaledSquareSum
// ---------------------------------------------------------------------------
/// `norm^2` accumulator with an explicit scale (LAPACK nrm2 / lassq):
/// `norm = scale * sqrt(square_sum)` with `scale = max_i |x_i|`, so `square_sum`
/// stays in [1, n] and cannot overflow/underflow.
template <typename T>
struct ScaledSquareSum
{
    T scale      = T{0};
    T square_sum = T{1};
};

namespace detail
{
/// Reference `make_scaled_square_sum` (cuda_tool/linear_reduction.h):
/// `magnitude == 0` yields the neutral {0, 1}.
template <typename T>
[[nodiscard]] ScaledSquareSum<T> make_scaled_square_sum(const T& value) noexcept
{
    const T magnitude = value < T{0} ? -value : value;
    return magnitude == T{0} ? ScaledSquareSum<T>{} : ScaledSquareSum<T>{magnitude, T{1}};
}

/// Persistent reduction storage of one scalar type (reference:
/// `details::LinearReductionStorage`, with the struct buffers split into scalar
/// arrays — see the header note).  Reused across calls; `retired` keeps a grown
/// allocation alive for commands an open frame still references.
template <typename T>
struct LinearReductionStorage
{
    /// dot: one partial per block, folded into `dot_result`.
    LDeviceBuffer<T> dot_partials;
    LDeviceVar<T>    dot_result;

    /// norm: one (scale, square_sum) partial per block, folded into the two
    /// result scalars.
    LDeviceBuffer<T> norm_scale_partials;
    LDeviceBuffer<T> norm_square_partials;
    LDeviceVar<T>    norm_scale_result;
    LDeviceVar<T>    norm_square_result;

    std::vector<std::shared_ptr<void>> retired;
};

/// Block size of the reference's reduction kernels (`linear_reduction_block_size`).
inline constexpr std::uint32_t kReductionBlockSize = 256u;

// ---- explicitly instantiated implementations (defined in blas.cpp) ---------
UIPC_LUISA_DEVICE_API double dot_impl(LStream& stream, const LCBufferView<double>& x,
                                      const LCBufferView<double>& y, LinearReductionStorage<double>& s);
UIPC_LUISA_DEVICE_API float dot_impl(LStream& stream, const LCBufferView<float>& x,
                                     const LCBufferView<float>& y, LinearReductionStorage<float>& s);
UIPC_LUISA_DEVICE_API double norm_impl(LStream& stream, const LCBufferView<double>& x,
                                       LinearReductionStorage<double>& s);
UIPC_LUISA_DEVICE_API float norm_impl(LStream& stream, const LCBufferView<float>& x,
                                      LinearReductionStorage<float>& s);

UIPC_LUISA_DEVICE_API void scale_impl(LStream& stream, double a, const LBufferView<double>& y);
UIPC_LUISA_DEVICE_API void scale_impl(LStream& stream, float a, const LBufferView<float>& y);
UIPC_LUISA_DEVICE_API void axpy_impl(LStream& stream, double a, const LCBufferView<double>& x,
                                     const LBufferView<double>& y);
UIPC_LUISA_DEVICE_API void axpy_impl(LStream& stream, float a, const LCBufferView<float>& x,
                                     const LBufferView<float>& y);
UIPC_LUISA_DEVICE_API void axpby_impl(LStream& stream, double a, const LCBufferView<double>& x,
                                      double b, const LBufferView<double>& y);
UIPC_LUISA_DEVICE_API void axpby_impl(LStream& stream, float a, const LCBufferView<float>& x,
                                      float b, const LBufferView<float>& y);
UIPC_LUISA_DEVICE_API void update_xr_impl(LStream& stream, double alpha,
                                          const LBufferView<double>& x, const LCBufferView<double>& p,
                                          const LBufferView<double>& r, const LCBufferView<double>& Ap);
UIPC_LUISA_DEVICE_API void update_xr_impl(LStream& stream, float alpha, const LBufferView<float>& x,
                                          const LCBufferView<float>& p, const LBufferView<float>& r,
                                          const LCBufferView<float>& Ap);
UIPC_LUISA_DEVICE_API void update_p_impl(LStream& stream, const LBufferView<double>& p,
                                         const LCBufferView<double>& z, double beta);
UIPC_LUISA_DEVICE_API void update_p_impl(LStream& stream, const LBufferView<float>& p,
                                         const LCBufferView<float>& z, float beta);

/// Per-stream reduction storage for the free-function `dot`/`norm` flavours: the
/// same reuse as `LinearSystemContext`, keyed on the caller's stream so a
/// primitive called inside an open frame keeps its partials alive (§7.10).
template <typename T>
[[nodiscard]] LinearReductionStorage<T>& default_reduction_storage(LStream& stream)
{
    static const char tag = 0;  // one distinct address per LinearReductionStorage<T>
    return *static_cast<LinearReductionStorage<T>*>(stream.cached_object(
        &tag,
        []() -> void* { return new LinearReductionStorage<T>{}; },
        [](void* p) { delete static_cast<LinearReductionStorage<T>*>(p); }));
}
}// namespace detail

// ---------------------------------------------------------------------------
//  Default stream — the reference's implicit "legacy default stream"
// ---------------------------------------------------------------------------
/// The process-wide `LStream` (`LStream::current()`, the backend's `frame_stream()`).
/// There must be exactly one: each `LStream` owns its own CommandList, so a second
/// one is a hidden batching split (its commands would submit immediately and jump
/// ahead of the frame's recorded work).  Consumer code inside a frame passes the
/// frame's `LStream` instead.
[[nodiscard]] UIPC_LUISA_DEVICE_API LStream& default_blas_stream();

// ---------------------------------------------------------------------------
//  LinearSystemContext — the reference's persistent dot/norm context
// ---------------------------------------------------------------------------
/// Owns one `details::LinearReductionStorage<T>` per scalar type (float, double),
/// so the partials and result scalars are reused across calls.  Non-copyable,
/// exactly like the reference.
class UIPC_LUISA_DEVICE_API LLinearSystemContext
{
  public:
    /// `stream` defaults to the process-lifetime default stream (the reference's
    /// `cudaStream_t s = default_stream()`).
    explicit LLinearSystemContext(LStream* stream = nullptr) noexcept
        : m_stream(stream)
    {
    }
    ~LLinearSystemContext()                                      = default;
    LLinearSystemContext(const LLinearSystemContext&)            = delete;
    LLinearSystemContext& operator=(const LLinearSystemContext&) = delete;

    [[nodiscard]] LStream& stream() noexcept
    {
        return m_stream ? *m_stream : default_blas_stream();
    }
    void set_stream(LStream& stream) noexcept { m_stream = &stream; }

    // -- dot: result = sum(x[i] * y[i]) -------------------------------------
    template <typename T>
    T dot(const LDenseVector<T>& x, const LDenseVector<T>& y)
    {
        return dot(x.cview(), y.cview());
    }
    template <typename T>
    T dot(const LCBufferView<T>& x, const LCBufferView<T>& y)
    {
        if(x.size() != y.size())
            throw std::invalid_argument{"LinearSystemContext::dot size mismatch"};
        if(x.size() == 0u)
            return T{0};
        return detail::dot_impl(stream(), x, y, reduction_storage<T>());
    }
    template <typename T>
    T dot(const LCDenseVectorView<T>& x, const LCDenseVectorView<T>& y)
    {
        return dot(x.buffer_view(), y.buffer_view());
    }

    // -- norm (nrm2) --------------------------------------------------------
    template <typename T>
    T norm(const LDenseVector<T>& x)
    {
        return norm(x.cview());
    }
    template <typename T>
    T norm(const LCBufferView<T>& x)
    {
        if(x.size() == 0u)
            return T{0};
        return detail::norm_impl(stream(), x, reduction_storage<T>());
    }
    template <typename T>
    T norm(const LCDenseVectorView<T>& x)
    {
        return norm(x.buffer_view());
    }

    /// `norm2` spelling (the plan's name for the same op).
    template <typename T>
    T norm2(const LCBufferView<T>& x)
    {
        return norm(x);
    }

  private:
    template <typename T>
    detail::LinearReductionStorage<T>& reduction_storage()
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "LinearSystemContext: only float/double reductions exist");
        if constexpr(std::is_same_v<T, float>)
            return m_float_reduction;
        else
            return m_double_reduction;
    }

    LStream*                               m_stream = nullptr;
    detail::LinearReductionStorage<float>  m_float_reduction;
    detail::LinearReductionStorage<double> m_double_reduction;
};

/// The reference's spelling (`cuda_tool::LinearSystemContext`).
using LinearSystemContext = LLinearSystemContext;

// ---------------------------------------------------------------------------
//  Free-function dot / norm (the reference context's ops, stateless spelling)
// ---------------------------------------------------------------------------
/// `sum_i x[i]*y[i]` (device buffers; deterministic two-level reduction).
template <typename T>
T dot(const LCBufferView<T>& x, const LCBufferView<T>& y, LStream& stream = default_blas_stream())
{
    if(x.size() != y.size())
        throw std::invalid_argument{"linalg::dot size mismatch"};
    if(x.size() == 0u)
        return T{0};
    return detail::dot_impl(stream, x, y, detail::default_reduction_storage<T>(stream));
}

/// LAPACK nrm2 of a device buffer.
template <typename T>
T norm(const LCBufferView<T>& x, LStream& stream = default_blas_stream())
{
    if(x.size() == 0u)
        return T{0};
    return detail::norm_impl(stream, x, detail::default_reduction_storage<T>(stream));
}

/// Alias kept because the plan spells the op `norm2`.
template <typename T>
T norm2(const LCBufferView<T>& x, LStream& stream = default_blas_stream())
{
    return norm(x, stream);
}

// ---------------------------------------------------------------------------
//  dense-vector kernels (the reference's per-iteration PCG helpers)
// ---------------------------------------------------------------------------
/// `y[i] = value` on the caller's stream.  LC has no fill command, and
/// `LBufferView::fill`'s stream parameter is what keeps this ordered inside an
/// open frame CommandList.
template <typename T>
void fill(const LBufferView<T>& y, const T& value, LStream& stream = default_blas_stream())
{
    y.fill(value, &stream);
}

/// `y[i] = a*y[i]` (`Spmv_scale_y_kernel`).
template <typename T>
void scale(const T& a, const LBufferView<T>& y, LStream& stream = default_blas_stream())
{
    detail::scale_impl(stream, a, y);
}

/// `y[i] += a*x[i]`.
template <typename T>
void axpy(const T& a, const LCBufferView<T>& x, const LBufferView<T>& y,
          LStream& stream = default_blas_stream())
{
    detail::axpy_impl(stream, a, x, y);
}

/// `y[i] = a*x[i] + b*y[i]`.
template <typename T>
void axpby(const T& a, const LCBufferView<T>& x, const T& b, const LBufferView<T>& y,
           LStream& stream = default_blas_stream())
{
    detail::axpby_impl(stream, a, x, b, y);
}

/// `dst[i] = src[i]` (device -> device, recorded on the caller's stream).
template <typename T>
void copy(const LBufferView<T>& dst, const LCBufferView<T>& src,
          LStream& stream = default_blas_stream())
{
    if(dst.size() != src.size())
        throw std::invalid_argument{"linalg::copy size mismatch"};
    if(dst.size() == 0u)
        return;
    stream.record(dst.native().copy_from(src.native()));
}

/// `x += alpha*p ; r -= alpha*Ap` (reference `update_xr_kernel`).
template <typename T>
void update_xr(const T& alpha, const LBufferView<T>& x, const LCBufferView<T>& p,
               const LBufferView<T>& r, const LCBufferView<T>& Ap,
               LStream& stream = default_blas_stream())
{
    detail::update_xr_impl(stream, alpha, x, p, r, Ap);
}

/// `p = z + beta*p` (reference `update_p_kernel`).
template <typename T>
void update_p(const LBufferView<T>& p, const LCBufferView<T>& z, const T& beta,
              LStream& stream = default_blas_stream())
{
    detail::update_p_impl(stream, p, z, beta);
}
}// namespace uipc::backend::luisa::luisa_device::linalg

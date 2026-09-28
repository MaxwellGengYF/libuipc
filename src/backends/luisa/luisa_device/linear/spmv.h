#pragma once
/********************************************************************
 * @file spmv.h
 * @brief T3.4 — sparse matrix-vector kernels: the LuisaCompute port of
 *        src/backends/cuda/linear_system/spmv.h + spmv.cu.
 *
 * The reference class is `uipc::backend::cuda::Spmv` with the five methods below
 * (all on `Float` = double, block size 3).  The port is the class template
 * `lla::LSpmv<T,N>` (so the block-1 and block-3 shapes of the T3.4 tests are the
 * same code), with `lla::Spmv = LSpmv<double,3>` keeping the reference spelling.
 * Method names, parameter order and the block-3 default are unchanged; the
 * reference's `cudaStream_t` parameter becomes an optional `LStream&`.
 *
 *   sym_spmv(a, A, x, b, y)          y = a*A_sym*x + b*y, upper-packed BCOO,
 *                                    on-the-fly mirrored expansion (atomic_add)
 *   rbk_spmv(a, A, x, b, y)          reduce-by-key SpMV over a *general* (row
 *                                    grouped) BCOO: y = a*A*x + b*y
 *   rbk_sym_spmv(a, A, x, b, y)      reduce-by-key symmetric SpMV, same product
 *                                    as sym_spmv but grouped per output row
 *   rbk_sym_spmv_dot(a, A, x, b, y, d_dot, count, capacity, stream)
 *                                    y = a*A_sym*x + b*y *and* d_dot = x^T(a*A_sym*x)
 *                                    in one pass; the triplet count is read on
 *                                    the device and the grid is sized by
 *                                    `triplet_capacity`, so a captured graph
 *                                    stays valid while nnz varies within the
 *                                    reserved capacity
 *   cpu_sym_spmv(a, A, x, b, y)      debug fallback (host, very slow)
 *
 * The symmetric convention (extracted from the reference kernels and preserved
 * exactly — cookbook §7.12):
 *
 *   * the BCOO stores the upper triangle (`col >= row` conceptually, one entry
 *     per (row,col) of the block matrix);
 *   * an entry (i, j, B) with **i == j** contributes `a * B * x_j` to `y_i`
 *     exactly once (the diagonal is neither halved nor doubled, and it has no
 *     mirrored copy);
 *   * an entry (i, j, B) with **i != j** contributes `a * B * x_j` to `y_i`
 *     *and* `a * B^T * x_i` to `y_j` (mirrored, transposed).  Several entries
 *     share an output row, which is what the reduce-by-key variants group;
 *   * duplicates of the same (i, j) are independent entries and simply sum
 *     (the reference accumulates them with atomic_add / a segmented fold);
 *   * `b` is applied to `y` **first** (`y = b*y`, or a zero fill when `b == 0`),
 *     and rows with no entry keep that scaled value;
 *   * `rbk_sym_spmv_dot` folds the *dot* contribution with the same convention:
 *     `a * x_i^T B x_i` for a diagonal entry, `2*a * x_i^T B x_j` for an
 *     off-diagonal one — i.e. `x^T (a*A_sym*x)`, independent of `b`.
 *
 * `rbk_spmv` is the *general* variant: it applies the stored entries as a plain
 * COO matrix (a symmetric upper-packed input therefore yields
 * `a*A_upper*x + b*y`, not the symmetric product) and assumes the entries of one
 * row are consecutive (the converter's sorted output).
 *
 * Deviations (each with its equivalence argument)
 * ----------------------------------------------
 *  * The reference's `rbk_*` kernels broadcast a warp segment's head
 *    (`cub::ShuffleIndex<32>` + the `__fns` trick) and let either the head
 *    (plain add) or a cross-warp lane (`atomic_add`) write.  The port keeps the
 *    identical grouping — a warp-level head-segmented fold over consecutive
 *    entries with the same output row (`detail::seg_fold_*`, the same
 *    Hillis-Steele transcription T3.2 uses for `FastSegmentalReduce`) — and has
 *    the head lane write with `atomic_add`.  The reference's non-atomic branch
 *    only applies when a row's entries all sit inside one warp, where no other
 *    lane writes that row, so `atomic_add` and the plain add are the same
 *    operation on the same value; for a row that crosses a warp boundary the
 *    reference is atomic as well.  The reduction *semantics* are therefore
 *    unchanged, and only the association order of a float sum can differ
 *    (inherent to the reference's cross-warp path too).
 *  * The four `y`-writing variants accumulate `y` with `atomic_add`, exactly like
 *    the reference.  When several warps/blocks contribute to the same output row
 *    the *order* of those atomics is not fixed by the programming model, so the
 *    last bits of `y` can differ between two runs of the same input (and between
 *    two launch shapes) — the reference has the same property, and the R6
 *    determinism requirement is met by the *reductions* (`dot`/`norm` and the
 *    fused `d_dot`, which are atomic-free two-pass folds).  See cookbook §7.12.
 *  * Device indices are unchecked (RULE M2), so every kernel clamps its
 *    triplet index to the *capacity* the caller reserved and predicates the
 *    contribution on `i < count`; that is what makes the
 *    "grid sized by capacity, count read on device" contract safe.
 *  * `cpu_sym_spmv` builds the symmetric expansion directly in host code
 *    (duplicates accumulated as they are visited) instead of going through
 *    `Eigen::SparseMatrix::setFromTriplets`; the expanded product is the same,
 *    up to the association order of the row sums.
 *********************************************************************/
#include <luisa_device/linear/blas.h>
#include <luisa_device/linear/views.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

namespace uipc::backend::luisa::luisa_device::linalg
{
/**
 * @brief `y = a*A*x + b*y` for the upper-packed symmetric / general block BCOO.
 *
 * @tparam T the scalar type (`double`; `float` is instantiated as well)
 * @tparam N the block size (`N x N` blocks; the reference is `N == 3`)
 */
template <typename T = double, int N = 3>
class UIPC_LUISA_DEVICE_API LSpmv
{
  public:
    static_assert(N >= 1, "LSpmv: block size must be positive");

    using ScalarT       = T;
    static constexpr int block_size = N;
    using MatrixView    = LCTripletMatrixView<T, N, N>;
    using VectorView    = LCDenseVectorView<T>;
    using MutVectorView = LDenseVectorView<T>;
    /// `cuda_tool::VarView<Float>` — a single-element device buffer.
    using DotView = LBufferView<T>;
    /// `cuda_tool::CDense<IndexT>` — the device-side triplet count.
    using CountView = LCBufferView<std::int32_t>;

    /// y = a * A_sym * x + b * y (on-the-fly mirrored expansion, atomic).
    void sym_spmv(T a, MatrixView A, VectorView x, T b, MutVectorView y,
                  LStream& stream = default_blas_stream());

    /// y = a * A * x + b * y for a general (row-grouped) BCOO (reduce by key).
    void rbk_spmv(T a, MatrixView A, VectorView x, T b, MutVectorView y,
                  LStream& stream = default_blas_stream());

    /// y = a * A_sym * x + b * y, grouped per output row (reduce by key).
    void rbk_sym_spmv(T a, MatrixView A, VectorView x, T b, MutVectorView y,
                      LStream& stream = default_blas_stream());

    /// Fused `y = a*A_sym*x + b*y` and `d_dot = x^T (a*A_sym*x)`.
    ///
    /// The triplet count is read *on the device* (through `d_triplet_count`) and
    /// the grid is sized by `triplet_capacity`, so the launch shape does not
    /// change when the matrix nnz varies within the reserved capacity (the
    /// property a captured CUDA graph relies on).
    void rbk_sym_spmv_dot(T a, MatrixView A, VectorView x, T b, MutVectorView y, DotView d_dot,
                          CountView d_triplet_count, std::size_t triplet_capacity,
                          LStream& stream = default_blas_stream());

    /// Debug fallback: host-side symmetric SpMV (very slow — use `sym_spmv`).
    void cpu_sym_spmv(T a, MatrixView A, VectorView x, T b, MutVectorView y);

  private:
    /// `y = b*y`, or a zero fill when `b == 0` (the reference's
    /// `Spmv_scale_y_kernel` / `BufferLaunch().fill(y, 0)`).  Shared by every
    /// variant.
    void scale_or_zero(T b, MutVectorView y, LStream& stream);

    /// Fused-dot block partials + the allocations a growth retired (cookbook §7.10
    /// `detail::carve`): the second pass is a deterministic fold (R6), and a grown
    /// scratch must not invalidate a handle an already-recorded command uses.
    LDeviceBuffer<T>                   m_dot_partials;
    std::vector<std::shared_ptr<void>> m_dot_retired;
};

/// The reference's spelling (`cuda_tool::Spmv`, i.e. `Float` = double, block 3).
using Spmv = LSpmv<double, 3>;
}// namespace uipc::backend::luisa::luisa_device::linalg

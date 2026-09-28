#pragma once
/********************************************************************
 * @file matrix_assembler.h
 * @brief T5.2 — DoubletVectorAssembler / TripletMatrixAssembler (port of
 *        src/backends/cuda/utils/matrix_assembler.h).
 *
 * The reference's two assemblers are UIPC_GENERIC helper classes that map an
 * element-local gradient/Hessian (a `Vector<N*SegmentDim>` / a
 * `Matrix<N*BlockDim, N*BlockDim>`) onto the global doublet/triplet rows of a
 * `cuda_tool::DoubletVectorViewer` / `TripletMatrixViewer`.  The port is generic
 * over the *view type* (`lla::DoubletVectorViewT<false, T, N, Storage>` /
 * `lla::TripletMatrixViewT<false, T, M, N, Storage>`), so one implementation
 * serves the device (`storage::device`, built inside a kernel from its buffer
 * arguments -- cookbook §7.9 rule 2) and host evaluation (`storage::memory`,
 * used by the unit tests).
 *
 * Mapping / deviations (all reported in the T5.2 report):
 *   * every `UIPC_KERNEL_ASSERT` is kept *in place*: a `lcrt::device_assert` for
 *     the device flavour, a host check (`lla::detail::kernel_assert`) for the
 *     other flavours;
 *   * `TripletMatrixAssembler::half_block`'s `upper_LR` cannot branch on device
 *     data (RULE M2: `l_indices(I) + offset.x < r_indices(J) + offset.y` is a
 *     *device* predicate).  The port evaluates both candidate blocks (host
 *     constant indices) and *selects* the block and the two global indices
 *     element-wise -- a select passes its chosen operand through untouched, so
 *     the written (row, col, block) triples are exactly the reference's;
 *   * the reference's `offset++` becomes `m_I + counter` (the port's device index
 *     type is a DSL value, which has no compound assignment), and the assembler's
 *     index argument is normalised with the view's `Storage::as_index`, so a
 *     kernel may pass `dispatch_x()` directly;
 *   * `zero_out` and the `ignore`-flavoured `write` overloads of the reference
 *     are **not** ported: no 3D-volumetric constitution uses them (they first
 *     appear in the constraint/contact code) and their reference bodies branch on
 *     device data (`if(ignore(ii)) G.setZero();`), which needs its own
 *     select-based formulation.  They arrive with their first consumer.
 ********************************************************************/
#include <luisa_device/linear/linear.h>
#include <luisa_device/math/math.h>

namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

namespace matrix_assembler_detail
{
/**
 * @brief `pred ? when_true : when_false` for either a host or a DSL scalar.
 *
 * `lps::select` is host-scalar-only on purpose -- LC's own `select` has a
 * custom-struct overload that would also match a container, which is why the
 * numerics layer refuses DSL operands (cookbook §7.6 trap 3) -- so the port
 * dispatches explicitly instead of relying on unqualified lookup + ADL (a nested
 * `ProxyRange` class is not inside this namespace, so a using-declaration here
 * would not be visible there anyway; measured C3861).
 */
template <typename F, typename T, typename P>
[[nodiscard]] inline auto select_scalar(const F& f, const T& t, const P& p) noexcept
{
    if constexpr(lps::is_dsl_value_v<F> || lps::is_dsl_value_v<T> || lps::is_dsl_value_v<P>)
        return lcrt::select(f, t, p);  // `p ? t : f` (LC's own order)
    else
        return lps::select(f, t, p);
}

/**
 * @brief Element-wise container select: `pred ? when_true : when_false`.
 *
 * The numerics layer's `select` deliberately refuses containers (LC's own
 * `select` has a custom-struct overload that would also match a container and
 * silently build a `Var<LVec<...>>`), so a container select is per element.
 */
template <typename M, typename P>
[[nodiscard]] inline M select_mat(const M& when_false, const M& when_true, const P& pred) noexcept
{
    using T = lps::mat_element_t<M>;
    return lps::make_mat<T, lps::mat_rows_v<M>, lps::mat_cols_v<M>>(
        [&](int r, int c)
        { return select_scalar(when_false(r, c), when_true(r, c), pred); });
}
}  // namespace matrix_assembler_detail

/// A range of `N` consecutive doublets of a `DoubletVectorView` (the reference's
/// `DoubletVectorAssembler::ProxyRange`).
template <typename ViewT>
class DoubletVectorAssembler
{
  public:
    using View          = ViewT;
    using Storage       = typename View::storage_type;
    using index_type    = typename View::index_type;
    using ValueT        = typename View::ValueT;  // LVec<T, SegmentDim>
    static constexpr int SegmentDim = View::segment_dim;
    static_assert(SegmentDim > 1,
                  "DoubletVectorAssembler: the reference's SegmentDim == 1 flavour is unused by the "
                  "3D constitutions (call the view's operator() directly instead)");
    using element_type = lps::vec_element_t<ValueT>;

    using ElementVector = lps::LVec<element_type, SegmentDim>;

    template <int N>
        requires(N >= 1)
    class ProxyRange
    {
      public:
        using SegmentVector = lps::LVec<element_type, N * SegmentDim>;

        template <typename I>
        ProxyRange(const DoubletVectorAssembler& assembler, I&& index)
            : m_assembler(assembler)
            , m_I(Storage::as_index(std::forward<I>(index)))
        {
            // reference: UIPC_KERNEL_ASSERT(I + N <= doublet_count(), ...)
            if constexpr(Storage::is_device)
            {
                lcrt::device_assert(m_I + N <= m_assembler.m_doublet.doublet_count(),
                                    "Doublet out of range");
            }
            else
            {
                lla::detail::kernel_assert(m_I + N <= m_assembler.m_doublet.doublet_count(),
                                           "Doublet out of range");
            }
        }

        /// Reference: `write(const Eigen::Vector<IndexT, N>& indices, const SegmentVector& value)`
        /// The index vector's element type is generic (a host int on the host path,
        /// a DSL scalar inside a kernel).
        template <typename IVec>
            requires(lps::is_vec_like_v<IVec> && lps::vec_dim_v<IVec> == N)
        void write(const IVec& indices, const SegmentVector& value) &&
        {
            for(int ii = 0; ii < N; ++ii)
            {
                ElementVector G = value.segment<SegmentDim>(ii * SegmentDim);
                m_assembler.m_doublet(m_I + ii).write(indices(ii), G);
            }
        }

      private:
        const DoubletVectorAssembler& m_assembler;
        index_type                    m_I;
    };

    /// Reference: `template <int N> ProxyRange<N> segment(IndexT I)`
    template <int N, typename I>
    [[nodiscard]] ProxyRange<N> segment(I&& index) const
    {
        return ProxyRange<N>(*this, std::forward<I>(index));
    }
    /// Reference: `ProxyRange<1> operator()(IndexT I)`
    template <typename I>
    [[nodiscard]] ProxyRange<1> operator()(I&& index) const
    {
        return ProxyRange<1>(*this, std::forward<I>(index));
    }

    explicit DoubletVectorAssembler(ViewT doublet)
        : m_doublet(doublet)
    {
    }

  private:
    View m_doublet;
};

template <typename ViewT>
DoubletVectorAssembler(ViewT) -> DoubletVectorAssembler<ViewT>;

/// A range of `N x N` (or `N (N + 1) / 2`) consecutive triplets of a
/// `TripletMatrixView` (the reference's `TripletMatrixAssembler`).
template <typename ViewT>
class TripletMatrixAssembler
{
  public:
    using View      = ViewT;
    using Storage   = typename View::storage_type;
    using index_type = typename View::index_type;
    using ValueT    = typename View::ValueT;  // LMat<T, BlockDim, BlockDim>
    static constexpr int BlockDim = View::block_rows;
    static_assert(View::block_rows == View::block_cols,
                  "TripletMatrixAssembler: the reference only assembles square blocks");
    using element_type = lps::mat_element_t<ValueT>;

    using ElementMatrix = lps::LMat<element_type, BlockDim, BlockDim>;

    template <int N>
        requires(N >= 1)
    class ProxyRange
    {
      public:
        using BlockMatrix = lps::LMat<element_type, N * BlockDim, N * BlockDim>;

        template <typename I>
        ProxyRange(const TripletMatrixAssembler& assembler, I&& index)
            : m_assembler(assembler)
            , m_I(Storage::as_index(std::forward<I>(index)))
        {
            // reference: UIPC_KERNEL_ASSERT(I + (N * N) <= triplet_count(), ...)
            if constexpr(Storage::is_device)
            {
                lcrt::device_assert(m_I + N * N <= m_assembler.m_triplet.triplet_count(),
                                    "Triplet out of range");
            }
            else
            {
                lla::detail::kernel_assert(m_I + N * N <= m_assembler.m_triplet.triplet_count(),
                                           "Triplet out of range");
            }
        }

        template <typename LIVec, typename RIVec>
            requires(lps::is_vec_like_v<LIVec> && lps::vec_dim_v<LIVec> == N
                     && lps::is_vec_like_v<RIVec> && lps::vec_dim_v<RIVec> == N)
        void write(const LIVec& l_indices, const RIVec& r_indices, const BlockMatrix& value) &&
        {
            auto& tri     = m_assembler.m_triplet;
            auto  counter = 0;
            for(int ii = 0; ii < N; ++ii)
                for(int jj = 0; jj < N; ++jj)
                {
                    ElementMatrix H =
                        value.block<BlockDim, BlockDim>(ii * BlockDim, jj * BlockDim);
                    tri(m_I + counter++).write(l_indices(ii), r_indices(jj), H);
                }
        }

        template <typename IVec>
            requires(lps::is_vec_like_v<IVec> && lps::vec_dim_v<IVec> == N)
        void write(const IVec& indices, const BlockMatrix& value) &&
        {
            std::move(*this).write(indices, indices, value);
        }

      private:
        const TripletMatrixAssembler& m_assembler;
        index_type                    m_I;
    };

    /// Reference: `half_block<N>` -- only the upper triangle of the *global*
    /// matrix is written (`upper_LR`: a pair is swapped when its left index is not
    /// smaller than its right index in the global numbering).
    template <int N>
        requires(N >= 1)
    class ProxyRangeHalf
    {
      public:
        using BlockMatrix = lps::LMat<element_type, N * BlockDim, N * BlockDim>;

        template <typename I>
        ProxyRangeHalf(const TripletMatrixAssembler& assembler, I&& index)
            : m_assembler(assembler)
            , m_I(Storage::as_index(std::forward<I>(index)))
        {
            // reference: UIPC_KERNEL_ASSERT(I + (N * (N + 1)) / 2 <= triplet_count(), ...)
            if constexpr(Storage::is_device)
            {
                lcrt::device_assert(m_I + (N * (N + 1)) / 2
                                        <= m_assembler.m_triplet.triplet_count(),
                                    "Triplet out of range");
            }
            else
            {
                lla::detail::kernel_assert(m_I + (N * (N + 1)) / 2
                                               <= m_assembler.m_triplet.triplet_count(),
                                           "Triplet out of range");
            }
        }

        /// Reference: `write(const Eigen::Vector<IndexT, N>& indices, const BlockMatrix& value)`
        /// with the `upper_LR` swap rewritten as a select (see the file comment).
        template <typename IVec>
            requires(lps::is_vec_like_v<IVec> && lps::vec_dim_v<IVec> == N)
        void write(const IVec& indices, const BlockMatrix& value) &&
        {
            auto& tri = m_assembler.m_triplet;

            const auto submatrix_offset = tri.submatrix_offset();
            if constexpr(!Storage::is_device)
            {
                // reference: UIPC_KERNEL_ASSERT(submatrix_offset.x == submatrix_offset.y, ...)
                // (a host invariant: skipped on the device path, where both are DSL
                //  values and cannot drive host control flow)
                lla::detail::kernel_assert(submatrix_offset.x == submatrix_offset.y,
                                           "Symmetric assembly requires a square submatrix view");
            }

            auto counter = 0;
            for(int ii = 0; ii < N; ++ii)
                for(int jj = ii; jj < N; ++jj)
                {
                    auto l = indices(ii);
                    auto r = indices(jj);
                    // reference: `if (l_indices(I) + offset.x < r_indices(J) + offset.y)`
                    //            `{ ret.L = I; ret.R = J; } else { ret.L = J; ret.R = I; }`
                    auto keep = l + submatrix_offset.x < r + submatrix_offset.y;
                    // select(f, t, p) is p ? t : f (cookbook §7.6 trap 1)
                    auto row = matrix_assembler_detail::select_scalar(r, l, keep);
                    auto col = matrix_assembler_detail::select_scalar(l, r, keep);
                    auto H   = matrix_assembler_detail::select_mat(
                        value.block<BlockDim, BlockDim>(jj * BlockDim, ii * BlockDim),
                        value.block<BlockDim, BlockDim>(ii * BlockDim, jj * BlockDim),
                        keep);
                    tri(m_I + counter++).write(row, col, H);
                }
        }

      private:
        const TripletMatrixAssembler& m_assembler;
        index_type                    m_I;
    };

    /// Reference: `template <int M, int N> ProxyRange<N> block(IndexT I)` (M == N)
    template <int M, int N, typename I>
        requires(M == N)
    [[nodiscard]] ProxyRange<N> block(I&& index) const
    {
        return ProxyRange<N>(*this, std::forward<I>(index));
    }
    /// Reference: `template <int N> ProxyRangeHalf<N> half_block(IndexT I)`
    template <int N, typename I>
    [[nodiscard]] ProxyRangeHalf<N> half_block(I&& index) const
    {
        return ProxyRangeHalf<N>(*this, std::forward<I>(index));
    }
    /// Reference: `ProxyRange<1> operator()(IndexT I)`
    template <typename I>
    [[nodiscard]] ProxyRange<1> operator()(I&& index) const
    {
        return ProxyRange<1>(*this, std::forward<I>(index));
    }

    explicit TripletMatrixAssembler(ViewT triplet)
        : m_triplet(triplet)
    {
    }

  private:
    View m_triplet;
};

template <typename ViewT>
TripletMatrixAssembler(ViewT) -> TripletMatrixAssembler<ViewT>;
}  // namespace uipc::backend::luisa

#pragma once
#include <type_define.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
/**
 * @brief A helper class to unpack dense matrix blocks into triplet (COO) format sparse matrix.
 * 
 * This is the LuisaCompute refactored version of the original CUDA/muda implementation.
 * It uses LuisaCompute's buffer views and core types.
 * 
 * @tparam T The value type (float, double, etc.)
 * @tparam BlockDim The block dimension for blocked sparse matrices (1 for scalar)
 */
template <typename T, int BlockDim>
class TripletMatrixUnpacker
{
  public:
    /**
     * @brief Triplet entry structure for COO format
     */
    struct TripletEntry
    {
        IndexT row;
        IndexT col;
        Eigen::Matrix<T, BlockDim, BlockDim> value;
    };

    /**
     * @brief View type for accessing triplet entries in kernels
     */
    using TripletView = luisa::compute::BufferView<TripletEntry>;

    LUISA_GENERIC TripletMatrixUnpacker(TripletView triplets, IndexT triplet_count)
        : m_triplets(triplets)
        , m_triplet_count(triplet_count)
    {
    }

    template <int M, int N>
        requires(M >= 1 && N >= 1)
    class ProxyRange
    {
      public:
        LUISA_GENERIC ProxyRange(const TripletMatrixUnpacker& unpacker, IndexT I)
            : m_unpacker(unpacker)
            , m_I(I)
        {
            LUISA_ASSERT(I + (M * N) <= m_unpacker.m_triplet_count,
                        "Triplet out of range, I = {}, Count={} * {}, total={}",
                        I, M, N, m_unpacker.m_triplet_count);
        }

        /**
         * @brief Write a block matrix value for blocked sparse matrices (BlockDim > 1)
         * 
         * @param i Row index (block index)
         * @param j Column index (block index)
         * @param value The M*N block matrix value to write
         */
        LUISA_GENERIC void write(IndexT i,
                                IndexT j,
                                const Eigen::Matrix<T, BlockDim * M, BlockDim * N>& value)
            requires(BlockDim > 1 && (M > 1 || N > 1))
        {
            IndexT offset = m_I;
            for(IndexT ii = 0; ii < M; ++ii)
            {
                for(IndexT jj = 0; jj < N; ++jj)
                {
                    auto& entry = m_unpacker.m_triplets[offset];
                    entry.row = i + ii;
                    entry.col = j + jj;
                    entry.value = value.template block<BlockDim, BlockDim>(ii * BlockDim, jj * BlockDim);
                    ++offset;
                }
            }
        }

        /**
         * @brief Write a dense matrix value for scalar sparse matrices (BlockDim == 1)
         */
        LUISA_GENERIC void write(IndexT i, IndexT j, const Eigen::Matrix<T, M, N>& value)
            requires(BlockDim == 1 && (M > 1 || N > 1))
        {
            IndexT offset = m_I;
            for(IndexT ii = 0; ii < M; ++ii)
            {
                for(IndexT jj = 0; jj < N; ++jj)
                {
                    auto& entry = m_unpacker.m_triplets[offset];
                    entry.row = i + ii;
                    entry.col = j + jj;
                    entry.value(0, 0) = value(ii, jj);
                    ++offset;
                }
            }
        }

        /**
         * @brief Write a single block matrix value for blocked sparse matrices
         */
        LUISA_GENERIC void write(IndexT i, IndexT j, const Eigen::Matrix<T, BlockDim, BlockDim>& value)
            requires(BlockDim > 1 && M == 1 && N == 1)
        {
            auto& entry = m_unpacker.m_triplets[m_I];
            entry.row = i;
            entry.col = j;
            entry.value = value;
        }

        /**
         * @brief Write a scalar value for scalar sparse matrices
         */
        LUISA_GENERIC void write(IndexT i, IndexT j, const T& value)
            requires(BlockDim == 1 && M == 1 && N == 1)
        {
            auto& entry = m_unpacker.m_triplets[m_I];
            entry.row = i;
            entry.col = j;
            entry.value(0, 0) = value;
        }

      private:
        const TripletMatrixUnpacker& m_unpacker;
        IndexT                       m_I;
    };

    /**
     * @brief Proxy range for symmetric (half) matrix assembly
     * Only writes to the upper triangular part.
     */
    template <int N>
        requires(N >= 1)
    class ProxyRangeHalf
    {
      public:
        struct UpperIJ
        {
            IndexT dst_i;
            IndexT dst_j;
            IndexT ii;
            IndexT jj;
        };

        using BlockMatrix = Eigen::Matrix<T, N * BlockDim, N * BlockDim>;

        LUISA_GENERIC ProxyRangeHalf(const TripletMatrixUnpacker& unpacker, IndexT I)
            : m_unpacker(unpacker)
            , m_I(I)
        {
            LUISA_ASSERT(I + (N * (N + 1)) / 2 <= m_unpacker.m_triplet_count,
                        "Triplet out of range, I = {}, Count={} * {} / 2, total={}",
                        I, N, N, m_unpacker.m_triplet_count);
        }

        /**
         * @brief Only write to the upper triangular part of the global matrix.
         */
        LUISA_GENERIC void write(IndexT i, IndexT j, const BlockMatrix& value)
        {
            IndexT offset = m_I;
            for(IndexT ii = 0; ii < N; ++ii)
            {
                for(IndexT jj = ii; jj < N; ++jj)
                {
                    auto [dst_i, dst_j, ii_, jj_] = upper_ij(i, j, ii, jj);

                    auto& entry = m_unpacker.m_triplets[offset++];
                    entry.row = dst_i;
                    entry.col = dst_j;
                    entry.value = value.template block<BlockDim, BlockDim>(ii_ * BlockDim, jj_ * BlockDim);
                }
            }
        }

      private:
        LUISA_GENERIC UpperIJ upper_ij(const IndexT& i,
                                      const IndexT& j,
                                      const IndexT& ii,
                                      const IndexT& jj)
        {
            UpperIJ ret;
            // keep it in upper triangular in the global matrix
            auto dst_i = (i + ii);
            auto dst_j = (j + jj);
            if(dst_i < dst_j)
            {
                ret.dst_i = dst_i;
                ret.dst_j = dst_j;
                ret.ii    = ii;
                ret.jj    = jj;
            }
            else
            {
                ret.dst_i = dst_j;
                ret.dst_j = dst_i;
                ret.ii    = jj;
                ret.jj    = ii;
            }
            return ret;
        }

        const TripletMatrixUnpacker& m_unpacker;
        IndexT                       m_I;
    };

    /** 
     * @brief Take a range of [I, I + M * N) from the triplets.
     */
    template <int M, int N>
    LUISA_GENERIC ProxyRange<M, N> block(IndexT I) const
    {
        return ProxyRange<M, N>(*this, I);
    }

    /**
     * @brief Take a range of [I, I + N * (N+1) / 2) from the triplets for symmetric assembly.
     */
    template <int N>
    LUISA_GENERIC ProxyRangeHalf<N> half_block(IndexT I) const
    {
        return ProxyRangeHalf<N>(*this, I);
    }

    /**
     * @brief Take a range of [I, I + N) from the triplets.
     */
    template <int N>
    LUISA_GENERIC ProxyRange<N, 1> segment(IndexT I) const
    {
        return ProxyRange<N, 1>(*this, I);
    }

    /** 
     * @brief Take a range of [I, I + 1) from the triplets.
     */
    LUISA_GENERIC ProxyRange<1, 1> operator()(IndexT I) const
    {
        return ProxyRange<1, 1>(*this, I);
    }

    /**
     * @brief Get the triplet count
     */
    LUISA_GENERIC IndexT triplet_count() const
    {
        return m_triplet_count;
    }

    /**
     * @brief Get the triplet view
     */
    LUISA_GENERIC TripletView triplets() const
    {
        return m_triplets;
    }

  private:
    TripletView m_triplets;
    IndexT      m_triplet_count;
};

// CTAD helper
template <typename T, int BlockDim>
TripletMatrixUnpacker(luisa::compute::BufferView<typename TripletMatrixUnpacker<T, BlockDim>::TripletEntry>, IndexT)
    -> TripletMatrixUnpacker<T, BlockDim>;

}  // namespace uipc::backend::luisa

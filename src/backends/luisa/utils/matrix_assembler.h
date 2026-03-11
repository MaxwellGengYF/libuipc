#pragma once
#include <type_define.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
using namespace luisa;
using namespace luisa::compute;

template <int M, int N>
LC_GENERIC void zero_out(Vector<Float, M>& Vec, const Vector<IndexT, N>& zero_out_flag)
    requires(M % N == 0)
{
    constexpr int Segment = M / N;
    for(int i = 0; i < N; ++i)
    {
        if(zero_out_flag(i))
            Vec.template segment<Segment>(i * Segment).setZero();
    }
}

template <int M, int N>
LC_GENERIC void zero_out(Matrix<Float, M, M>& Mat, const Vector<IndexT, N>& zero_out_flag)
    requires(M % N == 0)
{
    constexpr int Segment = M / N;
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            if(zero_out_flag(i) || zero_out_flag(j))
                Mat.template block<Segment, Segment>(i * Segment, j * Segment).setZero();
        }
    }
}

template <typename T>
class DenseVectorAssembler
{
  public:
    LC_GENERIC DenseVectorAssembler(const BufferVar<T>& dense_buffer, uint offset, uint size)
        : m_dense_buffer(dense_buffer)
        , m_offset(offset)
        , m_size(size)
    {
    }

    template <int M, int N>
    LC_DEVICE void atomic_add(const Vector<IndexT, N>& indices,
                              const Vector<IndexT, N>& ignore,
                              const Vector<Float, M>&  G3N)
        requires(N >= 2 && M % N == 0)
    {
        constexpr int SegmentDim = M / N;
        using SegmentVector      = Vector<Float, SegmentDim>;
#pragma unroll
        for(int i = 0; i < N; ++i)
        {
            int dst = indices(i);
            if(ignore(i))
                continue;
            SegmentVector G = G3N.template segment<SegmentDim>(i * SegmentDim);

            // Atomic add to buffer at segment
            auto atomic_ref = m_dense_buffer->atomic(m_offset + dst * SegmentDim);
            for(int j = 0; j < SegmentDim; ++j)
            {
                atomic_ref.fetch_add(G[j]);
            }
        }
    }

    template <int M, int N>
    LC_DEVICE void atomic_add(const Vector<IndexT, N>& indices,
                              const Vector<Float, M>&  G3N)
        requires(N >= 2 && M % N == 0)
    {
        constexpr int SegmentDim = M / N;
        using SegmentVector      = Vector<Float, SegmentDim>;
#pragma unroll
        for(int i = 0; i < N; ++i)
        {
            int           dst = indices(i);
            SegmentVector G = G3N.template segment<SegmentDim>(i * SegmentDim);

            // Atomic add to buffer at segment
            auto atomic_ref = m_dense_buffer->atomic(m_offset + dst * SegmentDim);
            for(int j = 0; j < SegmentDim; ++j)
            {
                atomic_ref.fetch_add(G[j]);
            }
        }
    }

  private:
    BufferVar<T> m_dense_buffer;
    uint         m_offset;
    uint         m_size;
};

// CTAD
template <typename T>
DenseVectorAssembler(const BufferVar<T>&, uint, uint) -> DenseVectorAssembler<T>;

template <typename T, int SegmentDim>
class DoubletVectorAssembler
{
  public:
    using ElementVector = Eigen::Matrix<T, SegmentDim, 1>;

    LC_GENERIC DoubletVectorAssembler(const BufferVar<T>& doublet_buffer,
                                      const BufferVar<IndexT>& index_buffer,
                                      uint offset,
                                      uint doublet_count)
        : m_doublet_buffer(doublet_buffer)
        , m_index_buffer(index_buffer)
        , m_offset(offset)
        , m_doublet_count(doublet_count)
    {
    }

    template <int N>
        requires(N >= 1)
    class ProxyRange
    {
      public:
        using SegmentVector = Eigen::Vector<T, N * SegmentDim>;

        LC_GENERIC ProxyRange(DoubletVectorAssembler& assembler, IndexT I)
            : m_assembler(assembler)
            , m_I(I)
        {
            LC_ASSERT(I + N <= m_assembler.m_doublet_count,
                      "Doublet out of range");
        }

        LC_GENERIC void write(const Eigen::Vector<IndexT, N>& indices,
                              const SegmentVector&            value)
            requires(N > 1)
        {
            IndexT offset = m_I;
            for(IndexT ii = 0; ii < N; ++ii)
            {
                ElementVector G = value.template segment<SegmentDim>(ii * SegmentDim);
                // Write index and value
                m_assembler.m_index_buffer->write(m_assembler.m_offset + offset, indices(ii));
                for(int j = 0; j < SegmentDim; ++j)
                {
                    m_assembler.m_doublet_buffer->write(m_assembler.m_offset + offset * SegmentDim + j, G[j]);
                }
                offset++;
            }
        }

        LC_GENERIC void write(const Eigen::Vector<IndexT, N>& indices,
                              const Eigen::Vector<IndexT, N>& ignore,
                              const SegmentVector&            value)
            requires(N > 1)
        {
            IndexT offset = m_I;
            for(IndexT ii = 0; ii < N; ++ii)
            {
                ElementVector G = value.template segment<SegmentDim>(ii * SegmentDim);
                if(ignore(ii))
                    G.setZero();
                // Write index and value
                m_assembler.m_index_buffer->write(m_assembler.m_offset + offset, indices(ii));
                for(int j = 0; j < SegmentDim; ++j)
                {
                    m_assembler.m_doublet_buffer->write(m_assembler.m_offset + offset * SegmentDim + j, G[j]);
                }
                offset++;
            }
        }

        LC_GENERIC void write(IndexT indices, const ElementVector& value)
            requires(N == 1)
        {
            // Write index and value
            m_assembler.m_index_buffer->write(m_assembler.m_offset + m_I, indices);
            for(int j = 0; j < SegmentDim; ++j)
            {
                m_assembler.m_doublet_buffer->write(m_assembler.m_offset + m_I * SegmentDim + j, value[j]);
            }
        }

        LC_GENERIC void write(IndexT indices, IndexT ignore, const ElementVector& value)
            requires(N == 1)
        {
            ElementVector G = value;
            if(ignore)
                G.setZero();
            // Write index and value
            m_assembler.m_index_buffer->write(m_assembler.m_offset + m_I, indices);
            for(int j = 0; j < SegmentDim; ++j)
            {
                m_assembler.m_doublet_buffer->write(m_assembler.m_offset + m_I * SegmentDim + j, G[j]);
            }
        }

      private:
        DoubletVectorAssembler& m_assembler;
        IndexT                  m_I;
    };


    /** 
     * @brief Take a range of [I, I + N) from the doublets.
     */
    template <int N>
    LC_GENERIC ProxyRange<N> segment(IndexT I)
    {
        return ProxyRange<N>(*this, I);
    }

    /** 
     * @brief Take a range of [I, I + 1) from the doublets.
     */
    LC_GENERIC ProxyRange<1> operator()(IndexT I)
    {
        return ProxyRange<1>(*this, I);
    }

  private:
    BufferVar<T>      m_doublet_buffer;
    BufferVar<IndexT> m_index_buffer;
    uint              m_offset;
    uint              m_doublet_count;
};

// CTAD
template <typename T, int SegmentDim>
DoubletVectorAssembler(const BufferVar<T>&, const BufferVar<IndexT>&, uint, uint)
    -> DoubletVectorAssembler<T, SegmentDim>;


template <typename T, int BlockDim>
class TripletMatrixAssembler
{
  public:
    using ElementMatrix = Eigen::Matrix<T, BlockDim, BlockDim>;


    LC_GENERIC TripletMatrixAssembler(const BufferVar<T>& value_buffer,
                                      const BufferVar<IndexT>& row_index_buffer,
                                      const BufferVar<IndexT>& col_index_buffer,
                                      uint offset,
                                      uint triplet_count)
        : m_value_buffer(value_buffer)
        , m_row_index_buffer(row_index_buffer)
        , m_col_index_buffer(col_index_buffer)
        , m_offset(offset)
        , m_triplet_count(triplet_count)
    {
    }

    template <int N>
        requires(N >= 1)
    class ProxyRange
    {
      public:
        using BlockMatrix = Eigen::Matrix<T, N * BlockDim, N * BlockDim>;

        LC_GENERIC ProxyRange(TripletMatrixAssembler& assembler, IndexT I)
            : m_assembler(assembler)
            , m_I(I)
        {
            LC_ASSERT(I + (N * N) <= m_assembler.m_triplet_count,
                      "Triplet out of range");
        }


        LC_GENERIC void write(const Eigen::Vector<IndexT, N>& l_indices,
                              const Eigen::Vector<IndexT, N>& r_indices,
                              const BlockMatrix&              value)
            requires(N > 1)
        {
            IndexT offset = m_I;
            for(IndexT ii = 0; ii < N; ++ii)
            {
                for(IndexT jj = 0; jj < N; ++jj)
                {
                    ElementMatrix H =
                        value.template block<BlockDim, BlockDim>(ii * BlockDim, jj * BlockDim);

                    m_assembler.m_row_index_buffer->write(m_assembler.m_offset + offset, l_indices(ii));
                    m_assembler.m_col_index_buffer->write(m_assembler.m_offset + offset, r_indices(jj));
                    for(int i = 0; i < BlockDim; ++i)
                    {
                        for(int j = 0; j < BlockDim; ++j)
                        {
                            m_assembler.m_value_buffer->write(
                                m_assembler.m_offset + offset * BlockDim * BlockDim + i * BlockDim + j, 
                                H(i, j));
                        }
                    }
                    offset++;
                }
            }
        }

        LC_GENERIC void write(const Eigen::Vector<IndexT, N>& l_indices,
                              const Eigen::Vector<int8_t, N>& l_ignore,
                              const Eigen::Vector<IndexT, N>& r_indices,
                              const Eigen::Vector<int8_t, N>& r_ignore,
                              const BlockMatrix&              value)
            requires(N > 1)
        {
            IndexT offset = m_I;
            for(IndexT ii = 0; ii < N; ++ii)
            {
                for(IndexT jj = 0; jj < N; ++jj)
                {
                    ElementMatrix H;
                    if(l_ignore(ii) || r_ignore(jj))
                        H.setZero();
                    else
                        H = value.template block<BlockDim, BlockDim>(ii * BlockDim,
                                                                     jj * BlockDim);

                    m_assembler.m_row_index_buffer->write(m_assembler.m_offset + offset, l_indices(ii));
                    m_assembler.m_col_index_buffer->write(m_assembler.m_offset + offset, r_indices(jj));
                    for(int i = 0; i < BlockDim; ++i)
                    {
                        for(int j = 0; j < BlockDim; ++j)
                        {
                            m_assembler.m_value_buffer->write(
                                m_assembler.m_offset + offset * BlockDim * BlockDim + i * BlockDim + j, 
                                H(i, j));
                        }
                    }
                    offset++;
                }
            }
        }


        LC_GENERIC void write(const Eigen::Vector<IndexT, N>& indices, const BlockMatrix& value)
            requires(N > 1)
        {
            write(indices, indices, value);
        }


        LC_GENERIC void write(const Eigen::Vector<IndexT, N>& indices,
                              const Eigen::Vector<int8_t, N>  ignore,
                              const BlockMatrix&              value)
            requires(N > 1)
        {
            write(indices, ignore, indices, ignore, value);
        }

        LC_GENERIC void write(IndexT indices, const ElementMatrix& value)
            requires(N == 1)
        {
            IndexT offset = m_I;
            m_assembler.m_row_index_buffer->write(m_assembler.m_offset + offset, indices);
            m_assembler.m_col_index_buffer->write(m_assembler.m_offset + offset, indices);
            for(int i = 0; i < BlockDim; ++i)
            {
                for(int j = 0; j < BlockDim; ++j)
                {
                    m_assembler.m_value_buffer->write(
                        m_assembler.m_offset + offset * BlockDim * BlockDim + i * BlockDim + j, 
                        value(i, j));
                }
            }
        }

        LC_GENERIC void write(IndexT indices, IndexT ignore, const ElementMatrix& value)
            requires(N == 1)
        {
            IndexT        offset = m_I;
            ElementMatrix H      = value;
            if(ignore)
                H.setZero();
            m_assembler.m_row_index_buffer->write(m_assembler.m_offset + offset, indices);
            m_assembler.m_col_index_buffer->write(m_assembler.m_offset + offset, indices);
            for(int i = 0; i < BlockDim; ++i)
            {
                for(int j = 0; j < BlockDim; ++j)
                {
                    m_assembler.m_value_buffer->write(
                        m_assembler.m_offset + offset * BlockDim * BlockDim + i * BlockDim + j, 
                        H(i, j));
                }
            }
        }

      private:
        TripletMatrixAssembler& m_assembler;
        IndexT                  m_I;
    };

    template <int N>
        requires(N >= 1)
    class ProxyRangeHalf
    {
      public:
        struct UpperLR
        {
            IndexT L;
            IndexT R;
        };

        using BlockMatrix = Eigen::Matrix<T, N * BlockDim, N * BlockDim>;
        LC_GENERIC ProxyRangeHalf(const TripletMatrixAssembler& assembler, IndexT I)
            : m_assembler(assembler)
            , m_I(I)
        {
            LC_ASSERT(I + (N * (N + 1)) / 2 <= m_assembler.m_triplet_count,
                      "Triplet out of range for half block");
        }

        /**
         * @brief Only write to the upper triangular part of the global matrix. (not the submatrix)
         */
        LC_GENERIC void write(const Eigen::Vector<IndexT, N>& indices, const BlockMatrix& value)
        {
            IndexT offset = m_I;
            for(IndexT ii = 0; ii < N; ++ii)
            {
                for(IndexT jj = ii; jj < N; ++jj)
                {

                    auto [L, R] = upper_LR(indices, ii, jj);

                    ElementMatrix H =
                        value.template block<BlockDim, BlockDim>(L * BlockDim, R * BlockDim);

                    m_assembler.m_row_index_buffer->write(m_assembler.m_offset + offset, indices(L));
                    m_assembler.m_col_index_buffer->write(m_assembler.m_offset + offset, indices(R));
                    for(int i = 0; i < BlockDim; ++i)
                    {
                        for(int j = 0; j < BlockDim; ++j)
                        {
                            m_assembler.m_value_buffer->write(
                                m_assembler.m_offset + offset * BlockDim * BlockDim + i * BlockDim + j, 
                                H(i, j));
                        }
                    }
                    offset++;
                }
            }
        }

        /**
         * @brief Only write to the upper triangular part of the global matrix. (not the submatrix)
         */
        LC_GENERIC void write(const Eigen::Vector<IndexT, N>& l_indices,
                              const Eigen::Vector<IndexT, N>& r_indices,
                              const BlockMatrix&              value)
        {
            IndexT offset = m_I;
            for(IndexT ii = 0; ii < N; ++ii)
            {
                for(IndexT jj = ii; jj < N; ++jj)
                {

                    auto [L, R] = upper_LR(l_indices, r_indices, ii, jj);

                    ElementMatrix H =
                        value.template block<BlockDim, BlockDim>(L * BlockDim, R * BlockDim);

                    m_assembler.m_row_index_buffer->write(m_assembler.m_offset + offset, l_indices(L));
                    m_assembler.m_col_index_buffer->write(m_assembler.m_offset + offset, r_indices(R));
                    for(int i = 0; i < BlockDim; ++i)
                    {
                        for(int j = 0; j < BlockDim; ++j)
                        {
                            m_assembler.m_value_buffer->write(
                                m_assembler.m_offset + offset * BlockDim * BlockDim + i * BlockDim + j, 
                                H(i, j));
                        }
                    }
                    offset++;
                }
            }
        }

        /**
         * @brief Only write to the upper triangular part of the global matrix. (not the submatrix)
         * 
         * Constraints: if either side is ignored, write zero.
         */
        LC_GENERIC void write(const Eigen::Vector<IndexT, N>& indices,
                              const Eigen::Vector<int8_t, N>  ignore,
                              const BlockMatrix&              value)
        {
            IndexT offset = m_I;
            for(IndexT ii = 0; ii < N; ++ii)
            {
                for(IndexT jj = ii; jj < N; ++jj)
                {
                    auto [L, R] = upper_LR(indices, ii, jj);

                    ElementMatrix H;
                    if(ignore(L) || ignore(R))
                    {
                        H.setZero();
                    }
                    else
                    {
                        H = value.template block<BlockDim, BlockDim>(L * BlockDim, R * BlockDim);
                    }
                    m_assembler.m_row_index_buffer->write(m_assembler.m_offset + offset, indices(L));
                    m_assembler.m_col_index_buffer->write(m_assembler.m_offset + offset, indices(R));
                    for(int i = 0; i < BlockDim; ++i)
                    {
                        for(int j = 0; j < BlockDim; ++j)
                        {
                            m_assembler.m_value_buffer->write(
                                m_assembler.m_offset + offset * BlockDim * BlockDim + i * BlockDim + j, 
                                H(i, j));
                        }
                    }
                    offset++;
                }
            }
        }

      private:
        LC_GENERIC UpperLR upper_LR(const Eigen::Vector<IndexT, N>& indices,
                                      const IndexT&                   I,
                                      const IndexT&                   J)
        {
            return upper_LR(indices, indices, I, J);
        }

        LC_GENERIC UpperLR upper_LR(const Eigen::Vector<IndexT, N>& l_indices,
                                      const Eigen::Vector<IndexT, N>& r_indices,
                                      const IndexT&                   I,
                                      const IndexT&                   J)
        {
            // For simplicity, we assume submatrix_offset is (0,0)
            // In a full implementation, this would need to be passed in
            UpperLR ret;
            // keep it in upper triangular in the global matrix (not the submatrix)
            if(l_indices(I) < r_indices(J))
            {
                ret.L = I;
                ret.R = J;
            }
            else
            {
                ret.L = J;
                ret.R = I;
            }
            return ret;
        }

        const TripletMatrixAssembler& m_assembler;
        IndexT                        m_I;
    };


    /** 
     * @brief Take a range of [I, I + N * N) from the triplets.
     */
    template <int M, int N>
    LC_GENERIC ProxyRange<N> block(IndexT I)
        requires(M == N)
    {
        return ProxyRange<N>(*this, I);
    }

    /** 
     * @brief Take a range of [I, I + N * (N + 1) / 2) from the triplets.
     */
    template <int N>
    LC_GENERIC ProxyRangeHalf<N> half_block(IndexT I)
    {
        return ProxyRangeHalf<N>(*this, I);
    }

    /** 
     * @brief Take a range of [I, I + 1) from the triplets.
     */
    LC_GENERIC ProxyRange<1> operator()(IndexT I)
    {
        return ProxyRange<1>(*this, I);
    }

  private:
    BufferVar<T>      m_value_buffer;
    BufferVar<IndexT> m_row_index_buffer;
    BufferVar<IndexT> m_col_index_buffer;
    uint              m_offset;
    uint              m_triplet_count;
};

// CTAD
template <typename T, int BlockDim>
TripletMatrixAssembler(const BufferVar<T>&, const BufferVar<IndexT>&, const BufferVar<IndexT>&, uint, uint)
    -> TripletMatrixAssembler<T, BlockDim>;
}  // namespace uipc::backend::luisa

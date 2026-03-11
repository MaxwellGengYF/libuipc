#pragma once
#include <luisa/luisa-compute.h>
#include <Eigen/Core>

namespace uipc::backend::luisa
{
/**
 * @brief Fast segmental reduce implementation using Luisa Compute
 * 
 * This class provides segmented reduction operations on GPU using Luisa Compute.
 * It reduces segments of data based on a key array, where consecutive elements
 * with the same key belong to the same segment.
 * 
 * This implementation uses warp-level shuffle operations for efficient 
 * segmented reduction without explicit shared memory management.
 * 
 * @tparam BlockSize The CUDA block size (default: 128)
 * @tparam WarpSize The CUDA warp size (default: 32)
 */
template <int BlockSize = 128, int WarpSize = 32>
class FastSegmentalReduce
{
    struct Flags
    {
        union
        {
            struct
            {
                luisa::uint is_head;
                luisa::uint is_cross_warp;
                luisa::uint is_valid;
            };
            luisa::uint flags;
        };

        void b2i() noexcept
        {
            is_head       = is_head ? 1u : 0u;
            is_cross_warp = is_cross_warp ? 1u : 0u;
            is_valid      = is_valid ? 1u : 0u;
        }
    };

  public:
    /**
     * @brief Construct a FastSegmentalReduce object
     * 
     * @param device The Luisa Compute device
     * @param stream The stream for kernel execution
     */
    FastSegmentalReduce(luisa::compute::Device& device, luisa::compute::Stream& stream);

    /**
     * @brief Perform segmental reduction using offset and input buffers
     * 
     * Example:
     * when ReduceOp = std::plus
     * dst = [0, 1, 1, 2, 2, 2]
     * in  = [1, 1, 1, 1, 1, 1]
     * out = [1, 2, 3]
     * 
     * @tparam T The element type
     * @tparam ReduceOp The reduction operation type (default: std::plus<T>)
     * @param dst The destination key buffer (maps input indices to segment indices)
     * @param in The input buffer
     * @param out The output buffer for reduced values
     * @param op The reduction operation
     * @return Reference to this object for method chaining
     */
    template <typename T, typename ReduceOp = std::plus<T>>
    FastSegmentalReduce& reduce(luisa::compute::BufferView<int> dst,
                                luisa::compute::BufferView<T>   in,
                                luisa::compute::BufferView<T>   out,
                                ReduceOp op = ReduceOp{});

    /**
     * @brief Perform segmental reduction on Eigen matrices
     * 
     * @tparam T The scalar type of the matrix
     * @tparam M The number of rows in the matrix
     * @tparam N The number of columns in the matrix
     * @tparam ReduceOp The reduction operation type (default: std::plus<T>)
     * @param dst The destination key buffer
     * @param in The input buffer of matrices
     * @param out The output buffer for reduced matrices
     * @param op The reduction operation
     * @return Reference to this object for method chaining
     */
    template <typename T, int M, int N, typename ReduceOp = std::plus<T>>
    FastSegmentalReduce& reduce(luisa::compute::BufferView<int>                    dst,
                                luisa::compute::BufferView<Eigen::Matrix<T, M, N>> in,
                                luisa::compute::BufferView<Eigen::Matrix<T, M, N>> out,
                                ReduceOp op = ReduceOp{});

    /**
     * @brief Perform segmental reduction using custom key and value accessors
     * 
     * @note This overload is NOT supported in Luisa Compute backend.
     *       Luisa Compute does not support capturing arbitrary host functors
     *       into device kernels. Use the buffer-based overload instead.
     * 
     * @tparam T The element type
     * @tparam GetKeyOp The key accessor functor type
     * @tparam GetValueOp The value accessor functor type
     * @tparam ReduceOp The reduction operation type
     */
    template <typename T, typename GetKeyOp, typename GetValueOp, typename ReduceOp = std::plus<T>>
    [[deprecated("Functor-based reduce() is not supported in Luisa Compute backend. Use buffer-based overload.")]]
    FastSegmentalReduce& reduce(luisa::uint                     in_size,
                                luisa::compute::BufferView<T>   out,
                                GetKeyOp                        get_key_op,
                                GetValueOp                      get_value_op,
                                ReduceOp                        op = ReduceOp{});

    /**
     * @brief Perform segmental reduction on Eigen matrices with custom accessors
     * 
     * @note This overload is NOT supported in Luisa Compute backend.
     *       Luisa Compute does not support capturing arbitrary host functors
     *       into device kernels. Use the buffer-based overload instead.
     * 
     * @tparam T The scalar type of the matrix
     * @tparam M The number of rows in the matrix
     * @tparam N The number of columns in the matrix
     * @tparam GetKeyOp The key accessor functor type
     * @tparam GetValueOp The value accessor functor type
     * @tparam ReduceOp The reduction operation type
     */
    template <typename T, int M, int N, typename GetKeyOp, typename GetValueOp, typename ReduceOp = std::plus<T>>
    [[deprecated("Functor-based reduce() is not supported in Luisa Compute backend. Use buffer-based overload.")]]
    FastSegmentalReduce& reduce(luisa::uint                                        in_size,
                                luisa::compute::BufferView<Eigen::Matrix<T, M, N>> out,
                                GetKeyOp                                           get_key_op,
                                GetValueOp                                         get_value_op,
                                ReduceOp                                           op = ReduceOp{});

  private:
    luisa::compute::Device&  m_device;
    luisa::compute::Stream&  m_stream;
    luisa::compute::Buffer<Flags> m_flags_buffer;
};
}  // namespace uipc::backend::luisa

#include "details/fast_segmental_reduce.inl"

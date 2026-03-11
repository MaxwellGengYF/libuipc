#pragma once
#include <type_traits>

namespace uipc::backend::luisa
{
namespace detail
{
    inline int b2i(bool b) noexcept
    {
        return b ? 1 : 0;
    }
}

template <int BlockSize, int WarpSize>
FastSegmentalReduce<BlockSize, WarpSize>::FastSegmentalReduce(
    luisa::compute::Device& device, 
    luisa::compute::Stream& stream)
    : m_device(device)
    , m_stream(stream)
    , m_flags_buffer(device.create_buffer<typename FastSegmentalReduce<BlockSize, WarpSize>::Flags>(BlockSize))
{
}

template <int BlockSize, int WarpSize>
template <typename T, typename ReduceOp>
FastSegmentalReduce<BlockSize, WarpSize>& FastSegmentalReduce<BlockSize, WarpSize>::reduce(
    luisa::compute::BufferView<int> dst,
    luisa::compute::BufferView<T>   in,
    luisa::compute::BufferView<T>   out,
    ReduceOp                        op)
{
    using ValueT = T;

    constexpr int warp_size = WarpSize;
    constexpr int block_dim = BlockSize;

    // Clear output buffer
    m_stream << out.fill(ValueT{0});

    using namespace luisa;
    using namespace luisa::compute;

    // Kernel with buffers passed as parameters
    Kernel1D kernel = [&](BufferVar<int> dst_buf, BufferVar<T> in_buf, BufferVar<T> out_buf) {
        auto global_thread_id = dispatch_id().x;
        auto thread_id_in_block = global_thread_id % block_dim;
        auto lane_id = thread_id_in_block % warp_size;

        Var<int> prev_i = -1;
        Var<int> next_i = -1;
        Var<int> i = -1;
        
        Flags flags;
        Var<ValueT> value = ValueT{0};
        flags.is_cross_warp = 0;

        $if(global_thread_id > 0 && global_thread_id < in_buf.size())
        {
            prev_i = dst_buf.read(global_thread_id - 1);
        };

        $if(global_thread_id < in_buf.size() - 1)
        {
            next_i = dst_buf.read(global_thread_id + 1);
        };

        $if(global_thread_id < in_buf.size())
        {
            i = dst_buf.read(global_thread_id);
            value = in_buf.read(global_thread_id);
            flags.is_valid = 1;
        }
        $else
        {
            i = -1;
            flags.is_valid = 0;
            flags.is_cross_warp = 0;
        };

        $if(lane_id == 0)
        {
            flags.is_head = 1;
            flags.is_cross_warp = detail::b2i(prev_i == i);
        }
        $else
        {
            flags.is_head = detail::b2i(prev_i != i);

            $if(lane_id == warp_size - 1)
            {
                flags.is_cross_warp = detail::b2i(next_i == i);
            };
        };

        // Convert boolean flags to integers for reduction
        flags.b2i();

        // Warp-level segmented reduction for flags using shuffle
        Var<luisa::uint> reduce_flags = flags.flags;
        for (int offset = 1; offset < warp_size; offset *= 2)
        {
            auto shuffled = warp::shfl_up(reduce_flags, offset);
            $if(lane_id >= offset)
            {
                reduce_flags = op(reduce_flags, shuffled);
            };
        }
        flags.flags = reduce_flags;

        // Warp-level segmented reduction for value
        Var<ValueT> reduce_value = value;
        for (int offset = 1; offset < warp_size; offset *= 2)
        {
            auto shuffled = warp::shfl_up(reduce_value, offset);
            $if(lane_id >= offset && !flags.is_head)
            {
                reduce_value = op(reduce_value, shuffled);
            };
        }
        value = reduce_value;

        // Write results
        $if(flags.is_head && flags.is_valid)
        {
            $if(flags.is_cross_warp)
            {
                // Atomic add for cross-warp segments
                out_buf.atomic(i).fetch_add(value);
            }
            $else
            {
                out_buf.write(i, value);
            };
        };
    };

    // Compile and dispatch kernel
    auto shader = m_device.compile(kernel);
    auto block_count = (in.size() + block_dim - 1) / block_dim;
    m_stream << shader(dst, in, out).dispatch(block_count * block_dim);

    return *this;
}

template <int BlockSize, int WarpSize>
template <typename T, int M, int N, typename ReduceOp>
FastSegmentalReduce<BlockSize, WarpSize>& FastSegmentalReduce<BlockSize, WarpSize>::reduce(
    luisa::compute::BufferView<int>                    dst,
    luisa::compute::BufferView<Eigen::Matrix<T, M, N>> in,
    luisa::compute::BufferView<Eigen::Matrix<T, M, N>> out,
    ReduceOp                                           op)
{
    using Matrix = Eigen::Matrix<T, M, N>;

    constexpr int warp_size = WarpSize;
    constexpr int block_dim = BlockSize;

    // Clear output buffer
    m_stream << out.fill(Matrix::Zero().eval());

    using namespace luisa;
    using namespace luisa::compute;

    Kernel1D kernel = [&](BufferVar<int> dst_buf, BufferVar<Matrix> in_buf, BufferVar<Matrix> out_buf) {
        auto global_thread_id = dispatch_id().x;
        auto thread_id_in_block = global_thread_id % block_dim;
        auto lane_id = thread_id_in_block % warp_size;

        Var<int> prev_i = -1;
        Var<int> next_i = -1;
        Var<int> i = -1;
        
        Flags flags;
        Var<Matrix> value = Matrix::Zero();
        flags.is_cross_warp = 0;

        $if(global_thread_id > 0 && global_thread_id < in_buf.size())
        {
            prev_i = dst_buf.read(global_thread_id - 1);
        };

        $if(global_thread_id < in_buf.size() - 1)
        {
            next_i = dst_buf.read(global_thread_id + 1);
        };

        $if(global_thread_id < in_buf.size())
        {
            i = dst_buf.read(global_thread_id);
            value = in_buf.read(global_thread_id);
            flags.is_valid = 1;
        }
        $else
        {
            i = -1;
            value = Matrix::Zero();
            flags.is_valid = 0;
            flags.is_cross_warp = 0;
        };

        $if(lane_id == 0)
        {
            flags.is_head = 1;
            flags.is_cross_warp = detail::b2i(prev_i == i);
        }
        $else
        {
            flags.is_head = detail::b2i(prev_i != i);

            $if(lane_id == warp_size - 1)
            {
                flags.is_cross_warp = detail::b2i(next_i == i);
            };
        };

        // Convert boolean flags to integers for reduction
        flags.b2i();

        // Warp-level segmented reduction for flags using shuffle
        Var<luisa::uint> reduce_flags = flags.flags;
        for (int offset = 1; offset < warp_size; offset *= 2)
        {
            auto shuffled = warp::shfl_up(reduce_flags, offset);
            $if(lane_id >= offset)
            {
                reduce_flags = op(reduce_flags, shuffled);
            };
        }
        flags.flags = reduce_flags;

        // Warp-level segmented reduction for each matrix element
        for (int row = 0; row < M; ++row)
        {
            for (int col = 0; col < N; ++col)
            {
                Var<T> reduce_elem = value(row, col);
                for (int offset = 1; offset < warp_size; offset *= 2)
                {
                    auto shuffled = warp::shfl_up(reduce_elem, offset);
                    $if(lane_id >= offset && !flags.is_head)
                    {
                        reduce_elem = op(reduce_elem, shuffled);
                    };
                }
                value(row, col) = reduce_elem;
            }
        }

        // Write results
        $if(flags.is_head && flags.is_valid)
        {
            $if(flags.is_cross_warp)
            {
                // For cross-warp segments, read-modify-write
                // Note: Atomic operations on matrices need special handling
                // This is a simplified version that reads and adds
                auto existing = out_buf.read(i);
                out_buf.write(i, existing + value);
            }
            $else
            {
                out_buf.write(i, value);
            };
        };
    };

    // Compile and dispatch kernel
    auto shader = m_device.compile(kernel);
    auto block_count = (in.size() + block_dim - 1) / block_dim;
    m_stream << shader(dst, in, out).dispatch(block_count * block_dim);

    return *this;
}

// The functor-based overloads are disabled as LC doesn't support capturing
// host functors into device kernels directly. Use the buffer-based versions instead.

template <int BlockSize, int WarpSize>
template <typename T, typename GetKeyOp, typename GetValueOp, typename ReduceOp>
FastSegmentalReduce<BlockSize, WarpSize>& FastSegmentalReduce<BlockSize, WarpSize>::reduce(
    luisa::uint                     in_size,
    luisa::compute::BufferView<T>   out,
    GetKeyOp                        get_key_op,
    GetValueOp                      get_value_op,
    ReduceOp                        op)
{
    // Luisa Compute does not support capturing arbitrary host functors
    // into device kernels. Use the buffer-based reduce() overload instead.
    static_assert(std::is_void_v<T>, 
        "Functor-based reduce() is not supported in Luisa Compute backend. "
        "Please use the buffer-based overload: reduce(dst, in, out, op)");
    return *this;
}

template <int BlockSize, int WarpSize>
template <typename T, int M, int N, typename GetKeyOp, typename GetValueOp, typename ReduceOp>
FastSegmentalReduce<BlockSize, WarpSize>& FastSegmentalReduce<BlockSize, WarpSize>::reduce(
    luisa::uint                                        in_size,
    luisa::compute::BufferView<Eigen::Matrix<T, M, N>> out,
    GetKeyOp                                           get_key_op,
    GetValueOp                                         get_value_op,
    ReduceOp                                           op)
{
    // Luisa Compute does not support capturing arbitrary host functors
    // into device kernels. Use the buffer-based reduce() overload instead.
    static_assert(std::is_void_v<T>, 
        "Functor-based reduce() is not supported in Luisa Compute backend. "
        "Please use the buffer-based overload: reduce(dst, in, out, op)");
    return *this;
}

}  // namespace uipc::backend::luisa

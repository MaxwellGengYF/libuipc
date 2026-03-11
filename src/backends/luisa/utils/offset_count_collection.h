#pragma once
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/device.h>
#include <luisa/core/stl/vector.h>
#include <uipc/common/span.h>
#include <uipc/common/type_define.h>
#include <uipc/common/log.h>

namespace uipc::backend::luisa
{
/*
 * @brief A Offsets/Counts Collection
 * 
 * You can only setup the `Counts`, and get the `Offsets` after calling `scan()`.
 * The `scan()` function will compute the offsets based on the counts.
 * 
 * This is a LuisaCompute refactored version of the CUDA backend's OffsetCountCollection.
 * It uses LuisaCompute's Buffer for GPU memory management.
 * 
 * The buffers are allocated with size N+1 where the last element (index N) stores
 * the total count for consistency with the CUDA backend.
 * 
 * @tparam T The integer type for counts and offsets (must be integral)
 */
template <std::integral T>
class OffsetCountCollection
{
  public:
    /**
     * @brief Resize the collection to hold N elements
     * @param device The LuisaCompute device for buffer allocation
     * @param N The number of elements
     */
    void resize(luisa::compute::Device& device, SizeT N);

    /**
     * @brief Get a mutable view of the counts buffer (size N, excludes the padding element)
     * @return Buffer view for writing counts
     */
    luisa::compute::BufferView<T> counts();

    /**
     * @brief Perform exclusive prefix sum scan to compute offsets from counts
     * @param stream The stream for kernel execution
     * @param init The initial value for the scan (default: 0)
     */
    void scan(luisa::compute::Stream& stream, const T& init = T{0});

    /**
     * @brief Get the total count (sum of all counts)
     * @return The total count (read from offsets[N])
     */
    SizeT total_count() const;

    /**
     * @brief Get a const view of the counts buffer (size N, excludes the padding element)
     * @return Const buffer view of counts
     */
    luisa::compute::BufferView<T> counts() const;

    /**
     * @brief Get a const view of the offsets buffer (size N, excludes the padding element)
     * @return Const buffer view of offsets
     */
    luisa::compute::BufferView<T> offsets() const;

    /**
     * @brief Get the offset and count for a specific index
     * @param i The index
     * @return Tuple of (offset, count) for the index
     * @note This downloads data from GPU and is slow. Use for debugging only.
     */
    std::tuple<T, T> operator[](SizeT i) const;

    /**
     * @brief Get the number of elements
     * @return The size of the collection
     */
    SizeT size() const noexcept { return m_N; }

  private:
    luisa::compute::Buffer<T> m_counts;    ///< Device buffer for counts (size N+1, last element is padding)
    luisa::compute::Buffer<T> m_offsets;   ///< Device buffer for offsets (size N+1, last element stores total)
    SizeT m_N = 0;                         ///< Number of elements
};

template <std::integral T>
void OffsetCountCollection<T>::resize(luisa::compute::Device& device, SizeT N)
{
    m_N = N;
    auto N_1 = N + 1;
    if(N > 0)
    {
        m_counts  = device.create_buffer<T>(N_1);
        m_offsets = device.create_buffer<T>(N_1);
    }
    else
    {
        m_counts  = luisa::compute::Buffer<T>{};
        m_offsets = luisa::compute::Buffer<T>{};
    }
}

template <std::integral T>
luisa::compute::BufferView<T> OffsetCountCollection<T>::counts()
{
    // Return view of size N (exclude the padding element at index N)
    return m_counts.view(0, m_N);
}

template <std::integral T>
void OffsetCountCollection<T>::scan(luisa::compute::Stream& stream, const T& init)
{
    if(m_N == 0)
    {
        return;
    }

    // Read counts to host, compute offsets on host, then upload
    // This mirrors the CUDA backend's std::exclusive_scan approach
    // For production, a GPU-based exclusive scan kernel should be used
    luisa::vector<T> host_counts(m_N + 1);
    luisa::vector<T> host_offsets(m_N + 1);

    stream << m_counts.view().copy_to(host_counts.data()) << luisa::compute::synchronize();

    // Set the padding element to 0 for proper scan
    host_counts[m_N] = 0;

    // Compute exclusive prefix sum (same as std::exclusive_scan)
    T running_sum = init;
    for(SizeT i = 0; i <= m_N; ++i)
    {
        host_offsets[i] = running_sum;
        running_sum += host_counts[i];
    }

    // Upload offsets back to GPU
    stream << m_offsets.view().copy_from(host_offsets.data()) << luisa::compute::synchronize();
}

template <std::integral T>
SizeT OffsetCountCollection<T>::total_count() const
{
    // Read the total count from the last element of offsets buffer
    // This mirrors the CUDA backend's behavior: return m_offsets[m_N];
    if(m_N == 0 || !m_offsets)
        return 0;
    
    luisa::vector<T> last_offset(1);
    auto stream = m_offsets.device()->create_stream();
    stream << m_offsets.view(m_N, 1).copy_to(last_offset.data())
           << luisa::compute::synchronize();
    return static_cast<SizeT>(last_offset[0]);
}

template <std::integral T>
luisa::compute::BufferView<T> OffsetCountCollection<T>::counts() const
{
    // Return view of size N (exclude the padding element at index N)
    return m_counts.view(0, m_N);
}

template <std::integral T>
luisa::compute::BufferView<T> OffsetCountCollection<T>::offsets() const
{
    // Return view of size N (exclude the total count element at index N)
    return m_offsets.view(0, m_N);
}

template <std::integral T>
std::tuple<T, T> OffsetCountCollection<T>::operator[](SizeT i) const
{
    UIPC_ASSERT(i < m_N, "Index out of bounds for OffsetCountCollection, i = {}, N = {}", i, m_N);
    
    // Download data from GPU to get specific values
    // Note: This is slow and should be avoided in performance-critical code
    luisa::vector<T> count_val(1);
    luisa::vector<T> offset_val(1);

    auto stream = m_counts.device()->create_stream();
    stream << m_counts.view(i, 1).copy_to(count_val.data())
           << m_offsets.view(i, 1).copy_to(offset_val.data())
           << luisa::compute::synchronize();

    return {offset_val[0], count_val[0]};
}

}  // namespace uipc::backend::luisa

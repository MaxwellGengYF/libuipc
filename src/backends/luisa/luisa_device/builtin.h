#pragma once
#include <luisa/luisa-compute.h>
/**
 * @brief LuisaCompute Device Built-in Functions Header File
 * 
 * This header provides declarations for device-side built-in functions using
 * LuisaCompute's compute DSL and runtime APIs.
 * 
 * Based on the original CUDA intrinsics header, adapted for LuisaCompute which
 * supports multiple backends (CUDA, DirectX, Metal, CPU) through a unified
 * programming interface.
 * 
 * @see https://github.com/LuisaGroup/LuisaCompute
 */

namespace uipc::backend::luisa
{
/**
 * @brief Device math functions namespace
 * 
 * These functions map to LuisaCompute's math library which provides
 * cross-platform GPU-accelerated math operations.
 */
namespace device_math
{
    // ============================================================================
    // Bit Manipulation Functions
    // ============================================================================

    /**
     * @brief Reverse the bit order of a 32-bit unsigned integer
     * @param x 32-bit unsigned integer
     * @return Bit-reversed value of x
     */
    [[nodiscard]] inline auto brev(unsigned int x) noexcept
    {
        return luisa::compute::math::reverse_bits(x);
    }

    /**
     * @brief Reverse the bit order of a 64-bit unsigned integer
     * @param x 64-bit unsigned integer
     * @return Bit-reversed value of x
     */
    [[nodiscard]] inline auto brevll(unsigned long long int x) noexcept
    {
        return luisa::compute::math::reverse_bits(x);
    }

    /**
     * @brief Return the number of consecutive high-order zero bits in a 32-bit integer
     * @param x 32-bit integer
     * @return Number of leading zero bits (0-32)
     */
    [[nodiscard]] inline auto clz(unsigned int x) noexcept
    {
        return luisa::compute::math::clz(x);
    }

    /**
     * @brief Count the number of consecutive high-order zero bits in a 64-bit integer
     * @param x 64-bit integer
     * @return Number of leading zero bits (0-64)
     */
    [[nodiscard]] inline auto clzll(unsigned long long int x) noexcept
    {
        return luisa::compute::math::clz(x);
    }

    /**
     * @brief Count the number of bits that are set to 1 in a 32-bit integer
     * @param x 32-bit unsigned integer
     * @return Number of set bits (0-32)
     */
    [[nodiscard]] inline auto popc(unsigned int x) noexcept
    {
        return luisa::compute::math::popcount(x);
    }

    /**
     * @brief Count the number of bits that are set to 1 in a 64-bit integer
     * @param x 64-bit unsigned integer
     * @return Number of set bits (0-64)
     */
    [[nodiscard]] inline auto popcll(unsigned long long int x) noexcept
    {
        return luisa::compute::math::popcount(x);
    }

    // ============================================================================
    // Arithmetic Functions
    // ============================================================================

    /**
     * @brief Compute average of signed input arguments, avoiding overflow
     * @param x First signed integer
     * @param y Second signed integer
     * @return (x + y) >> 1
     */
    [[nodiscard]] inline auto hadd(int x, int y) noexcept
    {
        return (x & y) + ((x ^ y) >> 1);
    }

    /**
     * @brief Compute average of unsigned input arguments, avoiding overflow
     * @param x First unsigned integer
     * @param y Second unsigned integer
     * @return (x + y) >> 1
     */
    [[nodiscard]] inline auto uhadd(unsigned int x, unsigned int y) noexcept
    {
        return (x & y) + ((x ^ y) >> 1);
    }

    /**
     * @brief Compute rounded average of signed input arguments, avoiding overflow
     * @param x First signed integer
     * @param y Second signed integer
     * @return (x + y + 1) >> 1
     */
    [[nodiscard]] inline auto rhadd(int x, int y) noexcept
    {
        return (x | y) - ((x ^ y) >> 1);
    }

    /**
     * @brief Compute rounded average of unsigned input arguments, avoiding overflow
     * @param x First unsigned integer
     * @param y Second unsigned integer
     * @return (x + y + 1) >> 1
     */
    [[nodiscard]] inline auto urhadd(unsigned int x, unsigned int y) noexcept
    {
        return (x | y) - ((x ^ y) >> 1);
    }

    /**
     * @brief Calculate the most significant 32 bits of the product of two 32-bit integers
     * @param x First 32-bit integer
     * @param y Second 32-bit integer
     * @return Most significant 32 bits of 64-bit product
     */
    [[nodiscard]] inline auto mulhi(int x, int y) noexcept
    {
        return static_cast<int>((static_cast<long long>(x) * static_cast<long long>(y)) >> 32);
    }

    /**
     * @brief Calculate the most significant 32 bits of the product of two 32-bit unsigned integers
     * @param x First 32-bit unsigned integer
     * @param y Second 32-bit unsigned integer
     * @return Most significant 32 bits of 64-bit product
     */
    [[nodiscard]] inline auto umulhi(unsigned int x, unsigned int y) noexcept
    {
        return static_cast<unsigned int>((static_cast<unsigned long long>(x) * static_cast<unsigned long long>(y)) >> 32);
    }

    // ============================================================================
    // Type Casting Functions
    // ============================================================================

    /**
     * @brief Convert a float to a signed integer in round-to-nearest-even mode
     * @param x Single-precision floating-point value
     * @return Converted signed integer value
     */
    [[nodiscard]] inline auto float2int_rn(float x) noexcept
    {
        return luisa::compute::cast<int>(luisa::compute::math::round(x));
    }

    /**
     * @brief Convert a float to a signed integer in round-down mode
     * @param x Single-precision floating-point value
     * @return Converted signed integer value
     */
    [[nodiscard]] inline auto float2int_rd(float x) noexcept
    {
        return luisa::compute::cast<int>(luisa::compute::math::floor(x));
    }

    /**
     * @brief Convert a float to a signed integer in round-up mode
     * @param x Single-precision floating-point value
     * @return Converted signed integer value
     */
    [[nodiscard]] inline auto float2int_ru(float x) noexcept
    {
        return luisa::compute::cast<int>(luisa::compute::math::ceil(x));
    }

    /**
     * @brief Convert a float to a signed integer in round-towards-zero mode
     * @param x Single-precision floating-point value
     * @return Converted signed integer value
     */
    [[nodiscard]] inline auto float2int_rz(float x) noexcept
    {
        return luisa::compute::cast<int>(x);
    }

    /**
     * @brief Reinterpret bits in a float as a signed integer
     * @param x Single-precision floating-point value
     * @return Reinterpreted signed integer value
     */
    [[nodiscard]] inline auto float_as_int(float x) noexcept
    {
        return luisa::compute::bit_cast<int>(x);
    }

    /**
     * @brief Reinterpret bits in a float as an unsigned integer
     * @param x Single-precision floating-point value
     * @return Reinterpreted unsigned integer value
     */
    [[nodiscard]] inline auto float_as_uint(float x) noexcept
    {
        return luisa::compute::bit_cast<unsigned int>(x);
    }

    /**
     * @brief Reinterpret bits in an integer as a float
     * @param x Signed integer value
     * @return Reinterpreted single-precision floating-point value
     */
    [[nodiscard]] inline auto int_as_float(int x) noexcept
    {
        return luisa::compute::bit_cast<float>(x);
    }

    /**
     * @brief Reinterpret bits in an unsigned integer as a float
     * @param x Unsigned integer value
     * @return Reinterpreted single-precision floating-point value
     */
    [[nodiscard]] inline auto uint_as_float(unsigned int x) noexcept
    {
        return luisa::compute::bit_cast<float>(x);
    }

}  // namespace device_math

/**
 * @brief Device synchronization namespace
 */
namespace device_sync
{
    /**
     * @brief Memory fence at block scope
     * 
     * Ensures that all writes to all memory made by the calling thread before
     * the call are observed by all threads in the block.
     */
    inline void threadfence_block() noexcept
    {
        luisa::compute::thread_fence(luisa::compute::MemoryOrder::SEQ_CST, 
                                     luisa::compute::MemoryScope::BLOCK);
    }

    /**
     * @brief Memory fence at device scope
     * 
     * Ensures that no writes to all memory made by the calling thread after
     * the call are observed by any thread in the device as occurring before any
     * write made by the calling thread before the call.
     */
    inline void threadfence() noexcept
    {
        luisa::compute::thread_fence(luisa::compute::MemoryOrder::SEQ_CST,
                                     luisa::compute::MemoryScope::DEVICE);
    }

    /**
     * @brief Memory fence at system scope
     * 
     * Ensures that all writes to all memory made by the calling thread before
     * the call are observed by all threads in the device, host threads, and all
     * threads in peer devices.
     */
    inline void threadfence_system() noexcept
    {
        luisa::compute::thread_fence(luisa::compute::MemoryOrder::SEQ_CST,
                                     luisa::compute::MemoryScope::SYSTEM);
    }

    /**
     * @brief Synchronize all threads in the thread block
     * 
     * Waits until all threads in the thread block have reached this point and
     * all global and shared memory accesses made by these threads prior to
     * the call are visible to all threads in the block.
     */
    inline void syncthreads() noexcept
    {
        luisa::compute::sync_block();
    }

}  // namespace device_sync

/**
 * @brief Device atomic operations namespace
 * 
 * These operations use LuisaCompute's atomic API which provides
 * consistent atomic operations across all supported backends.
 */
namespace device_atomic
{
    /**
     * @brief Atomic addition
     * @tparam T Value type (int, unsigned int, unsigned long long, float, double)
     * @param ptr Address in global or shared memory
     * @param val Value to add
     * @return Old value at address
     */
    template<typename T>
    [[nodiscard]] inline auto add(T* ptr, T val) noexcept
    {
        return luisa::compute::atomic_fetch_add(ptr, val);
    }

    /**
     * @brief Atomic subtraction
     * @tparam T Value type (int, unsigned int)
     * @param ptr Address in global or shared memory
     * @param val Value to subtract
     * @return Old value at address
     */
    template<typename T>
    [[nodiscard]] inline auto sub(T* ptr, T val) noexcept
    {
        return luisa::compute::atomic_fetch_sub(ptr, val);
    }

    /**
     * @brief Atomic exchange
     * @tparam T Value type (int, unsigned int, unsigned long long, float)
     * @param ptr Address in global or shared memory
     * @param val Value to store
     * @return Old value at address
     */
    template<typename T>
    [[nodiscard]] inline auto exch(T* ptr, T val) noexcept
    {
        return luisa::compute::atomic_exchange(ptr, val);
    }

    /**
     * @brief Atomic minimum
     * @tparam T Value type (int, unsigned int, unsigned long long, long long)
     * @param ptr Address in global or shared memory
     * @param val Value to compare
     * @return Old value at address
     */
    template<typename T>
    [[nodiscard]] inline auto min(T* ptr, T val) noexcept
    {
        return luisa::compute::atomic_fetch_min(ptr, val);
    }

    /**
     * @brief Atomic maximum
     * @tparam T Value type (int, unsigned int, unsigned long long, long long)
     * @param ptr Address in global or shared memory
     * @param val Value to compare
     * @return Old value at address
     */
    template<typename T>
    [[nodiscard]] inline auto max(T* ptr, T val) noexcept
    {
        return luisa::compute::atomic_fetch_max(ptr, val);
    }

    /**
     * @brief Atomic compare and swap
     * @tparam T Value type (int, unsigned int, unsigned long long, unsigned short)
     * @param ptr Address in global or shared memory
     * @param compare Value to compare with
     * @param val Value to store if compare matches
     * @return Old value at address
     */
    template<typename T>
    [[nodiscard]] inline auto CAS(T* ptr, T compare, T val) noexcept
    {
        return luisa::compute::atomic_compare_exchange(ptr, compare, val);
    }

    /**
     * @brief Atomic bitwise AND
     * @tparam T Value type (int, unsigned int, unsigned long long)
     * @param ptr Address in global or shared memory
     * @param val Value to AND with
     * @return Old value at address
     */
    template<typename T>
    [[nodiscard]] inline auto And(T* ptr, T val) noexcept
    {
        return luisa::compute::atomic_fetch_and(ptr, val);
    }

    /**
     * @brief Atomic bitwise OR
     * @tparam T Value type (int, unsigned int, unsigned long long)
     * @param ptr Address in global or shared memory
     * @param val Value to OR with
     * @return Old value at address
     */
    template<typename T>
    [[nodiscard]] inline auto Or(T* ptr, T val) noexcept
    {
        return luisa::compute::atomic_fetch_or(ptr, val);
    }

    /**
     * @brief Atomic bitwise XOR
     * @tparam T Value type (int, unsigned int, unsigned long long)
     * @param ptr Address in global or shared memory
     * @param val Value to XOR with
     * @return Old value at address
     */
    template<typename T>
    [[nodiscard]] inline auto Xor(T* ptr, T val) noexcept
    {
        return luisa::compute::atomic_fetch_xor(ptr, val);
    }

    /**
     * @brief Atomic increment (wraps to 0 when val is reached)
     * @param ptr Address in global or shared memory
     * @param val Maximum value (wraps to 0 when old >= val)
     * @return Old value at address
     */
    [[nodiscard]] inline auto Inc(unsigned int* ptr, unsigned int val) noexcept
    {
        auto old = *ptr;
        auto new_val = (old >= val) ? 0u : old + 1u;
        while(!luisa::compute::atomic_compare_exchange(ptr, old, new_val)) {
            old = *ptr;
            new_val = (old >= val) ? 0u : old + 1u;
        }
        return old;
    }

    /**
     * @brief Atomic decrement (wraps to val when 0 is reached)
     * @param ptr Address in global or shared memory
     * @param val Value to wrap to when old == 0 or old > val
     * @return Old value at address
     */
    [[nodiscard]] inline auto Dec(unsigned int* ptr, unsigned int val) noexcept
    {
        auto old = *ptr;
        auto new_val = ((old == 0u) || (old > val)) ? val : old - 1u;
        while(!luisa::compute::atomic_compare_exchange(ptr, old, new_val)) {
            old = *ptr;
            new_val = ((old == 0u) || (old > val)) ? val : old - 1u;
        }
        return old;
    }

}  // namespace device_atomic

/**
 * @brief Warp-level primitives namespace
 * 
 * Note: LuisaCompute provides warp-level primitives through its DSL.
 * These may have different semantics depending on the backend.
 */
namespace device_warp
{
    /**
     * @brief Warp shuffle - get value from specific lane
     * @tparam T Value type
     * @param var Variable to shuffle
     * @param srcLane Source lane index
     * @return Value from source lane
     */
    template<typename T>
    [[nodiscard]] inline auto shuffle(T var, int srcLane) noexcept
    {
        return luisa::compute::warp_shuffle(var, srcLane);
    }

    /**
     * @brief Warp shuffle up
     * @tparam T Value type
     * @param var Variable to shuffle
     * @param delta Offset to shuffle up by
     * @return Value from lane (laneId - delta)
     */
    template<typename T>
    [[nodiscard]] inline auto shuffle_up(T var, unsigned int delta) noexcept
    {
        return luisa::compute::warp_shuffle_up(var, delta);
    }

    /**
     * @brief Warp shuffle down
     * @tparam T Value type
     * @param var Variable to shuffle
     * @param delta Offset to shuffle down by
     * @return Value from lane (laneId + delta)
     */
    template<typename T>
    [[nodiscard]] inline auto shuffle_down(T var, unsigned int delta) noexcept
    {
        return luisa::compute::warp_shuffle_down(var, delta);
    }

    /**
     * @brief Warp shuffle XOR
     * @tparam T Value type
     * @param var Variable to shuffle
     * @param laneMask Lane mask for XOR shuffle
     * @return Value from lane (laneId XOR laneMask)
     */
    template<typename T>
    [[nodiscard]] inline auto shuffle_xor(T var, int laneMask) noexcept
    {
        return luisa::compute::warp_shuffle_xor(var, laneMask);
    }

    /**
     * @brief Warp-wide all-reduce AND
     * @param predicate Predicate to evaluate
     * @return Non-zero if predicate is true for all active threads, zero otherwise
     */
    [[nodiscard]] inline auto all(int predicate) noexcept
    {
        return luisa::compute::warp_all(predicate);
    }

    /**
     * @brief Warp-wide any-reduce OR
     * @param predicate Predicate to evaluate
     * @return Non-zero if predicate is true for any active thread, zero otherwise
     */
    [[nodiscard]] inline auto any(int predicate) noexcept
    {
        return luisa::compute::warp_any(predicate);
    }

    /**
     * @brief Warp ballot - return bitmask of threads with predicate true
     * @param predicate Predicate to evaluate
     * @return Bitmask with bit N set if predicate is true for thread N
     */
    [[nodiscard]] inline auto ballot(int predicate) noexcept
    {
        return luisa::compute::warp_ballot(predicate);
    }

    /**
     * @brief Get current lane ID within warp
     * @return Lane ID (0-31)
     */
    [[nodiscard]] inline auto lane_id() noexcept
    {
        return luisa::compute::warp_lane_id();
    }

    /**
     * @brief Get warp size
     * @return Number of threads in warp (typically 32)
     */
    [[nodiscard]] inline auto size() noexcept
    {
        return luisa::compute::warp_size();
    }

    /**
     * @brief Warp prefix sum (inclusive scan)
     * @tparam T Value type
     * @param val Value to include in prefix sum
     * @return Sum of values from lane 0 to current lane (inclusive)
     */
    template<typename T>
    [[nodiscard]] inline auto prefix_sum(T val) noexcept
    {
        return luisa::compute::warp_prefix_sum(val);
    }

    /**
     * @brief Warp active count
     * @return Number of active threads in warp
     */
    [[nodiscard]] inline auto active_count() noexcept
    {
        return luisa::compute::warp_active_count();
    }

}  // namespace device_warp

/**
 * @brief Block-level primitives namespace
 */
namespace device_block
{
    /**
     * @brief Get current thread index within block
     * @return Thread index (0 to block_size - 1)
     */
    [[nodiscard]] inline auto thread_index() noexcept
    {
        return luisa::compute::block_thread_index();
    }

    /**
     * @brief Get block size
     * @return Number of threads in block
     */
    [[nodiscard]] inline auto size() noexcept
    {
        return luisa::compute::block_size();
    }

    /**
     * @brief Block-wide all-reduce AND
     * @param predicate Predicate to evaluate
     * @return Non-zero if predicate is true for all threads, zero otherwise
     */
    [[nodiscard]] inline auto all(int predicate) noexcept
    {
        return luisa::compute::block_all(predicate);
    }

    /**
     * @brief Block-wide any-reduce OR
     * @param predicate Predicate to evaluate
     * @return Non-zero if predicate is true for any thread, zero otherwise
     */
    [[nodiscard]] inline auto any(int predicate) noexcept
    {
        return luisa::compute::block_any(predicate);
    }

}  // namespace device_block

/**
 * @brief Device info namespace
 */
namespace device_info
{
    /**
     * @brief Get current block index
     * @return Block index within grid
     */
    [[nodiscard]] inline auto block_id() noexcept
    {
        return luisa::compute::dispatch_id();
    }

    /**
     * @brief Get total number of blocks
     * @return Grid size in blocks
     */
    [[nodiscard]] inline auto num_blocks() noexcept
    {
        return luisa::compute::dispatch_size();
    }

    /**
     * @brief Get global thread ID
     * @return Global unique thread identifier
     */
    [[nodiscard]] inline auto global_thread_id() noexcept
    {
        return luisa::compute::dispatch_id() * luisa::compute::block_size() 
               + luisa::compute::block_thread_index();
    }

}  // namespace device_info

// ============================================================================
// Backward compatibility aliases (CUDA-style naming)
// ============================================================================

namespace compat
{
    using namespace device_math;
    using namespace device_sync;
    using namespace device_info;

    // Atomic operations with CUDA-style naming
    template<typename T>
    [[nodiscard]] inline auto atomicAdd(T* ptr, T val) noexcept
    {
        return device_atomic::add(ptr, val);
    }

    template<typename T>
    [[nodiscard]] inline auto atomicSub(T* ptr, T val) noexcept
    {
        return device_atomic::sub(ptr, val);
    }

    template<typename T>
    [[nodiscard]] inline auto atomicExch(T* ptr, T val) noexcept
    {
        return device_atomic::exch(ptr, val);
    }

    template<typename T>
    [[nodiscard]] inline auto atomicMin(T* ptr, T val) noexcept
    {
        return device_atomic::min(ptr, val);
    }

    template<typename T>
    [[nodiscard]] inline auto atomicMax(T* ptr, T val) noexcept
    {
        return device_atomic::max(ptr, val);
    }

    template<typename T>
    [[nodiscard]] inline auto atomicAnd(T* ptr, T val) noexcept
    {
        return device_atomic::And(ptr, val);
    }

    template<typename T>
    [[nodiscard]] inline auto atomicOr(T* ptr, T val) noexcept
    {
        return device_atomic::Or(ptr, val);
    }

    template<typename T>
    [[nodiscard]] inline auto atomicXor(T* ptr, T val) noexcept
    {
        return device_atomic::Xor(ptr, val);
    }

    template<typename T>
    [[nodiscard]] inline auto atomicCAS(T* ptr, T compare, T val) noexcept
    {
        return device_atomic::CAS(ptr, compare, val);
    }

    [[nodiscard]] inline auto atomicInc(unsigned int* ptr, unsigned int val) noexcept
    {
        return device_atomic::Inc(ptr, val);
    }

    [[nodiscard]] inline auto atomicDec(unsigned int* ptr, unsigned int val) noexcept
    {
        return device_atomic::Dec(ptr, val);
    }

    // Warp operations with CUDA-style naming
    template<typename T>
    [[nodiscard]] inline auto __shfl_sync(unsigned int /*mask*/, T var, int srcLane, int /*width*/ = 32) noexcept
    {
        return device_warp::shuffle(var, srcLane);
    }

    template<typename T>
    [[nodiscard]] inline auto __shfl_up_sync(unsigned int /*mask*/, T var, unsigned int delta, int /*width*/ = 32) noexcept
    {
        return device_warp::shuffle_up(var, delta);
    }

    template<typename T>
    [[nodiscard]] inline auto __shfl_down_sync(unsigned int /*mask*/, T var, unsigned int delta, int /*width*/ = 32) noexcept
    {
        return device_warp::shuffle_down(var, delta);
    }

    template<typename T>
    [[nodiscard]] inline auto __shfl_xor_sync(unsigned int /*mask*/, T var, int laneMask, int /*width*/ = 32) noexcept
    {
        return device_warp::shuffle_xor(var, laneMask);
    }

    [[nodiscard]] inline auto __all_sync(unsigned int /*mask*/, int predicate) noexcept
    {
        return device_warp::all(predicate);
    }

    [[nodiscard]] inline auto __any_sync(unsigned int /*mask*/, int predicate) noexcept
    {
        return device_warp::any(predicate);
    }

    [[nodiscard]] inline auto __ballot_sync(unsigned int /*mask*/, int predicate) noexcept
    {
        return device_warp::ballot(predicate);
    }

    inline void __syncwarp(unsigned int /*mask*/ = 0xffffffff) noexcept
    {
        device_sync::threadfence_block();
    }

    inline void __threadfence_block() noexcept
    {
        device_sync::threadfence_block();
    }

    inline void __threadfence() noexcept
    {
        device_sync::threadfence();
    }

    inline void __threadfence_system() noexcept
    {
        device_sync::threadfence_system();
    }

    inline void __syncthreads() noexcept
    {
        device_sync::syncthreads();
    }

}  // namespace compat

}  // namespace uipc::backend::luisa

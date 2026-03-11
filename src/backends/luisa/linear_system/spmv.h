#pragma once
#include <type_define.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>

namespace uipc::backend::luisa
{
/**
 * @brief Sparse Matrix-Vector multiplication (SpMV) operations
 * 
 * Refactored from CUDA backend to use LuisaCompute's unified compute API.
 * Provides various SpMV kernels for block sparse matrices with 3x3 blocks.
 */
class Spmv
{
  public:
    /**
     * @brief Construct Spmv with a stream for kernel execution
     */
    explicit Spmv(luisa::compute::Stream* stream)
        : _stream(stream)
    {}

    /**
     * @brief Symmetric BCOO SpMV: y = a * A * x + b * y
     * 
     * For symmetric matrices, only the lower/upper triangle is stored.
     * The operation computes both A*x and A^T*x contributions for off-diagonal blocks.
     * 
     * @param a Scalar multiplier for A * x
     * @param A Block COO matrix view (symmetric, stored triangle only)
     * @param x Input dense vector
     * @param b Scalar multiplier for y (y = b*y + a*A*x)
     * @param y Output dense vector (modified in-place)
     */
    void sym_spmv(Float                           a,
                  CBCOOMatrixView<Float, 3>       A,
                  luisa::compute::BufferView<const Float>   x,
                  Float                           b,
                  luisa::compute::BufferView<Float>         y);

    /**
     * @brief Reduce-by-key SpMV: y = a * A * x + b * y
     * 
     * Uses reduce-by-key algorithm for matrix-vector multiplication.
     * Currently delegates to sym_spmv.
     * 
     * @param a Scalar multiplier for A * x
     * @param A Block COO matrix view
     * @param x Input dense vector
     * @param b Scalar multiplier for y
     * @param y Output dense vector
     */
    void rbk_spmv(Float                           a,
                  CBCOOMatrixView<Float, 3>       A,
                  luisa::compute::BufferView<const Float>   x,
                  Float                           b,
                  luisa::compute::BufferView<Float>         y);

    /**
     * @brief Reduce-by-key symmetric SpMV: y = a * A * x + b * y
     * 
     * For symmetric matrices using reduce-by-key algorithm.
     * 
     * @param a Scalar multiplier for A * x
     * @param A Block COO matrix view (symmetric, stored triangle only)
     * @param x Input dense vector
     * @param b Scalar multiplier for y
     * @param y Output dense vector
     */
    void rbk_sym_spmv(Float                           a,
                      CBCOOMatrixView<Float, 3>       A,
                      luisa::compute::BufferView<const Float>   x,
                      Float                           b,
                      luisa::compute::BufferView<Float>         y);

    /**
     * @brief Reduce-by-key symmetric SpMV with fused dot product
     * 
     * Computes y = a * A * x + b * y AND d_dot = x^T * (a * A * x) in a single pass.
     * This is useful for Krylov subspace methods like CG.
     * 
     * @param a Scalar multiplier for A * x
     * @param A Block COO matrix view (symmetric, stored triangle only)
     * @param x Input dense vector
     * @param b Scalar multiplier for y
     * @param y Output dense vector
     * @param d_dot Buffer to store the computed dot product (size should be at least 1)
     */
    void rbk_sym_spmv_dot(Float                           a,
                          CBCOOMatrixView<Float, 3>       A,
                          luisa::compute::BufferView<const Float>   x,
                          Float                           b,
                          luisa::compute::BufferView<Float>         y,
                          luisa::compute::BufferView<Float>         d_dot);

    /**
     * @brief Debug fallback CPU SpMV (very slow, only for debug)
     * 
     * Performs SpMV on CPU for verification purposes. Downloads data from GPU,
     * computes on CPU, then uploads result back to GPU.
     * 
     * @param a Scalar multiplier for A * x
     * @param A Block COO matrix view (symmetric, stored triangle only)
     * @param x Input dense vector
     * @param b Scalar multiplier for y
     * @param y Output dense vector
     */
    void cpu_sym_spmv(Float                           a,
                      CBCOOMatrixView<Float, 3>       A,
                      luisa::compute::BufferView<const Float>   x,
                      Float                           b,
                      luisa::compute::BufferView<Float>         y);

  private:
    luisa::compute::Stream* _stream = nullptr;
};
}  // namespace uipc::backend::luisa

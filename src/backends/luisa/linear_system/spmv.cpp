#include <linear_system/spmv.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
using namespace luisa::compute;

// Type aliases matching the header
using CDenseVectorView = BufferView<const Float>;
using DenseVectorView  = BufferView<Float>;
using VarView          = BufferView<Float>;

// Helper function to zero out a buffer
static void buffer_zero(DenseVectorView y, Stream& stream)
{
    stream << y.fill(0.0f);
}

// Helper: y = b * y
static void scale_vector(DenseVectorView y, Float b, Stream& stream)
{
    if(b == 1.0f)
        return;

    Kernel1D scale_kernel = [&](BufferVar<Float> y_buf, Float scale) noexcept
    {
        auto i = dispatch_id().x;
        $if(i < y_buf.size())
        {
            y_buf.write(i, scale * y_buf.read(i));
        };
    };

    auto shader = stream.device().compile(scale_kernel);
    stream << shader(y, b).dispatch(y.size());
}

// Helper to reinterpret Eigen::Matrix buffer as Matrix3x3 buffer
// Both have the same memory layout (9 floats in column-major order)
static BufferView<const Matrix3x3> reinterpret_as_matrix3x3(
    BufferView<const Eigen::Matrix<Float, 3, 3>> eigen_buffer)
{
    // Both Eigen::Matrix<Float, 3, 3> and Matrix3x3 (float3x3) are 
    // 9 floats in column-major order, so we can safely reinterpret
    return BufferView<const Matrix3x3>(
        eigen_buffer.handle(),
        eigen_buffer.offset(),
        eigen_buffer.size());
}

void Spmv::sym_spmv(Float                           a,
                    CBCOOMatrixView<Float, 3>       A,
                    CDenseVectorView                x,
                    Float                           b,
                    DenseVectorView                 y)
{
    constexpr uint N = 3;
    auto& stream = *_stream;

    // Initialize y: y = b * y or y = 0
    if(b != 0.0f)
    {
        scale_vector(y, b, stream);
    }
    else
    {
        buffer_zero(y, stream);
    }

    // Reinterpret the Eigen matrix buffer as Matrix3x3 buffer
    auto block_values_view = reinterpret_as_matrix3x3(A.block_values);

    // Kernel for symmetric SpMV: y += a * A * x
    Kernel1D sym_spmv_kernel = [&](BufferVar<const int>    row_indices,
                                   BufferVar<const int>    col_indices,
                                   BufferVar<const Matrix3x3> block_values,
                                   BufferVar<const Float>   x_buf,
                                   BufferVar<Float>         y_buf,
                                   Float                    alpha,
                                   UInt                     block_count) noexcept
    {
        auto idx = dispatch_id().x;
        $if(idx < block_count)
        {
            auto i = row_indices.read(idx);
            auto j = col_indices.read(idx);
            auto mat = block_values.read(idx);

            // Load x segment for column j
            Float3 x_j;
            x_j.x = x_buf.read(j * N + 0);
            x_j.y = x_buf.read(j * N + 1);
            x_j.z = x_buf.read(j * N + 2);

            // Compute block * x_j (matrix is column-major: mat[col][row])
            Float3 result_i;
            result_i.x = mat[0][0] * x_j.x + mat[1][0] * x_j.y + mat[2][0] * x_j.z;
            result_i.y = mat[0][1] * x_j.x + mat[1][1] * x_j.y + mat[2][1] * x_j.z;
            result_i.z = mat[0][2] * x_j.x + mat[1][2] * x_j.y + mat[2][2] * x_j.z;

            // Atomic add to y[i]
            auto y_i_base = i * N;
            y_buf.atomic(y_i_base + 0).fetch_add(alpha * result_i.x);
            y_buf.atomic(y_i_base + 1).fetch_add(alpha * result_i.y);
            y_buf.atomic(y_i_base + 2).fetch_add(alpha * result_i.z);

            // For off-diagonal blocks, also add transpose contribution to y[j]
            $if(i != j)
            {
                // Load x segment for column i (for transpose multiplication)
                Float3 x_i;
                x_i.x = x_buf.read(i * N + 0);
                x_i.y = x_buf.read(i * N + 1);
                x_i.z = x_buf.read(i * N + 2);

                // Compute block^T * x_i
                Float3 result_j;
                result_j.x = mat[0][0] * x_i.x + mat[0][1] * x_i.y + mat[0][2] * x_i.z;
                result_j.y = mat[1][0] * x_i.x + mat[1][1] * x_i.y + mat[1][2] * x_i.z;
                result_j.z = mat[2][0] * x_i.x + mat[2][1] * x_i.y + mat[2][2] * x_i.z;

                // Atomic add to y[j]
                auto y_j_base = j * N;
                y_buf.atomic(y_j_base + 0).fetch_add(alpha * result_j.x);
                y_buf.atomic(y_j_base + 1).fetch_add(alpha * result_j.y);
                y_buf.atomic(y_j_base + 2).fetch_add(alpha * result_j.z);
            };
        };
    };

    auto shader = stream.device().compile(sym_spmv_kernel);
    stream << shader(A.block_row_indices,
                     A.block_col_indices,
                     block_values_view,
                     x,
                     y,
                     a,
                     static_cast<uint>(A.block_count))
                .dispatch(A.block_count);
}

void Spmv::rbk_spmv(Float                           a,
                    CBCOOMatrixView<Float, 3>       A,
                    CDenseVectorView                x,
                    Float                           b,
                    DenseVectorView                 y)
{
    // For simplicity, delegate to sym_spmv
    // A full implementation would use warp-level primitives for segmented reduction
    sym_spmv(a, A, x, b, y);
}

void Spmv::rbk_sym_spmv(Float                           a,
                        CBCOOMatrixView<Float, 3>       A,
                        CDenseVectorView                x,
                        Float                           b,
                        DenseVectorView                 y)
{
    constexpr uint N = 3;
    auto& stream = *_stream;

    // Initialize y: y = b * y or y = 0
    if(b != 0.0f)
    {
        scale_vector(y, b, stream);
    }
    else
    {
        buffer_zero(y, stream);
    }

    // Reinterpret the Eigen matrix buffer as Matrix3x3 buffer
    auto block_values_view = reinterpret_as_matrix3x3(A.block_values);

    // Kernel for symmetric reduce-by-key SpMV
    Kernel1D rbk_sym_kernel = [&](BufferVar<const int>    row_indices,
                                  BufferVar<const int>    col_indices,
                                  BufferVar<const Matrix3x3> block_values,
                                  BufferVar<const Float>   x_buf,
                                  BufferVar<Float>         y_buf,
                                  Float                    alpha,
                                  UInt                     block_count) noexcept
    {
        auto idx = dispatch_id().x;
        $if(idx < block_count)
        {
            auto i = row_indices.read(idx);
            auto j = col_indices.read(idx);
            auto mat = block_values.read(idx);

            // Load x_j and compute contribution
            Float3 x_j;
            x_j.x = x_buf.read(j * N + 0);
            x_j.y = x_buf.read(j * N + 1);
            x_j.z = x_buf.read(j * N + 2);

            // Compute block * x_j
            Float3 vec;
            vec.x = mat[0][0] * x_j.x + mat[1][0] * x_j.y + mat[2][0] * x_j.z;
            vec.y = mat[0][1] * x_j.x + mat[1][1] * x_j.y + mat[2][1] * x_j.z;
            vec.z = mat[0][2] * x_j.x + mat[1][2] * x_j.y + mat[2][2] * x_j.z;

            // For off-diagonal: also compute transpose contribution to row j
            $if(i != j)
            {
                Float3 x_i;
                x_i.x = x_buf.read(i * N + 0);
                x_i.y = x_buf.read(i * N + 1);
                x_i.z = x_buf.read(i * N + 2);

                Float3 vec_t;
                vec_t.x = mat[0][0] * x_i.x + mat[0][1] * x_i.y + mat[0][2] * x_i.z;
                vec_t.y = mat[1][0] * x_i.x + mat[1][1] * x_i.y + mat[1][2] * x_i.z;
                vec_t.z = mat[2][0] * x_i.x + mat[2][1] * x_i.y + mat[2][2] * x_i.z;

                auto y_j_base = j * N;
                y_buf.atomic(y_j_base + 0).fetch_add(alpha * vec_t.x);
                y_buf.atomic(y_j_base + 1).fetch_add(alpha * vec_t.y);
                y_buf.atomic(y_j_base + 2).fetch_add(alpha * vec_t.z);
            };

            // Always use atomic add (same row may be processed by different warps)
            auto y_i_base = i * N;
            y_buf.atomic(y_i_base + 0).fetch_add(alpha * vec.x);
            y_buf.atomic(y_i_base + 1).fetch_add(alpha * vec.y);
            y_buf.atomic(y_i_base + 2).fetch_add(alpha * vec.z);
        };
    };

    auto shader = stream.device().compile(rbk_sym_kernel);
    stream << shader(A.block_row_indices,
                     A.block_col_indices,
                     block_values_view,
                     x,
                     y,
                     a,
                     static_cast<uint>(A.block_count))
                .dispatch(A.block_count);
}

void Spmv::rbk_sym_spmv_dot(Float                           a,
                            CBCOOMatrixView<Float, 3>       A,
                            CDenseVectorView                x,
                            Float                           b,
                            DenseVectorView                 y,
                            DenseVectorView                 d_dot)
{
    constexpr uint N = 3;
    auto& stream = *_stream;

    // Initialize y: y = b * y or y = 0
    if(b != 0.0f)
    {
        scale_vector(y, b, stream);
    }
    else
    {
        buffer_zero(y, stream);
    }

    // Zero out d_dot (assuming d_dot has size 1)
    buffer_zero(d_dot, stream);

    // Reinterpret the Eigen matrix buffer as Matrix3x3 buffer
    auto block_values_view = reinterpret_as_matrix3x3(A.block_values);

    // Kernel for symmetric SpMV with fused dot product
    Kernel1D rbk_sym_dot_kernel = [&](BufferVar<const int>    row_indices,
                                      BufferVar<const int>    col_indices,
                                      BufferVar<const Matrix3x3> block_values,
                                      BufferVar<const Float>   x_buf,
                                      BufferVar<Float>         y_buf,
                                      BufferVar<Float>         dot_buf,
                                      Float                    alpha,
                                      UInt                     block_count) noexcept
    {
        auto idx = dispatch_id().x;
        $if(idx < block_count)
        {
            auto i = row_indices.read(idx);
            auto j = col_indices.read(idx);
            auto mat = block_values.read(idx);

            // Load x_j
            Float3 x_j;
            x_j.x = x_buf.read(j * N + 0);
            x_j.y = x_buf.read(j * N + 1);
            x_j.z = x_buf.read(j * N + 2);

            // Compute block * x_j
            Float3 vec;
            vec.x = mat[0][0] * x_j.x + mat[1][0] * x_j.y + mat[2][0] * x_j.z;
            vec.y = mat[0][1] * x_j.x + mat[1][1] * x_j.y + mat[2][1] * x_j.z;
            vec.z = mat[0][2] * x_j.x + mat[1][2] * x_j.y + mat[2][2] * x_j.z;

            // Compute local dot product contribution
            Float dot_local = 0.0f;
            $if(i == j)
            {
                // Diagonal: dot = x_j^T * (block * x_j)
                dot_local = alpha * (x_j.x * vec.x + x_j.y * vec.y + x_j.z * vec.z);
            }
            $else
            {
                // Off-diagonal: dot = 2 * x_i^T * (block * x_j)
                Float3 x_i;
                x_i.x = x_buf.read(i * N + 0);
                x_i.y = x_buf.read(i * N + 1);
                x_i.z = x_buf.read(i * N + 2);

                dot_local = 2.0f * alpha * (x_i.x * vec.x + x_i.y * vec.y + x_i.z * vec.z);

                // Also add transpose contribution to y[j]
                Float3 vec_t;
                vec_t.x = mat[0][0] * x_i.x + mat[0][1] * x_i.y + mat[0][2] * x_i.z;
                vec_t.y = mat[1][0] * x_i.x + mat[1][1] * x_i.y + mat[1][2] * x_i.z;
                vec_t.z = mat[2][0] * x_i.x + mat[2][1] * x_i.y + mat[2][2] * x_i.z;

                auto y_j_base = j * N;
                y_buf.atomic(y_j_base + 0).fetch_add(alpha * vec_t.x);
                y_buf.atomic(y_j_base + 1).fetch_add(alpha * vec_t.y);
                y_buf.atomic(y_j_base + 2).fetch_add(alpha * vec_t.z);
            };

            // Atomic add to y[i]
            auto y_i_base = i * N;
            y_buf.atomic(y_i_base + 0).fetch_add(alpha * vec.x);
            y_buf.atomic(y_i_base + 1).fetch_add(alpha * vec.y);
            y_buf.atomic(y_i_base + 2).fetch_add(alpha * vec.z);

            // Atomic add to global dot product
            dot_buf.atomic(0).fetch_add(dot_local);
        };
    };

    auto shader = stream.device().compile(rbk_sym_dot_kernel);
    stream << shader(A.block_row_indices,
                     A.block_col_indices,
                     block_values_view,
                     x,
                     y,
                     d_dot,
                     a,
                     static_cast<uint>(A.block_count))
                .dispatch(A.block_count);
}

void Spmv::cpu_sym_spmv(Float                           a,
                        CBCOOMatrixView<Float, 3>       A,
                        CDenseVectorView                x,
                        Float                           b,
                        DenseVectorView                 y)
{
    constexpr uint N = 3;
    auto& stream = *_stream;

    // Download data from device
    std::vector<int>      row_indices(A.block_count);
    std::vector<int>      col_indices(A.block_count);
    std::vector<Eigen::Matrix<Float, 3, 3>> block_values(A.block_count);
    std::vector<Float>    x_host(x.size());
    std::vector<Float>    y_host(y.size());

    stream << A.block_row_indices.copy_to(row_indices.data())
           << A.block_col_indices.copy_to(col_indices.data())
           << A.block_values.copy_to(block_values.data())
           << x.copy_to(x_host.data())
           << y.copy_to(y_host.data())
           << synchronize();

    // Apply y = b * y
    if(b != 0.0f)
    {
        for(size_t i = 0; i < y_host.size(); ++i)
        {
            y_host[i] *= b;
        }
    }
    else
    {
        std::fill(y_host.begin(), y_host.end(), 0.0f);
    }

    // Compute y += a * A * x (symmetric)
    for(size_t idx = 0; idx < A.block_count; ++idx)
    {
        auto i = row_indices[idx];
        auto j = col_indices[idx];
        const auto& eigen_mat = block_values[idx];

        // Load x_j
        Eigen::Vector3f x_j;
        x_j[0] = x_host[j * N + 0];
        x_j[1] = x_host[j * N + 1];
        x_j[2] = x_host[j * N + 2];

        // Compute block * x_j using Eigen
        Eigen::Vector3f vec = eigen_mat * x_j;

        // Add to y[i]
        auto y_i_base = i * N;
        y_host[y_i_base + 0] += a * vec[0];
        y_host[y_i_base + 1] += a * vec[1];
        y_host[y_i_base + 2] += a * vec[2];

        // For off-diagonal, also add transpose contribution to y[j]
        if(i != j)
        {
            Eigen::Vector3f x_i;
            x_i[0] = x_host[i * N + 0];
            x_i[1] = x_host[i * N + 1];
            x_i[2] = x_host[i * N + 2];

            Eigen::Vector3f vec_t = eigen_mat.transpose() * x_i;

            auto y_j_base = j * N;
            y_host[y_j_base + 0] += a * vec_t[0];
            y_host[y_j_base + 1] += a * vec_t[1];
            y_host[y_j_base + 2] += a * vec_t[2];
        }
    }

    // Upload result back to device
    stream << y.copy_from(y_host.data());
}
}  // namespace uipc::backend::luisa

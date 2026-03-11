#include <luisa/luisa-compute.h>
#include <luisa/dsl/syntax.h>
#include <Eigen/Dense>

namespace uipc::backend::luisa
{
using namespace luisa;
using namespace luisa::compute;

// Type aliases for compatibility
using Float = Var<float>;
using Vector3 = Var<float3>;
using Vector12 = Var<std::array<float, 12>>;
using Vector9 = Var<std::array<float, 9>>;
using Matrix3x3 = Var<float3x3>;
using Matrix9x9 = std::array<std::array<float, 9>, 9>;

// Helper function to extract segment from Vector12 (3 elements starting at offset)
inline float3 segment3(const Vector12& q, uint offset) noexcept
{
    return make_float3(q[offset], q[offset + 1], q[offset + 2]);
}

// Helper function to set segment in Vector9
inline void set_segment3(Vector9& out, uint offset, const float3& v) noexcept
{
    out[offset + 0] = v.x;
    out[offset + 1] = v.y;
    out[offset + 2] = v.z;
}

// Helper for outer product (vector * vector^T)
inline float3x3 outer_product(const float3& a, const float3& b) noexcept
{
    return make_float3x3(
        a.x * b.x, a.y * b.x, a.z * b.x,
        a.x * b.y, a.y * b.y, a.z * b.y,
        a.x * b.z, a.y * b.z, a.z * b.z
    );
}

// Compute shape energy for affine body dynamics
Callable shape_energy = [](Vector12 q) noexcept -> Float
{
    Float ret = 0.0f;
    
    // Extract a1, a2, a3 from q (at offsets 3, 6, 9)
    float3 a1 = segment3(q, 3);
    float3 a2 = segment3(q, 6);
    float3 a3 = segment3(q, 9);

    // Compute squared norms
    Float a1_norm_sq = dot(a1, a1);
    Float a2_norm_sq = dot(a2, a2);
    Float a3_norm_sq = dot(a3, a3);

    // Energy terms for orthogonality constraints
    ret += (a1_norm_sq - 1.0f) * (a1_norm_sq - 1.0f);
    ret += (a2_norm_sq - 1.0f) * (a2_norm_sq - 1.0f);
    ret += (a3_norm_sq - 1.0f) * (a3_norm_sq - 1.0f);

    // Energy terms for perpendicularity constraints
    Float a1_dot_a2 = dot(a1, a2);
    Float a2_dot_a3 = dot(a2, a3);
    Float a3_dot_a1 = dot(a3, a1);

    ret += a1_dot_a2 * a1_dot_a2 * 2.0f;
    ret += a2_dot_a3 * a2_dot_a3 * 2.0f;
    ret += a3_dot_a1 * a3_dot_a1 * 2.0f;

    return ret;
};

// Compute gradient of shape energy (returns 9 elements for a1, a2, a3)
Callable shape_energy_gradient = [](Vector12 q) noexcept -> Vector9
{
    Vector9 ret;

    // Extract a1, a2, a3 from q
    float3 a1 = segment3(q, 3);
    float3 a2 = segment3(q, 6);
    float3 a3 = segment3(q, 9);

    Float a1_norm_sq = dot(a1, a1);
    Float a2_norm_sq = dot(a2, a2);
    Float a3_norm_sq = dot(a3, a3);

    Float a1_dot_a2 = dot(a1, a2);
    Float a2_dot_a3 = dot(a2, a3);
    Float a3_dot_a1 = dot(a3, a1);

    // dE/da1 = 4.0 * (a1.squaredNorm() - 1.0) * a1 + 4.0 * a2.dot(a1) * a2 + 4.0 * (a3.dot(a1)) * a3
    float3 dEda1 = 4.0f * (a1_norm_sq - 1.0f) * a1 + 4.0f * a1_dot_a2 * a2 + 4.0f * a3_dot_a1 * a3;

    // dE/da2 = 4.0 * (a2.squaredNorm() - 1.0) * a2 + 4.0 * a3.dot(a2) * a3 + 4.0 * a1.dot(a2) * a1
    float3 dEda2 = 4.0f * (a2_norm_sq - 1.0f) * a2 + 4.0f * a2_dot_a3 * a3 + 4.0f * a1_dot_a2 * a1;

    // dE/da3 = 4.0 * (a3.squaredNorm() - 1.0) * a3 + 4.0 * a1.dot(a3) * a1 + 4.0 * a2.dot(a3) * a2
    float3 dEda3 = 4.0f * (a3_norm_sq - 1.0f) * a3 + 4.0f * a3_dot_a1 * a1 + 4.0f * a2_dot_a3 * a2;

    set_segment3(ret, 0, dEda1);
    set_segment3(ret, 3, dEda2);
    set_segment3(ret, 6, dEda3);

    return ret;
};

// Helper: compute ddV/ddai (diagonal Hessian block for ai)
Callable ddV_ddai_callable = [](float3 ai, float3 aj, float3 ak) noexcept -> Matrix3x3
{
    Float ai_norm_sq = dot(ai, ai);
    
    // ddV_ddai = 8.0 * ai * ai.transpose()
    //          + 4.0 * (ai.squaredNorm() - 1) * I
    //          + 4.0 * aj * aj.transpose() 
    //          + 4.0 * ak * ak.transpose();
    float3x3 result = 8.0f * outer_product(ai, ai)
                    + 4.0f * (ai_norm_sq - 1.0f) * make_float3x3(1.0f)
                    + 4.0f * outer_product(aj, aj)
                    + 4.0f * outer_product(ak, ak);
    
    return result;
};

// Helper: compute ddV/daidaj (off-diagonal Hessian block)
Callable ddV_daidaj_callable = [](float3 ai, float3 aj, float3 ak) noexcept -> Matrix3x3
{
    Float ai_dot_aj = dot(ai, aj);
    
    // ddV_daidaj = 4.0 * aj * ai.transpose() + 4.0 * ai.dot(aj) * I
    float3x3 result = 4.0f * outer_product(aj, ai) + 4.0f * ai_dot_aj * make_float3x3(1.0f);
    
    return result;
};

// Compute Hessian blocks of shape energy
Callable shape_energy_hessian_blocks = [](Vector12 q) noexcept 
    -> std::tuple<Matrix3x3, Matrix3x3, Matrix3x3, Matrix3x3, Matrix3x3, Matrix3x3>
{
    // Extract a1, a2, a3 from q
    float3 a1 = segment3(q, 3);
    float3 a2 = segment3(q, 6);
    float3 a3 = segment3(q, 9);

    Matrix3x3 ddVdda1 = ddV_ddai_callable(a1, a2, a3);
    Matrix3x3 ddVdda2 = ddV_ddai_callable(a2, a3, a1);
    Matrix3x3 ddVdda3 = ddV_ddai_callable(a3, a1, a2);

    Matrix3x3 ddVda1da2 = ddV_daidaj_callable(a1, a2, a3);
    Matrix3x3 ddVda1da3 = ddV_daidaj_callable(a1, a3, a2);
    Matrix3x3 ddVda2da3 = ddV_daidaj_callable(a2, a3, a1);

    return {ddVdda1, ddVdda2, ddVdda3, ddVda1da2, ddVda1da3, ddVda2da3};
};

// Helper to get element from matrix
inline float get_matrix_element(const float3x3& m, uint row, uint col) noexcept
{
    // float3x3 is column-major, m[col] gives the column vector
    float3 column = m[col];
    if(row == 0) return column.x;
    if(row == 1) return column.y;
    return column.z;
}

// Compute full 9x9 Hessian matrix as array
Callable shape_energy_hessian = [](Vector12 q) noexcept -> Matrix9x9
{
    Matrix9x9 H{};
    
    // Get Hessian blocks
    auto [ddVdda1, ddVdda2, ddVdda3, ddVda1da2, ddVda1da3, ddVda2da3] = shape_energy_hessian_blocks(q);

    // Assemble 9x9 Hessian matrix
    // Layout:
    // [ ddVdda1    ddVda1da2  ddVda1da3 ]
    // [ ddVda1da2^T ddVdda2   ddVda2da3 ]
    // [ ddVda1da3^T ddVda2da3^T ddVdda3 ]
    
    for(uint i = 0; i < 3; i++)
    {
        for(uint j = 0; j < 3; j++)
        {
            // Block (0,0): ddVdda1
            H[i][j] = get_matrix_element(ddVdda1, i, j);
            // Block (0,1): ddVda1da2
            H[i][j + 3] = get_matrix_element(ddVda1da2, i, j);
            // Block (0,2): ddVda1da3
            H[i][j + 6] = get_matrix_element(ddVda1da3, i, j);
            
            // Block (1,0): ddVda1da2^T
            H[i + 3][j] = get_matrix_element(ddVda1da2, j, i);
            // Block (1,1): ddVdda2
            H[i + 3][j + 3] = get_matrix_element(ddVdda2, i, j);
            // Block (1,2): ddVda2da3
            H[i + 3][j + 6] = get_matrix_element(ddVda2da3, i, j);
            
            // Block (2,0): ddVda1da3^T
            H[i + 6][j] = get_matrix_element(ddVda1da3, j, i);
            // Block (2,1): ddVda2da3^T
            H[i + 6][j + 3] = get_matrix_element(ddVda2da3, j, i);
            // Block (2,2): ddVdda3
            H[i + 6][j + 6] = get_matrix_element(ddVdda3, i, j);
        }
    }

    return H;
};

}  // namespace uipc::backend::luisa

#include <affine_body/utils.h>
#include <luisa/core/mathematics.h>

namespace uipc::backend::luisa
{
UIPC_GENERIC Matrix3x3 q_to_A(const Vector12& q)
{
    Matrix3x3 A = Matrix3x3::fill(0.0f);
    // Row 0: q[3], q[4], q[5]
    A[0][0] = q[3];
    A[0][1] = q[4];
    A[0][2] = q[5];
    // Row 1: q[6], q[7], q[8]
    A[1][0] = q[6];
    A[1][1] = q[7];
    A[1][2] = q[8];
    // Row 2: q[9], q[10], q[11]
    A[2][0] = q[9];
    A[2][1] = q[10];
    A[2][2] = q[11];
    return A;
}

UIPC_GENERIC Vector9 A_to_q(const Matrix3x3& A)
{
    Vector9 q = Vector9::zeros();
    // Row 0 -> q[0..2]
    q[0] = A[0][0];
    q[1] = A[0][1];
    q[2] = A[0][2];
    // Row 1 -> q[3..5]
    q[3] = A[1][0];
    q[4] = A[1][1];
    q[5] = A[1][2];
    // Row 2 -> q[6..8]
    q[6] = A[2][0];
    q[7] = A[2][1];
    q[8] = A[2][2];
    return q;
}

UIPC_GENERIC Vector9 F_to_A(const Vector9& F)
{
    Vector9 A;
    // Reorder from column-major F to row-major A
    A[0] = F[0];
    A[1] = F[3];
    A[2] = F[6];
    A[3] = F[1];
    A[4] = F[4];
    A[5] = F[7];
    A[6] = F[2];
    A[7] = F[5];
    A[8] = F[8];
    return A;
}

UIPC_GENERIC Matrix9x9 HF_to_HA(const Matrix9x9& HF)
{
    Matrix9x9 HA;
    // Mapping from column-major F indices to row-major A indices
    // A[0]=F[0], A[1]=F[3], A[2]=F[6]
    // A[3]=F[1], A[4]=F[4], A[5]=F[7]
    // A[6]=F[2], A[7]=F[5], A[8]=F[8]
    const int map[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
    
    for(int i = 0; i < 9; ++i)
    {
        for(int j = 0; j < 9; ++j)
        {
            HA(map[i], map[j]) = HF(i, j);
        }
    }
    return HA;
}

UIPC_GENERIC Matrix4x4 q_to_transform(const Vector12& q)
{
    Matrix4x4 trans = Matrix4x4::fill(0.0f);
    // translation: q[0], q[1], q[2] -> column 3, rows 0-2
    trans[3][0] = q[0];
    trans[3][1] = q[1];
    trans[3][2] = q[2];
    
    // rotation matrix (rows 0-2 of A) stored in columns 0-2
    // Row 0 of A: q[3], q[4], q[5]
    trans[0][0] = q[3];
    trans[1][0] = q[4];
    trans[2][0] = q[5];
    
    // Row 1 of A: q[6], q[7], q[8]
    trans[0][1] = q[6];
    trans[1][1] = q[7];
    trans[2][1] = q[8];
    
    // Row 2 of A: q[9], q[10], q[11]
    trans[0][2] = q[9];
    trans[1][2] = q[10];
    trans[2][2] = q[11];
    
    // last row: 0, 0, 0, 1
    trans[3][3] = 1.0f;
    
    return trans;
}

UIPC_GENERIC Vector12 transform_to_q(const Matrix4x4& trans)
{
    Vector12 q;
    // translation from column 3, rows 0-2
    q[0] = trans[3][0];
    q[1] = trans[3][1];
    q[2] = trans[3][2];
    
    // rotation: columns 0-2, rows 0-2 stored as rows in q
    // Row 0: column 0, rows 0-2
    q[3] = trans[0][0];
    q[4] = trans[1][0];
    q[5] = trans[2][0];
    
    // Row 1: column 1, rows 0-2
    q[6] = trans[0][1];
    q[7] = trans[1][1];
    q[8] = trans[2][1];
    
    // Row 2: column 2, rows 0-2
    q[9] = trans[0][2];
    q[10] = trans[1][2];
    q[11] = trans[2][2];
    
    return q;
}

UIPC_GENERIC Matrix4x4 q_v_to_transform_v(const Vector12& q)
{
    Matrix4x4 trans = Matrix4x4::fill(0.0f);
    // translation: q[0], q[1], q[2] -> column 3, rows 0-2
    trans[3][0] = q[0];
    trans[3][1] = q[1];
    trans[3][2] = q[2];
    
    // rotation matrix (rows 0-2 of A) stored in columns 0-2
    // Row 0 of A: q[3], q[4], q[5]
    trans[0][0] = q[3];
    trans[1][0] = q[4];
    trans[2][0] = q[5];
    
    // Row 1 of A: q[6], q[7], q[8]
    trans[0][1] = q[6];
    trans[1][1] = q[7];
    trans[2][1] = q[8];
    
    // Row 2 of A: q[9], q[10], q[11]
    trans[0][2] = q[9];
    trans[1][2] = q[10];
    trans[2][2] = q[11];
    
    // last row fill zero (already zero from initialization)
    
    return trans;
}

UIPC_GENERIC Vector12 transform_v_to_q_v(const Matrix4x4& transform_v)
{
    // the same to transform_to_q
    return transform_to_q(transform_v);
}

UIPC_GENERIC void orthonormal_basis(Vector3& t, Vector3& n, Vector3& b)
{
    t = luisa::normalize(t);
    Vector3 unit_x{1.0f, 0.0f, 0.0f};
    Vector3 unit_y{0.0f, 1.0f, 0.0f};
    Vector3 test_vector = unit_x;
    if(std::abs(luisa::dot(t, test_vector)) > 0.9f)
    {
        test_vector = unit_y;
    }
    n = luisa::normalize(luisa::cross(t, test_vector));
    b = luisa::normalize(luisa::cross(t, n));
}
}  // namespace uipc::backend::luisa

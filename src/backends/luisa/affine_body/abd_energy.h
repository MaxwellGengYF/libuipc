#pragma once
#include <luisa/core/basic_types.h>
#include <luisa/core/mathematics.h>
#include <array>

namespace uipc::backend::luisa
{
// Generic device/host function decorator
#ifndef LUISA_GENERIC
#define LUISA_GENERIC inline
#endif

// Fixed-size vector/matrix types for affine body dynamics
using Vector12 = std::array<double, 12>;
using Vector9  = std::array<double, 9>;
using Matrix9x9 = std::array<std::array<double, 9>, 9>;

// Helper: compute dot product for 3D vectors
LUISA_GENERIC double dot3(const luisa::double3& a, const luisa::double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Helper: compute outer product of two 3D vectors
LUISA_GENERIC luisa::double3x3 outer3(const luisa::double3& a, const luisa::double3& b)
{
    return luisa::double3x3{
        luisa::double3{a.x * b.x, a.y * b.x, a.z * b.x},
        luisa::double3{a.x * b.y, a.y * b.y, a.z * b.y},
        luisa::double3{a.x * b.z, a.y * b.z, a.z * b.z}};
}

//tex:
//$$ S = \frac{V_{\perp}}{\kappa v} $$
//we don't include the $\kappa$ and $v$ calculate it by yourself and multiply it

//tex:
//$$
// \frac{V_{\perp}}{\kappa v} =\sum\left(a_{i} \cdot a_{i}-1\right)^{2}
// +\sum_{i \neq j}\left(a_{i} \cdot a_{j}\right)^{2}
//$$
LUISA_GENERIC double shape_energy(const Vector12& q)
{
    // Extract column vectors from affine matrix A (stored as [a0, a1, a2] in q[0..8])
    luisa::double3 a0{q[0], q[1], q[2]};
    luisa::double3 a1{q[3], q[4], q[5]};
    luisa::double3 a2{q[6], q[7], q[8]};

    double d0 = dot3(a0, a0) - 1.0;
    double d1 = dot3(a1, a1) - 1.0;
    double d2 = dot3(a2, a2) - 1.0;

    double d01 = dot3(a0, a1);
    double d02 = dot3(a0, a2);
    double d12 = dot3(a1, a2);

    // Energy = sum of (|ai|^2 - 1)^2 + sum of (ai·aj)^2 for i != j
    return d0 * d0 + d1 * d1 + d2 * d2 + d01 * d01 + d02 * d02 + d12 * d12;
}

//tex:
// $$\frac{1}{\kappa v}\frac{\partial V_{\perp}}{\partial a_{i}}=
//2 \left(2\left(a_{i} \cdot a_{i}-1\right) a_{i}
//+ \sum a_{j}  (a_{j} \cdot a_{i})\right)$$
LUISA_GENERIC Vector9 shape_energy_gradient(const Vector12& q)
{
    // Extract column vectors from affine matrix A
    luisa::double3 a0{q[0], q[1], q[2]};
    luisa::double3 a1{q[3], q[4], q[5]};
    luisa::double3 a2{q[6], q[7], q[8]};

    // Precompute dot products
    double a0_dot_a0 = dot3(a0, a0);
    double a1_dot_a1 = dot3(a1, a1);
    double a2_dot_a2 = dot3(a2, a2);
    double a0_dot_a1 = dot3(a0, a1);
    double a0_dot_a2 = dot3(a0, a2);
    double a1_dot_a2 = dot3(a1, a2);

    Vector9 grad;

    // Gradient w.r.t. a0: 2 * (2*(a0·a0-1)*a0 + a1*(a1·a0) + a2*(a2·a0))
    double scale0 = 2.0 * (a0_dot_a0 - 1.0);
    grad[0] = 2.0 * (scale0 * a0.x + a1.x * a0_dot_a1 + a2.x * a0_dot_a2);
    grad[1] = 2.0 * (scale0 * a0.y + a1.y * a0_dot_a1 + a2.y * a0_dot_a2);
    grad[2] = 2.0 * (scale0 * a0.z + a1.z * a0_dot_a1 + a2.z * a0_dot_a2);

    // Gradient w.r.t. a1: 2 * (2*(a1·a1-1)*a1 + a0*(a0·a1) + a2*(a2·a1))
    double scale1 = 2.0 * (a1_dot_a1 - 1.0);
    grad[3] = 2.0 * (scale1 * a1.x + a0.x * a0_dot_a1 + a2.x * a1_dot_a2);
    grad[4] = 2.0 * (scale1 * a1.y + a0.y * a0_dot_a1 + a2.y * a1_dot_a2);
    grad[5] = 2.0 * (scale1 * a1.z + a0.z * a0_dot_a1 + a2.z * a1_dot_a2);

    // Gradient w.r.t. a2: 2 * (2*(a2·a2-1)*a2 + a0*(a0·a2) + a1*(a1·a2))
    double scale2 = 2.0 * (a2_dot_a2 - 1.0);
    grad[6] = 2.0 * (scale2 * a2.x + a0.x * a0_dot_a2 + a1.x * a1_dot_a2);
    grad[7] = 2.0 * (scale2 * a2.y + a0.y * a0_dot_a2 + a1.y * a1_dot_a2);
    grad[8] = 2.0 * (scale2 * a2.z + a0.z * a0_dot_a2 + a1.z * a1_dot_a2);

    return grad;
}

LUISA_GENERIC Matrix9x9 shape_energy_hessian(const Vector12& q)
{
    // Extract column vectors from affine matrix A
    luisa::double3 a0{q[0], q[1], q[2]};
    luisa::double3 a1{q[3], q[4], q[5]};
    luisa::double3 a2{q[6], q[7], q[8]};

    // Precompute dot products
    double a0_dot_a0 = dot3(a0, a0);
    double a1_dot_a1 = dot3(a1, a1);
    double a2_dot_a2 = dot3(a2, a2);
    double a0_dot_a1 = dot3(a0, a1);
    double a0_dot_a2 = dot3(a0, a2);
    double a1_dot_a2 = dot3(a1, a2);

    Matrix9x9 H{};

    // Diagonal blocks
    // H_00 = 2 * (2*(a0·a0-1)*I + 4*a0*a0^T + 2*(a1·a1 + a2·a2)*I)
    double diag0_scale = 4.0 * (a0_dot_a0 - 1.0) + 4.0 * (a1_dot_a1 + a2_dot_a2);
    H[0][0] = diag0_scale + 8.0 * a0.x * a0.x;
    H[0][1] = 8.0 * a0.x * a0.y;
    H[0][2] = 8.0 * a0.x * a0.z;
    H[1][0] = 8.0 * a0.y * a0.x;
    H[1][1] = diag0_scale + 8.0 * a0.y * a0.y;
    H[1][2] = 8.0 * a0.y * a0.z;
    H[2][0] = 8.0 * a0.z * a0.x;
    H[2][1] = 8.0 * a0.z * a0.y;
    H[2][2] = diag0_scale + 8.0 * a0.z * a0.z;

    // H_11 = 2 * (2*(a1·a1-1)*I + 4*a1*a1^T + 2*(a0·a0 + a2·a2)*I)
    double diag1_scale = 4.0 * (a1_dot_a1 - 1.0) + 4.0 * (a0_dot_a0 + a2_dot_a2);
    H[3][3] = diag1_scale + 8.0 * a1.x * a1.x;
    H[3][4] = 8.0 * a1.x * a1.y;
    H[3][5] = 8.0 * a1.x * a1.z;
    H[4][3] = 8.0 * a1.y * a1.x;
    H[4][4] = diag1_scale + 8.0 * a1.y * a1.y;
    H[4][5] = 8.0 * a1.y * a1.z;
    H[5][3] = 8.0 * a1.z * a1.x;
    H[5][4] = 8.0 * a1.z * a1.y;
    H[5][5] = diag1_scale + 8.0 * a1.z * a1.z;

    // H_22 = 2 * (2*(a2·a2-1)*I + 4*a2*a2^T + 2*(a0·a0 + a1·a1)*I)
    double diag2_scale = 4.0 * (a2_dot_a2 - 1.0) + 4.0 * (a0_dot_a0 + a1_dot_a1);
    H[6][6] = diag2_scale + 8.0 * a2.x * a2.x;
    H[6][7] = 8.0 * a2.x * a2.y;
    H[6][8] = 8.0 * a2.x * a2.z;
    H[7][6] = 8.0 * a2.y * a2.x;
    H[7][7] = diag2_scale + 8.0 * a2.y * a2.y;
    H[7][8] = 8.0 * a2.y * a2.z;
    H[8][6] = 8.0 * a2.z * a2.x;
    H[8][7] = 8.0 * a2.z * a2.y;
    H[8][8] = diag2_scale + 8.0 * a2.z * a2.z;

    // Off-diagonal blocks (symmetric)
    // H_01 = H_10^T = 2 * (2*a0*a1^T + 2*(a0·a1)*I)
    H[0][3] = 4.0 * (a0.x * a1.x + a0_dot_a1);
    H[0][4] = 4.0 * a0.x * a1.y;
    H[0][5] = 4.0 * a0.x * a1.z;
    H[1][3] = 4.0 * a0.y * a1.x;
    H[1][4] = 4.0 * (a0.y * a1.y + a0_dot_a1);
    H[1][5] = 4.0 * a0.y * a1.z;
    H[2][3] = 4.0 * a0.z * a1.x;
    H[2][4] = 4.0 * a0.z * a1.y;
    H[2][5] = 4.0 * (a0.z * a1.z + a0_dot_a1);

    H[3][0] = 4.0 * (a1.x * a0.x + a0_dot_a1);
    H[3][1] = 4.0 * a1.x * a0.y;
    H[3][2] = 4.0 * a1.x * a0.z;
    H[4][0] = 4.0 * a1.y * a0.x;
    H[4][1] = 4.0 * (a1.y * a0.y + a0_dot_a1);
    H[4][2] = 4.0 * a1.y * a0.z;
    H[5][0] = 4.0 * a1.z * a0.x;
    H[5][1] = 4.0 * a1.z * a0.y;
    H[5][2] = 4.0 * (a1.z * a0.z + a0_dot_a1);

    // H_02 = H_20^T = 2 * (2*a0*a2^T + 2*(a0·a2)*I)
    H[0][6] = 4.0 * (a0.x * a2.x + a0_dot_a2);
    H[0][7] = 4.0 * a0.x * a2.y;
    H[0][8] = 4.0 * a0.x * a2.z;
    H[1][6] = 4.0 * a0.y * a2.x;
    H[1][7] = 4.0 * (a0.y * a2.y + a0_dot_a2);
    H[1][8] = 4.0 * a0.y * a2.z;
    H[2][6] = 4.0 * a0.z * a2.x;
    H[2][7] = 4.0 * a0.z * a2.y;
    H[2][8] = 4.0 * (a0.z * a2.z + a0_dot_a2);

    H[6][0] = 4.0 * (a2.x * a0.x + a0_dot_a2);
    H[6][1] = 4.0 * a2.x * a0.y;
    H[6][2] = 4.0 * a2.x * a0.z;
    H[7][0] = 4.0 * a2.y * a0.x;
    H[7][1] = 4.0 * (a2.y * a0.y + a0_dot_a2);
    H[7][2] = 4.0 * a2.y * a0.z;
    H[8][0] = 4.0 * a2.z * a0.x;
    H[8][1] = 4.0 * a2.z * a0.y;
    H[8][2] = 4.0 * (a2.z * a0.z + a0_dot_a2);

    // H_12 = H_21^T = 2 * (2*a1*a2^T + 2*(a1·a2)*I)
    H[3][6] = 4.0 * (a1.x * a2.x + a1_dot_a2);
    H[3][7] = 4.0 * a1.x * a2.y;
    H[3][8] = 4.0 * a1.x * a2.z;
    H[4][6] = 4.0 * a1.y * a2.x;
    H[4][7] = 4.0 * (a1.y * a2.y + a1_dot_a2);
    H[4][8] = 4.0 * a1.y * a2.z;
    H[5][6] = 4.0 * a1.z * a2.x;
    H[5][7] = 4.0 * a1.z * a2.y;
    H[5][8] = 4.0 * (a1.z * a2.z + a1_dot_a2);

    H[6][3] = 4.0 * (a2.x * a1.x + a1_dot_a2);
    H[6][4] = 4.0 * a2.x * a1.y;
    H[6][5] = 4.0 * a2.x * a1.z;
    H[7][3] = 4.0 * a2.y * a1.x;
    H[7][4] = 4.0 * (a2.y * a1.y + a1_dot_a2);
    H[7][5] = 4.0 * a2.y * a1.z;
    H[8][3] = 4.0 * a2.z * a1.x;
    H[8][4] = 4.0 * a2.z * a1.y;
    H[8][5] = 4.0 * (a2.z * a1.z + a1_dot_a2);

    return H;
}

}  // namespace uipc::backend::luisa

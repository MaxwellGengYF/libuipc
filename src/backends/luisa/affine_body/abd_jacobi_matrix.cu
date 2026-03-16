#include <affine_body/abd_jacobi_matrix.h>

namespace uipc::backend::luisa
{
__attribute__((always_inline)) Vector12 operator*(const ABDJacobi::ABDJacobiT& j, const float3& g)
{
    Vector12    g12;
    const auto& x     = j.J().m_x_bar;
    g12.set_segment3(0, g);
    g12.set_segment3(3, x * g.x);
    g12.set_segment3(6, x * g.y);
    g12.set_segment3(9, x * g.z);
    return g12;
}

__attribute__((always_inline)) float3 operator*(const ABDJacobi& j, const Vector12& q)
{
    const auto& t  = q.segment3(0);
    const auto& a1 = q.segment3(3);
    const auto& a2 = q.segment3(6);
    const auto& a3 = q.segment3(9);
    const auto& x  = j.m_x_bar;
    return float3{dot(x, a1), dot(x, a2), dot(x, a3)} + t;
}

// no translation component
__attribute__((always_inline)) float3 ABDJacobi::vec_x(const Vector12& q) const
{
    const auto& a1 = q.segment3(3);
    const auto& a2 = q.segment3(6);
    const auto& a3 = q.segment3(9);
    const auto& x  = m_x_bar;
    return float3{dot(x, a1), dot(x, a2), dot(x, a3)};
}

//tex:
//$$
//\left[\begin{array}{ccc|ccc:ccc:ccc}
//1 &   &   & \bar{x}_1 & \bar{x}_2 & \bar{x}_3 &  &  &  &  &  & \\
//& 1 &   &  &  &  & \bar{x}_1 & \bar{x}_2 & \bar{x}_3 &  &  &  \\
//&   & 1 &  &  &  &  &  &  &  \bar{x}_1 & \bar{x}_2 & \bar{x}_3\\
//\end{array}\right]
//$$
__attribute__((always_inline)) Matrix3x12 ABDJacobi::to_mat() const
{
    Matrix3x12  ret;
    // Initialize to zero
    for(int i = 0; i < 36; ++i) ret.data[i] = 0.0f;
    
    const auto& x = m_x_bar;
    // Identity for translation part (rows 0,1,2 cols 0,1,2)
    ret(0, 0) = 1.0f;
    ret(1, 1) = 1.0f;
    ret(2, 2) = 1.0f;
    
    // Row 0: x^T at cols 3,4,5
    ret(0, 3) = x.x;
    ret(0, 4) = x.y;
    ret(0, 5) = x.z;
    
    // Row 1: x^T at cols 6,7,8
    ret(1, 6) = x.x;
    ret(1, 7) = x.y;
    ret(1, 8) = x.z;
    
    // Row 2: x^T at cols 9,10,11
    ret(2, 9) = x.x;
    ret(2, 10) = x.y;
    ret(2, 11) = x.z;
    
    return ret;
}

__attribute__((always_inline)) Matrix12x12 ABDJacobi::JT_H_J(const ABDJacobiT& lhs_J_T,
                                           const float3x3&  Hessian,
                                           const ABDJacobi&  rhs_J)
{
    //tex:
    //$$
    //\begin{bmatrix}
    //\mathbf{H} & \mathbf{c}_1\cdot\bar{\mathbf{y}}^{T} & \mathbf{c}_2\cdot\bar{\mathbf{y}}^{T} & \mathbf{c}_3\cdot\bar{\mathbf{y}}^{T}\\
    //\bar{\mathbf{x}}\cdot\mathbf{r}_1 & H_{11}\cdot\bar{\mathbf{x}}\cdot\bar{\mathbf{y}}^{T}  & H_{12}\cdot \bar{\mathbf{x}}\cdot\bar{\mathbf{y}}^{T} & H_{13}\cdot \bar{\mathbf{x}}\cdot\bar{\mathbf{y}}^{T}\\
    //\bar{\mathbf{x}}\cdot\mathbf{r}_2 & H_{21}\cdot \bar{\mathbf{x}}\cdot\bar{\mathbf{y}}^{T} & H_{22}\cdot \bar{\mathbf{x}}\cdot\bar{\mathbf{y}}^{T} & H_{23}\cdot \bar{\mathbf{x}}\cdot\bar{\mathbf{y}}^{T}\\
    //\bar{\mathbf{x}}\cdot\mathbf{r}_3 & H_{31}\cdot \bar{\mathbf{x}}\cdot\bar{\mathbf{y}}^{T} & H_{32}\cdot \bar{\mathbf{x}}\cdot\bar{\mathbf{y}}^{T} & H_{33}\cdot \bar{\mathbf{x}}\cdot\bar{\mathbf{y}}^{T}\\
    //\end{bmatrix}
    //$$

    Matrix12x12 ret = Matrix12x12::zeros();
    auto        x   = lhs_J_T.J().m_x_bar;
    auto        y   = rhs_J.m_x_bar;
    
    // ret.block<3, 3>(0, 0) = Hessian
    ret.set_block3x3(0, 0, Hessian);

    // Hessian Col * y
    // ret.block<3, 3>(0, 3) = Hessian.block<3, 1>(0, 0) * y.transpose();
    float3 col0 = float3{Hessian[0][0], Hessian[0][1], Hessian[0][2]};
    ret.set_row_block3(0, 3, col0 * y);
    
    // ret.block<3, 3>(0, 6) = Hessian.block<3, 1>(0, 1) * y.transpose();
    float3 col1 = float3{Hessian[1][0], Hessian[1][1], Hessian[1][2]};
    ret.set_row_block3(0, 6, col1 * y);
    
    // ret.block<3, 3>(0, 9) = Hessian.block<3, 1>(0, 2) * y.transpose();
    float3 col2 = float3{Hessian[2][0], Hessian[2][1], Hessian[2][2]};
    ret.set_row_block3(0, 9, col2 * y);

    // x * Hessian Row
    // ret.block<3, 3>(3, 0) = x * Hessian.block<1, 3>(0, 0);
    float3 row0 = float3{Hessian[0][0], Hessian[1][0], Hessian[2][0]};
    ret.set_col_block3(3, 0, x * row0);
    
    // ret.block<3, 3>(6, 0) = x * Hessian.block<1, 3>(1, 0);
    float3 row1 = float3{Hessian[0][1], Hessian[1][1], Hessian[2][1]};
    ret.set_col_block3(6, 0, x * row1);
    
    // ret.block<3, 3>(9, 0) = x * Hessian.block<1, 3>(2, 0);
    float3 row2 = float3{Hessian[0][2], Hessian[1][2], Hessian[2][2]};
    ret.set_col_block3(9, 0, x * row2);

    float3x3 x_y = x * transpose(float3x3(y));

    // kronecker product
    //tex: $$ \mathbf{H} \otimes (\bar{\mathbf{x}}\cdot\bar{\mathbf{y}}^{T})$$
    ret.add_block3x3(3, 3, x_y * Hessian[0][0]);
    ret.add_block3x3(3, 6, x_y * Hessian[0][1]);
    ret.add_block3x3(3, 9, x_y * Hessian[0][2]);

    ret.add_block3x3(6, 3, x_y * Hessian[1][0]);
    ret.add_block3x3(6, 6, x_y * Hessian[1][1]);
    ret.add_block3x3(6, 9, x_y * Hessian[1][2]);

    ret.add_block3x3(9, 3, x_y * Hessian[2][0]);
    ret.add_block3x3(9, 6, x_y * Hessian[2][1]);
    ret.add_block3x3(9, 9, x_y * Hessian[2][2]);

    return ret;
}

__attribute__((always_inline)) Vector12 operator*(const ABDJacobiDyadicMass& JTJ, const Vector12& p)
{
    Vector12    ret;
    const auto& m = JTJ.m_mass;
    const auto& D = JTJ.m_mass_times_dyadic_x_bar;
    const auto& x = JTJ.m_mass_times_x_bar;

    const auto& p_p  = p.segment3(0);
    const auto& p_a1 = p.segment3(3);
    const auto& p_a2 = p.segment3(6);
    const auto& p_a3 = p.segment3(9);

    //tex:
    //$$
    //\left[\begin{array}{c}
    //\mathbf{p}_{\mathbf{a}} \mathbf{x} + m\mathbf{p}_{\mathbf{p}}\\
    //\mathbf{D} \cdot \mathbf{p}_{\mathbf{a}_1} + \mathbf{x} \cdot p_1\\
    //\mathbf{D} \cdot \mathbf{p}_{\mathbf{a}_2} + \mathbf{x} \cdot p_2\\
    //\mathbf{D} \cdot \mathbf{p}_{\mathbf{a}_3} + \mathbf{x} \cdot p_3\\
    //\end{array}\right]_{12\times1}
    //$$

    ret[0] = dot(x, p_a1) + m * p_p.x;
    ret[1] = dot(x, p_a2) + m * p_p.y;
    ret[2] = dot(x, p_a3) + m * p_p.z;

    ret.set_segment3(3, D * p_a1 + x * p_p.x);
    ret.set_segment3(6, D * p_a2 + x * p_p.y);
    ret.set_segment3(9, D * p_a3 + x * p_p.z);

    return ret;
}

__attribute__((always_inline)) ABDJacobiDyadicMass& ABDJacobiDyadicMass::operator+=(const ABDJacobiDyadicMass& rhs)
{
    m_mass += rhs.m_mass;
    m_mass_times_x_bar += rhs.m_mass_times_x_bar;
    m_mass_times_dyadic_x_bar += rhs.m_mass_times_dyadic_x_bar;
    return *this;
}

//tex:
//$$
//\mathbf{J}^T\mathbf{J}
//= \left[\begin{array}{cccccccccccc}
//1 & 0 & 0 & \bar{x}_{1} & \bar{x}_{2} & \bar{x}_{3} & 0 & 0 & 0 & 0 & 0 & 0\\
//0 & 1 & 0 & 0 & 0 & 0 & \bar{x}_{1} & \bar{x}_{2} & \bar{x}_{3} & 0 & 0 & 0\\
//0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & \bar{x}_{1} & \bar{x}_{2} & \bar{x}_{3}\\
//\bar{x}_{1} & 0 & 0 & \bar{x}_{1}^{2} & \bar{x}_{1} \bar{x}_{2} & \bar{x}_{1} \bar{x}_{3} & 0 & 0 & 0 & 0 & 0 & 0\\
//\bar{x}_{2} & 0 & 0 & \bar{x}_{1} \bar{x}_{2} & \bar{x}_{2}^{2} & \bar{x}_{2} \bar{x}_{3} & 0 & 0 & 0 & 0 & 0 & 0\\
//\bar{x}_{3} & 0 & 0 & \bar{x}_{1} \bar{x}_{3} & \bar{x}_{2} \bar{x}_{3} & \bar{x}_{3}^{2} & 0 & 0 & 0 & 0 & 0 & 0\\
//0 & \bar{x}_{1} & 0 & 0 & 0 & 0 & \bar{x}_{1}^{2} & \bar{x}_{1} \bar{x}_{2} & \bar{x}_{1} \bar{x}_{3} & 0 & 0 & 0\\
//0 & \bar{x}_{2} & 0 & 0 & 0 & 0 & \bar{x}_{1} \bar{x}_{2} & \bar{x}_{2}^{2} & \bar{x}_{2} \bar{x}_{3} & 0 & 0 & 0\\
//0 & \bar{x}_{3} & 0 & 0 & 0 & 0 & \bar{x}_{1} \bar{x}_{3} & \bar{x}_{2} \bar{x}_{3} & \bar{x}_{3}^{2} & 0 & 0 & 0\\
//0 & 0 & \bar{x}_{1} & 0 & 0 & 0 & 0 & 0 & 0 & \bar{x}_{1}^{2} & \bar{x}_{1} \bar{x}_{2} & \bar{x}_{1} \bar{x}_{3}\\
//0 & 0 & \bar{x}_{2} & 0 & 0 & 0 & 0 & 0 & 0 & \bar{x}_{1} \bar{x}_{2} & \bar{x}_{2}^{2} & \bar{x}_{2} \bar{x}_{3}\\
//0 & 0 & \bar{x}_{3} & 0 & 0 & 0 & 0 & 0 & 0 & \bar{x}_{1} \bar{x}_{3} & \bar{x}_{2} \bar{x}_{3} & \bar{x}_{3}^{2}\end{array}\right]
//$$
__attribute__((always_inline)) void ABDJacobiDyadicMass::add_to(Matrix12x12& h) const
{
    // row 0 + col 0
    h(0, 0) += m_mass;
    h.add_row_block3(0, 3, m_mass_times_x_bar);
    h.add_col_block3(3, 0, m_mass_times_x_bar);

    // row 1 + col 1
    h(1, 1) += m_mass;
    h.add_row_block3(1, 6, m_mass_times_x_bar);
    h.add_col_block3(6, 1, m_mass_times_x_bar);

    // row 2 + col 2
    h(2, 2) += m_mass;
    h.add_row_block3(2, 9, m_mass_times_x_bar);
    h.add_col_block3(9, 2, m_mass_times_x_bar);

    // block<3,3> at (3,3)  (6,6)  (9,9)
    h.add_block3x3(3, 3, m_mass_times_dyadic_x_bar);
    h.add_block3x3(6, 6, m_mass_times_dyadic_x_bar);
    h.add_block3x3(9, 9, m_mass_times_dyadic_x_bar);
}

__attribute__((always_inline)) Matrix12x12 ABDJacobiDyadicMass::to_mat() const
{
    Matrix12x12 h = Matrix12x12::zeros();
    add_to(h);
    return h;
}

__attribute__((always_inline)) ABDJacobiDyadicMass ABDJacobiDyadicMass::atomic_add(ABDJacobiDyadicMass& dst,
                                                                const ABDJacobiDyadicMass& src)
{
    ABDJacobiDyadicMass ret;
    
    // Atomic add for mass (double) - returns old value
    ret.m_mass = luisa::atomic_add(&dst.m_mass, src.m_mass);
    
    // Atomic add for float3 - returns old values
    ret.m_mass_times_x_bar.x = luisa::atomic_add(&dst.m_mass_times_x_bar.x, src.m_mass_times_x_bar.x);
    ret.m_mass_times_x_bar.y = luisa::atomic_add(&dst.m_mass_times_x_bar.y, src.m_mass_times_x_bar.y);
    ret.m_mass_times_x_bar.z = luisa::atomic_add(&dst.m_mass_times_x_bar.z, src.m_mass_times_x_bar.z);
    
    // Atomic add for float3x3 (column-major matrix) - returns old values
    // Column 0
    ret.m_mass_times_dyadic_x_bar[0][0] = luisa::atomic_add(&dst.m_mass_times_dyadic_x_bar[0][0], src.m_mass_times_dyadic_x_bar[0][0]);
    ret.m_mass_times_dyadic_x_bar[0][1] = luisa::atomic_add(&dst.m_mass_times_dyadic_x_bar[0][1], src.m_mass_times_dyadic_x_bar[0][1]);
    ret.m_mass_times_dyadic_x_bar[0][2] = luisa::atomic_add(&dst.m_mass_times_dyadic_x_bar[0][2], src.m_mass_times_dyadic_x_bar[0][2]);
    // Column 1
    ret.m_mass_times_dyadic_x_bar[1][0] = luisa::atomic_add(&dst.m_mass_times_dyadic_x_bar[1][0], src.m_mass_times_dyadic_x_bar[1][0]);
    ret.m_mass_times_dyadic_x_bar[1][1] = luisa::atomic_add(&dst.m_mass_times_dyadic_x_bar[1][1], src.m_mass_times_dyadic_x_bar[1][1]);
    ret.m_mass_times_dyadic_x_bar[1][2] = luisa::atomic_add(&dst.m_mass_times_dyadic_x_bar[1][2], src.m_mass_times_dyadic_x_bar[1][2]);
    // Column 2
    ret.m_mass_times_dyadic_x_bar[2][0] = luisa::atomic_add(&dst.m_mass_times_dyadic_x_bar[2][0], src.m_mass_times_dyadic_x_bar[2][0]);
    ret.m_mass_times_dyadic_x_bar[2][1] = luisa::atomic_add(&dst.m_mass_times_dyadic_x_bar[2][1], src.m_mass_times_dyadic_x_bar[2][1]);
    ret.m_mass_times_dyadic_x_bar[2][2] = luisa::atomic_add(&dst.m_mass_times_dyadic_x_bar[2][2], src.m_mass_times_dyadic_x_bar[2][2]);
    
    return ret;
}
}  // namespace uipc::backend::luisa

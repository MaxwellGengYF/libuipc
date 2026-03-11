#pragma once
#include <luisa/luisa-compute.h>

// ref: https://github.com/ipc-sim/Codim-IPC/blob/main/Library/Math/DIhedral_ANGLE.h
namespace uipc::backend::luisa
{
/**
 * @brief Compute the dihedral angle between two planes defined by four points.
 * 
 * Mid-Edge: V1-V2
 * Opposite Points: V0, V3
 */
template <typename T>
inline void dihedral_angle(const luisa::Vector<T, 3>& v0,
                           const luisa::Vector<T, 3>& v1,
                           const luisa::Vector<T, 3>& v2,
                           const luisa::Vector<T, 3>& v3,
                           T&                       DA)
{
    using luisa::cross;
    using luisa::dot;
    using luisa::length;
    using luisa::sqrt;
    using luisa::acos;
    using luisa::min;
    using luisa::max;

    const luisa::Vector<T, 3> n1 = cross(v1 - v0, v2 - v0);
    const luisa::Vector<T, 3> n2 = cross(v2 - v3, v1 - v3);
    T n1_len = length(n1);
    T n2_len = length(n2);
    DA = acos(max(T(-1), min(T(1), dot(n1, n2) / (n1_len * n2_len))));
    if(dot(cross(n2, n1), v1 - v2) < 0)
    {
        DA = -DA;
    }
}

namespace detail
{
    // here we map our v order to rusmas' in this function for implementation convenience
    template <typename T>
    inline void dihedral_angle_gradient(const luisa::Vector<T, 3>& v2,  // input: V0
                                        const luisa::Vector<T, 3>& v0,  // input: V1
                                        const luisa::Vector<T, 3>& v1,  // input: V2
                                        const luisa::Vector<T, 3>& v3,  // input: V3
                                        luisa::Vector<T, 12>&      grad)
    {
        using luisa::cross;
        using luisa::dot;
        using luisa::length;
        using luisa::sqrt;

        luisa::Vector<T, 3> e0 = v1 - v0;
        luisa::Vector<T, 3> e1 = v2 - v0;
        luisa::Vector<T, 3> e2 = v3 - v0;
        luisa::Vector<T, 3> e3 = v2 - v1;
        luisa::Vector<T, 3> e4 = v3 - v1;
        luisa::Vector<T, 3> n1 = cross(e0, e1);
        luisa::Vector<T, 3> n2 = cross(e2, e0);

        T n1SqNorm = dot(n1, n1);
        T n2SqNorm = dot(n2, n2);
        T e0norm   = length(e0);

        // fill in gradient in order with g2, g0, g1, g3 in rusmas' doc
        // grad.segment(0, 3) = -e0norm / n1SqNorm * n1;
        grad[0] = -e0norm / n1SqNorm * n1.x;
        grad[1] = -e0norm / n1SqNorm * n1.y;
        grad[2] = -e0norm / n1SqNorm * n1.z;
        
        // grad.segment(3, 3) = -e0.dot(e3) / (e0norm * n1SqNorm) * n1 - e0.dot(e4) / (e0norm * n2SqNorm) * n2;
        T coeff1 = -dot(e0, e3) / (e0norm * n1SqNorm);
        T coeff2 = -dot(e0, e4) / (e0norm * n2SqNorm);
        grad[3] = coeff1 * n1.x + coeff2 * n2.x;
        grad[4] = coeff1 * n1.y + coeff2 * n2.y;
        grad[5] = coeff1 * n1.z + coeff2 * n2.z;
        
        // grad.segment(6, 3) = e0.dot(e1) / (e0norm * n1SqNorm) * n1 + e0.dot(e2) / (e0norm * n2SqNorm) * n2;
        T coeff3 = dot(e0, e1) / (e0norm * n1SqNorm);
        T coeff4 = dot(e0, e2) / (e0norm * n2SqNorm);
        grad[6] = coeff3 * n1.x + coeff4 * n2.x;
        grad[7] = coeff3 * n1.y + coeff4 * n2.y;
        grad[8] = coeff3 * n1.z + coeff4 * n2.z;
        
        // grad.segment(9, 3) = -e0norm / n2SqNorm * n2;
        grad[9]  = -e0norm / n2SqNorm * n2.x;
        grad[10] = -e0norm / n2SqNorm * n2.y;
        grad[11] = -e0norm / n2SqNorm * n2.z;
    }


    template <typename T>
    inline void compute_m_hat(const luisa::Vector<T, 3>& xp,
                              const luisa::Vector<T, 3>& xe0,
                              const luisa::Vector<T, 3>& xe1,
                              luisa::Vector<T, 3>&       mHat)
    {
        using luisa::dot;
        using luisa::normalize;

        luisa::Vector<T, 3> e = xe1 - xe0;
        mHat = normalize(xe0 + (xp - xe0) * dot(xp - xe0, e) / dot(e, e) - xp);
    }

    // here we map our v order to rusmas' in this function for implementation convenience
    template <typename T>
    inline void dihedral_angle_hessian(const luisa::Vector<T, 3>& v2,  // input: V0
                                       const luisa::Vector<T, 3>& v0,  // input: V1
                                       const luisa::Vector<T, 3>& v1,  // input: V2
                                       const luisa::Vector<T, 3>& v3,  // input: V3
                                       luisa::Matrix<T, 12>&      Hess)
    {
        using luisa::cross;
        using luisa::dot;
        using luisa::length;
        using luisa::sqrt;
        using luisa::outer_product;

        luisa::Vector<T, 3> e[5] = {v1 - v0, v2 - v0, v3 - v0, v2 - v1, v3 - v1};
        T norm_e[5] = {
            length(e[0]),
            length(e[1]),
            length(e[2]),
            length(e[3]),
            length(e[4]),
        };

        luisa::Vector<T, 3> n1     = cross(e[0], e[1]);
        luisa::Vector<T, 3> n2     = cross(e[2], e[0]);
        T                      n1norm = length(n1);
        T                      n2norm = length(n2);

        luisa::Vector<T, 3> mHat1, mHat2, mHat3, mHat4, mHat01, mHat02;
        compute_m_hat(v1, v0, v2, mHat1);
        compute_m_hat(v1, v0, v3, mHat2);
        compute_m_hat(v0, v1, v2, mHat3);
        compute_m_hat(v0, v1, v3, mHat4);
        compute_m_hat(v2, v0, v1, mHat01);
        compute_m_hat(v3, v0, v1, mHat02);

        T cosalpha1, cosalpha2, cosalpha3, cosalpha4;
        cosalpha1 = dot(e[0], e[1]) / (norm_e[0] * norm_e[1]);
        cosalpha2 = dot(e[0], e[2]) / (norm_e[0] * norm_e[2]);
        cosalpha3 = -dot(e[0], e[3]) / (norm_e[0] * norm_e[3]);
        cosalpha4 = -dot(e[0], e[4]) / (norm_e[0] * norm_e[4]);

        T h1, h2, h3, h4, h01, h02;
        h1  = n1norm / norm_e[1];
        h2  = n2norm / norm_e[2];
        h3  = n1norm / norm_e[3];
        h4  = n2norm / norm_e[4];
        h01 = n1norm / norm_e[0];
        h02 = n2norm / norm_e[0];

        // Helper lambda for outer product (n * m^T)
        auto outer = [](const luisa::Vector<T, 3>& a, const luisa::Vector<T, 3>& b) {
            luisa::Matrix<T, 3> m;
            m[0] = luisa::Vector<T, 3>{a.x * b.x, a.y * b.x, a.z * b.x};
            m[1] = luisa::Vector<T, 3>{a.x * b.y, a.y * b.y, a.z * b.y};
            m[2] = luisa::Vector<T, 3>{a.x * b.z, a.y * b.z, a.z * b.z};
            return m;
        };

        //TODO: can extract to functions
        luisa::Matrix<T, 3> N1_01 = outer(n1, mHat01 / (h01 * h01 * n1norm));
        luisa::Matrix<T, 3> N1_3 = outer(n1, mHat3 / (h01 * h3 * n1norm));
        luisa::Matrix<T, 3> N1_1 = outer(n1, mHat1 / (h01 * h1 * n1norm));
        luisa::Matrix<T, 3> N2_4 = outer(n2, mHat4 / (h02 * h4 * n2norm));
        luisa::Matrix<T, 3> N2_2 = outer(n2, mHat2 / (h02 * h2 * n2norm));
        luisa::Matrix<T, 3> N2_02 = outer(n2, mHat02 / (h02 * h02 * n2norm));
        luisa::Matrix<T, 3> M3_01_1 =
            outer(mHat01 * (cosalpha3 / (h3 * h01 * n1norm)), n1);
        luisa::Matrix<T, 3> M1_01_1 =
            outer(mHat01 * (cosalpha1 / (h1 * h01 * n1norm)), n1);
        luisa::Matrix<T, 3> M1_1_1 =
            outer(mHat1 * (cosalpha1 / (h1 * h1 * n1norm)), n1);
        luisa::Matrix<T, 3> M3_3_1 =
            outer(mHat3 * (cosalpha3 / (h3 * h3 * n1norm)), n1);
        luisa::Matrix<T, 3> M3_1_1 =
            outer(mHat3 * (cosalpha3 / (h3 * h1 * n1norm)), n1);
        luisa::Matrix<T, 3> M1_3_1 =
            outer(mHat1 * (cosalpha1 / (h1 * h3 * n1norm)), n1);
        luisa::Matrix<T, 3> M4_02_2 =
            outer(mHat02 * (cosalpha4 / (h4 * h02 * n2norm)), n2);
        luisa::Matrix<T, 3> M2_02_2 =
            outer(mHat02 * (cosalpha2 / (h2 * h02 * n2norm)), n2);
        luisa::Matrix<T, 3> M4_4_2 =
            outer(mHat4 * (cosalpha4 / (h4 * h4 * n2norm)), n2);
        luisa::Matrix<T, 3> M2_4_2 =
            outer(mHat4 * (cosalpha2 / (h2 * h4 * n2norm)), n2);
        luisa::Matrix<T, 3> M4_2_2 =
            outer(mHat2 * (cosalpha4 / (h4 * h2 * n2norm)), n2);
        luisa::Matrix<T, 3> M2_2_2 =
            outer(mHat2 * (cosalpha2 / (h2 * h2 * n2norm)), n2);
        luisa::Matrix<T, 3> B1 =
            outer(n1, mHat01 / (norm_e[0] * norm_e[0] * n1norm));
        luisa::Matrix<T, 3> B2 =
            outer(n2, mHat02 / (norm_e[0] * norm_e[0] * n2norm));

        // transpose for 3x3 matrices
        auto transpose3 = [](const luisa::Matrix<T, 3>& m) {
            luisa::Matrix<T, 3> t;
            t[0] = luisa::Vector<T, 3>{m[0].x, m[1].x, m[2].x};
            t[1] = luisa::Vector<T, 3>{m[0].y, m[1].y, m[2].y};
            t[2] = luisa::Vector<T, 3>{m[0].z, m[1].z, m[2].z};
            return t;
        };

        // Helper to set 3x3 block in 12x12 matrix
        auto set_block = [&Hess](int row, int col, const luisa::Matrix<T, 3>& m) {
            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 3; j++) {
                    Hess[row + i][col + j] = m[j][i];  // column-major to column-major
                }
            }
        };

        auto transpose_and_set = [&set_block, &transpose3](int row, int col, const luisa::Matrix<T, 3>& m) {
            set_block(row, col, transpose3(m));
        };

        // fill in Hessian in order with g2, g0, g1, g3 in rusmus' doc
        // Hess.block(0, 0, 3, 3) = -(N1_01 + N1_01.transpose());
        set_block(0, 0, -(N1_01 + transpose3(N1_01)));
        
        // Hess.block(3, 0, 3, 3) = M3_01_1 - N1_3;
        set_block(3, 0, M3_01_1 - N1_3);
        // Hess.block(0, 3, 3, 3) = Hess.block(3, 0, 3, 3).transpose();
        transpose_and_set(0, 3, M3_01_1 - N1_3);
        
        // Hess.block(6, 0, 3, 3) = M1_01_1 - N1_1;
        set_block(6, 0, M1_01_1 - N1_1);
        // Hess.block(0, 6, 3, 3) = Hess.block(6, 0, 3, 3).transpose();
        transpose_and_set(0, 6, M1_01_1 - N1_1);
        
        // Hess.block(0, 9, 3, 3).setZero();
        set_block(0, 9, luisa::Matrix<T, 3>::fill(T(0)));
        // Hess.block(9, 0, 3, 3).setZero();
        set_block(9, 0, luisa::Matrix<T, 3>::fill(T(0)));

        // Hess.block(3, 3, 3, 3) = M3_3_1 + M3_3_1.transpose() - B1 + M4_4_2 + M4_4_2.transpose() - B2;
        set_block(3, 3, M3_3_1 + transpose3(M3_3_1) - B1 + M4_4_2 + transpose3(M4_4_2) - B2);
        
        // Hess.block(3, 6, 3, 3) = M3_1_1 + M1_3_1.transpose() + B1 + M4_2_2 + M2_4_2.transpose() + B2;
        set_block(3, 6, M3_1_1 + transpose3(M1_3_1) + B1 + M4_2_2 + transpose3(M2_4_2) + B2);
        // Hess.block(6, 3, 3, 3) = Hess.block(3, 6, 3, 3).transpose();
        transpose_and_set(6, 3, M3_1_1 + transpose3(M1_3_1) + B1 + M4_2_2 + transpose3(M2_4_2) + B2);
        
        // Hess.block(3, 9, 3, 3) = M4_02_2 - N2_4;
        set_block(3, 9, M4_02_2 - N2_4);
        // Hess.block(9, 3, 3, 3) = Hess.block(3, 9, 3, 3).transpose();
        transpose_and_set(9, 3, M4_02_2 - N2_4);

        // Hess.block(6, 6, 3, 3) = M1_1_1 + M1_1_1.transpose() - B1 + M2_2_2 + M2_2_2.transpose() - B2;
        set_block(6, 6, M1_1_1 + transpose3(M1_1_1) - B1 + M2_2_2 + transpose3(M2_2_2) - B2);
        
        // Hess.block(6, 9, 3, 3) = M2_02_2 - N2_2;
        set_block(6, 9, M2_02_2 - N2_2);
        // Hess.block(9, 6, 3, 3) = Hess.block(6, 9, 3, 3).transpose();
        transpose_and_set(9, 6, M2_02_2 - N2_2);

        // Hess.block(9, 9, 3, 3) = -(N2_02 + N2_02.transpose());
        set_block(9, 9, -(N2_02 + transpose3(N2_02)));
    }
}  // namespace detail

/**
 * @brief Compute the gradient of the dihedral angle between two planes defined by four points.
 * 
 * Mid-Edge: V1-V2
 * Opposite Points: V0, V3
 */
template <typename T>
inline void dihedral_angle_gradient(const luisa::Vector<T, 3>& v0,
                                    const luisa::Vector<T, 3>& v1,
                                    const luisa::Vector<T, 3>& v2,
                                    const luisa::Vector<T, 3>& v3,
                                    luisa::Vector<T, 12>&      grad)
{
    detail::dihedral_angle_gradient(v0, v1, v2, v3, grad);
}


template <typename T>
inline void dihedral_angle_hessian(const luisa::Vector<T, 3>& v0,
                                   const luisa::Vector<T, 3>& v1,
                                   const luisa::Vector<T, 3>& v2,
                                   const luisa::Vector<T, 3>& v3,
                                   luisa::Matrix<T, 12>&      Hess)
{
    detail::dihedral_angle_hessian(v0, v1, v2, v3, Hess);
}
}  // namespace uipc::backend::luisa

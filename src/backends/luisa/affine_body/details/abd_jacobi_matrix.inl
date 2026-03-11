#include <luisa/core/mathematics.h>
namespace uipc::backend::luisa
{
template <size_t N>
__attribute__((always_inline)) Vector<float, 3 * N> ABDJacobiStack<N>::operator*(const Vector12& q) const
{
    Vector<float, 3 * N> ret;
#pragma unroll
    for(size_t i = 0; i < N; ++i)
    {
        float3 result = (*m_jacobis[i]) * q;
        ret.data[3 * i + 0] = result.x;
        ret.data[3 * i + 1] = result.y;
        ret.data[3 * i + 2] = result.z;
    }
    return ret;
}

template <size_t N>
__attribute__((always_inline)) Matrix<float, 3 * N, 12> ABDJacobiStack<N>::to_mat() const
{
    Matrix<float, 3 * N, 12> ret;
    for(size_t i = 0; i < N; ++i)
    {
        Matrix3x12 jacobi_mat = m_jacobis[i]->to_mat();
        // Copy 3x12 block to ret at row 3*i
        for(size_t c = 0; c < 12; ++c)
            for(size_t r = 0; r < 3; ++r)
                ret(r + 3 * i, c) = jacobi_mat(c * 3 + r);
    }
    return ret;
}

template <size_t N>
__attribute__((always_inline)) Vector12
ABDJacobiStack<N>::ABDJacobiStackT::operator*(const Vector<float, 3 * N>& g) const
{
    Vector12 ret = Vector12::zeros();
#pragma unroll
    for(size_t i = 0; i < N; ++i)
    {
        const ABDJacobi* jacobi = m_origin.m_jacobis[i];
        float3 g_segment{g.data[3 * i], g.data[3 * i + 1], g.data[3 * i + 2]};
        Vector12 partial = jacobi->T() * g_segment;
        for(int j = 0; j < 12; ++j)
            ret.data[j] += partial.data[j];
    }
    return ret;
}
}  // namespace uipc::backend::luisa

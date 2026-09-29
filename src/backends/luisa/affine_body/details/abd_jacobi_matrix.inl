/********************************************************************
 * @file affine_body/details/abd_jacobi_matrix.inl
 * @brief T6.1a — the bodies of the factored ABD jacobi/dyadic-mass routines
 * (transcription of src/backends/cuda/affine_body/abd_jacobi_matrix.cu).
 *
 * Included at the end of affine_body/abd_jacobi_matrix.h.  Every routine is
 * templated on the scalar (`S` = `double` on the host, `lcrt::Var<double>` in a
 * kernel) so that the shipped kernel, the host oracle and the device-vs-host
 * comparison run the *same* body (cookbook §7.16 rule 1).  The reference's
 * `#pragma unroll` is dropped (MSVC C4068, and the build must be warning-free).
 *********************************************************************/
#include <luisa_device/math/math.h>
#include <utils/scalar_util.h>
#include <uipc/common/config.h>
#include <uipc/common/log.h>

namespace uipc::backend::luisa
{
namespace abd_detail
{
/// The reference's `UIPC_KERNEL_ASSERT(cond, "...")`: `lcrt::device_assert` when
/// the condition is a device predicate, `UIPC_ASSERT` (which aborts, cookbook
/// §7.16 rule 7) when it is a host `bool` -- a host `if` cannot take a device
/// predicate (RULE M2).  The same helper shape as utils/codim_thickness.h.
template <typename Cond>
inline void assert_condition(Cond&& cond, const char* message) noexcept
{
    if constexpr(lps::is_dsl_value_v<std::remove_cvref_t<Cond>>)
        lcrt::device_assert(cond, message);
    else
        UIPC_ASSERT(static_cast<bool>(cond), "{}", message);
}
}  // namespace abd_detail

// ---------------------------------------------------------------------------
//  ABDJacobi
// ---------------------------------------------------------------------------
// reference: UIPC_GENERIC Vector12 operator*(const ABDJacobi::ABDJacobiT& j, const Vector3& g)
template <class S>
[[nodiscard]] inline lps::LVec<S, 12> abd_jacobi_T_mul(const ABDJacobiT<S>& j,
                                                       const lps::LVec<S, 3>& g) noexcept
{
    const auto&      x = j.x_bar;
    lps::LVec<S, 12> g12;
    g12.segment<3>(0) = g;
    g12.segment<3>(3) = x * g(0);
    g12.segment<3>(6) = x * g(1);
    g12.segment<3>(9) = x * g(2);
    return g12;
}

// reference: UIPC_GENERIC Vector3 operator*(const ABDJacobi& j, const Vector12& q)
template <class S>
[[nodiscard]] inline lps::LVec<S, 3> abd_jacobi_point(const ABDJacobiT<S>& j,
                                                      const lps::LVec<S, 12>& q) noexcept
{
    const auto& t  = q.segment<3>(0);
    const auto& a1 = q.segment<3>(3);
    const auto& a2 = q.segment<3>(6);
    const auto& a3 = q.segment<3>(9);
    const auto& x  = j.x_bar;
    // reference: `Vector3{x.dot(a1), x.dot(a2), x.dot(a3)} + t`
    return lps::LVec<S, 3>{x.dot(a1), x.dot(a2), x.dot(a3)} + t;
}

// reference: UIPC_GENERIC Vector3 ABDJacobi::vec_x(const Vector12& q) const  (no translation)
template <class S>
[[nodiscard]] inline lps::LVec<S, 3> abd_jacobi_vec_x(const ABDJacobiT<S>& j,
                                                      const lps::LVec<S, 12>& q) noexcept
{
    const auto& a1 = q.segment<3>(3);
    const auto& a2 = q.segment<3>(6);
    const auto& a3 = q.segment<3>(9);
    const auto& x  = j.x_bar;
    return lps::LVec<S, 3>{x.dot(a1), x.dot(a2), x.dot(a3)};
}

// reference: UIPC_GENERIC Matrix3x12 ABDJacobi::to_mat() const
template <class S>
[[nodiscard]] inline lps::LMat<S, 3, 12> abd_jacobi_to_mat(const ABDJacobiT<S>& j) noexcept
{
    lps::LMat<S, 3, 12> ret = lps::LMat<S, 3, 12>::Zero();
    const auto&         x   = j.x_bar;
    ret(0, 0)               = lps::one_of<S>();
    ret(1, 1)               = lps::one_of<S>();
    ret(2, 2)               = lps::one_of<S>();
    ret.block<1, 3>(0, 3)   = x.transpose();
    ret.block<1, 3>(1, 6)   = x.transpose();
    ret.block<1, 3>(2, 9)   = x.transpose();
    return ret;
}

// reference: UIPC_GENERIC Matrix12x12 ABDJacobi::JT_H_J(
//                const ABDJacobiT& lhs_J_T, const Matrix3x3& Hessian, const ABDJacobi& rhs_J)
template <class S>
[[nodiscard]] inline lps::LMat<S, 12, 12> abd_jacobi_JT_H_J(const ABDJacobiT<S>& lhs_J_T,
                                                            const lps::LMat<S, 3, 3>& Hessian,
                                                            const ABDJacobiT<S>& rhs_J) noexcept
{
    lps::LMat<S, 12, 12> ret = lps::LMat<S, 12, 12>::Zero();
    auto                 x   = lhs_J_T.x_bar;
    auto                 y   = rhs_J.x_bar;
    ret.block<3, 3>(0, 0)    = Hessian;

    // Hessian Col * y
    ret.block<3, 3>(0, 3) = Hessian.block<3, 1>(0, 0) * y.transpose();
    ret.block<3, 3>(0, 6) = Hessian.block<3, 1>(0, 1) * y.transpose();
    ret.block<3, 3>(0, 9) = Hessian.block<3, 1>(0, 2) * y.transpose();

    // x * Hessian Row
    ret.block<3, 3>(3, 0) = x * Hessian.block<1, 3>(0, 0);
    ret.block<3, 3>(6, 0) = x * Hessian.block<1, 3>(1, 0);
    ret.block<3, 3>(9, 0) = x * Hessian.block<1, 3>(2, 0);

    lps::LMat<S, 3, 3> x_y = x * y.transpose();

    // kronecker product: H (x) (x y^T)
    ret.block<3, 3>(3, 3) = x_y * Hessian(0, 0);
    ret.block<3, 3>(3, 6) = x_y * Hessian(0, 1);
    ret.block<3, 3>(3, 9) = x_y * Hessian(0, 2);

    ret.block<3, 3>(6, 3) = x_y * Hessian(1, 0);
    ret.block<3, 3>(6, 6) = x_y * Hessian(1, 1);
    ret.block<3, 3>(6, 9) = x_y * Hessian(1, 2);

    ret.block<3, 3>(9, 3) = x_y * Hessian(2, 0);
    ret.block<3, 3>(9, 6) = x_y * Hessian(2, 1);
    ret.block<3, 3>(9, 9) = x_y * Hessian(2, 2);

    return ret;
}

// ---------------------------------------------------------------------------
//  ABDJacobiStack<N> (reference: details/abd_jacobi_matrix.inl)
// ---------------------------------------------------------------------------
template <size_t N>
template <class S>
[[nodiscard]] inline lps::LVec<S, static_cast<int>(3 * N)> ABDJacobiStack<N>::operator*(
    const lps::LVec<S, 12>& q) const noexcept
{
    lps::LVec<S, static_cast<int>(3 * N)> ret;
    for(size_t i = 0; i < N; ++i)
    {
        ret.segment<3>(static_cast<int>(3 * i)) =
            abd_jacobi_point(to_generic(*m_jacobis[i]), q);
    }
    return ret;
}

template <size_t N>
template <class S>
[[nodiscard]] inline lps::LMat<S, static_cast<int>(3 * N), 12> ABDJacobiStack<N>::to_mat()
    const noexcept
{
    lps::LMat<S, static_cast<int>(3 * N), 12> ret;
    for(size_t i = 0; i < N; ++i)
    {
        ret.block<3, 12>(static_cast<int>(3 * i), 0) =
            abd_jacobi_to_mat(to_generic(*m_jacobis[i]));
    }
    return ret;
}

template <size_t N>
template <class S>
// NOTE: the reference's `ABDJacobiStackT::operator*` returns **Vector12** (it is
// the transpose map `J_i^T g_i` summed over the stack); the port's first
// declaration spelled the stack-map return type here, which the T6.1a test caught
// at its first instantiation (N = 2).  The return type below is the reference's.
[[nodiscard]] inline lps::LVec<S, 12> ABDJacobiStack<N>::ABDJacobiStackT::
    operator*(const lps::LVec<S, static_cast<int>(3 * N)>& g) const noexcept
{
    lps::LVec<S, 12> ret = lps::LVec<S, 12>::Zero();
    for(size_t i = 0; i < N; ++i)
    {
        const ABDJacobi* jacobi = m_origin.m_jacobis[i];
        ret += abd_jacobi_T_mul(to_generic(*jacobi),
                                g.segment<3>(static_cast<int>(3 * i)));
    }
    return ret;
}

// ---------------------------------------------------------------------------
//  ABDJacobiDyadicMass (reference: abd_jacobi_matrix.cu, lines 105-211)
// ---------------------------------------------------------------------------
// reference: UIPC_GENERIC Vector12 operator*(const ABDJacobiDyadicMass& JTJ, const Vector12& p)
template <class S>
[[nodiscard]] inline lps::LVec<S, 12> abd_dyadic_mass_mul(
    const ABDJacobiDyadicMassT<S>& mJTJ, const lps::LVec<S, 12>& p) noexcept
{
    lps::LVec<S, 12> ret;
    const auto&      m = mJTJ.m_mass;
    const auto&      D = mJTJ.m_mass_times_dyadic_x_bar;
    const auto&      x = mJTJ.m_mass_times_x_bar;

    const auto& p_p  = p.segment<3>(0);
    const auto& p_a1 = p.segment<3>(3);
    const auto& p_a2 = p.segment<3>(6);
    const auto& p_a3 = p.segment<3>(9);

    ret(0) = x.dot(p_a1) + m * p_p(0);
    ret(1) = x.dot(p_a2) + m * p_p(1);
    ret(2) = x.dot(p_a3) + m * p_p(2);

    ret.segment<3>(3) = D * p_a1 + x * p_p(0);
    ret.segment<3>(6) = D * p_a2 + x * p_p(1);
    ret.segment<3>(9) = D * p_a3 + x * p_p(2);

    return ret;
}

// reference: UIPC_GENERIC ABDJacobiDyadicMass& ABDJacobiDyadicMass::operator+=(...)
template <class S>
inline void abd_dyadic_mass_add(ABDJacobiDyadicMassT<S>&       dst,
                                const ABDJacobiDyadicMassT<S>& rhs) noexcept
{
    // a DSL *scalar* has no operator+= (cookbook §7.16 rule 3): detail::add
    dst.m_mass = lps::detail::add(dst.m_mass, rhs.m_mass);
    dst.m_mass_times_x_bar += rhs.m_mass_times_x_bar;
    dst.m_mass_times_dyadic_x_bar += rhs.m_mass_times_dyadic_x_bar;
}

// reference: UIPC_GENERIC void ABDJacobiDyadicMass::add_to(Matrix12x12& h) const
template <class S>
inline void abd_dyadic_mass_add_to(const ABDJacobiDyadicMassT<S>& mJTJ,
                                   lps::LMat<S, 12, 12>&          h) noexcept
{
    const auto& m_mass    = mJTJ.m_mass;
    const auto& m_x_bar   = mJTJ.m_mass_times_x_bar;
    const auto& m_dyadic  = mJTJ.m_mass_times_dyadic_x_bar;

    // row 0 + col 0   (the reference's `h(0,0) += ...`; a DSL *scalar* has no
    // operator+= (cookbook §7.16 rule 3), so the scalar entries accumulate
    // through lps::detail::add.  The block entries keep `+=`, and the three
    // `m_x_bar` columns go through the block's *column view*: Eigen's
    // `block<3,1>` and a `Vector3` are the same object there, while this layer's
    // LMat<3,1> and LVec<3> are distinct types (MatMap::operator+= only accepts
    // an LMat).)
    h(0, 0)             = lps::detail::add(h(0, 0), m_mass);
    h.block<1, 3>(0, 3) += m_x_bar.transpose();
    h.block<3, 1>(3, 0).col(0) += m_x_bar;

    // row 1 + col 1
    h(1, 1)             = lps::detail::add(h(1, 1), m_mass);
    h.block<1, 3>(1, 6) += m_x_bar.transpose();
    h.block<3, 1>(6, 1).col(0) += m_x_bar;

    // row 2 + col 2
    h(2, 2)             = lps::detail::add(h(2, 2), m_mass);
    h.block<1, 3>(2, 9) += m_x_bar.transpose();
    h.block<3, 1>(9, 2).col(0) += m_x_bar;

    // block<3,3> at (3,3)  (6,6)  (9,9)
    h.block<3, 3>(3, 3) += m_dyadic;
    h.block<3, 3>(6, 6) += m_dyadic;
    h.block<3, 3>(9, 9) += m_dyadic;
}

// reference: UIPC_GENERIC Matrix12x12 ABDJacobiDyadicMass::to_mat() const
template <class S>
[[nodiscard]] inline lps::LMat<S, 12, 12> abd_dyadic_mass_to_mat(
    const ABDJacobiDyadicMassT<S>& mJTJ) noexcept
{
    lps::LMat<S, 12, 12> h = lps::LMat<S, 12, 12>::Zero();
    abd_dyadic_mass_add_to(mJTJ, h);
    return h;
}

// reference: UIPC_GENERIC Vector3 ABDJacobiDyadicMass::center_of_mass() const
template <class S>
[[nodiscard]] inline lps::LVec<S, 3> abd_dyadic_mass_center_of_mass(
    const ABDJacobiDyadicMassT<S>& mJTJ) noexcept
{
    const auto& m = mJTJ.m_mass;
    abd_detail::assert_condition(
        m > lps::zero_of<S>(),
        "ABDJacobiDyadicMass::center_of_mass requires positive mass");
    return mJTJ.m_mass_times_x_bar / m;
}

// reference: UIPC_GENERIC Matrix3x3 ABDJacobiDyadicMass::inertia_tensor_cm() const
//   `if(m <= 0) return Matrix3x3::Zero();` -- a *device* predicate cannot drive
//   control flow in the DSL (RULE M2), so the reference's early return becomes a
//   select of the fully computed value (the discarded lanes are never read; the
//   same shape T5.4a chose for diag_inv_row).
template <class S>
[[nodiscard]] inline lps::LMat<S, 3, 3> abd_dyadic_mass_inertia_tensor_cm(
    const ABDJacobiDyadicMassT<S>& mJTJ) noexcept
{
    const auto& m    = mJTJ.m_mass;
    // the reference computes c inline here (it does *not* go through
    // center_of_mass(), whose positive-mass assert would fire first)
    const auto  c    = mJTJ.m_mass_times_x_bar / m;
    const auto& S_dy = mJTJ.m_mass_times_dyadic_x_bar;
    // I^O = tr(S) I_3 - S  (inertia about the origin)
    const auto  trS      = S_dy.trace();
    const auto  I_origin = trS * lps::LMat<S, 3, 3>::Identity() - S_dy;
    // I_cm = I^O - m (|c|^2 I_3 - c c^T)
    const auto c2    = c.squaredNorm();
    const auto value = I_origin - m * (c2 * lps::LMat<S, 3, 3>::Identity() - c * c.transpose());
    return scalar_detail::select_container(lps::LMat<S, 3, 3>::Zero(),
                                           value,
                                           m > lps::zero_of<S>());
}
}  // namespace uipc::backend::luisa

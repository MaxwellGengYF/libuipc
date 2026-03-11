#pragma once
#include <affine_body/type_define.h>

namespace uipc::backend::luisa
{
//tex: $$ \mathbf{J}_{3 \times 12} $$ or $$ (\mathbf{J}^T)_{12 \times 3} $$
class ABDJacobi  // for every point
{
  public:
    class ABDJacobiT
    {
        const ABDJacobi& m_j;

      public:
        explicit __attribute__((always_inline)) ABDJacobiT(const ABDJacobi& j)
            : m_j(j)
        {
        }
        __attribute__((always_inline)) friend Vector12 operator*(const ABDJacobiT& j, const float3& g);

        __attribute__((always_inline)) const auto& J() const { return m_j; }
    };
    __attribute__((always_inline)) ABDJacobi(const float3& x_bar)
        : m_x_bar(x_bar)
    {
    }

    __attribute__((always_inline)) ABDJacobi()
        : m_x_bar(float3{0.0f, 0.0f, 0.0f})
    {
    }

    __attribute__((always_inline)) friend float3 operator*(const ABDJacobi& j, const Vector12& q);
    __attribute__((always_inline)) friend Vector12 operator*(const ABDJacobi::ABDJacobiT& j, const float3& g);

    __attribute__((always_inline)) float3 point_from_affine(const Vector12& q)
    {
        return (*this) * q;
    }

    __attribute__((always_inline)) float3 point_x(const Vector12& q) const
    {
        return (*this) * q;
    };

    // without translation, only rotation and scaling
    __attribute__((always_inline)) float3 vec_x(const Vector12& q) const;

    __attribute__((always_inline)) Matrix3x12 to_mat() const;

    __attribute__((always_inline)) ABDJacobiT T() const { return ABDJacobiT(*this); }

    __attribute__((always_inline)) const float3& x_bar() const { return m_x_bar; }

    //tex: $$ \mathbf{J}^T\mathbf{H}\mathbf{J} $$
    static __attribute__((always_inline)) Matrix12x12 JT_H_J(const ABDJacobiT& lhs_J_T,
                                           const float3x3&  Hessian,
                                           const ABDJacobi&  rhs_J);

  private:
    //tex: $$ \bar{\mathbf{x}} $$
    float3 m_x_bar;
};

template <size_t N>
class ABDJacobiStack
{
  protected:
    const ABDJacobi* m_jacobis[N];

  public:
    class ABDJacobiStackT
    {
        const ABDJacobiStack& m_origin;

      public:
        __attribute__((always_inline)) ABDJacobiStackT(const ABDJacobiStack& j)
            : m_origin(j)
        {
        }
        __attribute__((always_inline)) Vector12 operator*(const Vector<float, 3 * N>& g) const;
    };

    __attribute__((always_inline)) Vector<float, 3 * N> operator*(const Vector12& q) const;

    __attribute__((always_inline)) Matrix<float, 3 * N, 12> to_mat() const;

    __attribute__((always_inline)) ABDJacobiStackT T() const { return ABDJacobiStackT(*this); }
};

class ABDJacobiStack2 : public ABDJacobiStack<2>
{
  public:
    __attribute__((always_inline)) ABDJacobiStack2(const ABDJacobi& j1, const ABDJacobi& j2)
    {
        m_jacobis[0] = &j1;
        m_jacobis[1] = &j2;
    }
};

class ABDJacobiStack3 : public ABDJacobiStack<3>
{
  public:
    __attribute__((always_inline)) ABDJacobiStack3(const ABDJacobi& j1, const ABDJacobi& j2, const ABDJacobi& j3)
    {
        m_jacobis[0] = &j1;
        m_jacobis[1] = &j2;
        m_jacobis[2] = &j3;
    }
};

class ABDJacobiStack4 : public ABDJacobiStack<4>
{
  public:
    __attribute__((always_inline)) ABDJacobiStack4(const ABDJacobi& j1,
                                 const ABDJacobi& j2,
                                 const ABDJacobi& j3,
                                 const ABDJacobi& j4)
    {
        m_jacobis[0] = &j1;
        m_jacobis[1] = &j2;
        m_jacobis[2] = &j3;
        m_jacobis[3] = &j4;
    }
};
//tex:
// $$
//\mathbf{g}^{\text{Affine}}_k = \sum_{i\in \mathscr{C}_k \cap \mathscr{A}}
//\mathbf{J}_i^T \frac{\partial B}{\partial\mathbf{x}_i}
//= \sum_{i\in \mathscr{C}_k \cap \mathscr{A}}
//
//\begin{bmatrix}
//g_{1}\\
//g_{2}\\
//g_{3}\\
//\hline
//
//\bar{x}_1 g_{1}\\
//\bar{x}_2 g_{1}\\
//\bar{x}_3 g_{1}\\
//\hdashline
//
//\bar{x}_1 g_{2}\\
//\bar{x}_2 g_{2}\\
//\bar{x}_3 g_{2}\\
//\hdashline
//
//\bar{x}_1 g_{3}\\
//\bar{x}_2 g_{3}\\
//\bar{x}_3 g_{3}
//
//\end{bmatrix}_{i}
//
//=
//\sum_{i\in \mathscr{C}_k \cap \mathscr{A}}
//
//\begin{bmatrix}
//\mathbf{g}\\
//\hline
//
//g_{1} \bar{\mathbf{x}}\\
//\hdashline
//
//g_{2} \bar{\mathbf{x}}\\
//\hdashline
//
//g_{3} \bar{\mathbf{x}}\\
//
//\end{bmatrix}_{i}
// $$

//tex:
// where $\mathscr{C}_k$ is the $k$-th contact pair, and $\mathscr{A}$ represents the point set of all affine bodies.
//

//tex: $$\mathbf{J}^T\mathbf{M}_i\mathbf{J} $$
class ABDJacobiDyadicMass
{
  public:
    __attribute__((always_inline)) ABDJacobiDyadicMass()
        : m_mass(0)
        , m_mass_times_x_bar(float3{0.0f, 0.0f, 0.0f})
        , m_mass_times_dyadic_x_bar(float3x3::zeros())
    {
    }

    __attribute__((always_inline)) static ABDJacobiDyadicMass from_dyadic_mass(Float sum_m,
                                                             const float3& sum_m_x_bar,
                                                             const float3x3& sum_m_x_bar_x_bar)
    {
        ABDJacobiDyadicMass ret;
        ret.m_mass                    = sum_m;
        ret.m_mass_times_x_bar        = sum_m_x_bar;
        ret.m_mass_times_dyadic_x_bar = sum_m_x_bar_x_bar;
        return ret;
    }

    __attribute__((always_inline)) ABDJacobiDyadicMass(double node_mass, const float3& x_bar)
        : m_mass(node_mass)
        , m_mass_times_x_bar(float(node_mass) * x_bar)
        , m_mass_times_dyadic_x_bar((float(node_mass) * x_bar) * transpose(float3x3(x_bar)))
    {
    }

    __attribute__((always_inline)) friend Vector12 operator*(const ABDJacobiDyadicMass& mJTJ,
                                           const Vector12&            p);

    __attribute__((always_inline)) ABDJacobiDyadicMass& operator+=(const ABDJacobiDyadicMass& rhs);

    __attribute__((always_inline)) void add_to(Matrix12x12& h) const;

    __attribute__((always_inline)) Matrix12x12 to_mat() const;

    __attribute__((always_inline)) double mass() const { return m_mass; }

    static __attribute__((always_inline)) auto zero() { return ABDJacobiDyadicMass{}; }

    static __attribute__((always_inline)) ABDJacobiDyadicMass atomic_add(ABDJacobiDyadicMass& dst,
                                                      const ABDJacobiDyadicMass& src);

  private:
    double m_mass;
    //tex: $$ m\bar{\mathbf{x}} $$
    float3 m_mass_times_x_bar;
    //tex: $$ m\bar{\mathbf{x}} \otimes \bar{\mathbf{x}} $$
    float3x3 m_mass_times_dyadic_x_bar;
};
}  // namespace uipc::backend::luisa

#include "details/abd_jacobi_matrix.inl"

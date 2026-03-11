#pragma once
#include "givens.hpp"
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
namespace math
{
    using namespace luisa;
    using namespace luisa::compute;

    template <typename T>
    LUISA_DEVICE constexpr void swap(T& a, T& b)
    {
        auto s = a;
        a      = b;
        b      = s;
    }

    template <typename T>
    LUISA_DEVICE constexpr void polar_decomposition(const Matrix<T, 2>& A,
                                                    Matrix<T, 2>&       S,
                                                    GivensRotation<T>&  R) noexcept
    {
        S = A;
        Vector<T, 2> x{A[0][0] + A[1][1], A[0][1] - A[1][0]};
        auto           d = luisa::length(x);
        if(d != 0)
        {
            R.c = x.x / d;
            R.s = -x.y / d;
        }
        else
        {
            R.c = 1;
            R.s = 0;
        }
        R.rowRotation(S);
    }

    template <typename T>
    LUISA_DEVICE constexpr void qr_svd2x2(const Matrix<T, 2>& A,
                                          Vector<T, 2>&       S,
                                          GivensRotation<T>&  U,
                                          GivensRotation<T>&  V) noexcept
    {
        Matrix<T, 2> S_sym;
        polar_decomposition(A, S_sym, U);

        T    cosine{}, sine{};
        auto x{S_sym[0][0]}, y{S_sym[1][0]}, z{S_sym[1][1]};
        auto y2 = y * y;

        if(y2 == 0)
        {  // S is already diagonal
            cosine = 1;
            sine   = 0;
            S.x    = x;
            S.y    = z;
        }
        else
        {
            auto tau = (T)0.5 * (x - z);
            T    w{luisa::sqrt(tau * tau + y2)}, t{};
            if(tau > 0)  // tau + w > w > y > 0 ==> division is safe
                t = y / (tau + w);
            else  // tau - w < -w < -y < 0 ==> division is safe
                t = y / (tau - w);
            cosine = 1 / luisa::sqrt(t * t + (T)1);
            sine   = -t * cosine;

            T c2    = cosine * cosine;
            T _2csy = 2 * cosine * sine * y;
            T s2    = sine * sine;

            S.x = c2 * x - _2csy + s2 * z;
            S.y = s2 * x + _2csy + c2 * z;
        }

        // Sorting
        if(S.x < S.y)
        {
            swap(S.x, S.y);
            V.c = -sine;
            V.s = cosine;
        }
        else
        {
            V.c = cosine;
            V.s = sine;
        }
        U *= V;
    }

    // Custom copysign since luisa::core doesn't have it
    template <typename T>
    LUISA_DEVICE constexpr T copysign(T x, T y) noexcept
    {
        return y >= 0 ? luisa::abs(x) : -luisa::abs(x);
    }

    template <typename T>
    LUISA_DEVICE constexpr T wilkinson_shift(const T a1, const T b1, const T a2) noexcept
    {
        T d  = (T)0.5 * (a1 - a2);
        T bs = b1 * b1;
        T mu = a2 - copysign(bs / (luisa::abs(d) + luisa::sqrt(d * d + bs)), d);
        return mu;
    }

    template <typename T>
    LUISA_DEVICE constexpr void flip_sign(int j, Matrix<T, 3>& U, Vector<T, 3>& S) noexcept
    {
        S[j]     = -S[j];
        U[j]     = -U[j];  // Column j of U
    }

    template <int t, typename T>
    LUISA_DEVICE constexpr void sort_sigma(Matrix<T, 3>& U,
                                           Vector<T, 3>& sigma,
                                           Matrix<T, 3>& V) noexcept
    {
        /// t == 0
        if constexpr(t == 0)
        {
            if(luisa::abs(sigma.y) >= luisa::abs(sigma.z))
            {
                if(sigma.y < 0)
                {
                    flip_sign(1, U, sigma);
                    flip_sign(2, U, sigma);
                }
                return;
            }

            if(sigma.z < 0)
            {
                flip_sign(1, U, sigma);
                flip_sign(2, U, sigma);
            }

            swap(sigma.y, sigma.z);
            // Swap column 1 and column 2
            swap(U[1], U[2]);
            swap(V[1], V[2]);

            if(sigma.y > sigma.x)
            {
                swap(sigma.x, sigma.y);
                swap(U[0], U[1]);
                swap(V[0], V[1]);
            }
            else
            {
                U[2] = -U[2];
                V[2] = -V[2];
            }
        }
        /// t == 1
        else if constexpr(t == 1)
        {
            if(luisa::abs(sigma.x) >= sigma.y)
            {
                if(sigma.x < 0)
                {
                    flip_sign(0, U, sigma);
                    flip_sign(2, U, sigma);
                }
                return;
            }

            swap(sigma.x, sigma.y);
            swap(U[0], U[1]);
            swap(V[0], V[1]);

            if(luisa::abs(sigma.y) < luisa::abs(sigma.z))
            {
                swap(sigma.y, sigma.z);
                swap(U[1], U[2]);
                swap(V[1], V[2]);
            }
            else
            {
                U[1] = -U[1];
                V[1] = -V[1];
            }

            if(sigma.y < 0)
            {
                flip_sign(1, U, sigma);
                flip_sign(2, U, sigma);
            }
        }
    }

    template <int t, typename T>
    LUISA_DEVICE constexpr void process(Matrix<T, 3>& B,
                                        Matrix<T, 3>& U,
                                        Vector<T, 3>& S,
                                        Matrix<T, 3>& V) noexcept
    {
        GivensRotation<T> u{0, 1};
        GivensRotation<T> v{0, 1};
        constexpr int     other   = (t == 1) ? 0 : 2;
        S[other]                  = B[other][other];
        
        // Extract 2x2 block from B
        Matrix<T, 2> B_;
        for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 2; ++j)
                B_[j][i] = B[t+j][t+i];
                
        Vector<T, 2> S_;
        qr_svd2x2(B_, S_, u, v);
        
        S[t]     = S_.x;
        S[t + 1] = S_.y;

        u.rowi += t;
        u.rowk += t;
        v.rowi += t;
        v.rowk += t;
        u.columnRotation(U);
        v.columnRotation(V);
    }

    // Machine epsilon for float/double
    template <typename T>
    LUISA_DEVICE constexpr T machine_epsilon() {
        if constexpr(std::is_same_v<T, float>) {
            return 1.1920929e-07f;  // FLT_EPSILON
        } else {
            return 2.2204460492503131e-16;  // DBL_EPSILON
        }
    }

    template <typename T>
    LUISA_DEVICE constexpr void qr_svd(const Matrix<T, 3>& A,
                                       Vector<T, 3>&       S,
                                       Matrix<T, 3>&       U,
                                       Matrix<T, 3>&       V) noexcept
    {
        // Set identity matrices
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                U[j][i] = V[j][i] = (i == j) ? T(1) : T(0);

        Matrix<T, 3> B = A;

        upper_bidiagonalize(B, U, V);

        GivensRotation<T> r{0, 1};

        T alpha[3] = {B[0][0], B[1][1], B[2][2]};
        T beta[2]  = {B[1][0], B[2][1]};
        T gamma[2] = {alpha[0] * beta[0], alpha[1] * beta[1]};

        constexpr auto eta = machine_epsilon<T>() * (T)128;
        T              tol = eta
                * luisa::max((T)0.5
                               * luisa::sqrt(alpha[0] * alpha[0] + alpha[1] * alpha[1]
                                           + alpha[2] * alpha[2]
                                           + beta[0] * beta[0] + beta[1] * beta[1]),
                           (T)1);

        while(luisa::abs(alpha[0]) > tol && luisa::abs(alpha[1]) > tol && luisa::abs(alpha[2]) > tol
              && luisa::abs(beta[0]) > tol && luisa::abs(beta[1]) > tol)
        {
            auto mu = wilkinson_shift(alpha[1] * alpha[1] + beta[0] * beta[0],
                                      gamma[1],
                                      alpha[2] * alpha[2] + beta[1] * beta[1]);

            r.computeConventional(alpha[0] * alpha[0] - mu, gamma[0]);
            r.columnRotation(B);
            r.columnRotation(V);
            zero_chasing(B, U, V);

            alpha[0] = B[0][0];
            alpha[1] = B[1][1];
            alpha[2] = B[2][2];
            beta[0]  = B[1][0];
            beta[1]  = B[2][1];
            gamma[0] = alpha[0] * beta[0];
            gamma[1] = alpha[1] * beta[1];
        }

        if(luisa::abs(beta[1]) <= tol)
        {
            process<0>(B, U, S, V);
            sort_sigma<0>(U, S, V);
        }
        else if(luisa::abs(beta[0]) <= tol)
        {
            process<1>(B, U, S, V);
            sort_sigma<1>(U, S, V);
        }
        else if(luisa::abs(alpha[1]) <= tol)
        {
            GivensRotation<T> r1(1, 2);
            r1.computeUnconventional(B[1][2], B[2][2]);
            r1.rowRotation(B);
            r1.columnRotation(U);
            process<0>(B, U, S, V);
            sort_sigma<0>(U, S, V);
        }
        else if(luisa::abs(alpha[2]) <= tol)
        {
            GivensRotation<T> r1(1, 2);
            r1.computeConventional(B[1][1], B[2][1]);
            r1.columnRotation(B);
            r1.columnRotation(V);

            GivensRotation<T> r2(0, 2);
            r2.computeConventional(B[0][0], B[2][0]);
            r2.columnRotation(B);
            r2.columnRotation(V);

            process<0>(B, U, S, V);
            sort_sigma<0>(U, S, V);
        }
        else if(luisa::abs(alpha[0]) <= tol)
        {
            GivensRotation<T> r1(0, 1);
            r1.computeUnconventional(B[0][1], B[1][1]);
            r1.rowRotation(B);
            r1.columnRotation(U);

            GivensRotation<T> r2(0, 2);
            r2.computeUnconventional(B[0][2], B[2][2]);
            r2.rowRotation(B);
            r2.columnRotation(U);

            process<1>(B, U, S, V);
            sort_sigma<1>(U, S, V);
        }
    }
}  // namespace math

}  // namespace uipc::backend::luisa

#pragma once
/********************************************************************
 * @file mat.h
 * @brief T2.1 — fixed-size matrix `LMat<T,R,C>` for the generic numerics layer.
 *
 * `LMat<T,R,C>` replaces every `Eigen::Matrix<T,R,C>` (`uipc::{Matrix2x2,3x3,4x4,
 * 6x6,9x9,12x12}`, ...) in *device* code paths.  As with `LVec`, the element type
 * is generic: `double` on the host, `lcrt::Var<double>` on the device (see
 * math/math.h for why `Var` and not `Expr`).
 *
 * Storage is **column-major** (`index(r,c) == c * R + r`) exactly like Eigen's
 * default, so a matrix can be memcpy'd into/out of a buffer without shuffling.
 *
 * Everything that Eigen returns as a *block expression* (`block<>`, `col()`,
 * `row()`, `diagonal()`) is returned as a lightweight strided view that is
 * assignable and implicitly convertible to a value — the same contract as
 * `LVec`'s `segment<>`.
 *
 * `determinant()` / `inverse()` / decompositions deliberately live in
 * `math/matrix_ops.h` (T2.2) and `math/svd.h` (T2.2), not here.
 *********************************************************************/
#include <luisa_device/math/vec.h>

#include <array>
#include <cstddef>
#include <utility>

namespace uipc::backend::luisa::luisa_device::math
{
/// Mutable, doubly-strided view into a run of matrix elements.
template <typename T, int BR, int BC>
struct MatMap;

/// Element-wise (Eigen `.array()`) matrix view; `T` may be const.
template <typename T, int BR, int BC>
struct MatArray;

// ---------------------------------------------------------------------------
//  Traits
// ---------------------------------------------------------------------------
template <typename M>
struct mat_like
{
    static constexpr bool value = false;
    static constexpr int  rows  = 0;
    static constexpr int  cols  = 0;
    using element               = void;
};

template <typename T, int R, int C>
struct mat_like<LMat<T, R, C>>
{
    static constexpr bool value = true;
    static constexpr int  rows  = R;
    static constexpr int  cols  = C;
    using element               = T;
};

template <typename T, int BR, int BC>
struct mat_like<MatMap<T, BR, BC>>
{
    static constexpr bool value = true;
    static constexpr int  rows  = BR;
    static constexpr int  cols  = BC;
    using element               = T;
};

template <typename T, int BR, int BC>
struct mat_like<MatArray<T, BR, BC>>
{
    static constexpr bool value = true;
    static constexpr int  rows  = BR;
    static constexpr int  cols  = BC;
    using element               = std::remove_const_t<T>;
};

template <typename M>
inline constexpr bool is_mat_like_v = mat_like<std::remove_cvref_t<M>>::value;

template <typename M>
inline constexpr int mat_rows_v = mat_like<std::remove_cvref_t<M>>::rows;

template <typename M>
inline constexpr int mat_cols_v = mat_like<std::remove_cvref_t<M>>::cols;

template <typename M>
using mat_element_t = typename mat_like<std::remove_cvref_t<M>>::element;

template <typename M>
struct is_mat_array : std::false_type
{
};
template <typename T, int BR, int BC>
struct is_mat_array<MatArray<T, BR, BC>> : std::true_type
{
};
template <typename M>
inline constexpr bool is_mat_array_v = is_mat_array<std::remove_cvref_t<M>>::value;

// Tell the scalar `select` to keep its hands off these types.
template <typename T, int R, int C>
struct is_container_like<LMat<T, R, C>> : std::true_type
{
};
template <typename T, int BR, int BC>
struct is_container_like<MatMap<T, BR, BC>> : std::true_type
{
};
template <typename T, int BR, int BC>
struct is_container_like<MatArray<T, BR, BC>> : std::true_type
{
};

/// Index of element (r, c) in column-major storage.
[[nodiscard]] constexpr std::size_t mat_index(int r, int c, int rows) noexcept
{
    return static_cast<std::size_t>(c) * static_cast<std::size_t>(rows)
           + static_cast<std::size_t>(r);
}

namespace detail
{
template <typename R, int NR, int NC, typename F, std::size_t... I>
[[nodiscard]] inline LMat<R, NR, NC> make_mat_impl(F&& f, std::index_sequence<I...>) noexcept
{
    // Component order is row-major, matching Eigen's `Matrix(...)` constructor.
    return LMat<R, NR, NC>{static_cast<R>(f(static_cast<int>(I) / NC, static_cast<int>(I) % NC))...};
}
}// namespace detail

/// Build an LMat element-wise from a factory `f(int r, int c) -> scalar`.
template <typename R, int NR, int NC, typename F>
[[nodiscard]] inline LMat<R, NR, NC> make_mat(F&& f) noexcept
{
    return detail::make_mat_impl<R, NR, NC>(std::forward<F>(f),
                                            std::make_index_sequence<static_cast<std::size_t>(NR * NC)>{});
}

// ---------------------------------------------------------------------------
//  LMat
// ---------------------------------------------------------------------------
template <typename T, int R, int C>
struct LMat
{
    static_assert(R > 0 && C > 0, "LMat needs positive dimensions");
    static_assert(std::is_trivially_destructible_v<T>,
                  "LC buffer element types must be trivially destructible");

    using element_type = T;
    static constexpr int rows = R;
    static constexpr int cols = C;

    /// Column-major storage (`index(r,c) == c * R + r`), zero-initialised.
    std::array<T, static_cast<std::size_t>(R) * static_cast<std::size_t>(C)> data{};

    // -- construction --------------------------------------------------------
    LMat() noexcept = default;

    /// Component-wise in **row-major** order, matching Eigen's constructor.
    template <typename... U>
        requires(sizeof...(U) == static_cast<std::size_t>(R) * static_cast<std::size_t>(C)
                 && (... && std::is_convertible_v<U, T>))
    LMat(U&&... u) noexcept
    {
        const T values[sizeof...(U)]{static_cast<T>(std::forward<U>(u))...};
        int   k = 0;
        for(int r = 0; r < R; ++r)
            for(int c = 0; c < C; ++c)
                data[mat_index(r, c, R)] = values[k++];
    }

    [[nodiscard]] static LMat Zero() noexcept { return LMat{}; }

    [[nodiscard]] static LMat Ones() noexcept
    {
        return make_mat<T, R, C>([](int, int) { return one_of<T>(); });
    }

    [[nodiscard]] static LMat Identity() noexcept
    {
        return make_mat<T, R, C>([](int r, int c)
                                 { return r == c ? one_of<T>() : zero_of<T>(); });
    }

    [[nodiscard]] static LMat Constant(const T& v) noexcept
    {
        return make_mat<T, R, C>([&](int, int) { return v; });
    }

    /// Wrap a raw element pointer (the `Eigen::Map<Matrix..>` analogue).
    [[nodiscard]] static MatMap<T, R, C> Map(T* p, int row_stride = 1, int col_stride = R) noexcept
    {
        return MatMap<T, R, C>{p, row_stride, col_stride};
    }

    // -- element access ------------------------------------------------------
    [[nodiscard]] T& operator()(int r, int c) noexcept
    {
        return data[mat_index(r, c, R)];
    }
    [[nodiscard]] const T& operator()(int r, int c) const noexcept
    {
        return data[mat_index(r, c, R)];
    }
    /// Linear (column-major) access.
    [[nodiscard]] T& operator[](int i) noexcept { return data[static_cast<std::size_t>(i)]; }
    [[nodiscard]] const T& operator[](int i) const noexcept
    {
        return data[static_cast<std::size_t>(i)];
    }
    [[nodiscard]] T& coeff(int i) noexcept { return (*this)[i]; }
    [[nodiscard]] const T& coeff(int i) const noexcept { return (*this)[i]; }

    // -- views ---------------------------------------------------------------
    /// Mutable view of the `BR x BC` block whose top-left corner is (r0, c0).
    template <int BR, int BC>
    [[nodiscard]] MatMap<T, BR, BC> block(int r0, int c0) & noexcept
    {
        static_assert(BR > 0 && BC > 0, "block<BR,BC> needs positive sizes");
        return MatMap<T, BR, BC>{data.data() + mat_index(r0, c0, R), 1, R};
    }
    /// Copy of the block. A const parent — or an rvalue such as `(A * B)` —
    /// binds here too and gets a safe copy.
    template <int BR, int BC>
    [[nodiscard]] LMat<T, BR, BC> block(int r0, int c0) const& noexcept
    {
        return make_mat<T, BR, BC>([&](int r, int c)
                                   { return data[mat_index(r0 + r, c0 + c, R)]; });
    }

    /// Column j as a vector view / value.
    [[nodiscard]] VecMap<T, R> col(int j) & noexcept
    {
        return VecMap<T, R>{data.data() + mat_index(0, j, R), 1};
    }
    [[nodiscard]] LVec<T, R> col(int j) const& noexcept
    {
        return make_vec<T, R>([&](int r) { return data[mat_index(r, j, R)]; });
    }

    /// Row i as a vector view / value.
    [[nodiscard]] VecMap<T, C> row(int i) & noexcept
    {
        return VecMap<T, C>{data.data() + mat_index(i, 0, R), R};
    }
    [[nodiscard]] LVec<T, C> row(int i) const& noexcept
    {
        return make_vec<T, C>([&](int c) { return data[mat_index(i, c, R)]; });
    }

    /// Diagonal as a vector view / value.
    [[nodiscard]] VecMap<T, (R < C ? R : C)> diagonal() & noexcept
    {
        return VecMap<T, (R < C ? R : C)>{data.data(), R + 1};
    }
    [[nodiscard]] LVec<T, (R < C ? R : C)> diagonal() const& noexcept
    {
        return make_vec<T, (R < C ? R : C)>([&](int i) { return data[mat_index(i, i, R)]; });
    }

    /// Element-wise view (Eigen `.array()`).
    [[nodiscard]] MatArray<T, R, C> array() noexcept { return MatArray<T, R, C>{data.data(), 1, R}; }
    [[nodiscard]] MatArray<const T, R, C> array() const noexcept
    {
        return MatArray<const T, R, C>{data.data(), 1, R};
    }

    /// Determinant / inverse (square only) -- defined in matrix_ops.h (T2.2).
    [[nodiscard]] auto determinant() const noexcept;
    [[nodiscard]] LMat inverse() const noexcept;

    /// Transposed copy.
    [[nodiscard]] LMat<T, C, R> transpose() const noexcept
    {
        return make_mat<T, C, R>([&](int r, int c) { return (*this)(c, r); });
    }

    [[nodiscard]] auto trace() const noexcept
    {
        auto s = (*this)(0, 0);
        for(int i = 1; i < (R < C ? R : C); ++i)
            s = detail::add(s, (*this)(i, i));
        return stored(s);
    }

    /// Frobenius norm.
    [[nodiscard]] auto squaredNorm() const noexcept
    {
        auto s = detail::mul((*this)(0, 0), (*this)(0, 0));
        for(int c = 0; c < C; ++c)
            for(int r = 0; r < R; ++r)
                if(r != 0 || c != 0)
                    s = detail::add(s, detail::mul((*this)(r, c), (*this)(r, c)));
        return stored(s);
    }
    [[nodiscard]] auto norm() const noexcept { return stored(sqrt(squaredNorm())); }

    /// Column-matrix -> vector (Eigen allows assigning `Matrix<T,N,1>` to `Vector<T,N>`).
    template <int C_ = C>
        requires(C_ == 1)
    [[nodiscard]] operator LVec<T, R>() const noexcept
    {
        return make_vec<T, R>([&](int r) { return (*this)(r, 0); });
    }

    // -- assignment ----------------------------------------------------------
    template <typename U>
        requires(!std::is_same_v<U, T>)
    LMat& operator=(const LMat<U, R, C>& rhs) noexcept
    {
        for(int c = 0; c < C; ++c)
            for(int r = 0; r < R; ++r)
                (*this)(r, c) = rhs(r, c);
        return *this;
    }
    template <typename U>
    LMat& operator=(const MatMap<U, R, C>& rhs) noexcept
    {
        for(int c = 0; c < C; ++c)
            for(int r = 0; r < R; ++r)
                (*this)(r, c) = rhs(r, c);
        return *this;
    }
    template <typename U>
    LMat& operator=(const MatArray<U, R, C>& rhs) noexcept
    {
        for(int c = 0; c < C; ++c)
            for(int r = 0; r < R; ++r)
                (*this)(r, c) = rhs(r, c);
        return *this;
    }

    template <typename U>
    LMat& operator+=(const LMat<U, R, C>& rhs) noexcept
    {
        for(int c = 0; c < C; ++c)
            for(int r = 0; r < R; ++r)
                (*this)(r, c) = detail::add((*this)(r, c), rhs(r, c));
        return *this;
    }
    template <typename U>
    LMat& operator-=(const LMat<U, R, C>& rhs) noexcept
    {
        for(int c = 0; c < C; ++c)
            for(int r = 0; r < R; ++r)
                (*this)(r, c) = detail::sub((*this)(r, c), rhs(r, c));
        return *this;
    }
    template <typename S>
        requires(is_scalar_like_v<S>)
    LMat& operator*=(const S& s) noexcept
    {
        for(int c = 0; c < C; ++c)
            for(int r = 0; r < R; ++r)
                (*this)(r, c) = detail::mul((*this)(r, c), s);
        return *this;
    }
    template <typename S>
        requires(is_scalar_like_v<S>)
    LMat& operator/=(const S& s) noexcept
    {
        for(int c = 0; c < C; ++c)
            for(int r = 0; r < R; ++r)
                (*this)(r, c) = detail::div((*this)(r, c), s);
        return *this;
    }

    void setZero() noexcept
    {
        for(auto& v : data) v = zero_of<T>();
    }
    void setIdentity() noexcept
    {
        for(int c = 0; c < C; ++c)
            for(int r = 0; r < R; ++r)
                (*this)(r, c) = r == c ? one_of<T>() : zero_of<T>();
    }
    void setConstant(const T& v) noexcept
    {
        for(auto& e : data) e = v;
    }
};

// ---------------------------------------------------------------------------
//  MatMap — doubly-strided mutable view
// ---------------------------------------------------------------------------
template <typename T, int BR, int BC>
struct MatMap
{
    static_assert(BR > 0 && BC > 0, "MatMap needs positive dimensions");

    T*  base       = nullptr;
    int row_stride = 1;
    int col_stride = BR;

    using element_type        = T;
    static constexpr int rows = BR;
    static constexpr int cols = BC;

    [[nodiscard]] T& operator()(int r, int c) noexcept
    {
        return base[static_cast<std::ptrdiff_t>(r) * row_stride
                    + static_cast<std::ptrdiff_t>(c) * col_stride];
    }
    [[nodiscard]] const T& operator()(int r, int c) const noexcept
    {
        return base[static_cast<std::ptrdiff_t>(r) * row_stride
                    + static_cast<std::ptrdiff_t>(c) * col_stride];
    }

    [[nodiscard]] LMat<T, BR, BC> eval() const noexcept
    {
        return make_mat<T, BR, BC>([&](int r, int c) { return (*this)(r, c); });
    }
    [[nodiscard]] operator LMat<T, BR, BC>() const noexcept { return eval(); }

    [[nodiscard]] VecMap<T, BR> col(int j) const noexcept
    {
        return VecMap<T, BR>{base + static_cast<std::ptrdiff_t>(j) * col_stride, row_stride};
    }
    [[nodiscard]] VecMap<T, BC> row(int i) const noexcept
    {
        return VecMap<T, BC>{base + static_cast<std::ptrdiff_t>(i) * row_stride, col_stride};
    }

    /// Element-wise view of the same block of elements.
    [[nodiscard]] MatArray<T, BR, BC> array() noexcept
    {
        return MatArray<T, BR, BC>{base, row_stride, col_stride};
    }
    [[nodiscard]] MatArray<const T, BR, BC> array() const noexcept
    {
        return MatArray<const T, BR, BC>{base, row_stride, col_stride};
    }

    template <typename U>
    MatMap& operator=(const LMat<U, BR, BC>& rhs) noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = rhs(r, c);
        return *this;
    }
    void setZero() noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = zero_of<T>();
    }
    void setIdentity() noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = r == c ? one_of<T>() : zero_of<T>();
    }
    void setConstant(const T& v) noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = v;
    }
    template <typename U>
    MatMap& operator=(const MatMap<U, BR, BC>& rhs) noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = rhs(r, c);
        return *this;
    }
    template <typename U>
    MatMap& operator=(const MatArray<U, BR, BC>& rhs) noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = rhs(r, c);
        return *this;
    }
    template <typename S>
        requires(is_scalar_like_v<S>)
    MatMap& operator=(const S& s) noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = s;
        return *this;
    }

    template <typename U>
    MatMap& operator+=(const LMat<U, BR, BC>& rhs) noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = detail::add((*this)(r, c), rhs(r, c));
        return *this;
    }
    template <typename U>
    MatMap& operator-=(const LMat<U, BR, BC>& rhs) noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = detail::sub((*this)(r, c), rhs(r, c));
        return *this;
    }
    template <typename S>
        requires(is_scalar_like_v<S>)
    MatMap& operator*=(const S& s) noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = detail::mul((*this)(r, c), s);
        return *this;
    }
    template <typename S>
        requires(is_scalar_like_v<S>)
    MatMap& operator/=(const S& s) noexcept
    {
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                (*this)(r, c) = detail::div((*this)(r, c), s);
        return *this;
    }
};

// ---------------------------------------------------------------------------
//  MatArray — element-wise view
// ---------------------------------------------------------------------------
template <typename T, int BR, int BC>
struct MatArray
{
    static_assert(BR > 0 && BC > 0, "MatArray needs positive dimensions");

    using raw_element = std::remove_const_t<T>;

    T*  base       = nullptr;
    int row_stride = 1;
    int col_stride = BR;

    using element_type        = raw_element;
    static constexpr int rows = BR;
    static constexpr int cols = BC;

    [[nodiscard]] auto& operator()(int r, int c) noexcept
    {
        return base[static_cast<std::ptrdiff_t>(r) * row_stride
                    + static_cast<std::ptrdiff_t>(c) * col_stride];
    }
    [[nodiscard]] const raw_element& operator()(int r, int c) const noexcept
    {
        return base[static_cast<std::ptrdiff_t>(r) * row_stride
                    + static_cast<std::ptrdiff_t>(c) * col_stride];
    }

    [[nodiscard]] LMat<raw_element, BR, BC> eval() const noexcept
    {
        return make_mat<raw_element, BR, BC>([&](int r, int c) { return (*this)(r, c); });
    }
    [[nodiscard]] operator LMat<raw_element, BR, BC>() const noexcept { return eval(); }
    [[nodiscard]] LMat<raw_element, BR, BC> matrix() const noexcept { return eval(); }
    [[nodiscard]] LMat<raw_element, BR, BC> mat() const noexcept { return eval(); }

    [[nodiscard]] LMat<raw_element, BR, BC> abs() const noexcept
    {
        return make_mat<raw_element, BR, BC>([&](int r, int c) { return math::abs((*this)(r, c)); });
    }
    [[nodiscard]] LMat<raw_element, BR, BC> sqrt() const noexcept
    {
        return make_mat<raw_element, BR, BC>([&](int r, int c) { return math::sqrt((*this)(r, c)); });
    }
    [[nodiscard]] auto maxCoeff() const noexcept
    {
        auto m = (*this)(0, 0);
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                m = math::max(m, (*this)(r, c));
        return stored(m);
    }
    [[nodiscard]] auto minCoeff() const noexcept
    {
        auto m = (*this)(0, 0);
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                m = math::min(m, (*this)(r, c));
        return stored(m);
    }
    [[nodiscard]] auto sum() const noexcept
    {
        auto s = (*this)(0, 0);
        for(int c = 0; c < BC; ++c)
            for(int r = 0; r < BR; ++r)
                if(r != 0 || c != 0)
                    s = detail::add(s, (*this)(r, c));
        return stored(s);
    }
};

// ---------------------------------------------------------------------------
//  Matrix operators
// ---------------------------------------------------------------------------
#define UIPC_LUISA_DEFINE_MAT_BINARY(OP, TRAMPOLINE)                                                     \
    template <typename A, typename B>                                                        \
        requires(is_mat_like_v<A> && is_mat_like_v<B> && mat_rows_v<A> == mat_rows_v<B>       \
                 && mat_cols_v<A> == mat_cols_v<B>)                                          \
    [[nodiscard]] inline auto operator OP(const A& a, const B& b) noexcept                   \
    {                                                                                        \
        constexpr int NR = mat_rows_v<A>;                                                    \
        constexpr int NC = mat_cols_v<A>;                                                    \
        using R = detail::element_result_t<mat_element_t<A>, mat_element_t<B>>;              \
        return make_mat<R, NR, NC>([&](int r, int c) { return detail::TRAMPOLINE(a(r, c), b(r, c)); }); \
    }

UIPC_LUISA_DEFINE_MAT_BINARY(+, add)
UIPC_LUISA_DEFINE_MAT_BINARY(-, sub)

#undef UIPC_LUISA_DEFINE_MAT_BINARY

/// Element-wise product / quotient (Eigen: `a.array() * b.array()`).
#define UIPC_LUISA_DEFINE_MAT_ARRAY_BINARY(OP, TRAMPOLINE)                                               \
    template <typename A, typename B>                                                        \
        requires(is_mat_like_v<A> && is_mat_like_v<B> && mat_rows_v<A> == mat_rows_v<B>       \
                 && mat_cols_v<A> == mat_cols_v<B>                                           \
                 && (is_mat_array_v<A> || is_mat_array_v<B>))                                \
    [[nodiscard]] inline auto operator OP(const A& a, const B& b) noexcept                   \
    {                                                                                        \
        constexpr int NR = mat_rows_v<A>;                                                    \
        constexpr int NC = mat_cols_v<A>;                                                    \
        using R = detail::element_result_t<mat_element_t<A>, mat_element_t<B>>;              \
        return make_mat<R, NR, NC>([&](int r, int c) { return detail::TRAMPOLINE(a(r, c), b(r, c)); }); \
    }

UIPC_LUISA_DEFINE_MAT_ARRAY_BINARY(*, mul)
UIPC_LUISA_DEFINE_MAT_ARRAY_BINARY(/, div)

#undef UIPC_LUISA_DEFINE_MAT_ARRAY_BINARY

template <typename M, typename S>
    requires(is_mat_like_v<M> && is_scalar_like_v<S> && !is_mat_like_v<S> && !is_vec_like_v<S>)
[[nodiscard]] inline auto operator*(const M& m, const S& s) noexcept
{
    constexpr int NR = mat_rows_v<M>;
    constexpr int NC = mat_cols_v<M>;
    using R = detail::element_result_t<mat_element_t<M>, S>;
    return make_mat<R, NR, NC>([&](int r, int c) { return detail::mul(m(r, c), s); });
}

template <typename S, typename M>
    requires(is_mat_like_v<M> && is_scalar_like_v<S> && !is_mat_like_v<S> && !is_vec_like_v<S>)
[[nodiscard]] inline auto operator*(const S& s, const M& m) noexcept
{
    constexpr int NR = mat_rows_v<M>;
    constexpr int NC = mat_cols_v<M>;
    using R = detail::element_result_t<S, mat_element_t<M>>;
    return make_mat<R, NR, NC>([&](int r, int c) { return detail::mul(s, m(r, c)); });
}

template <typename M, typename S>
    requires(is_mat_like_v<M> && is_scalar_like_v<S> && !is_mat_like_v<S> && !is_vec_like_v<S>)
[[nodiscard]] inline auto operator/(const M& m, const S& s) noexcept
{
    constexpr int NR = mat_rows_v<M>;
    constexpr int NC = mat_cols_v<M>;
    using R = detail::element_result_t<mat_element_t<M>, S>;
    return make_mat<R, NR, NC>([&](int r, int c) { return detail::div(m(r, c), s); });
}

template <typename M>
    requires is_mat_like_v<M>
[[nodiscard]] inline auto operator-(const M& m) noexcept
{
    constexpr int NR = mat_rows_v<M>;
    constexpr int NC = mat_cols_v<M>;
    using R = mat_element_t<M>;
    return make_mat<R, NR, NC>([&](int r, int c) { return detail::neg(m(r, c)); });
}

/// Matrix product: (R x K) * (K x C) -> (R x C).
/// Disabled for `.array()` operands, which multiply element-wise instead.
template <typename A, typename B>
    requires(is_mat_like_v<A> && is_mat_like_v<B> && mat_cols_v<A> == mat_rows_v<B>
             && !is_mat_array_v<A> && !is_mat_array_v<B>)
[[nodiscard]] inline auto operator*(const A& a, const B& b) noexcept
{
    constexpr int NR = mat_rows_v<A>;
    constexpr int NK = mat_cols_v<A>;
    constexpr int NC = mat_cols_v<B>;
    using R = detail::element_result_t<mat_element_t<A>, mat_element_t<B>>;
    return make_mat<R, NR, NC>(
        [&](int r, int c)
        {
            auto s = detail::mul(a(r, 0), b(0, c));
            for(int k = 1; k < NK; ++k)
                s = detail::add(s, detail::mul(a(r, k), b(k, c)));
            return s;
        });
}

/// Matrix * vector: (R x C) * (C) -> (R).
template <typename M, typename V>
    requires(is_mat_like_v<M> && is_vec_like_v<V> && mat_cols_v<M> == vec_dim_v<V>)
[[nodiscard]] inline auto operator*(const M& m, const V& v) noexcept
{
    constexpr int NR = mat_rows_v<M>;
    constexpr int NC = mat_cols_v<M>;
    using R = detail::element_result_t<mat_element_t<M>, vec_element_t<V>>;
    return make_vec<R, NR>(
        [&](int r)
        {
            auto s = detail::mul(m(r, 0), v(0));
            for(int c = 1; c < NC; ++c)
                s = detail::add(s, detail::mul(m(r, c), v(c)));
            return s;
        });
}

/// Vector * row-matrix: (R) * (1 x C) -> (R x C), i.e. the outer product
/// `a * b.transpose()`.
template <typename V, typename M>
    requires(is_vec_like_v<V> && is_mat_like_v<M> && mat_rows_v<M> == 1)
[[nodiscard]] inline auto operator*(const V& v, const M& m) noexcept
{
    constexpr int NR = vec_dim_v<V>;
    constexpr int NC = mat_cols_v<M>;
    using R = detail::element_result_t<vec_element_t<V>, mat_element_t<M>>;
    return make_mat<R, NR, NC>([&](int r, int c) { return detail::mul(v(r), m(0, c)); });
}

// ---------------------------------------------------------------------------
//  Deferred LVec / VecMap members that need LMat to be complete
// ---------------------------------------------------------------------------
template <typename T, int N>
[[nodiscard]] inline LMat<T, 1, N> LVec<T, N>::transpose() const noexcept
{
    return make_mat<T, 1, N>([&](int, int c) { return data[static_cast<std::size_t>(c)]; });
}

template <typename T, int N>
[[nodiscard]] inline LMat<T, N, N> LVec<T, N>::asDiagonal() const noexcept
{
    return make_mat<T, N, N>([&](int r, int c)
                             { return r == c ? data[static_cast<std::size_t>(r)] : zero_of<T>(); });
}

template <typename T, int M>
[[nodiscard]] inline LMat<T, 1, M> VecMap<T, M>::transpose() const noexcept
{
    return make_mat<T, 1, M>([&](int, int c) { return (*this)(c); });
}
}// namespace uipc::backend::luisa::luisa_device::math

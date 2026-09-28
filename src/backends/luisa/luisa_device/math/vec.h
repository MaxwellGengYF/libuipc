#pragma once
/********************************************************************
 * @file vec.h
 * @brief T2.1 — fixed-size vector `LVec<T,N>` for the generic numerics layer.
 *
 * `LVec<T,N>` replaces the `Eigen::Vector<T,N>` aliases (`uipc::Vector2/3/4/6/9/
 * 12`, ...) in *device* code paths.  As with everything in `luisa_device/math`
 * the element type `T` is generic:
 *
 *   T = double             -> host evaluation
 *   T = lcrt::Var<double>  -> device evaluation (a DSL local variable)
 *
 * Why `Var` and not `Expr`: LC's `Expr` deletes copy assignment (single
 * assignment), while `Var` is a mutable DSL local.  Storing `Var` keeps
 * `v(i) = ...`, `ret.segment<3>(0) = ...` and `auto r = a + b;` all working.
 *
 * Storage is a plain `std::array<T,N>` in column-major-compatible order, so an
 * LVec can be embedded in a `LUISA_STRUCT` buffer element or memcpy'd as-is.
 *
 * Views (`VecMap`, `VecArray`) hold a raw element pointer plus a stride.  That is
 * safe in the LC DSL because kernel bodies are ordinary host C++: an `LVec` of
 * DSL variables is a *host* struct holding expression handles, so a pointer into
 * it behaves like a pointer into a host array.  The usual Eigen caveat applies —
 * a view must not outlive its parent, hence `segment()` on an rvalue is deleted.
 *********************************************************************/
#include <luisa_device/math/math.h>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uipc::backend::luisa::luisa_device::math
{
template <typename T, int N>
struct LVec;

/// Mutable, strided view over a contiguous run of elements of a vector.
template <typename T, int M>
struct VecMap;

/// Element-wise (Eigen `.array()`) view; `T` may be const for read-only views.
template <typename T, int M>
struct VecArray;

/// Row vector produced by `LVec::transpose()`; defined in mat.h.
template <typename T, int R, int C>
struct LMat;

// ---------------------------------------------------------------------------
//  Traits
// ---------------------------------------------------------------------------
/// Vector-ness / dimension / element type of any vector-like argument, so that
/// `LVec`, `VecMap` and `VecArray` can share one set of operators.
template <typename V>
struct vec_like
{
    static constexpr bool value = false;
    static constexpr int  dim   = 0;
    using element               = void;
};

template <typename T, int N>
struct vec_like<LVec<T, N>>
{
    static constexpr bool value = true;
    static constexpr int  dim   = N;
    using element               = T;
};

template <typename T, int M>
struct vec_like<VecMap<T, M>>
{
    static constexpr bool value = true;
    static constexpr int  dim   = M;
    using element               = T;
};

template <typename T, int M>
struct vec_like<VecArray<T, M>>
{
    static constexpr bool value = true;
    static constexpr int  dim   = M;
    using element               = std::remove_const_t<T>;
};

template <typename V>
inline constexpr bool is_vec_like_v = vec_like<std::remove_cvref_t<V>>::value;

template <typename V>
inline constexpr int vec_dim_v = vec_like<std::remove_cvref_t<V>>::dim;

template <typename V>
using vec_element_t = typename vec_like<std::remove_cvref_t<V>>::element;

/// Arrays (`x.array()`) are element-wise, so `*` and `/` are element-wise for them.
template <typename V>
struct is_vec_array : std::false_type
{
};
template <typename T, int M>
struct is_vec_array<VecArray<T, M>> : std::true_type
{
};
template <typename V>
inline constexpr bool is_vec_array_v = is_vec_array<std::remove_cvref_t<V>>::value;

/// Arithmetic or DSL scalar (but not a vector).
template <typename S>
inline constexpr bool is_scalar_like_v =
    std::is_arithmetic_v<std::remove_cvref_t<S>> || lcrt::is_dsl_v<std::remove_cvref_t<S>>;

/// Materialise a computed value into its *storable* type (identity on the host).
template <typename T>
[[nodiscard]] inline auto stored(T&& x) noexcept
{
    return static_cast<storage_of_t<T>>(std::forward<T>(x));
}

namespace detail
{
/// Result element type of an element-wise combination of two element types.
template <typename A, typename B>
using element_result_t = scalar_result_t<A, B>;

template <typename R, int N, typename F, std::size_t... I>
[[nodiscard]] inline LVec<R, N> make_vec_impl(F&& f, std::index_sequence<I...>) noexcept
{
    // Braced-init-list evaluation is left-to-right and the factory is pure.
    return LVec<R, N>{static_cast<R>(f(static_cast<int>(I)))...};
}
}// namespace detail

/// Build an LVec component-wise from a factory `f(int i) -> scalar`.
template <typename R, int N, typename F>
[[nodiscard]] inline LVec<R, N> make_vec(F&& f) noexcept
{
    return detail::make_vec_impl<R, N>(std::forward<F>(f),
                                       std::make_index_sequence<static_cast<std::size_t>(N)>{});
}

// ---------------------------------------------------------------------------
//  LVec
// ---------------------------------------------------------------------------
template <typename T, int N>
struct LVec
{
    static_assert(N > 0, "LVec needs at least one component");
    static_assert(std::is_trivially_destructible_v<T>,
                  "LC buffer element types must be trivially destructible");

    using element_type       = T;
    static constexpr int dim = N;

    /// Plain, column-major-compatible storage (zero-initialised by default).
    std::array<T, N> data{};

    // -- construction --------------------------------------------------------
    LVec() noexcept = default;

    /// Component-wise: `LVec<double,3>{x, y, z}`.
    template <typename... U>
        requires(sizeof...(U) == N && (... && std::is_convertible_v<U, T>))
    LVec(U&&... u) noexcept : data{static_cast<T>(std::forward<U>(u))...}
    {
    }

    [[nodiscard]] static LVec Zero() noexcept { return LVec{}; }

    [[nodiscard]] static LVec Ones() noexcept
    {
        return make_vec<T, N>([](int) { return one_of<T>(); });
    }

    [[nodiscard]] static LVec Constant(const T& v) noexcept
    {
        return make_vec<T, N>([&](int) { return v; });
    }

    /// Binary unit vector: component `I` is one, the rest are zero.
    template <int I>
    [[nodiscard]] static LVec Unit() noexcept
    {
        static_assert(I >= 0 && I < N, "Unit<I>() index out of range");
        return make_vec<T, N>([](int i) { return i == I ? one_of<T>() : zero_of<T>(); });
    }

    /// Wrap a raw element pointer (the `Eigen::Map<Vector..>` analogue).
    [[nodiscard]] static VecMap<T, N> Map(T* p, int stride = 1) noexcept
    {
        return VecMap<T, N>{p, stride};
    }

    // -- element access ------------------------------------------------------
    [[nodiscard]] T& operator()(int i) noexcept { return data[static_cast<std::size_t>(i)]; }
    [[nodiscard]] const T& operator()(int i) const noexcept
    {
        return data[static_cast<std::size_t>(i)];
    }
    [[nodiscard]] T& operator[](int i) noexcept { return (*this)(i); }
    [[nodiscard]] const T& operator[](int i) const noexcept { return (*this)(i); }

    // Eigen-style component accessors (instantiated only where actually used).
    [[nodiscard]] T& x() noexcept { return (*this)(0); }
    [[nodiscard]] const T& x() const noexcept { return (*this)(0); }
    [[nodiscard]] T& y() noexcept { return (*this)(1); }
    [[nodiscard]] const T& y() const noexcept { return (*this)(1); }
    [[nodiscard]] T& z() noexcept { return (*this)(2); }
    [[nodiscard]] const T& z() const noexcept { return (*this)(2); }
    [[nodiscard]] T& w() noexcept { return (*this)(3); }
    [[nodiscard]] const T& w() const noexcept { return (*this)(3); }

    // -- views ---------------------------------------------------------------
    /// Mutable view of `[offset, offset + M)`.
    template <int M>
    [[nodiscard]] VecMap<T, M> segment(int offset) & noexcept
    {
        static_assert(M > 0, "segment<M> needs M > 0");
        return VecMap<T, M>{data.data() + offset, 1};
    }
    /// Copy of `[offset, offset + M)`. A const parent — or an rvalue, which also
    /// binds here — cannot hand out a view, so a copy is the only safe answer.
    template <int M>
    [[nodiscard]] LVec<T, M> segment(int offset) const& noexcept
    {
        return make_vec<T, M>([&](int i) { return data[static_cast<std::size_t>(offset + i)]; });
    }

    template <int M>
    [[nodiscard]] VecMap<T, M> head() & noexcept
    {
        return segment<M>(0);
    }
    template <int M>
    [[nodiscard]] LVec<T, M> head() const& noexcept
    {
        return segment<M>(0);
    }

    template <int M>
    [[nodiscard]] VecMap<T, M> tail() & noexcept
    {
        return segment<M>(N - M);
    }
    template <int M>
    [[nodiscard]] LVec<T, M> tail() const& noexcept
    {
        return segment<M>(N - M);
    }

    /// Element-wise view (the `.array()` of Eigen).
    [[nodiscard]] VecArray<T, N> array() noexcept { return VecArray<T, N>{data.data(), 1}; }
    [[nodiscard]] VecArray<const T, N> array() const noexcept
    {
        return VecArray<const T, N>{data.data(), 1};
    }

    // -- derived values (defined below, after the free functions) ------------
    [[nodiscard]] LMat<T, N, N> asDiagonal() const noexcept;
    [[nodiscard]] LMat<T, 1, N> transpose() const noexcept;
    [[nodiscard]] auto          dot(const LVec& other) const noexcept;
    [[nodiscard]] auto          squaredNorm() const noexcept;
    [[nodiscard]] auto          norm() const noexcept;
    [[nodiscard]] auto          normalized() const noexcept;
    [[nodiscard]] auto          sum() const noexcept;
    [[nodiscard]] auto          maxCoeff() const noexcept;
    [[nodiscard]] auto          minCoeff() const noexcept;
    /// Eigen's `.array().abs().max()` chain lands here.
    [[nodiscard]] auto max() const noexcept { return maxCoeff(); }
    [[nodiscard]] auto min() const noexcept { return minCoeff(); }
    [[nodiscard]] auto prod() const noexcept
    {
        auto p = data[0];
        for(int i = 1; i < N; ++i)
            p = detail::mul(p, data[static_cast<std::size_t>(i)]);
        return stored(p);
    }

    // -- assignment ----------------------------------------------------------
    template <typename U>
        requires(!std::is_same_v<U, T>)
    LVec& operator=(const LVec<U, N>& rhs) noexcept
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = rhs(i);
        return *this;
    }

    template <typename U>
    LVec& operator=(const VecMap<U, N>& rhs) noexcept
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = rhs(i);
        return *this;
    }

    template <typename U>
    LVec& operator=(const VecArray<U, N>& rhs) noexcept
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = rhs(i);
        return *this;
    }

    // -- compound assignment -------------------------------------------------
    template <typename U>
    LVec& operator+=(const LVec<U, N>& rhs) noexcept
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = detail::add((*this)(i), rhs(i));
        return *this;
    }
    template <typename U>
    LVec& operator-=(const LVec<U, N>& rhs) noexcept
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = detail::sub((*this)(i), rhs(i));
        return *this;
    }
    template <typename S>
        requires(is_scalar_like_v<S>)
    LVec& operator*=(const S& s) noexcept
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = detail::mul((*this)(i), s);
        return *this;
    }
    template <typename S>
        requires(is_scalar_like_v<S>)
    LVec& operator/=(const S& s) noexcept
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = detail::div((*this)(i), s);
        return *this;
    }

    void setZero() noexcept
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = zero_of<T>();
    }
    void setOnes() noexcept
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = one_of<T>();
    }
    void setConstant(const T& v) noexcept
    {
        for(int i = 0; i < N; ++i)
            (*this)(i) = v;
    }
};

// ---------------------------------------------------------------------------
//  VecMap — strided mutable view
// ---------------------------------------------------------------------------
template <typename T, int M>
struct VecMap
{
    static_assert(M > 0, "VecMap needs at least one component");

    T*  base   = nullptr;
    int stride = 1;

    using element_type       = T;
    static constexpr int dim = M;

    [[nodiscard]] T& operator()(int i) noexcept
    {
        return base[static_cast<std::ptrdiff_t>(i) * stride];
    }
    [[nodiscard]] const T& operator()(int i) const noexcept
    {
        return base[static_cast<std::ptrdiff_t>(i) * stride];
    }

    /// Materialise the view.
    [[nodiscard]] LVec<T, M> eval() const noexcept
    {
        return make_vec<T, M>([&](int i) { return (*this)(i); });
    }
    [[nodiscard]] operator LVec<T, M>() const noexcept { return eval(); }

    /// Row vector (defined in mat.h, which owns the `LMat` definition).
    [[nodiscard]] LMat<T, 1, M> transpose() const noexcept;

    /// Element-wise view of the same run of elements.
    [[nodiscard]] VecArray<T, M> array() noexcept { return VecArray<T, M>{base, stride}; }
    [[nodiscard]] VecArray<const T, M> array() const noexcept
    {
        return VecArray<const T, M>{base, stride};
    }

    [[nodiscard]] VecMap submap(int offset, int stride_ = -1) const noexcept
    {
        return VecMap{base + static_cast<std::ptrdiff_t>(offset) * stride,
                      stride_ < 0 ? stride : stride_};
    }

    template <typename U>
    VecMap& operator=(const LVec<U, M>& rhs) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = rhs(i);
        return *this;
    }
    template <typename U>
    VecMap& operator=(const VecMap<U, M>& rhs) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = rhs(i);
        return *this;
    }
    template <typename U>
    VecMap& operator=(const VecArray<U, M>& rhs) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = rhs(i);
        return *this;
    }
    template <typename S>
        requires(is_scalar_like_v<S>)
    VecMap& operator=(const S& s) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = s;
        return *this;
    }

    template <typename U>
    VecMap& operator+=(const LVec<U, M>& rhs) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::add((*this)(i), rhs(i));
        return *this;
    }
    template <typename U>
    VecMap& operator-=(const LVec<U, M>& rhs) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::sub((*this)(i), rhs(i));
        return *this;
    }
    template <typename S>
        requires(is_scalar_like_v<S>)
    VecMap& operator*=(const S& s) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::mul((*this)(i), s);
        return *this;
    }
    template <typename S>
        requires(is_scalar_like_v<S>)
    VecMap& operator/=(const S& s) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::div((*this)(i), s);
        return *this;
    }
};

// ---------------------------------------------------------------------------
//  VecArray — element-wise view
// ---------------------------------------------------------------------------
template <typename T, int M>
struct VecArray
{
    static_assert(M > 0, "VecArray needs at least one component");

    using raw_element = std::remove_const_t<T>;

    T*  base   = nullptr;
    int stride = 1;

    using element_type       = raw_element;
    static constexpr int dim = M;

    [[nodiscard]] auto& operator()(int i) noexcept
    {
        return base[static_cast<std::ptrdiff_t>(i) * stride];
    }
    [[nodiscard]] const raw_element& operator()(int i) const noexcept
    {
        return base[static_cast<std::ptrdiff_t>(i) * stride];
    }

    [[nodiscard]] LVec<raw_element, M> eval() const noexcept
    {
        return make_vec<raw_element, M>([&](int i) { return (*this)(i); });
    }
    [[nodiscard]] operator LVec<raw_element, M>() const noexcept { return eval(); }

    /// Eigen spells the conversion back to a vector `.matrix()`.
    [[nodiscard]] LVec<raw_element, M> matrix() const noexcept { return eval(); }
    [[nodiscard]] LVec<raw_element, M> mat() const noexcept { return eval(); }

    // -- element-wise maths --------------------------------------------------
    [[nodiscard]] LVec<raw_element, M> abs() const noexcept
    {
        return make_vec<raw_element, M>([&](int i) { return math::abs((*this)(i)); });
    }
    [[nodiscard]] LVec<raw_element, M> sqrt() const noexcept
    {
        return make_vec<raw_element, M>([&](int i) { return math::sqrt((*this)(i)); });
    }
    [[nodiscard]] LVec<raw_element, M> square() const noexcept
    {
        return make_vec<raw_element, M>([&](int i)
                                        { return detail::mul((*this)(i), (*this)(i)); });
    }
    [[nodiscard]] LVec<raw_element, M> inverse() const noexcept
    {
        return make_vec<raw_element, M>([&](int i)
                                        { return detail::div(one_of<raw_element>(), (*this)(i)); });
    }
    [[nodiscard]] LVec<raw_element, M> sign() const noexcept
    {
        return make_vec<raw_element, M>([&](int i) { return math::sign((*this)(i)); });
    }
    template <typename S>
    [[nodiscard]] LVec<raw_element, M> pow(const S& p) const noexcept
    {
        return make_vec<raw_element, M>([&](int i) { return math::pow((*this)(i), p); });
    }
    [[nodiscard]] auto max() const noexcept
    {
        auto m = (*this)(0);
        for(int i = 1; i < M; ++i)
            m = math::max(m, (*this)(i));
        return stored(m);
    }
    [[nodiscard]] auto min() const noexcept
    {
        auto m = (*this)(0);
        for(int i = 1; i < M; ++i)
            m = math::min(m, (*this)(i));
        return stored(m);
    }
    [[nodiscard]] auto sum() const noexcept
    {
        auto s = (*this)(0);
        for(int i = 1; i < M; ++i)
            s = detail::add(s, (*this)(i));
        return stored(s);
    }
    [[nodiscard]] auto prod() const noexcept
    {
        auto p = (*this)(0);
        for(int i = 1; i < M; ++i)
            p = detail::mul(p, (*this)(i));
        return stored(p);
    }

    // -- compound assignment (write-through) ---------------------------------
    template <typename V>
        requires(!std::is_const_v<T> && is_vec_like_v<V> && vec_dim_v<V> == M)
    VecArray& operator+=(const V& rhs) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::add((*this)(i), rhs(i));
        return *this;
    }
    template <typename V>
        requires(!std::is_const_v<T> && is_vec_like_v<V> && vec_dim_v<V> == M)
    VecArray& operator-=(const V& rhs) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::sub((*this)(i), rhs(i));
        return *this;
    }
    template <typename V>
        requires(!std::is_const_v<T> && is_vec_like_v<V> && vec_dim_v<V> == M)
    VecArray& operator*=(const V& rhs) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::mul((*this)(i), rhs(i));
        return *this;
    }
    template <typename V>
        requires(!std::is_const_v<T> && is_vec_like_v<V> && vec_dim_v<V> == M)
    VecArray& operator/=(const V& rhs) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::div((*this)(i), rhs(i));
        return *this;
    }
    template <typename S>
        requires(!std::is_const_v<T> && is_scalar_like_v<S>)
    VecArray& operator+=(const S& s) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::add((*this)(i), s);
        return *this;
    }
    template <typename S>
        requires(!std::is_const_v<T> && is_scalar_like_v<S>)
    VecArray& operator-=(const S& s) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::sub((*this)(i), s);
        return *this;
    }
    template <typename S>
        requires(!std::is_const_v<T> && is_scalar_like_v<S>)
    VecArray& operator*=(const S& s) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::mul((*this)(i), s);
        return *this;
    }
    template <typename S>
        requires(!std::is_const_v<T> && is_scalar_like_v<S>)
    VecArray& operator/=(const S& s) noexcept
    {
        for(int i = 0; i < M; ++i)
            (*this)(i) = detail::div((*this)(i), s);
        return *this;
    }
};

// ---------------------------------------------------------------------------
//  Vector operators
// ---------------------------------------------------------------------------
// NOTE: the component expression goes through detail::add/sub — LC's DSL scalar
// operators live at global scope and are shadowed by our own overloads here (see
// math.h).
#define UIPC_LUISA_DEFINE_VEC_BINARY(OP, TRAMPOLINE)                            \
    template <typename A, typename B>                                             \
        requires(is_vec_like_v<A> && is_vec_like_v<B>                             \
                 && vec_dim_v<A> == vec_dim_v<B>)                                 \
    [[nodiscard]] inline auto operator OP(const A& a, const B& b) noexcept        \
    {                                                                             \
        constexpr int N = vec_dim_v<A>;                                           \
        using R = detail::element_result_t<vec_element_t<A>, vec_element_t<B>>;   \
        return make_vec<R, N>([&](int i) { return detail::TRAMPOLINE(a(i), b(i)); });\
    }

UIPC_LUISA_DEFINE_VEC_BINARY(+, add)
UIPC_LUISA_DEFINE_VEC_BINARY(-, sub)

#undef UIPC_LUISA_DEFINE_VEC_BINARY

/// Element-wise product of two `.array()` views (Eigen: `a.array() * b.array()`).
/// Only enabled when at least one operand is an array, so that vector `*` stays
/// undefined exactly as it is in Eigen.
template <typename A, typename B>
    requires(is_vec_like_v<A> && is_vec_like_v<B> && vec_dim_v<A> == vec_dim_v<B>
             && (is_vec_array_v<A> || is_vec_array_v<B>))
[[nodiscard]] inline auto operator*(const A& a, const B& b) noexcept
{
    constexpr int N = vec_dim_v<A>;
    using R = detail::element_result_t<vec_element_t<A>, vec_element_t<B>>;
    return make_vec<R, N>([&](int i) { return detail::mul(a(i), b(i)); });
}

template <typename A, typename B>
    requires(is_vec_like_v<A> && is_vec_like_v<B> && vec_dim_v<A> == vec_dim_v<B>
             && (is_vec_array_v<A> || is_vec_array_v<B>))
[[nodiscard]] inline auto operator/(const A& a, const B& b) noexcept
{
    constexpr int N = vec_dim_v<A>;
    using R = detail::element_result_t<vec_element_t<A>, vec_element_t<B>>;
    return make_vec<R, N>([&](int i) { return detail::div(a(i), b(i)); });
}

/// Vector op scalar (and the reversed spellings for `*` / `/`).
template <typename V, typename S>
    requires(is_vec_like_v<V> && is_scalar_like_v<S> && !is_vec_like_v<S>)
[[nodiscard]] inline auto operator*(const V& v, const S& s) noexcept
{
    constexpr int N = vec_dim_v<V>;
    using R = detail::element_result_t<vec_element_t<V>, S>;
    return make_vec<R, N>([&](int i) { return detail::mul(v(i), s); });
}

template <typename S, typename V>
    requires(is_vec_like_v<V> && is_scalar_like_v<S> && !is_vec_like_v<S>)
[[nodiscard]] inline auto operator*(const S& s, const V& v) noexcept
{
    constexpr int N = vec_dim_v<V>;
    using R = detail::element_result_t<S, vec_element_t<V>>;
    return make_vec<R, N>([&](int i) { return detail::mul(s, v(i)); });
}

template <typename V, typename S>
    requires(is_vec_like_v<V> && is_scalar_like_v<S> && !is_vec_like_v<S>)
[[nodiscard]] inline auto operator/(const V& v, const S& s) noexcept
{
    constexpr int N = vec_dim_v<V>;
    using R = detail::element_result_t<vec_element_t<V>, S>;
    return make_vec<R, N>([&](int i) { return detail::div(v(i), s); });
}

template <typename S, typename V>
    requires(is_vec_like_v<V> && is_scalar_like_v<S> && !is_vec_like_v<S>)
[[nodiscard]] inline auto operator/(const S& s, const V& v) noexcept
{
    constexpr int N = vec_dim_v<V>;
    using R = detail::element_result_t<S, vec_element_t<V>>;
    return make_vec<R, N>([&](int i) { return detail::div(s, v(i)); });
}

template <typename V>
    requires is_vec_like_v<V>
[[nodiscard]] inline auto operator-(const V& v) noexcept
{
    constexpr int N = vec_dim_v<V>;
    using R = vec_element_t<V>;
    return make_vec<R, N>([&](int i) { return detail::neg(v(i)); });
}

// ---------------------------------------------------------------------------
//  Vector geometry
// ---------------------------------------------------------------------------
/// Dot product (also available as `LVec::dot`).
template <typename A, typename B>
    requires(is_vec_like_v<A> && is_vec_like_v<B> && vec_dim_v<A> == vec_dim_v<B>)
[[nodiscard]] inline auto dot(const A& a, const B& b) noexcept
{
    constexpr int N = vec_dim_v<A>;
    auto          s = detail::mul(a(0), b(0));
    for(int i = 1; i < N; ++i)
        s = detail::add(s, detail::mul(a(i), b(i)));
    return stored(s);
}

/// Cross product (3-component vectors only).
template <typename A, typename B>
    requires(is_vec_like_v<A> && is_vec_like_v<B> && vec_dim_v<A> == vec_dim_v<B>)
[[nodiscard]] inline auto cross(const A& a, const B& b) noexcept
{
    static_assert(vec_dim_v<A> == 3, "cross is only defined for 3-component vectors");
    using R = detail::element_result_t<vec_element_t<A>, vec_element_t<B>>;
    return make_vec<R, 3>([&](int i)
                          {
                              const int j = (i + 1) % 3;
                              const int k = (i + 2) % 3;
                              return detail::sub(detail::mul(a(j), b(k)),
                                                  detail::mul(a(k), b(j)));
                          });
}

template <typename A>
    requires is_vec_like_v<A>
[[nodiscard]] inline auto squaredNorm(const A& a) noexcept
{
    return dot(a, a);
}

template <typename A>
    requires is_vec_like_v<A>
[[nodiscard]] inline auto norm(const A& a) noexcept
{
    return stored(sqrt(dot(a, a)));
}

template <typename A>
    requires is_vec_like_v<A>
[[nodiscard]] inline auto normalized(const A& a) noexcept
{
    using R = vec_element_t<A>;
    auto n  = norm(a);
    return make_vec<R, vec_dim_v<A>>([&](int i) { return detail::div(a(i), n); });
}

template <typename A>
    requires is_vec_like_v<A>
[[nodiscard]] inline auto sum(const A& a) noexcept
{
    auto s = a(0);
    for(int i = 1; i < vec_dim_v<A>; ++i)
        s = detail::add(s, a(i));
    return stored(s);
}

/// Component-wise min / max (free functions, mirroring `.array().min()`).
template <typename A>
    requires is_vec_like_v<A>
[[nodiscard]] inline auto maxCoeff(const A& a) noexcept
{
    auto m = a(0);
    for(int i = 1; i < vec_dim_v<A>; ++i)
        m = math::max(m, a(i));
    return stored(m);
}

template <typename A>
    requires is_vec_like_v<A>
[[nodiscard]] inline auto minCoeff(const A& a) noexcept
{
    auto m = a(0);
    for(int i = 1; i < vec_dim_v<A>; ++i)
        m = math::min(m, a(i));
    return stored(m);
}

// ---------------------------------------------------------------------------
//  LVec member definitions (need the free functions above)
// ---------------------------------------------------------------------------
template <typename T, int N>
[[nodiscard]] inline auto LVec<T, N>::dot(const LVec& other) const noexcept
{
    return math::dot(*this, other);
}

template <typename T, int N>
[[nodiscard]] inline auto LVec<T, N>::squaredNorm() const noexcept
{
    return math::dot(*this, *this);
}

template <typename T, int N>
[[nodiscard]] inline auto LVec<T, N>::norm() const noexcept
{
    return stored(sqrt(math::dot(*this, *this)));
}

template <typename T, int N>
[[nodiscard]] inline auto LVec<T, N>::normalized() const noexcept
{
    auto n = norm();
    return make_vec<T, N>([&](int i)
                          { return detail::div(data[static_cast<std::size_t>(i)], n); });
}

template <typename T, int N>
[[nodiscard]] inline auto LVec<T, N>::maxCoeff() const noexcept
{
    auto m = data[0];
    for(int i = 1; i < N; ++i)
        m = math::max(m, data[static_cast<std::size_t>(i)]);
    return stored(m);
}

template <typename T, int N>
[[nodiscard]] inline auto LVec<T, N>::minCoeff() const noexcept
{
    auto m = data[0];
    for(int i = 1; i < N; ++i)
        m = math::min(m, data[static_cast<std::size_t>(i)]);
    return stored(m);
}

template <typename T, int N>
[[nodiscard]] inline auto LVec<T, N>::sum() const noexcept
{
    auto s = data[0];
    for(int i = 1; i < N; ++i)
        s = detail::add(s, data[static_cast<std::size_t>(i)]);
    return stored(s);
}
}// namespace uipc::backend::luisa::luisa_device::math

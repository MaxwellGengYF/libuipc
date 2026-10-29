#pragma once
/********************************************************************
 * @file ccd.inl
 * @brief T2.3 — CCD broad phases and the four conservative-advancement
 * (additive CCD) routines.  Ported from
 * `src/backends/cuda/utils/distance/details/ccd.inl`.
 *
 * ## Mapping
 *   Eigen `Array<T,N>::max/min` (element-wise)   -> detail::array_max/array_min
 *   `(v.array() > d).any()`                      -> detail::any_gt(v, d)
 *   `||` of two device predicates                -> `|`   (cookbook §7.6 trap #2)
 *   `while` + data-dependent `return`            -> fixed-iteration predicated
 *                                                   loop (detail::ccd_loop)
 *
 * ## The predicated loop (detail::ccd_loop)
 * The reference's loop is
 *
 *     T toc_prev = toc; toc = 0;
 *     while(true) {
 *         if(max_iter >= 0) { if(--max_iter < 0) return true; }
 *         T tocLowerBound = (1 - eta) * gap / ((dist + thickness) * maxDispMag);
 *         <advance the simplex by tocLowerBound * dX>;
 *         <rebuild flags + distance>;
 *         if(toc && gap / (dist + thickness) < gap0) break;   // -> return true
 *         toc += tocLowerBound;
 *         if(toc > toc_prev) return false;
 *     }
 *     return true;
 *
 * `max_iter` is a *host* `int` in the reference (`if(max_iter >= 0)` is host
 * control flow), so the port keeps a host loop with a host bound
 * (`detail::ccd_iterations`).  Everything else — the two termination tests, the
 * positions, `toc` — is device data, so:
 *
 *  - `running` is the loop's `active` flag.  All state updates (`x`, `toc`) are
 *    predicated on it; once a termination test fires the state is frozen, which
 *    makes every further iteration a no-op *including* `toc` (it is only advanced
 *    while `running`) and the distance (recomputed from frozen positions gives the
 *    same value).  The result is therefore identical to the early-return form.
 *  - The two exits are read off the flags afterwards:
 *      gap test fired              -> break            -> true  (`hit`)
 *      `toc > toc_prev` fired      -> return false     -> false (`!running & !hit`)
 *      iteration budget exhausted  -> return true      -> true  (`running`)
 *      nothing moved (`maxDispMag`) -> return false    -> false (`!sleeping`)
 *    Note the budget exit returns *true* in the reference, so a run that never
 *    terminated is "found".
 *
 * Cost (OP-C3): on the device the loop is a runtime `$while`, emitted **once** as
 * control flow (it used to be unrolled `max_iter` times at trace time, which is what
 * forced the TOI kernels down to a reduced bound).  Shader size and JIT time are now
 * independent of `max_iter`, so the reference's budget can be passed: the port's TOI
 * kernels use `luisa_device::ccd_max_iter()` (default 64, the convergence-sized value
 * the parity cases are measured at -- see `luisa_device/tuning.h`).  The host
 * instantiation keeps the plain loop: there is no `$while` outside a traced kernel.
 *********************************************************************/
#include <luisa_device/math/distance/flagged.h>

namespace uipc::backend::luisa::luisa_device::math::distance
{
// RULE M1 spellings: the ported expressions below mix host and DSL scalars, and
// this namespace declares no operators of its own (its *vector* operators come
// from `math`, which is an enclosing namespace).
using math::detail::add;
using math::detail::div;
using math::detail::mul;
using math::detail::neg;
using math::detail::sub;

namespace detail
{
/// `.any()` of the element-wise `v > threshold` (Eigen `(v.array() > d).any()`).
template <typename V, typename S>
[[nodiscard]] inline auto any_gt(const V& v, const S& threshold) noexcept
{
    constexpr int N = math::vec_dim_v<V>;
    using R         = math::storage_of_t<std::remove_cvref_t<decltype(v(0) > threshold)>>;
    auto gt = math::make_vec<R, N>([&](int i) noexcept { return v(i) > threshold; });
    return any_of(gt);
}

/// Storable predicate (`bool` on the host, `Var<bool>` on the device): every
/// `bool` the reference computes from device data becomes one of these.
template <typename T>
using ccd_bool_t = math::storage_of_t<bool_of_t<T>>;

/// Host bound of the predicated CCD loop; see the file header.
inline constexpr int kCcdUnboundedIterations = 64;

/// The reference's `max_iter` is a host value: a non-negative budget bounds the
/// loop, a negative one means "iterate until a termination test fires", for which
/// the port uses `kCcdUnboundedIterations`.
///
/// OP-C3: the bound may also be a *device* value (`lcrt::Var<int>`), which is how a
/// kernel passes a runtime budget (see `luisa_device/ccd_config.h`).  The reference's
/// negative "unbounded" spelling is host control flow and has no device meaning, so a
/// device bound is used as it stands (the caller guarantees `>= 0`).
template <typename IterT>
[[nodiscard]] inline auto ccd_iterations(IterT&& max_iter) noexcept
{
    if constexpr(std::is_arithmetic_v<std::remove_cvref_t<IterT>>)
    {
        return max_iter >= 0 ? int(max_iter) : kCcdUnboundedIterations;
    }
    else
    {
        return static_cast<std::remove_cvref_t<IterT>>(max_iter);
    }
}

/// The reference's conservative-advancement loop with every data-dependent exit
/// predicated (see the file header for the equivalence argument).
///
/// @param x         simplex positions, advanced in place
/// @param dx        the common-motion-removed displacements, one per position
/// @param update    `update()` -> the current squared distance; it rebuilds the
///                  flags of the current configuration (and, for edge-edge,
///                  applies the reference's `dFunc <= 0` endpoint fallback)
template <typename T, int N, typename IterT, typename Update>
[[nodiscard]] inline bool_of_t<T> ccd_loop(LVec<T, 3>*       x,
                                           const LVec<T, 3>* dx,
                                           T                 eta,
                                           T                 thickness,
                                           T                 max_disp_mag,
                                           ccd_bool_t<T>     sleeping,
                                           IterT             max_iter,
                                           T&                toc,
                                           Update&&          update) noexcept
{
    const T thickness2 = mul(thickness, thickness);

    T dist2 = update();
    T dist  = math::sqrt(dist2);
    // gap0 = eta * (dist2 - thickness^2) / (dist + thickness)
    T gap = div(mul(eta, sub(dist2, thickness2)), add(dist, thickness));

    const T toc_prev = toc;

    // `running` == false freezes the whole iteration.
    //
    // `T` and `ccd_bool_t<T>` are `Var`s on the device (`math::storage_of_t`
    // materialises every DSL value into a local reference variable), so each
    // assignment below is a *store* and the state below is loop-carried -- which is
    // what lets the body be traced once inside a `$while` (OP-C3) instead of being
    // unrolled once per iteration.
    ccd_bool_t<T> running = !sleeping;
    ccd_bool_t<T> hit     = false;
    T             toc_cur = zero_of<T>();

    auto iters = ccd_iterations(max_iter);

    // One conservative-advancement iteration.  Shared verbatim by the two loop
    // spellings below, so the host instantiation and the device kernel cannot drift.
    auto iterate = [&]() noexcept
    {
        // tocLowerBound = (1 - eta) * (dist2 - th^2) / ((dist + th) * maxDispMag)
        auto toc_lower = div(mul(sub(1.0, eta), sub(dist2, thickness2)),
                             mul(add(dist, thickness), max_disp_mag));

        // advance the simplex; `step` is 0 once a termination test has fired
        auto step = select_scalar(zero_of<T>(), toc_lower, running);
        for(int i = 0; i < N; ++i)
            x[i] = x[i] + step * dx[i];

        dist2 = update();
        dist  = math::sqrt(dist2);

        // if(toc && (d2 - th^2)/(d + th) < gap) break;   (-> true)
        auto gap_fired = detail::pred_and(running,
                                          toc_cur != zero_of<T>(),
                                          div(sub(dist2, thickness2), add(dist, thickness)) < gap);

        // toc += tocLowerBound — skipped exactly when the gap test fired
        toc_cur = add(toc_cur, select_scalar(zero_of<T>(), toc_lower, running & !gap_fired));

        // if(toc > toc_prev) return false;
        auto fail_fired = detail::pred_and(running, !gap_fired, toc_cur > toc_prev);

        hit     = detail::pred_or(hit, gap_fired);
        running = detail::pred_and(running, !gap_fired, !fail_fired);
    };

    if constexpr(math::is_dsl_value_v<T>)
    {
        // OP-C3 — a runtime `$while`, not a trace-time unroll.  The loop is emitted
        // once as control flow, so shader size and JIT time stop scaling with the
        // iteration budget; that is what allows passing the reference's budget
        // (see `ccd_config.h`) instead of a reduced, fidelity-compromising one.
        lcrt::Var<int> it = 0;
        $while((it < iters) & running)
        {
            iterate();
            it = add(it, 1);  // the port's DSL scalar spelling (RULE M1), not `it + 1`
        };
    }
    else
    {
        // Host instantiation (the `_device_vs_host` comparison tests): there is no
        // `$while` outside a traced kernel, and the trace-time unroll is free here.
        for(int it = 0; it < iters; ++it)
        {
            iterate();
        }
    }

    // `toc` is written exactly where the reference writes `toc = 0` (i.e. not at
    // all when the routine returned early because nothing moves).
    toc = select_scalar(toc, toc_cur, !sleeping);

    // "found" unless nothing moved, or the `toc > toc_prev` test fired.
    ccd_bool_t<T> found = detail::pred_and(detail::pred_or(running, hit), !sleeping);
    return found;
}
}  // namespace detail

// ---------------------------------------------------------------------------
//  Broad phases
//
// Each of these has two flavours — the pinned host `bool` signature and the device
// flavour returning a DSL predicate — over one shared `_impl`, so the logic exists
// exactly once (see `ccd.h`).
// ---------------------------------------------------------------------------

namespace detail
{
/// Shared body of `point_edge_cd_broadphase`.
template <typename T>
[[nodiscard]] inline bool_of_t<T> point_edge_cd_broadphase_impl(const LVec<T, 3>& x0,
                                                                const LVec<T, 3>& x1,
                                                                const LVec<T, 3>& x2,
                                                                T dist)
{
    auto max_e = array_max(x1, x2);
    auto min_e = array_min(x1, x2);
    // if((x0 - max_e > dist).any() || (min_e - x0 > dist).any()) return false;
    return !(any_gt(x0 - max_e, dist) | any_gt(min_e - x0, dist));
}

/// Shared body of `point_edge_ccd_broadphase`.
template <typename T>
[[nodiscard]] inline bool_of_t<T> point_edge_ccd_broadphase2_impl(const LVec<T, 2>& p,
                                                                  const LVec<T, 2>& e0,
                                                                  const LVec<T, 2>& e1,
                                                                  const LVec<T, 2>& dp,
                                                                  const LVec<T, 2>& de0,
                                                                  const LVec<T, 2>& de1,
                                                                  T dist)
{
    auto max_p = array_max(p, p + dp);
    auto min_p = array_min(p, p + dp);
    auto max_e = array_max(array_max(e0, e1), array_max(e0 + de0, e1 + de1));
    auto min_e = array_min(array_min(e0, e1), array_min(e0 + de0, e1 + de1));
    return !(any_gt(min_p - max_e, dist) | any_gt(min_e - max_p, dist));
}

/// Shared body of `point_triangle_cd_broadphase`.
template <typename T>
[[nodiscard]] inline bool_of_t<T> point_triangle_cd_broadphase_impl(const LVec<T, 3>& p,
                                                                    const LVec<T, 3>& t0,
                                                                    const LVec<T, 3>& t1,
                                                                    const LVec<T, 3>& t2,
                                                                    T dist)
{
    auto max_tri = array_max(array_max(t0, t1), t2);
    auto min_tri = array_min(array_min(t0, t1), t2);
    return !(any_gt(p - max_tri, dist) | any_gt(min_tri - p, dist));
}

/// Shared body of `edge_edge_cd_broadphase`.
template <typename T>
[[nodiscard]] inline bool_of_t<T> edge_edge_cd_broadphase_impl(const LVec<T, 3>& ea0,
                                                               const LVec<T, 3>& ea1,
                                                               const LVec<T, 3>& eb0,
                                                               const LVec<T, 3>& eb1,
                                                               T dist)
{
    auto max_a = array_max(ea0, ea1);
    auto min_a = array_min(ea0, ea1);
    auto max_b = array_max(eb0, eb1);
    auto min_b = array_min(eb0, eb1);
    return !(any_gt(min_a - max_b, dist) | any_gt(min_b - max_a, dist));
}

/// Shared body of `point_triangle_ccd_broadphase`.
template <typename T>
[[nodiscard]] inline bool_of_t<T> point_triangle_ccd_broadphase_impl(const LVec<T, 3>& p,
                                                                     const LVec<T, 3>& t0,
                                                                     const LVec<T, 3>& t1,
                                                                     const LVec<T, 3>& t2,
                                                                     const LVec<T, 3>& dp,
                                                                     const LVec<T, 3>& dt0,
                                                                     const LVec<T, 3>& dt1,
                                                                     const LVec<T, 3>& dt2,
                                                                     T dist)
{
    auto max_p   = array_max(p, p + dp);
    auto min_p   = array_min(p, p + dp);
    auto max_tri = array_max(array_max(array_max(t0, t1), array_max(t2, t0 + dt0)),
                             array_max(t1 + dt1, t2 + dt2));
    auto min_tri = array_min(array_min(array_min(t0, t1), array_min(t2, t0 + dt0)),
                             array_min(t1 + dt1, t2 + dt2));
    return !(any_gt(min_p - max_tri, dist) | any_gt(min_tri - max_p, dist));
}

/// Shared body of `edge_edge_ccd_broadphase`.
template <typename T>
[[nodiscard]] inline bool_of_t<T> edge_edge_ccd_broadphase_impl(const LVec<T, 3>& ea0,
                                                                const LVec<T, 3>& ea1,
                                                                const LVec<T, 3>& eb0,
                                                                const LVec<T, 3>& eb1,
                                                                const LVec<T, 3>& dea0,
                                                                const LVec<T, 3>& dea1,
                                                                const LVec<T, 3>& deb0,
                                                                const LVec<T, 3>& deb1,
                                                                T dist)
{
    auto max_a = array_max(array_max(ea0, ea1), array_max(ea0 + dea0, ea1 + dea1));
    auto min_a = array_min(array_min(ea0, ea1), array_min(ea0 + dea0, ea1 + dea1));
    auto max_b = array_max(array_max(eb0, eb1), array_max(eb0 + deb0, eb1 + deb1));
    auto min_b = array_min(array_min(eb0, eb1), array_min(eb0 + deb0, eb1 + deb1));
    return !(any_gt(min_a - max_b, dist) | any_gt(min_b - max_a, dist));
}

/// Shared body of `point_edge_ccd_broadphase`.
template <typename T>
[[nodiscard]] inline bool_of_t<T> point_edge_ccd_broadphase3_impl(const LVec<T, 3>& p,
                                                                  const LVec<T, 3>& e0,
                                                                  const LVec<T, 3>& e1,
                                                                  const LVec<T, 3>& dp,
                                                                  const LVec<T, 3>& de0,
                                                                  const LVec<T, 3>& de1,
                                                                  T dist)
{
    auto max_p = array_max(p, p + dp);
    auto min_p = array_min(p, p + dp);
    auto max_e = array_max(array_max(e0, e1), array_max(e0 + de0, e1 + de1));
    auto min_e = array_min(array_min(e0, e1), array_min(e0 + de0, e1 + de1));
    return !(any_gt(min_p - max_e, dist) | any_gt(min_e - max_p, dist));
}

/// Shared body of `point_point_ccd_broadphase`.
template <typename T>
[[nodiscard]] inline bool_of_t<T> point_point_ccd_broadphase_impl(const LVec<T, 3>& p0,
                                                                  const LVec<T, 3>& p1,
                                                                  const LVec<T, 3>& dp0,
                                                                  const LVec<T, 3>& dp1,
                                                                  T dist)
{
    auto max_p0 = array_max(p0, p0 + dp0);
    auto min_p0 = array_min(p0, p0 + dp0);
    auto max_p1 = array_max(p1, p1 + dp1);
    auto min_p1 = array_min(p1, p1 + dp1);
    return !(any_gt(min_p0 - max_p1, dist) | any_gt(min_p1 - max_p0, dist));
}
}  // namespace detail

/// Host flavour (the pinned `bool` signature).
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_edge_cd_broadphase(const LVec<T, 3>& x0,
                              const LVec<T, 3>& x1,
                              const LVec<T, 3>& x2,
                              T dist)
{
    return detail::point_edge_cd_broadphase_impl(x0, x1, x2, dist);
}

/// Device flavour: the predicate is a device value.
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> point_edge_cd_broadphase(const LVec<T, 3>& x0,
                                      const LVec<T, 3>& x1,
                                      const LVec<T, 3>& x2,
                                      T dist)
{
    return detail::point_edge_cd_broadphase_impl(x0, x1, x2, dist);
}

/// Host flavour (the pinned `bool` signature).
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_edge_ccd_broadphase(const LVec<T, 2>& p,
                               const LVec<T, 2>& e0,
                               const LVec<T, 2>& e1,
                               const LVec<T, 2>& dp,
                               const LVec<T, 2>& de0,
                               const LVec<T, 2>& de1,
                               T dist)
{
    return detail::point_edge_ccd_broadphase2_impl(p, e0, e1, dp, de0, de1, dist);
}

/// Device flavour: the predicate is a device value.
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> point_edge_ccd_broadphase(const LVec<T, 2>& p,
                                       const LVec<T, 2>& e0,
                                       const LVec<T, 2>& e1,
                                       const LVec<T, 2>& dp,
                                       const LVec<T, 2>& de0,
                                       const LVec<T, 2>& de1,
                                       T dist)
{
    return detail::point_edge_ccd_broadphase2_impl(p, e0, e1, dp, de0, de1, dist);
}

/// Host flavour (the pinned `bool` signature).
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_triangle_cd_broadphase(const LVec<T, 3>& p,
                                  const LVec<T, 3>& t0,
                                  const LVec<T, 3>& t1,
                                  const LVec<T, 3>& t2,
                                  T dist)
{
    return detail::point_triangle_cd_broadphase_impl(p, t0, t1, t2, dist);
}

/// Device flavour: the predicate is a device value.
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> point_triangle_cd_broadphase(const LVec<T, 3>& p,
                                          const LVec<T, 3>& t0,
                                          const LVec<T, 3>& t1,
                                          const LVec<T, 3>& t2,
                                          T dist)
{
    return detail::point_triangle_cd_broadphase_impl(p, t0, t1, t2, dist);
}

/// Host flavour (the pinned `bool` signature).
template <typename T>
    requires std::is_arithmetic_v<T>
bool edge_edge_cd_broadphase(const LVec<T, 3>& ea0,
                             const LVec<T, 3>& ea1,
                             const LVec<T, 3>& eb0,
                             const LVec<T, 3>& eb1,
                             T dist)
{
    return detail::edge_edge_cd_broadphase_impl(ea0, ea1, eb0, eb1, dist);
}

/// Device flavour: the predicate is a device value.
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> edge_edge_cd_broadphase(const LVec<T, 3>& ea0,
                                     const LVec<T, 3>& ea1,
                                     const LVec<T, 3>& eb0,
                                     const LVec<T, 3>& eb1,
                                     T dist)
{
    return detail::edge_edge_cd_broadphase_impl(ea0, ea1, eb0, eb1, dist);
}

/// Host flavour (the pinned `bool` signature).
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_triangle_ccd_broadphase(const LVec<T, 3>& p,
                                   const LVec<T, 3>& t0,
                                   const LVec<T, 3>& t1,
                                   const LVec<T, 3>& t2,
                                   const LVec<T, 3>& dp,
                                   const LVec<T, 3>& dt0,
                                   const LVec<T, 3>& dt1,
                                   const LVec<T, 3>& dt2,
                                   T dist)
{
    return detail::point_triangle_ccd_broadphase_impl(p, t0, t1, t2, dp, dt0, dt1, dt2, dist);
}

/// Device flavour: the predicate is a device value.
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> point_triangle_ccd_broadphase(const LVec<T, 3>& p,
                                           const LVec<T, 3>& t0,
                                           const LVec<T, 3>& t1,
                                           const LVec<T, 3>& t2,
                                           const LVec<T, 3>& dp,
                                           const LVec<T, 3>& dt0,
                                           const LVec<T, 3>& dt1,
                                           const LVec<T, 3>& dt2,
                                           T dist)
{
    return detail::point_triangle_ccd_broadphase_impl(p, t0, t1, t2, dp, dt0, dt1, dt2, dist);
}

/// Host flavour (the pinned `bool` signature).
template <typename T>
    requires std::is_arithmetic_v<T>
bool edge_edge_ccd_broadphase(const LVec<T, 3>& ea0,
                              const LVec<T, 3>& ea1,
                              const LVec<T, 3>& eb0,
                              const LVec<T, 3>& eb1,
                              const LVec<T, 3>& dea0,
                              const LVec<T, 3>& dea1,
                              const LVec<T, 3>& deb0,
                              const LVec<T, 3>& deb1,
                              T dist)
{
    return detail::edge_edge_ccd_broadphase_impl(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, dist);
}

/// Device flavour: the predicate is a device value.
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> edge_edge_ccd_broadphase(const LVec<T, 3>& ea0,
                                      const LVec<T, 3>& ea1,
                                      const LVec<T, 3>& eb0,
                                      const LVec<T, 3>& eb1,
                                      const LVec<T, 3>& dea0,
                                      const LVec<T, 3>& dea1,
                                      const LVec<T, 3>& deb0,
                                      const LVec<T, 3>& deb1,
                                      T dist)
{
    return detail::edge_edge_ccd_broadphase_impl(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, dist);
}

/// Host flavour (the pinned `bool` signature).
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_edge_ccd_broadphase(const LVec<T, 3>& p,
                               const LVec<T, 3>& e0,
                               const LVec<T, 3>& e1,
                               const LVec<T, 3>& dp,
                               const LVec<T, 3>& de0,
                               const LVec<T, 3>& de1,
                               T dist)
{
    return detail::point_edge_ccd_broadphase3_impl(p, e0, e1, dp, de0, de1, dist);
}

/// Device flavour: the predicate is a device value.
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> point_edge_ccd_broadphase(const LVec<T, 3>& p,
                                       const LVec<T, 3>& e0,
                                       const LVec<T, 3>& e1,
                                       const LVec<T, 3>& dp,
                                       const LVec<T, 3>& de0,
                                       const LVec<T, 3>& de1,
                                       T dist)
{
    return detail::point_edge_ccd_broadphase3_impl(p, e0, e1, dp, de0, de1, dist);
}

/// Host flavour (the pinned `bool` signature).
template <typename T>
    requires std::is_arithmetic_v<T>
bool point_point_ccd_broadphase(const LVec<T, 3>& p0,
                                const LVec<T, 3>& p1,
                                const LVec<T, 3>& dp0,
                                const LVec<T, 3>& dp1,
                                T dist)
{
    return detail::point_point_ccd_broadphase_impl(p0, p1, dp0, dp1, dist);
}

/// Device flavour: the predicate is a device value.
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> point_point_ccd_broadphase(const LVec<T, 3>& p0,
                                        const LVec<T, 3>& p1,
                                        const LVec<T, 3>& dp0,
                                        const LVec<T, 3>& dp1,
                                        T dist)
{
    return detail::point_point_ccd_broadphase_impl(p0, p1, dp0, dp1, dist);
}

// ---------------------------------------------------------------------------
//  Additive CCD
// ---------------------------------------------------------------------------
namespace detail
{
template <typename T>
[[nodiscard]] inline bool_of_t<T> point_triangle_ccd_impl(LVec<T, 3> p,
                                                          LVec<T, 3> t0,
                                                          LVec<T, 3> t1,
                                                          LVec<T, 3> t2,
                                                          LVec<T, 3> dp,
                                                          LVec<T, 3> dt0,
                                                          LVec<T, 3> dt1,
                                                          LVec<T, 3> dt2,
                                                          T          eta,
                                                          T          thickness,
                                                          int        max_iter,
                                                          T&         toc) noexcept
{
    // mov = (dt0 + dt1 + dt2 + dp) / 4; dt0 -= mov; dt1 -= mov; dt2 -= mov; dp -= mov;
    auto mov = (dt0 + dt1 + dt2 + dp) / 4.0;
    dt0      = dt0 - mov;
    dt1      = dt1 - mov;
    dt2      = dt2 - mov;
    dp       = dp - mov;

    // dispMag2Vec = {|dt0|^2, |dt1|^2, |dt2|^2}; maxDispMag = |dp| + sqrt(max)
    auto disp = math::make_vec<T, 3>([&](int i) noexcept
                                     {
                                         return i == 0   ? math::squaredNorm(dt0)
                                                : i == 1 ? math::squaredNorm(dt1)
                                                         : math::squaredNorm(dt2);
                                     });
    T max_disp_mag = add(dp.norm(), math::sqrt(math::maxCoeff(disp)));

    // the reference's `if(maxDispMag <= T(0)) return false;`
    auto sleeping = max_disp_mag <= zero_of<T>();

    LVec<T, 3>       X[4]  = {p, t0, t1, t2};
    const LVec<T, 3> dX[4] = {dp, dt0, dt1, dt2};

    auto update = [&]() noexcept -> T
    {
        auto flag = point_triangle_distance_flag<T>(X[0], X[1], X[2], X[3]);
        T    d2;
        point_triangle_distance2(flag, X[0], X[1], X[2], X[3], d2);
        return d2;
    };

    return detail::ccd_loop<T, 4>(X, dX, eta, thickness, max_disp_mag, sleeping, max_iter, toc, update);
}

template <typename T>
[[nodiscard]] inline bool_of_t<T> edge_edge_ccd_impl(LVec<T, 3> ea0,
                                                     LVec<T, 3> ea1,
                                                     LVec<T, 3> eb0,
                                                     LVec<T, 3> eb1,
                                                     LVec<T, 3> dea0,
                                                     LVec<T, 3> dea1,
                                                     LVec<T, 3> deb0,
                                                     LVec<T, 3> deb1,
                                                     T          eta,
                                                     T          thickness,
                                                     int        max_iter,
                                                     T&         toc) noexcept
{
    // mov = (dea0 + dea1 + deb0 + deb1) / 4; dea0 -= mov; ...
    auto mov = (dea0 + dea1 + deb0 + deb1) / 4.0;
    dea0     = dea0 - mov;
    dea1     = dea1 - mov;
    deb0     = deb0 - mov;
    deb1     = deb1 - mov;

    // maxDispMag = sqrt(max(|dea0|^2, |dea1|^2)) + sqrt(max(|deb0|^2, |deb1|^2))
    T max_disp_mag = add(math::sqrt(math::max(math::squaredNorm(dea0), math::squaredNorm(dea1))),
                         math::sqrt(math::max(math::squaredNorm(deb0), math::squaredNorm(deb1))));

    auto sleeping = max_disp_mag == zero_of<T>();

    LVec<T, 3>       X[4]  = {ea0, ea1, eb0, eb1};
    const LVec<T, 3> dX[4] = {dea0, dea1, deb0, deb1};

    auto update = [&]() noexcept -> T
    {
        auto flag = edge_edge_distance_flag<T>(X[0], X[1], X[2], X[3]);
        T    d2;
        edge_edge_distance2(flag, X[0], X[1], X[2], X[3], d2);

        // dFunc = d2 - thickness^2; if(dFunc <= 0) d2 = min endpoint distance^2
        // ("since we ensured other places that all dist smaller than dHat are
        // positive, this must be some far away nearly parallel edges")
        auto d_func = sub(d2, mul(thickness, thickness));
        auto dists  = math::make_vec<T, 4>([&](int i) noexcept
                                          {
                                              return i == 0   ? math::squaredNorm(X[0] - X[2])
                                                     : i == 1 ? math::squaredNorm(X[0] - X[3])
                                                     : i == 2 ? math::squaredNorm(X[1] - X[2])
                                                              : math::squaredNorm(X[1] - X[3]);
                                          });
        return select_scalar(d2, math::minCoeff(dists), d_func <= zero_of<T>());
    };

    return detail::ccd_loop<T, 4>(X, dX, eta, thickness, max_disp_mag, sleeping, max_iter, toc, update);
}

template <typename T>
[[nodiscard]] inline bool_of_t<T> point_edge_ccd_impl(LVec<T, 3> p,
                                                      LVec<T, 3> e0,
                                                      LVec<T, 3> e1,
                                                      LVec<T, 3> dp,
                                                      LVec<T, 3> de0,
                                                      LVec<T, 3> de1,
                                                      T          eta,
                                                      T          thickness,
                                                      int        max_iter,
                                                      T&         toc) noexcept
{
    // mov = (dp + de0 + de1) / 3; de0 -= mov; de1 -= mov; dp -= mov;
    auto mov = (dp + de0 + de1) / 3.0;
    de0      = de0 - mov;
    de1      = de1 - mov;
    dp       = dp - mov;

    // maxDispMag = |dp| + sqrt(max(|de0|^2, |de1|^2))
    T max_disp_mag = add(dp.norm(),
                         math::sqrt(math::max(math::squaredNorm(de0), math::squaredNorm(de1))));

    auto sleeping = max_disp_mag == zero_of<T>();

    LVec<T, 3>       X[3]  = {p, e0, e1};
    const LVec<T, 3> dX[3] = {dp, de0, de1};

    auto update = [&]() noexcept -> T
    {
        auto flag = point_edge_distance_flag<T>(X[0], X[1], X[2]);
        T    d2;
        point_edge_distance2(flag, X[0], X[1], X[2], d2);
        return d2;
    };

    return detail::ccd_loop<T, 3>(X, dX, eta, thickness, max_disp_mag, sleeping, max_iter, toc, update);
}

template <typename T>
[[nodiscard]] inline bool_of_t<T> point_point_ccd_impl(LVec<T, 3> p0,
                                                       LVec<T, 3> p1,
                                                       LVec<T, 3> dp0,
                                                       LVec<T, 3> dp1,
                                                       T          eta,
                                                       T          thickness,
                                                       int        max_iter,
                                                       T&         toc) noexcept
{
    // mov = (dp0 + dp1) / 2; dp1 -= mov; dp0 -= mov;
    auto mov = (dp0 + dp1) / 2.0;
    dp1      = dp1 - mov;
    dp0      = dp0 - mov;

    // maxDispMag = |dp0| + |dp1|
    T max_disp_mag = add(dp0.norm(), dp1.norm());

    auto sleeping = max_disp_mag == zero_of<T>();

    LVec<T, 3>       X[2]  = {p0, p1};
    const LVec<T, 3> dX[2] = {dp0, dp1};

    auto update = [&]() noexcept -> T
    {
        auto flag = point_point_distance_flag<T>(X[0], X[1]);
        T    d2;
        point_point_distance2(flag, X[0], X[1], d2);
        return d2;
    };

    return detail::ccd_loop<T, 2>(X, dX, eta, thickness, max_disp_mag, sleeping, max_iter, toc, update);
}
}  // namespace detail

/// Device flavour of `point_triangle_ccd` (see `ccd.h`).
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> point_triangle_ccd(LVec<T, 3> p,
                                LVec<T, 3> t0,
                                LVec<T, 3> t1,
                                LVec<T, 3> t2,
                                LVec<T, 3> dp,
                                LVec<T, 3> dt0,
                                LVec<T, 3> dt1,
                                LVec<T, 3> dt2,
                                T          eta,
                                T          thickness,
                                int        max_iter,
                                T&         toc)
{
    return detail::point_triangle_ccd_impl<T>(
        p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness, max_iter, toc);
}

/// Device flavour of `edge_edge_ccd` (see `ccd.h`).
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> edge_edge_ccd(LVec<T, 3> ea0,
                           LVec<T, 3> ea1,
                           LVec<T, 3> eb0,
                           LVec<T, 3> eb1,
                           LVec<T, 3> dea0,
                           LVec<T, 3> dea1,
                           LVec<T, 3> deb0,
                           LVec<T, 3> deb1,
                           T          eta,
                           T          thickness,
                           int        max_iter,
                           T&         toc)
{
    return detail::edge_edge_ccd_impl<T>(
        ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, eta, thickness, max_iter, toc);
}

/// Device flavour of `point_edge_ccd` (see `ccd.h`).
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> point_edge_ccd(LVec<T, 3> p,
                            LVec<T, 3> e0,
                            LVec<T, 3> e1,
                            LVec<T, 3> dp,
                            LVec<T, 3> de0,
                            LVec<T, 3> de1,
                            T          eta,
                            T          thickness,
                            int        max_iter,
                            T&         toc)
{
    return detail::point_edge_ccd_impl<T>(p, e0, e1, dp, de0, de1, eta, thickness, max_iter, toc);
}

/// Device flavour of `point_point_ccd` (see `ccd.h`).
template <typename T>
    requires(!std::is_arithmetic_v<T>)
bool_of_t<T> point_point_ccd(LVec<T, 3> p0,
                             LVec<T, 3> p1,
                             LVec<T, 3> dp0,
                             LVec<T, 3> dp1,
                             T          eta,
                             T          thickness,
                             int        max_iter,
                             T&         toc)
{
    return detail::point_point_ccd_impl<T>(p0, p1, dp0, dp1, eta, thickness, max_iter, toc);
}

// ---------------------------------------------------------------------------
//  Additive CCD — host flavours (the pinned `bool` signatures)
// ---------------------------------------------------------------------------

template <typename T>
    requires std::is_arithmetic_v<T>
bool point_triangle_ccd(LVec<T, 3> p,
                        LVec<T, 3> t0,
                        LVec<T, 3> t1,
                        LVec<T, 3> t2,
                        LVec<T, 3> dp,
                        LVec<T, 3> dt0,
                        LVec<T, 3> dt1,
                        LVec<T, 3> dt2,
                        T          eta,
                        T          thickness,
                        int        max_iter,
                        T&         toc)
{
    return detail::point_triangle_ccd_impl<T>(p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness, max_iter, toc);
}

template <typename T>
    requires std::is_arithmetic_v<T>
bool edge_edge_ccd(LVec<T, 3> ea0,
                   LVec<T, 3> ea1,
                   LVec<T, 3> eb0,
                   LVec<T, 3> eb1,
                   LVec<T, 3> dea0,
                   LVec<T, 3> dea1,
                   LVec<T, 3> deb0,
                   LVec<T, 3> deb1,
                   T          eta,
                   T          thickness,
                   int        max_iter,
                   T&         toc)
{
    return detail::edge_edge_ccd_impl<T>(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, eta, thickness, max_iter, toc);
}

template <typename T>
    requires std::is_arithmetic_v<T>
bool point_edge_ccd(LVec<T, 3> p,
                    LVec<T, 3> e0,
                    LVec<T, 3> e1,
                    LVec<T, 3> dp,
                    LVec<T, 3> de0,
                    LVec<T, 3> de1,
                    T          eta,
                    T          thickness,
                    int        max_iter,
                    T&         toc)
{
    return detail::point_edge_ccd_impl<T>(p, e0, e1, dp, de0, de1, eta, thickness, max_iter, toc);
}

template <typename T>
    requires std::is_arithmetic_v<T>
bool point_point_ccd(LVec<T, 3> p0,
                     LVec<T, 3> p1,
                     LVec<T, 3> dp0,
                     LVec<T, 3> dp1,
                     T          eta,
                     T          thickness,
                     int        max_iter,
                     T&         toc)
{
    return detail::point_point_ccd_impl<T>(p0, p1, dp0, dp1, eta, thickness, max_iter, toc);
}
}  // namespace uipc::backend::luisa::luisa_device::math::distance

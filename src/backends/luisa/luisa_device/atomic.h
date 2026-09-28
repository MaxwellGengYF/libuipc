#pragma once
/********************************************************************
 * @file atomic.h
 * @brief Atomic shims closing LuisaCompute DSL gaps (plan risk R9).
 *
 * LC specializes `detail::AtomicRef<T>` for int/uint/slong/ulong and float, but
 * NOT for double. The backend codegen already maps AtomicOp::ATOMIC_FETCH_ADD
 * to CUDA's `atomicAdd`, which has a double overload, so only the DSL type is
 * missing. We add the specialization here (a legal explicit specialization in
 * the type's own namespace) instead of patching the pinned LC install.
 *
 * This is the single place that knows about the gap; see the cookbook §3.3.
 *********************************************************************/
#include <luisa/luisa-compute.h>

#include <luisa_device/compat.h>

namespace luisa::compute::detail
{
// ---- double: mirrors the float specialization ---------------------------
template <>
class AtomicRef<double> : private AtomicRefBase
{
  public:
    explicit AtomicRef(const AtomicRefNode* access_chain) noexcept
        : AtomicRefBase{access_chain}
    {
    }

    auto exchange(Expr<double> desired) noexcept
    {
        return def<double>(
            access_chain()->operate(CallOp::ATOMIC_EXCHANGE, {desired.expression()}));
    }
    auto compare_exchange(Expr<double> expected, Expr<double> desired) noexcept
    {
        return def<double>(access_chain()->operate(
            CallOp::ATOMIC_COMPARE_EXCHANGE,
            {expected.expression(), desired.expression()}));
    }
    auto fetch_add(Expr<double> val) noexcept
    {
        return def<double>(
            access_chain()->operate(CallOp::ATOMIC_FETCH_ADD, {val.expression()}));
    }
    auto fetch_sub(Expr<double> val) noexcept
    {
        return def<double>(
            access_chain()->operate(CallOp::ATOMIC_FETCH_SUB, {val.expression()}));
    }
    auto fetch_min(Expr<double> val) noexcept
    {
        return def<double>(
            access_chain()->operate(CallOp::ATOMIC_FETCH_MIN, {val.expression()}));
    }
    auto fetch_max(Expr<double> val) noexcept
    {
        return def<double>(
            access_chain()->operate(CallOp::ATOMIC_FETCH_MAX, {val.expression()}));
    }
};
}// namespace luisa::compute::detail

namespace uipc::backend::luisa::luisa_device
{
// `atomic_add(&v(i), x)` in the reference backend becomes
// `v.atomic(i).fetch_add(x)` in a ported kernel. These tiny wrappers keep the
// intent readable and give one place to swap in an emulation should an LC
// backend lack a native float/double atomic.
[[nodiscard]] inline auto atomic_add(lcrt::detail::AtomicRef<float> ref, lcrt::Expr<float> v) noexcept
{
    return ref.fetch_add(v);
}

[[nodiscard]] inline auto atomic_add(lcrt::detail::AtomicRef<double> ref, lcrt::Expr<double> v) noexcept
{
    return ref.fetch_add(v);
}

[[nodiscard]] inline auto atomic_max(lcrt::detail::AtomicRef<double> ref, lcrt::Expr<double> v) noexcept
{
    return ref.fetch_max(v);
}

[[nodiscard]] inline auto atomic_min(lcrt::detail::AtomicRef<double> ref, lcrt::Expr<double> v) noexcept
{
    return ref.fetch_min(v);
}
}// namespace uipc::backend::luisa::luisa_device

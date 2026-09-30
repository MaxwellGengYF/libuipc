/********************************************************************
 * @file linear_bvh.inl
 * @brief LinearBVH — the parts that must be visible to a kernel body (port of
 *        src/backends/cuda/collision_detection/details/linear_bvh.inl).
 *
 * Included at the end of linear_bvh.h.  Contents:
 *   1. `detail::{morton_index, common_upper_bits, expand_bits, morton_code}` — the
 *      reference's `namespace detail` Morton-code arithmetic, written once for the
 *      host (`float`/`double`, the [bvh] oracles) and the device (`Var<float>`);
 *   2. `detail::{determine_range, find_split}` — Karras' range/split search (device);
 *   3. `LinearBVHDeviceViewer` — the reference's `LinearBVHViewer::query` traversal,
 *      as a kernel-body object;
 *   4. the host-side `LinearBVHViewer`/`LinearBVHVisitor` inline bodies.
 *
 * The traversal is the *contract* M8 depends on: the stack order (push left, then
 * right; pop from the top) fixes the order in which candidate indices are reported,
 * and the node test is the only filter.  Transcribed statement by statement from the
 * reference; the only additions are the Local bounds check (see
 * `LinearBVHDeviceViewer::query_impl`) and the LC spelling of the intrinsics.
 *********************************************************************/
#pragma once

#include <bit>
#include <cstdint>

namespace uipc::backend::luisa
{
namespace detail
{
template <typename M, typename I>
[[nodiscard]] inline auto morton_index(const M& morton, const I& idx) noexcept
{
    if constexpr(lps::is_dsl_value_v<M> || lps::is_dsl_value_v<I>)
    {
        return (lcrt::cast<std::uint64_t>(morton) << 32u)
               | lcrt::cast<std::uint64_t>(idx);
    }
    else
    {
        return (static_cast<std::uint64_t>(morton) << 32u)
               | static_cast<std::uint64_t>(idx);
    }
}

template <typename V>
[[nodiscard]] inline auto common_upper_bits(const V& lhs, const V& rhs) noexcept
{
    if constexpr(lps::is_dsl_value_v<V>)
    {
        // `::__clzll(lhs ^ rhs)` -- LC's clz on a 64-bit value lowers to __clzll.
        return lcrt::clz(lhs ^ rhs);
    }
    else
    {
        return static_cast<std::uint32_t>(std::countl_zero(lhs ^ rhs));
    }
}

template <typename V>
[[nodiscard]] inline auto expand_bits(V&& v) noexcept
{
    if constexpr(lps::is_dsl_value_v<V>)
    {
        // LC's Expr is single-assignment, so the running value is a Var local (the
        // same reason the numerics layer stores into `storage_of_t`).
        lcrt::Var<std::uint32_t> x = std::forward<V>(v);
        x                          = (x * 0x00010001u) & 0xFF0000FFu;
        x                          = (x * 0x00000101u) & 0x0F00F00Fu;
        x                          = (x * 0x00000011u) & 0xC30C30C3u;
        x                          = (x * 0x00000005u) & 0x49249249u;
        return x;
    }
    else
    {
        std::uint32_t x = static_cast<std::uint32_t>(std::forward<V>(v));
        x               = (x * 0x00010001u) & 0xFF0000FFu;
        x               = (x * 0x00000101u) & 0x0F00F00Fu;
        x               = (x * 0x00000011u) & 0xC30C30C3u;
        x               = (x * 0x00000005u) & 0x49249249u;
        return x;
    }
}

/// One Morton component: `expand_bits(static_cast<uint32_t>(xyz * 1024.0))` where the
/// multiply is `float * double -> double` (the reference's
/// `static_cast<std::uint32_t>(xyz.x() * 1024.0)`) followed by a truncating cast.
template <typename T>
[[nodiscard]] inline auto morton_component(const T& x) noexcept
{
    if constexpr(lps::is_dsl_value_v<T>)
    {
        return expand_bits(lcrt::cast<std::uint32_t>(lcrt::cast<double>(x) * 1024.0));
    }
    else
    {
        return expand_bits(
            static_cast<std::uint32_t>(static_cast<double>(x) * 1024.0));
    }
}

/// The reference's `detail::morton_code(Eigen::Vector3f xyz)`:
///   xyz = xyz.cwiseMin(1.0).cwiseMax(0.0);
///   xx = expand_bits(uint32(xyz.x() * 1024.0)); ... ; return xx*4 + yy*2 + zz;
/// Eigen's `mini<float>`/`maxi<float>` are `fminf`/`fmaxf`, i.e. exactly LC's
/// `lc_min`/`lc_max` on the device (and the host `lps::min`/`lps::max` agree for the
/// reachable NaN case: `fminf(NaN, 1) == 1`, and the second operand is never NaN).
template <typename T>
[[nodiscard]] inline auto morton_code(const lps::LVec<T, 3>& xyz) noexcept
{
    // `cwiseMin(1.0).cwiseMax(0.0)` in float (Eigen's Scalar is float here).
    lps::LVec<T, 3> c;
    for(int k = 0; k < 3; ++k)
        c(k) = lps::max(lps::min(xyz(k), 1.0f), 0.0f);

    auto xx = morton_component(c(0));
    auto yy = morton_component(c(1));
    auto zz = morton_component(c(2));
    return xx * 4u + yy * 2u + zz;
}

template <typename IdxT>
[[nodiscard]] inline LinearBVHMortonRange determine_range(
    const lcrt::BufferVar<std::uint64_t>& codes,
    lcrt::Var<std::uint32_t>              num_leaves,
    const IdxT&                           idx) noexcept
{
    LinearBVHMortonRange r;
    $if(idx == 0u)
    {
        r.x = 0u;
        r.y = num_leaves - 1u;
    }
    $else
    {
        // `int` arithmetic in the reference; the indices stay below 2^31.
        auto idx_i = lcrt::cast<std::int32_t>(idx);
        auto nl_i  = lcrt::cast<std::int32_t>(num_leaves);

        // determine direction of the range
        const auto self_code = codes->read(idx);
        const auto L_delta   = lcrt::cast<std::int32_t>(
            common_upper_bits(self_code, codes->read(idx - 1u)));
        const auto R_delta = lcrt::cast<std::int32_t>(
            common_upper_bits(self_code, codes->read(idx + 1u)));
        const auto d = lcrt::select(std::int32_t{-1}, std::int32_t{1}, R_delta > L_delta);

        // Compute upper bound for the length of the range
        const auto delta_min = lcrt::min(L_delta, R_delta);

        lcrt::Var<std::int32_t> l_max{2};
        lcrt::Var<std::int32_t> delta{-1};
        lcrt::Var<std::int32_t> i_tmp = idx_i + d * l_max;
        $if((i_tmp >= 0) & (i_tmp < nl_i))
        {
            delta = lcrt::cast<std::int32_t>(common_upper_bits(
                self_code, codes->read(lcrt::cast<std::uint32_t>(i_tmp))));
        };
        $while(delta > delta_min)
        {
            l_max = l_max << 1;
            i_tmp = idx_i + d * l_max;
            delta = -1;
            $if((i_tmp >= 0) & (i_tmp < nl_i))
            {
                delta = lcrt::cast<std::int32_t>(common_upper_bits(
                    self_code, codes->read(lcrt::cast<std::uint32_t>(i_tmp))));
            };
        };

        // Find the other end by binary search
        lcrt::Var<std::int32_t> l{0};
        lcrt::Var<std::int32_t> t = l_max >> 1;
        $while(t > 0)
        {
            i_tmp = idx_i + (l + t) * d;
            delta = -1;
            $if((i_tmp >= 0) & (i_tmp < nl_i))
            {
                delta = lcrt::cast<std::int32_t>(common_upper_bits(
                    self_code, codes->read(lcrt::cast<std::uint32_t>(i_tmp))));
            };
            $if(delta > delta_min)
            {
                l = l + t;
            };
            t = t >> 1;
        };

        const auto jdx = idx_i + l * d;
        // if(d < 0) { swap(idx, jdx); }  -- make sure that r.x < r.y
        r.x = lcrt::select(lcrt::cast<std::uint32_t>(idx_i),
                           lcrt::cast<std::uint32_t>(jdx),
                           d < 0);
        r.y = lcrt::select(lcrt::cast<std::uint32_t>(jdx),
                           lcrt::cast<std::uint32_t>(idx_i),
                           d < 0);
    };
    return r;
}

template <typename IdxT>
[[nodiscard]] inline lcrt::Var<std::uint32_t> find_split(
    const lcrt::BufferVar<std::uint64_t>& codes,
    lcrt::Var<std::uint32_t>              num_leaves,
    const IdxT&                           first,
    const IdxT&                           last) noexcept
{
    (void)num_leaves;  // unused in the reference too
    const auto first_code = codes->read(first);
    const auto last_code  = codes->read(last);

    lcrt::Var<std::uint32_t> split{0u};
    // `if(first_code == last_code) return (first + last) >> 1;` -- a *dead* branch in
    // practice: the 64-bit words differ whenever the positions differ, and the
    // single-leaf case is the only one with equal words.  Transcribed as it stands.
    $if(first_code == last_code)
    {
        split = (first + last) >> 1u;
    }
    $else
    {
        const auto delta_node = lcrt::cast<std::int32_t>(
            common_upper_bits(first_code, last_code));

        auto first_i = lcrt::cast<std::int32_t>(first);
        auto last_i  = lcrt::cast<std::int32_t>(last);

        // binary search...
        lcrt::Var<std::int32_t> split_i = first_i;
        lcrt::Var<std::int32_t> stride  = last_i - first_i;
        // do { ... } while(stride > 1)
        $while(true)
        {
            stride         = (stride + 1) >> 1;
            const auto mid = split_i + stride;
            $if(mid < last_i)
            {
                const auto delta = lcrt::cast<std::int32_t>(common_upper_bits(
                    first_code, codes->read(lcrt::cast<std::uint32_t>(mid))));
                $if(delta > delta_node)
                {
                    split_i = mid;
                };
            };
            $if(stride <= 1)
            {
                $break;
            };
        };
        split = lcrt::cast<std::uint32_t>(split_i);
    };
    return split;
}
}// namespace detail

/**
 * @brief The kernel-side traversal handle (the reference's `LinearBVHViewer` seen
 *        from inside a kernel).
 *
 * Construct it *inside* the kernel body from the arguments:
 * @code
 *   auto view = LinearBVHDeviceViewer{nodes, viewer.nodes(), viewer.aabbs(), n};
 * @endcode
 * (`viewer.nodes()`/`viewer.aabbs()` are the host `LBufferView`s, which convert to
 * kernel arguments; inside the kernel they are the `BufferVar`s.)
 */
class LinearBVHDeviceViewer
{
  public:
    static constexpr std::uint32_t DEFAULT_STACK_SIZE = 64u;

    /// The buffer handles are held by *reference*: LC's `Var<Buffer<T>>` has a deleted
    /// copy constructor (resource.h), so a kernel-body object must not copy them
    /// (`nodes`/`aabbs` are the kernel lambda's parameters and stay alive for the whole
    /// body).
    LinearBVHDeviceViewer(const lcrt::Var<std::uint32_t>&        num_nodes,
                          const lcrt::Var<std::uint32_t>&        num_objects,
                          const lcrt::BufferVar<LinearBVHNode>&  nodes,
                          const lcrt::BufferVar<LinearBVHAABB>&  aabbs) noexcept
        : m_num_nodes{num_nodes}
        , m_num_objects{num_objects}
        , m_nodes{nodes}
        , m_aabbs{aabbs}
    {
    }

    [[nodiscard]] const lcrt::Var<std::uint32_t>& num_nodes() const noexcept
    {
        return m_num_nodes;
    }
    [[nodiscard]] const lcrt::Var<std::uint32_t>& num_objects() const noexcept
    {
        return m_num_objects;
    }

    /// Query the AABBs that contain the point `q` (the reference's
    /// `query(const Vector3&, CallbackF)`: `node.contains(q.cast<float>())`).
    template <std::uint32_t StackNum = DEFAULT_STACK_SIZE, typename CallbackF>
    [[nodiscard]] lcrt::Var<std::uint32_t> query(
        const lps::LVec<lcrt::Var<float>, 3>& q, CallbackF&& cb) const noexcept
    {
        lcrt::Local<std::uint32_t> stack{StackNum};
        lcrt::Var<std::uint32_t>   sp{0u};
        return query_point_with_stack<StackNum>(q, stack, sp,
                                                std::forward<CallbackF>(cb));
    }

    /// The explicit-stack overload (the reference's `(q, stack, stack_num, cb)`): the
    /// caller owns the traversal stack, which LC spells `Local<uint32_t>`.
    template <std::uint32_t StackNum = DEFAULT_STACK_SIZE, typename CallbackF>
    [[nodiscard]] lcrt::Var<std::uint32_t> query(
        const lps::LVec<lcrt::Var<float>, 3>& q,
        lcrt::Local<std::uint32_t>&           stack,
        lcrt::Var<std::uint32_t>&             sp,
        CallbackF&&                           cb) const noexcept
    {
        return query_point_with_stack<StackNum>(q, stack, sp,
                                                std::forward<CallbackF>(cb));
    }

    /// Query the AABBs that overlap `q` (the reference's
    /// `query(const LinearBVHAABB&, CallbackF)`: `node.intersects(Q)`).
    template <std::uint32_t StackNum = DEFAULT_STACK_SIZE, typename CallbackF>
    [[nodiscard]] lcrt::Var<std::uint32_t> query_aabb(
        const LBox<lcrt::Var<float>>& q, CallbackF&& cb) const noexcept
    {
        lcrt::Local<std::uint32_t> stack{StackNum};
        lcrt::Var<std::uint32_t>   sp{0u};
        return query_aabb_with_stack<StackNum>(q, stack, sp, std::forward<CallbackF>(cb));
    }

    template <std::uint32_t StackNum = DEFAULT_STACK_SIZE, typename CallbackF>
    [[nodiscard]] lcrt::Var<std::uint32_t> query_aabb(
        const LBox<lcrt::Var<float>>& q,
        lcrt::Local<std::uint32_t>&   stack,
        lcrt::Var<std::uint32_t>&     sp,
        CallbackF&&                   cb) const noexcept
    {
        return query_aabb_with_stack<StackNum>(q, stack, sp, std::forward<CallbackF>(cb));
    }

    // NOTE: the reference's `stack_overflow()` member (`mutable int`) is NOT ported:
    // a viewer copied into a kernel cannot report anything back to the host, so the
    // flag was already unobservable there.  The traversal consequence is ported (an
    // overflowing stack truncates the candidate set) and pinned by the [bvh] test.

  private:
    template <std::uint32_t StackNum, typename CallbackF>
    [[nodiscard]] lcrt::Var<std::uint32_t> query_point_with_stack(
        const lps::LVec<lcrt::Var<float>, 3>& q,
        lcrt::Local<std::uint32_t>&           stack,
        lcrt::Var<std::uint32_t>&             sp,
        CallbackF&&                           cb) const noexcept
    {
        return query_impl<StackNum>(
            stack, sp,
            [&](const lcrt::Expr<LAABB>& node_box) noexcept
            { return box_contains(load_aabb(node_box), q); },
            std::forward<CallbackF>(cb));
    }

    template <std::uint32_t StackNum, typename CallbackF>
    [[nodiscard]] lcrt::Var<std::uint32_t> query_aabb_with_stack(
        const LBox<lcrt::Var<float>>& q,
        lcrt::Local<std::uint32_t>&   stack,
        lcrt::Var<std::uint32_t>&     sp,
        CallbackF&&                   cb) const noexcept
    {
        return query_impl<StackNum>(
            stack, sp,
            [&](const lcrt::Expr<LAABB>& node_box) noexcept
            { return box_intersects(load_aabb(node_box), q); },
            std::forward<CallbackF>(cb));
    }

    template <std::uint32_t StackNum, typename IntersectF, typename CallbackF>
    [[nodiscard]] lcrt::Var<std::uint32_t> query_impl(lcrt::Local<std::uint32_t>& stack,
                                                      lcrt::Var<std::uint32_t>&     sp,
                                                      IntersectF&& intersect,
                                                      CallbackF&&   cb) const noexcept
    {
        lcrt::Var<std::uint32_t> num_found{0u};

        sp       = 0u;
        stack[0] = 0u;  // root node is always 0
        sp       = 1u;

        $if(m_num_objects == 0u)
        {
            // the reference returns 0 right after pushing the root
            num_found = 0u;
        }
        $elif(m_num_objects == 1u)
        {
            $if(intersect(m_aabbs->read(0u)))
            {
                cb(m_nodes->read(0u).object_idx);
                num_found = 1u;
            };
        }
        $else
        {
            $while(sp > 0u)
            {
                sp        = sp - 1u;
                const auto node  = stack[sp];
                const auto L_idx = m_nodes->read(node).left_idx;
                const auto R_idx = m_nodes->read(node).right_idx;

                $if(intersect(m_aabbs->read(L_idx)))
                {
                    const auto obj_idx = m_nodes->read(L_idx).object_idx;
                    $if(obj_idx != 0xFFFFFFFFu)
                    {
                        cb(obj_idx);
                        num_found = num_found + 1u;
                    }
                    $else  // the node is not a leaf.
                    {
                        $if(sp < StackNum)
                        {
                            stack[sp] = L_idx;
                            sp        = sp + 1u;
                        };
                    };
                };
                $if(intersect(m_aabbs->read(R_idx)))
                {
                    const auto obj_idx = m_nodes->read(R_idx).object_idx;
                    $if(obj_idx != 0xFFFFFFFFu)
                    {
                        cb(obj_idx);
                        num_found = num_found + 1u;
                    }
                    $else  // the node is not a leaf.
                    {
                        $if(sp < StackNum)
                        {
                            stack[sp] = R_idx;
                            sp        = sp + 1u;
                        };
                    };
                };
                // the reference: `if(stack_ptr >= stack_end) { stack_overflow(); break; }`
                // (the port never wrote past the Local array, see the class note)
                $if(sp >= StackNum)
                {
                    $break;
                };
            };
        };
        return num_found;
    }

    // The counts are DSL values: inside a kernel they are the kernel's own arguments
    // (the host-side counts live in `LinearBVHViewer`).
    lcrt::Var<std::uint32_t>                     m_num_nodes;
    lcrt::Var<std::uint32_t>                     m_num_objects;
    const lcrt::BufferVar<LinearBVHNode>&        m_nodes;
    const lcrt::BufferVar<LinearBVHAABB>&        m_aabbs;
};

// ===========================================================================
//  Host-side viewer / visitor inline bodies
// ===========================================================================
inline LinearBVHViewer LinearBVH::viewer() const noexcept
{
    return LinearBVHViewer{static_cast<std::uint32_t>(m_nodes.size()),
                           static_cast<std::uint32_t>(m_mortons.size()),
                           m_nodes.cview(),
                           m_aabbs.cview()};
}

inline lud::LCBufferView<LinearBVHNode> LinearBVHVisitor::nodes() const noexcept
{
    return m_bvh.m_nodes.cview();
}

inline lud::LCBufferView<LinearBVHNode> LinearBVHVisitor::object_nodes() const noexcept
{
    auto object_count = m_bvh.m_indices.size();
    auto node_offset  = m_bvh.m_nodes.size() - object_count;
    return m_bvh.m_nodes.cview(node_offset, object_count);
}

inline lud::LCBufferView<LinearBVHAABB> LinearBVHVisitor::aabbs() const noexcept
{
    return m_bvh.m_aabbs.cview();
}

inline lud::LCBufferView<LinearBVHAABB> LinearBVHVisitor::top_aabb() const noexcept
{
    return m_bvh.m_max_aabb.cview();
}
}// namespace uipc::backend::luisa

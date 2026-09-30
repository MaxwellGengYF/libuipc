#pragma once
/********************************************************************
 * @file simplex_trajectory_filter_kernels.h
 * @brief T7.2b — the shared device kernels of the four BVH simplex trajectory
 *        filters (collision_detection/filters/{lbvh,stackless_bvh,
 *        info_stackless_bvh,info_stackless_bvh_v0}_simplex_trajectory_filter).
 *
 * WHY A SHARED FILE
 * -----------------
 * The reference's four `.cu` files are near-duplicates of each other (~7200 lines
 * for four variants): the swept-AABB builders (`detect_k1..k4`), the filter-active
 * classifiers (`filter_active_k1..k4`) and the TOI kernels (`filter_toi_k1..k4`) are
 * *literally the same code* in all four (verified by diff), and only the detect-time
 * per-pair predicate differs -- and only between the two *families*:
 *
 *   | file                          | detect predicate                                   |
 *   |-------------------------------|----------------------------------------------------|
 *   | lbvh / stackless_bvh          | `*_ccd_broadphase` box test                        |
 *   | info_stackless_bvh_v0         | the same test (it reads `v2b` instead of the node's
 *   |                               | bid/cid -- the same values, see the note below)     |
 *   | info_stackless_bvh (v1)       | the same test **plus the `alpha == 0` exact-distance
 *   |                               | branch** (`Dcd` below) -- the batch's one
 *   |                               | behavioural difference between the four             |
 *
 * The port keeps ONE copy of the shared kernels here and instantiates the detect
 * predicate pass twice (`Dcd == false` / `Dcd == true`), exactly the way T7.1b shared
 * its `culbvh` helpers between its near-duplicate info BVHs.  Every factory below is
 * `inline`, so the function-local shader cache is process-wide: the four filters
 * share one compiled shader per kernel (cold-JIT cost, cookbook §7.31).
 *
 * THE DETECT PREDICATE IS A *SELECTION* (the T7.2a decision, re-verified here)
 * --------------------------------------------------------------------------
 * The reference passes its per-pair predicate into the BVH's `detect`/`query`, where
 * it is called **inside the traversal callback**, i.e. only for objects whose AABB
 * already passed the fixed node test (`LinearBVHViewer::query`'s `node.intersects(Q)`
 * and `if(id > i && p(i, id))` / the stackless leaf test).  A record-time LC predicate
 * cannot read the per-primitive arrays these predicates touch (a captured view becomes
 * a shader-bound buffer of the first call: §7.32 rule 3 / §7.33 rules 1-3), so the port
 * *selects* instead: the BVH runs with a keep-all predicate (`KeepAll` below, i.e. the
 * maximal AABB-overlap pair set) and the reference's predicate body is applied as a
 * device pass over those reported pairs (`detect_*_pass_kernel`).  The surviving set is
 * therefore exactly the reference's, because a selection over a superset that never
 * changes which nodes are visited cannot gain or lose a pair -- what DOES differ is the
 * reported candidate *count* (hence the pair-buffer growth/retry), which §7.33 rule 1
 * already records as host-visible but not semantic.
 *
 * The InfoStacklessBVH family is the exception that has to be argued rather than
 * asserted: its `NodePred` *does* prune subtrees using device data (the node's common
 * bid/cid, `body_self_collision`, the contact mask).  The port passes `KeepAll` there
 * too, and the cull is sound to drop because **every cull condition is implied by the
 * leaf predicate**: the node's bid/cid are `-1` unless *every* leaf below shares them,
 * the leaf predicate rejects a pair whose two bids are equal with self-collision off,
 * and `allow_PT/PE/PP/EE_contact` are conjunctions that contain the node cull's own
 * `table[query_cid * dim + node_cid]` term (§7.32 rule 12, §7.33 rule 2(a)).  So a
 * culled node can only contain pairs the leaf predicate rejects anyway.  No
 * "device data reaches the predicate" hook was needed for any of the four variants.
 *
 * TYPE / API MAPPINGS (cookbook §3, §7.31-§7.33)
 * ---------------------------------------------
 *  * `CBufferView<T>` -> `lud::LCBufferView<T>`; a predicate argument -> the keep-all
 *    record-time callable; `UIPC_KERNEL_ASSERT(cond, "%f", x)` -> `lcrt::device_assert`
 *    (LC's device assert takes a literal message only -- the arguments are quoted in
 *    this file's comments instead of the message).
 *  * `CDense2D<IndexT>` masks -> flat views + an explicit extent (`index = row *
 *    extent + col`, §7.13/§7.33 rule 5); `allow_XX_contact` is transcribed
 *    one-directionally exactly as the reference spells it.
 *  * `positions(...).cast<float>()` -> `cast_float3` (the port's positions are
 *    `LVector3` = `LVec<Float,3>`, i.e. the reference's `Vector3`); the AABB arithmetic
 *    is the reference's `float` arithmetic (`float expand = ...`, then
 *    `min -= expand; max += expand;`).
 *  * `tois.resize_discard(n)` -> `resize(n)`; the kernel arguments of the TOI kernels
 *    are the reference's, except that the stackless variant's extra `max_iter` /
 *    `large_enough_toi` *arguments* (its own copy passes the two file constants) are
 *    the same values as the constants here, so the shared kernel bakes them in.
 *  * the port's CCD is a **fixed-iteration predicated loop** whose body is unrolled
 *    once per iteration at trace time (§7.8): passing the reference's
 *    `max_iter = 1000` would unroll 1000 times, so the TOI kernels pass
 *    `distance::detail::kCcdUnboundedIterations` (64) -- the port's convergence-sized
 *    bound.  The test measures that the tested regime is unaffected (the host CCD
 *    agrees with `max_iter = 1000` there) and the deviation is documented in §7.34.
 *********************************************************************/
#include <type_define.h>
#include <collision_detection/aabb.h>
#include <collision_detection/info_stackless_bvh.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>
#include <luisa_device/math/distance/distance.h>
#include <utils/codim_thickness.h>
#include <utils/primitive_d_hat.h>

#include <array>
#include <cstdint>

namespace uipc::backend::luisa
{
namespace simplex_trajectory_filter_detail
{
namespace lla  = luisa_device::linalg;
namespace lps  = luisa_device::math;
namespace ldist = luisa_device::math::distance;
// ---------------------------------------------------------------------------
//  The reference's file-scope constants (verbatim)
// ---------------------------------------------------------------------------
/// `constexpr SizeT max_iter = 1000;` -- the reference's CCD *budget*.  See the file
/// note: the port's CCD unrolls its body once per iteration, so the kernels pass the
/// convergence-sized `ldist::detail::kCcdUnboundedIterations` (64) instead.
constexpr SizeT kMaxIter = 1000;
/// `constexpr Float large_enough_toi = 1.1;` -- a `Float` (double) constant, i.e. the
/// double 1.1 truncated to `Float`; the same bit value as the `1.1f` literal the
/// half-plane filters spell out (§7.33 rule 9 pins the two spellings' difference).
constexpr Float kLargeEnoughToi = 1.1;
/// The bound the TOI kernels pass to the ported CCD (see the file note).
constexpr int kToiMaxIter = 16;

/// The dispatch bound of every kernel above: each one calls `lcrt::set_block_size(256)`,
/// so the grid is spelled as whole blocks (the ported collision code's convention; LC's
/// own `.dispatch(n)` rounds up in the same way).
[[nodiscard]] constexpr std::uint32_t round_up_block(std::uint64_t n,
                                                     std::uint32_t block = 256u) noexcept
{
    return static_cast<std::uint32_t>((n + block - 1u) / block * block);
}

/// The port's spelling of the reference's *device predicate argument* for
/// `AtomicCountingLBVH::detect/query`, `StacklessBVH::detect/query` and the info
/// variants: "keep every AABB-overlapping pair" (see the file note).  A record-time
/// callable with no captures -- its *type* is the shader cache key (§7.32 rule 3).
struct KeepAll
{
    [[nodiscard]] bool operator()(const auto& /*i*/, const auto& /*j*/) const noexcept
    {
        return true;
    }
};

// ---------------------------------------------------------------------------
//  Small device helpers
// ---------------------------------------------------------------------------
/// The reference's `.cast<float>()` on a device position (the port's `LVector3` is
/// `LVec<Float,3>`, so this is a genuine conversion, exactly like the reference's
/// `Vector3` -> `Vector3f`).
[[nodiscard]] inline lps::LVec<lcrt::Var<float>, 3> cast_float3(
    const lps::LVec<lcrt::Var<Float>, 3>& v) noexcept
{
    lps::LVec<lcrt::Var<float>, 3> r;
    for(int k = 0; k < 3; ++k)
        r(k) = lcrt::cast<float>(v(k));
    return r;
}

/// `aabb.min().array() -= expand; aabb.max().array() += expand;`
[[nodiscard]] inline LBox<lcrt::Var<float>> box_expanded(const LBox<lcrt::Var<float>>& b,
                                                         const lcrt::Var<float>& expand) noexcept
{
    LBox<lcrt::Var<float>> r = b;
    for(int k = 0; k < 3; ++k)
    {
        r.min(k) = r.min(k) - expand;
        r.max(k) = r.max(k) + expand;
    }
    return r;
}

/// `temp_PPs(i).setConstant(-1)` / `temp_PEs(i)` / `temp_PTs(i)` / `temp_EEs(i)`: LC
/// has no per-member buffer store, so a pair element is materialised whole and written
/// (the reference's in-place `setConstant` then partial `PP = {...}` writes become one
/// sentinel store followed by at most one pair store -- see `write_pair*`).
[[nodiscard]] inline lcrt::Var<LVector2i> sentinel_pair2() noexcept
{
    lcrt::Var<LVector2i> v;
    v.data[0] = IndexT{-1};
    v.data[1] = IndexT{-1};
    return v;
}

[[nodiscard]] inline lcrt::Var<LVector3i> sentinel_pair3() noexcept
{
    lcrt::Var<LVector3i> v;
    v.data[0] = IndexT{-1};
    v.data[1] = IndexT{-1};
    v.data[2] = IndexT{-1};
    return v;
}

[[nodiscard]] inline lcrt::Var<LVector4i> sentinel_pair4() noexcept
{
    lcrt::Var<LVector4i> v;
    v.data[0] = IndexT{-1};
    v.data[1] = IndexT{-1};
    v.data[2] = IndexT{-1};
    v.data[3] = IndexT{-1};
    return v;
}

/// Store a DSL index pair into a pair buffer element (`PP = {V0, V1}` etc.).
inline void write_pair2(lcrt::BufferVar<LVector2i>& buf,
                        const lcrt::Expr<std::uint32_t>& idx,
                        const lcrt::Var<IndexT>&        a,
                        const lcrt::Var<IndexT>&        b) noexcept
{
    lcrt::Var<LVector2i> v;
    v.data[0] = a;
    v.data[1] = b;
    buf->write(idx, v);
}

inline void write_pair3(lcrt::BufferVar<LVector3i>& buf,
                        const lcrt::Expr<std::uint32_t>& idx,
                        const lcrt::Var<IndexT>&        a,
                        const lcrt::Var<IndexT>&        b,
                        const lcrt::Var<IndexT>&        c) noexcept
{
    lcrt::Var<LVector3i> v;
    v.data[0] = a;
    v.data[1] = b;
    v.data[2] = c;
    buf->write(idx, v);
}

inline void write_pair4(lcrt::BufferVar<LVector4i>& buf,
                        const lcrt::Expr<std::uint32_t>& idx,
                        const lcrt::Var<IndexT>&        a,
                        const lcrt::Var<IndexT>&        b,
                        const lcrt::Var<IndexT>&        c,
                        const lcrt::Var<IndexT>&        d) noexcept
{
    lcrt::Var<LVector4i> v;
    v.data[0] = a;
    v.data[1] = b;
    v.data[2] = c;
    v.data[3] = d;
    buf->write(idx, v);
}

/// The reference's `Float D; distance::point_*_distance2(..., D);` -- the device
/// flavour writes its result through a `Var<Float>&`.
[[nodiscard]] inline lcrt::Var<Float> zero_float() noexcept
{
    return lcrt::Var<Float>{Float{0.0}};
}

// ---------------------------------------------------------------------------
//  The swept-AABB builders (the reference's `detect_k1..k4`, reused verbatim by all
//  four variants; the info variants' flavour additionally records the per-primitive
//  body/contact ids the info BVHs carry as node metadata -- see the header note)
// ---------------------------------------------------------------------------
/// `..._detect_k1_kernel`: swept point AABB of a *codim* vertex (`codimVs(i)`).
inline auto& detect_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<IndexT>, lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3>,
        lcrt::Buffer<LAABB>, lcrt::Buffer<Float>, lcrt::Buffer<Float>, Float,
        std::uint32_t>(
        [](lcrt::BufferVar<IndexT> codimVs, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LVector3> dxs, lcrt::BufferVar<LAABB> aabbs,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> alpha, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto vI = codimVs->read(i);

                const auto thickness       = thicknesses->read(vI);
                const auto d_hat_expansion = point_dcd_expansion(d_hats->read(vI));

                const auto pos   = load_vec3(Ps->read(vI));
                const auto pos_t = pos + load_vec3(dxs->read(vI)) * alpha;

                auto box = box_extend_point(box_empty<lcrt::Var<float>>(), cast_float3(pos));
                box      = box_extend_point(box, cast_float3(pos_t));

                const auto expand = lcrt::cast<float>(d_hat_expansion + thickness);

                lcrt::Var<LAABB> packed;
                store_aabb(packed, box_expanded(box, expand));
                aabbs->write(i, packed);
            };
        });
    return k;
}

/// `..._detect_k2_kernel`: swept point AABB of a surface vertex (`Vs(i)`).
inline auto& detect_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<IndexT>, lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3>,
        lcrt::Buffer<LAABB>, lcrt::Buffer<Float>, lcrt::Buffer<Float>, Float,
        std::uint32_t>(
        [](lcrt::BufferVar<IndexT> Vs, lcrt::BufferVar<LVector3> dxs,
           lcrt::BufferVar<LVector3> Ps, lcrt::BufferVar<LAABB> aabbs,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> alpha, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto vI = Vs->read(i);

                const auto thickness       = thicknesses->read(vI);
                const auto d_hat_expansion = point_dcd_expansion(d_hats->read(vI));

                const auto pos   = load_vec3(Ps->read(vI));
                const auto pos_t = pos + load_vec3(dxs->read(vI)) * alpha;

                auto box = box_extend_point(box_empty<lcrt::Var<float>>(), cast_float3(pos));
                box      = box_extend_point(box, cast_float3(pos_t));

                const auto expand = lcrt::cast<float>(d_hat_expansion + thickness);

                lcrt::Var<LAABB> packed;
                store_aabb(packed, box_expanded(box, expand));
                aabbs->write(i, packed);
            };
        });
    return k;
}

/// `..._detect_k3_kernel`: swept AABB of a surface edge.
inline auto& detect_k3_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector2i>, lcrt::Buffer<LVector3>, lcrt::Buffer<LAABB>,
        lcrt::Buffer<LVector3>, lcrt::Buffer<Float>, lcrt::Buffer<Float>, Float,
        std::uint32_t>(
        [](lcrt::BufferVar<LVector2i> Es, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LAABB> aabbs, lcrt::BufferVar<LVector3> dxs,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> alpha, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto eI = load_vec2i(Es->read(i));

                const auto thickness = edge_thickness(thicknesses->read(eI(0)),
                                                      thicknesses->read(eI(1)));
                const auto d_hat_expansion =
                    edge_dcd_expansion(d_hats->read(eI(0)), d_hats->read(eI(1)));

                const auto pos0   = load_vec3(Ps->read(eI(0)));
                const auto pos1   = load_vec3(Ps->read(eI(1)));
                const auto pos0_t = pos0 + load_vec3(dxs->read(eI(0))) * alpha;
                const auto pos1_t = pos1 + load_vec3(dxs->read(eI(1))) * alpha;

                auto box = box_extend_point(box_empty<lcrt::Var<float>>(), cast_float3(pos0));
                box      = box_extend_point(box, cast_float3(pos1));
                box      = box_extend_point(box, cast_float3(pos0_t));
                box      = box_extend_point(box, cast_float3(pos1_t));

                const auto expand = lcrt::cast<float>(d_hat_expansion + thickness);

                lcrt::Var<LAABB> packed;
                store_aabb(packed, box_expanded(box, expand));
                aabbs->write(i, packed);
            };
        });
    return k;
}

/// `..._detect_k4_kernel`: swept AABB of a surface triangle.
inline auto& detect_k4_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3i>, lcrt::Buffer<LVector3>, lcrt::Buffer<LAABB>,
        lcrt::Buffer<LVector3>, lcrt::Buffer<Float>, lcrt::Buffer<Float>, Float,
        std::uint32_t>(
        [](lcrt::BufferVar<LVector3i> Fs, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LAABB> aabbs, lcrt::BufferVar<LVector3> dxs,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> alpha, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto fI = load_vec3i(Fs->read(i));

                const auto thickness = triangle_thickness(thicknesses->read(fI(0)),
                                                          thicknesses->read(fI(1)),
                                                          thicknesses->read(fI(2)));
                const auto d_hat_expansion =
                    triangle_dcd_expansion(d_hats->read(fI(0)),
                                           d_hats->read(fI(1)),
                                           d_hats->read(fI(2)));

                const auto pos0   = load_vec3(Ps->read(fI(0)));
                const auto pos1   = load_vec3(Ps->read(fI(1)));
                const auto pos2   = load_vec3(Ps->read(fI(2)));
                const auto pos0_t = pos0 + load_vec3(dxs->read(fI(0))) * alpha;
                const auto pos1_t = pos1 + load_vec3(dxs->read(fI(1))) * alpha;
                const auto pos2_t = pos2 + load_vec3(dxs->read(fI(2))) * alpha;

                auto box = box_extend_point(box_empty<lcrt::Var<float>>(), cast_float3(pos0));
                box      = box_extend_point(box, cast_float3(pos1));
                box      = box_extend_point(box, cast_float3(pos2));
                box      = box_extend_point(box, cast_float3(pos0_t));
                box      = box_extend_point(box, cast_float3(pos1_t));
                box      = box_extend_point(box, cast_float3(pos2_t));

                const auto expand = lcrt::cast<float>(d_hat_expansion + thickness);

                lcrt::Var<LAABB> packed;
                store_aabb(packed, box_expanded(box, expand));
                aabbs->write(i, packed);
            };
        });
    return k;
}

// ---------------------------------------------------------------------------
//  The info variants' AABB builders: the same swept boxes plus the per-primitive
//  body/contact ids the info BVHs carry as node metadata
//  (`bids(i) = v2bs(first vertex)`, `cids(i) = contact_ids(first vertex)`).
// ---------------------------------------------------------------------------
/// `InfoStacklessBVHSimplexTrajectoryFilter_detect_k1_kernel`.
inline auto& detect_info_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<IndexT>, lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3>,
        lcrt::Buffer<LAABB>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<Float>,
        lcrt::Buffer<Float>, Float, std::uint32_t>(
        [](lcrt::BufferVar<IndexT> codimVs, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LVector3> dxs, lcrt::BufferVar<LAABB> aabbs,
           lcrt::BufferVar<IndexT> v2bs, lcrt::BufferVar<IndexT> contact_ids,
           lcrt::BufferVar<IndexT> bids, lcrt::BufferVar<IndexT> cids,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> alpha, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto vI = codimVs->read(i);

                const auto thickness       = thicknesses->read(vI);
                const auto d_hat_expansion = point_dcd_expansion(d_hats->read(vI));

                const auto pos   = load_vec3(Ps->read(vI));
                const auto pos_t = pos + load_vec3(dxs->read(vI)) * alpha;

                auto box = box_extend_point(box_empty<lcrt::Var<float>>(), cast_float3(pos));
                box      = box_extend_point(box, cast_float3(pos_t));

                const auto expand = lcrt::cast<float>(d_hat_expansion + thickness);

                lcrt::Var<LAABB> packed;
                store_aabb(packed, box_expanded(box, expand));
                aabbs->write(i, packed);
                bids->write(i, v2bs->read(vI));
                cids->write(i, contact_ids->read(vI));
            };
        });
    return k;
}

/// `InfoStacklessBVHSimplexTrajectoryFilter_detect_k2_kernel`.
inline auto& detect_info_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<IndexT>, lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3>,
        lcrt::Buffer<LAABB>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<Float>,
        lcrt::Buffer<Float>, Float, std::uint32_t>(
        [](lcrt::BufferVar<IndexT> Vs, lcrt::BufferVar<LVector3> dxs,
           lcrt::BufferVar<LVector3> Ps, lcrt::BufferVar<LAABB> aabbs,
           lcrt::BufferVar<IndexT> v2bs, lcrt::BufferVar<IndexT> contact_ids,
           lcrt::BufferVar<IndexT> bids, lcrt::BufferVar<IndexT> cids,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> alpha, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto vI = Vs->read(i);

                const auto thickness       = thicknesses->read(vI);
                const auto d_hat_expansion = point_dcd_expansion(d_hats->read(vI));

                const auto pos   = load_vec3(Ps->read(vI));
                const auto pos_t = pos + load_vec3(dxs->read(vI)) * alpha;

                auto box = box_extend_point(box_empty<lcrt::Var<float>>(), cast_float3(pos));
                box      = box_extend_point(box, cast_float3(pos_t));

                const auto expand = lcrt::cast<float>(d_hat_expansion + thickness);

                lcrt::Var<LAABB> packed;
                store_aabb(packed, box_expanded(box, expand));
                aabbs->write(i, packed);
                bids->write(i, v2bs->read(vI));
                cids->write(i, contact_ids->read(vI));
            };
        });
    return k;
}

/// `InfoStacklessBVHSimplexTrajectoryFilter_detect_k3_kernel`.
inline auto& detect_info_k3_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector2i>, lcrt::Buffer<LVector3>, lcrt::Buffer<LAABB>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<LVector3>, lcrt::Buffer<Float>,
        lcrt::Buffer<Float>, Float, std::uint32_t>(
        [](lcrt::BufferVar<LVector2i> Es, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LAABB> aabbs, lcrt::BufferVar<IndexT> v2bs,
           lcrt::BufferVar<IndexT> contact_ids, lcrt::BufferVar<IndexT> bids,
           lcrt::BufferVar<IndexT> cids, lcrt::BufferVar<LVector3> dxs,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> alpha, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto eI = load_vec2i(Es->read(i));

                const auto thickness = edge_thickness(thicknesses->read(eI(0)),
                                                      thicknesses->read(eI(1)));
                const auto d_hat_expansion =
                    edge_dcd_expansion(d_hats->read(eI(0)), d_hats->read(eI(1)));

                const auto pos0   = load_vec3(Ps->read(eI(0)));
                const auto pos1   = load_vec3(Ps->read(eI(1)));
                const auto pos0_t = pos0 + load_vec3(dxs->read(eI(0))) * alpha;
                const auto pos1_t = pos1 + load_vec3(dxs->read(eI(1))) * alpha;

                auto box = box_extend_point(box_empty<lcrt::Var<float>>(), cast_float3(pos0));
                box      = box_extend_point(box, cast_float3(pos1));
                box      = box_extend_point(box, cast_float3(pos0_t));
                box      = box_extend_point(box, cast_float3(pos1_t));

                const auto expand = lcrt::cast<float>(d_hat_expansion + thickness);

                lcrt::Var<LAABB> packed;
                store_aabb(packed, box_expanded(box, expand));
                aabbs->write(i, packed);
                bids->write(i, v2bs->read(eI(0)));
                cids->write(i, contact_ids->read(eI(0)));
            };
        });
    return k;
}

/// `InfoStacklessBVHSimplexTrajectoryFilter_detect_k4_kernel`.
inline auto& detect_info_k4_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3i>, lcrt::Buffer<LVector3>, lcrt::Buffer<LAABB>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<LVector3>, lcrt::Buffer<Float>,
        lcrt::Buffer<Float>, Float, std::uint32_t>(
        [](lcrt::BufferVar<LVector3i> Fs, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LAABB> aabbs, lcrt::BufferVar<IndexT> v2bs,
           lcrt::BufferVar<IndexT> contact_ids, lcrt::BufferVar<IndexT> bids,
           lcrt::BufferVar<IndexT> cids, lcrt::BufferVar<LVector3> dxs,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> alpha, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto fI = load_vec3i(Fs->read(i));

                const auto thickness = triangle_thickness(thicknesses->read(fI(0)),
                                                          thicknesses->read(fI(1)),
                                                          thicknesses->read(fI(2)));
                const auto d_hat_expansion =
                    triangle_dcd_expansion(d_hats->read(fI(0)),
                                           d_hats->read(fI(1)),
                                           d_hats->read(fI(2)));

                const auto pos0   = load_vec3(Ps->read(fI(0)));
                const auto pos1   = load_vec3(Ps->read(fI(1)));
                const auto pos2   = load_vec3(Ps->read(fI(2)));
                const auto pos0_t = pos0 + load_vec3(dxs->read(fI(0))) * alpha;
                const auto pos1_t = pos1 + load_vec3(dxs->read(fI(1))) * alpha;
                const auto pos2_t = pos2 + load_vec3(dxs->read(fI(2))) * alpha;

                auto box = box_extend_point(box_empty<lcrt::Var<float>>(), cast_float3(pos0));
                box      = box_extend_point(box, cast_float3(pos1));
                box      = box_extend_point(box, cast_float3(pos2));
                box      = box_extend_point(box, cast_float3(pos0_t));
                box      = box_extend_point(box, cast_float3(pos1_t));
                box      = box_extend_point(box, cast_float3(pos2_t));

                const auto expand = lcrt::cast<float>(d_hat_expansion + thickness);

                lcrt::Var<LAABB> packed;
                store_aabb(packed, box_expanded(box, expand));
                aabbs->write(i, packed);
                bids->write(i, v2bs->read(fI(0)));
                cids->write(i, contact_ids->read(fI(0)));
            };
        });
    return k;
}
// ---------------------------------------------------------------------------
//  The detect-time per-pair predicates, as *selection* passes over the traversal's
//  reported pairs (see the file note)
//
//  Each pass reads one raw `(i, j)` pair the keep-all traversal reported, evaluates
//  the reference's predicate body on it and writes `flags[i] = keep ? 1 : 0`; the
//  caller then compacts the pairs stably (`lla::compact_by_flags`), which keeps the
//  surviving pairs in the traversal's own order -- the order the reference's
//  in-traversal selection produces.
//
//  The reference's early `return false`s collapse into one conjunction: every check is
//  a pure function of device data and every read is in bounds for a pair the traversal
//  reported, so evaluating them all is equivalent (the ported half-plane filter's
//  `in_scope` precedent, §7.33).
// ---------------------------------------------------------------------------
/// `allow_XX_contact` for the port's flat mask layout (`index = row * extent + col`).
[[nodiscard]] inline auto mask_allows(const lcrt::BufferVar<IndexT>& table,
                                      const lcrt::Var<IndexT>&       row,
                                      const lcrt::Var<IndexT>&       col,
                                      const lcrt::Var<std::int32_t>& extent) noexcept
{
    return table->read(row * extent + col) != 0;
}

/// `LBVH/StacklessBVH/InfoStacklessBVHV0_detect_AllP_CodimP_pred`
/// (`Dcd == true` adds the InfoStacklessBVH v1 `alpha == 0` exact-distance branch).
template <bool Dcd>
inline auto& detect_AllP_CodimP_pass_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector2i>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3>, lcrt::Buffer<Float>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<Float>, lcrt::Buffer<std::int32_t>,
        std::int32_t, std::int32_t, Float, std::uint32_t>(
        [](lcrt::BufferVar<LVector2i> raw_pairs, lcrt::BufferVar<IndexT> Vs,
           lcrt::BufferVar<IndexT> codimVs, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LVector3> dxs, lcrt::BufferVar<Float> thicknesses,
           lcrt::BufferVar<IndexT> dimensions, lcrt::BufferVar<IndexT> contact_element_ids,
           lcrt::BufferVar<IndexT> contact_mask_tabular,
           lcrt::BufferVar<IndexT> subscene_element_ids,
           lcrt::BufferVar<IndexT> subscene_mask_tabular, lcrt::BufferVar<IndexT> v2b,
           lcrt::BufferVar<IndexT> body_self_collision, lcrt::BufferVar<Float> d_hats,
           lcrt::BufferVar<std::int32_t> flags,
           lcrt::Var<std::int32_t> contact_mask_size,
           lcrt::Var<std::int32_t> subscene_mask_size, lcrt::Var<Float> alpha,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto pair   = raw_pairs->read(i);
                const auto V      = Vs->read(pair.data[0]);
                const auto codimV = codimVs->read(pair.data[1]);

                const auto cid_V      = contact_element_ids->read(V);
                const auto cid_codimV = contact_element_ids->read(codimV);
                const auto scid_V     = subscene_element_ids->read(V);
                const auto scid_codimV = subscene_element_ids->read(codimV);

                // discard if the contact is disabled
                auto keep = mask_allows(subscene_mask_tabular, scid_V, scid_codimV,
                                        subscene_mask_size)
                            & mask_allows(contact_mask_tabular, cid_V, cid_codimV,
                                          contact_mask_size);

                // discard duplicate CodimP-CodimP pairs
                keep = keep
                       & !((dimensions->read(V) <= 2) & (V >= codimV));

                const auto body_i = v2b->read(V);
                const auto body_j = v2b->read(codimV);
                // skip self-collision for the same body if self collision off
                keep = keep
                       & !((body_i == body_j)
                           & (body_self_collision->read(body_i) == IndexT{0}));

                const auto P0 = load_vec3(Ps->read(V));
                const auto P1 = load_vec3(Ps->read(codimV));

                const auto thickness =
                    PP_thickness(thicknesses->read(V), thicknesses->read(codimV));
                const auto d_hat = PP_d_hat(d_hats->read(V), d_hats->read(codimV));

                const auto expand = d_hat + thickness;

                if constexpr(Dcd)
                {
                    $if(alpha == 0.0)
                    {
                        // DCD detect pass: the exact distance test (the v1 flavour only)
                        // NOTE: evaluated even when `keep` is already false -- the
                        // distance of an always-non-negative quantity cannot trap.
                        auto D2 = zero_float();
                        ldist::point_point_distance2(P0, P1, D2);
                        keep = keep & (D2 < expand * expand);
                    }
                    $else
                    {
                        const auto dP0 = alpha * load_vec3(dxs->read(V));
                        const auto dP1 = alpha * load_vec3(dxs->read(codimV));
                        keep = keep
                               & ldist::point_point_ccd_broadphase(P0, P1, dP0, dP1, expand);
                    };
                }
                else
                {
                    const auto dP0 = alpha * load_vec3(dxs->read(V));
                    const auto dP1 = alpha * load_vec3(dxs->read(codimV));
                    keep = keep
                           & ldist::point_point_ccd_broadphase(P0, P1, dP0, dP1, expand);
                }

                flags->write(i, lcrt::select(std::int32_t{0}, std::int32_t{1}, keep));
            };
        });
    return k;
}

/// `*_detect_CodimP_AllE_pred` (i = the CodimP query slot, j = the AllE object slot).
template <bool Dcd>
inline auto& detect_CodimP_AllE_pass_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector2i>, lcrt::Buffer<IndexT>, lcrt::Buffer<LVector2i>,
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3>, lcrt::Buffer<Float>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<Float>, lcrt::Buffer<std::int32_t>, std::int32_t, std::int32_t,
        Float, std::uint32_t>(
        [](lcrt::BufferVar<LVector2i> raw_pairs, lcrt::BufferVar<IndexT> codimVs,
           lcrt::BufferVar<LVector2i> Es, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LVector3> dxs, lcrt::BufferVar<Float> thicknesses,
           lcrt::BufferVar<IndexT> contact_element_ids,
           lcrt::BufferVar<IndexT> contact_mask_tabular,
           lcrt::BufferVar<IndexT> subscene_element_ids,
           lcrt::BufferVar<IndexT> subscene_mask_tabular, lcrt::BufferVar<IndexT> v2b,
           lcrt::BufferVar<IndexT> body_self_collision, lcrt::BufferVar<Float> d_hats,
           lcrt::BufferVar<std::int32_t> flags,
           lcrt::Var<std::int32_t> contact_mask_size,
           lcrt::Var<std::int32_t> subscene_mask_size, lcrt::Var<Float> alpha,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto pair   = raw_pairs->read(i);
                const auto codimV = codimVs->read(pair.data[0]);
                const auto E      = load_vec2i(Es->read(pair.data[1]));

                const auto cid_codimV = contact_element_ids->read(codimV);
                const auto cid_E0     = contact_element_ids->read(E(0));
                const auto cid_E1     = contact_element_ids->read(E(1));
                const auto scid_codimV = subscene_element_ids->read(codimV);
                const auto scid_E0     = subscene_element_ids->read(E(0));
                const auto scid_E1     = subscene_element_ids->read(E(1));

                // discard if the contact is disabled (allow_PE_contact: the two terms)
                auto keep = mask_allows(subscene_mask_tabular, scid_codimV, scid_E0,
                                        subscene_mask_size)
                            & mask_allows(subscene_mask_tabular, scid_codimV, scid_E1,
                                          subscene_mask_size)
                            & mask_allows(contact_mask_tabular, cid_codimV, cid_E0,
                                          contact_mask_size)
                            & mask_allows(contact_mask_tabular, cid_codimV, cid_E1,
                                          contact_mask_size);

                // discard if the vertex is on the edge
                keep = keep & !((E(0) == codimV) | (E(1) == codimV));

                const auto body_i = v2b->read(codimV);
                const auto body_j = v2b->read(E(0));
                keep = keep
                       & !((body_i == body_j)
                           & (body_self_collision->read(body_i) == IndexT{0}));

                const auto P  = load_vec3(Ps->read(codimV));
                const auto E0 = load_vec3(Ps->read(E(0)));
                const auto E1 = load_vec3(Ps->read(E(1)));

                const auto thickness = PE_thickness(thicknesses->read(codimV),
                                                    thicknesses->read(E(0)),
                                                    thicknesses->read(E(1)));
                const auto d_hat = PE_d_hat(d_hats->read(codimV),
                                            d_hats->read(E(0)),
                                            d_hats->read(E(1)));

                const auto expand = d_hat + thickness;

                if constexpr(Dcd)
                {
                    $if(alpha == 0.0)
                    {
                        auto           D2   = zero_float();
                        const auto     flag = ldist::point_edge_distance_flag(P, E0, E1);
                        ldist::point_edge_distance2(flag, P, E0, E1, D2);
                        keep = keep & (D2 < expand * expand);
                    }
                    $else
                    {
                        const auto dP  = alpha * load_vec3(dxs->read(codimV));
                        const auto dE0 = alpha * load_vec3(dxs->read(E(0)));
                        const auto dE1 = alpha * load_vec3(dxs->read(E(1)));
                        keep = keep
                               & ldist::point_edge_ccd_broadphase(P, E0, E1, dP, dE0, dE1,
                                                                  expand);
                    };
                }
                else
                {
                    const auto dP  = alpha * load_vec3(dxs->read(codimV));
                    const auto dE0 = alpha * load_vec3(dxs->read(E(0)));
                    const auto dE1 = alpha * load_vec3(dxs->read(E(1)));
                    keep = keep
                           & ldist::point_edge_ccd_broadphase(P, E0, E1, dP, dE0, dE1,
                                                              expand);
                }

                flags->write(i, lcrt::select(std::int32_t{0}, std::int32_t{1}, keep));
            };
        });
    return k;
}

/// `*_detect_AllE_AllE_pred` (i, j are edge slots, reported ordered `i < j`).
template <bool Dcd>
inline auto& detect_AllE_AllE_pass_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector2i>, lcrt::Buffer<LVector2i>, lcrt::Buffer<LVector3>,
        lcrt::Buffer<LVector3>, lcrt::Buffer<Float>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<Float>,
        lcrt::Buffer<std::int32_t>, std::int32_t, std::int32_t, Float, std::uint32_t>(
        [](lcrt::BufferVar<LVector2i> raw_pairs, lcrt::BufferVar<LVector2i> Es,
           lcrt::BufferVar<LVector3> Ps, lcrt::BufferVar<LVector3> dxs,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<IndexT> contact_element_ids,
           lcrt::BufferVar<IndexT> contact_mask_tabular,
           lcrt::BufferVar<IndexT> subscene_element_ids,
           lcrt::BufferVar<IndexT> subscene_mask_tabular, lcrt::BufferVar<IndexT> v2b,
           lcrt::BufferVar<IndexT> body_self_collision, lcrt::BufferVar<Float> d_hats,
           lcrt::BufferVar<std::int32_t> flags,
           lcrt::Var<std::int32_t> contact_mask_size,
           lcrt::Var<std::int32_t> subscene_mask_size, lcrt::Var<Float> alpha,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto pair = raw_pairs->read(i);
                const auto E0   = load_vec2i(Es->read(pair.data[0]));
                const auto E1   = load_vec2i(Es->read(pair.data[1]));

                const auto cid_E0_0 = contact_element_ids->read(E0(0));
                const auto cid_E0_1 = contact_element_ids->read(E0(1));
                const auto cid_E1_0 = contact_element_ids->read(E1(0));
                const auto cid_E1_1 = contact_element_ids->read(E1(1));
                const auto scid_E0_0 = subscene_element_ids->read(E0(0));
                const auto scid_E0_1 = subscene_element_ids->read(E0(1));
                const auto scid_E1_0 = subscene_element_ids->read(E1(0));
                const auto scid_E1_1 = subscene_element_ids->read(E1(1));

                // discard if the contact is disabled (allow_EE_contact: four terms)
                auto keep = mask_allows(subscene_mask_tabular, scid_E0_0, scid_E1_0,
                                        subscene_mask_size)
                            & mask_allows(subscene_mask_tabular, scid_E0_0, scid_E1_1,
                                          subscene_mask_size)
                            & mask_allows(subscene_mask_tabular, scid_E0_1, scid_E1_0,
                                          subscene_mask_size)
                            & mask_allows(subscene_mask_tabular, scid_E0_1, scid_E1_1,
                                          subscene_mask_size)
                            & mask_allows(contact_mask_tabular, cid_E0_0, cid_E1_0,
                                          contact_mask_size)
                            & mask_allows(contact_mask_tabular, cid_E0_0, cid_E1_1,
                                          contact_mask_size)
                            & mask_allows(contact_mask_tabular, cid_E0_1, cid_E1_0,
                                          contact_mask_size)
                            & mask_allows(contact_mask_tabular, cid_E0_1, cid_E1_1,
                                          contact_mask_size);

                // discard if the edges share a vertex
                keep = keep
                       & !((E0(0) == E1(0)) | (E0(0) == E1(1)) | (E0(1) == E1(0))
                           | (E0(1) == E1(1)));

                const auto body_i = v2b->read(E0(0));
                const auto body_j = v2b->read(E1(0));
                keep = keep
                       & !((body_i == body_j)
                           & (body_self_collision->read(body_i) == IndexT{0}));

                const auto E0_0 = load_vec3(Ps->read(E0(0)));
                const auto E0_1 = load_vec3(Ps->read(E0(1)));
                const auto E1_0 = load_vec3(Ps->read(E1(0)));
                const auto E1_1 = load_vec3(Ps->read(E1(1)));

                const auto thickness = EE_thickness(thicknesses->read(E0(0)),
                                                    thicknesses->read(E0(1)),
                                                    thicknesses->read(E1(0)),
                                                    thicknesses->read(E1(1)));
                const auto d_hat = EE_d_hat(d_hats->read(E0(0)),
                                            d_hats->read(E0(1)),
                                            d_hats->read(E1(0)),
                                            d_hats->read(E1(1)));

                const auto expand = d_hat + thickness;

                if constexpr(Dcd)
                {
                    $if(alpha == 0.0)
                    {
                        auto       D2   = zero_float();
                        const auto flag = ldist::edge_edge_distance_flag(E0_0, E0_1, E1_0, E1_1);
                        ldist::edge_edge_distance2(flag, E0_0, E0_1, E1_0, E1_1, D2);
                        keep = keep & (D2 < expand * expand);
                    }
                    $else
                    {
                        const auto dE0_0 = alpha * load_vec3(dxs->read(E0(0)));
                        const auto dE0_1 = alpha * load_vec3(dxs->read(E0(1)));
                        const auto dE1_0 = alpha * load_vec3(dxs->read(E1(0)));
                        const auto dE1_1 = alpha * load_vec3(dxs->read(E1(1)));
                        keep = keep
                               & ldist::edge_edge_ccd_broadphase(E0_0, E0_1, E1_0, E1_1,
                                                                 dE0_0, dE0_1, dE1_0,
                                                                 dE1_1, expand);
                    };
                }
                else
                {
                    const auto dE0_0 = alpha * load_vec3(dxs->read(E0(0)));
                    const auto dE0_1 = alpha * load_vec3(dxs->read(E0(1)));
                    const auto dE1_0 = alpha * load_vec3(dxs->read(E1(0)));
                    const auto dE1_1 = alpha * load_vec3(dxs->read(E1(1)));
                    keep = keep
                           & ldist::edge_edge_ccd_broadphase(E0_0, E0_1, E1_0, E1_1,
                                                             dE0_0, dE0_1, dE1_0, dE1_1,
                                                             expand);
                }

                flags->write(i, lcrt::select(std::int32_t{0}, std::int32_t{1}, keep));
            };
        });
    return k;
}

/// `*_detect_AllP_AllT_pred` (i = the AllP query slot, j = the AllT object slot).
template <bool Dcd>
inline auto& detect_AllP_AllT_pass_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector2i>, lcrt::Buffer<IndexT>, lcrt::Buffer<LVector3i>,
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3>, lcrt::Buffer<Float>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<Float>, lcrt::Buffer<std::int32_t>, std::int32_t, std::int32_t,
        Float, std::uint32_t>(
        [](lcrt::BufferVar<LVector2i> raw_pairs, lcrt::BufferVar<IndexT> Vs,
           lcrt::BufferVar<LVector3i> Fs, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LVector3> dxs, lcrt::BufferVar<Float> thicknesses,
           lcrt::BufferVar<IndexT> contact_element_ids,
           lcrt::BufferVar<IndexT> contact_mask_tabular,
           lcrt::BufferVar<IndexT> subscene_element_ids,
           lcrt::BufferVar<IndexT> subscene_mask_tabular, lcrt::BufferVar<IndexT> v2b,
           lcrt::BufferVar<IndexT> body_self_collision, lcrt::BufferVar<Float> d_hats,
           lcrt::BufferVar<std::int32_t> flags,
           lcrt::Var<std::int32_t> contact_mask_size,
           lcrt::Var<std::int32_t> subscene_mask_size, lcrt::Var<Float> alpha,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto pair = raw_pairs->read(i);
                const auto V    = Vs->read(pair.data[0]);
                const auto F    = load_vec3i(Fs->read(pair.data[1]));

                const auto cid_V  = contact_element_ids->read(V);
                const auto cid_F0 = contact_element_ids->read(F(0));
                const auto cid_F1 = contact_element_ids->read(F(1));
                const auto cid_F2 = contact_element_ids->read(F(2));
                const auto scid_V  = subscene_element_ids->read(V);
                const auto scid_F0 = subscene_element_ids->read(F(0));
                const auto scid_F1 = subscene_element_ids->read(F(1));
                const auto scid_F2 = subscene_element_ids->read(F(2));

                // discard if the contact is disabled (allow_PT_contact: three terms)
                auto keep = mask_allows(subscene_mask_tabular, scid_V, scid_F0,
                                        subscene_mask_size)
                            & mask_allows(subscene_mask_tabular, scid_V, scid_F1,
                                          subscene_mask_size)
                            & mask_allows(subscene_mask_tabular, scid_V, scid_F2,
                                          subscene_mask_size)
                            & mask_allows(contact_mask_tabular, cid_V, cid_F0,
                                          contact_mask_size)
                            & mask_allows(contact_mask_tabular, cid_V, cid_F1,
                                          contact_mask_size)
                            & mask_allows(contact_mask_tabular, cid_V, cid_F2,
                                          contact_mask_size);

                // discard if the point is on the triangle
                keep = keep & !((F(0) == V) | (F(1) == V) | (F(2) == V));

                const auto body_i = v2b->read(V);
                const auto body_j = v2b->read(F(0));
                keep = keep
                       & !((body_i == body_j)
                           & (body_self_collision->read(body_i) == IndexT{0}));

                const auto P  = load_vec3(Ps->read(V));
                const auto F0 = load_vec3(Ps->read(F(0)));
                const auto F1 = load_vec3(Ps->read(F(1)));
                const auto F2 = load_vec3(Ps->read(F(2)));

                const auto thickness = PT_thickness(thicknesses->read(V),
                                                    thicknesses->read(F(0)),
                                                    thicknesses->read(F(1)),
                                                    thicknesses->read(F(2)));
                const auto d_hat = PT_d_hat(d_hats->read(V),
                                            d_hats->read(F(0)),
                                            d_hats->read(F(1)),
                                            d_hats->read(F(2)));

                const auto expand = d_hat + thickness;

                if constexpr(Dcd)
                {
                    $if(alpha == 0.0)
                    {
                        auto       D2   = zero_float();
                        const auto flag = ldist::point_triangle_distance_flag(P, F0, F1, F2);
                        ldist::point_triangle_distance2(flag, P, F0, F1, F2, D2);
                        keep = keep & (D2 < expand * expand);
                    }
                    $else
                    {
                        const auto dP  = alpha * load_vec3(dxs->read(V));
                        const auto dF0 = alpha * load_vec3(dxs->read(F(0)));
                        const auto dF1 = alpha * load_vec3(dxs->read(F(1)));
                        const auto dF2 = alpha * load_vec3(dxs->read(F(2)));
                        keep = keep
                               & ldist::point_triangle_ccd_broadphase(P, F0, F1, F2, dP, dF0,
                                                                      dF1, dF2, expand);
                    };
                }
                else
                {
                    const auto dP  = alpha * load_vec3(dxs->read(V));
                    const auto dF0 = alpha * load_vec3(dxs->read(F(0)));
                    const auto dF1 = alpha * load_vec3(dxs->read(F(1)));
                    const auto dF2 = alpha * load_vec3(dxs->read(F(2)));
                    keep = keep
                           & ldist::point_triangle_ccd_broadphase(P, F0, F1, F2, dP, dF0,
                                                                  dF1, dF2, expand);
                }

                flags->write(i, lcrt::select(std::int32_t{0}, std::int32_t{1}, keep));
            };
        });
    return k;
}
// ---------------------------------------------------------------------------
//  Filter Active (the reference's `filter_active_k1..k4`, identical in all four
//  variants) -- each classifies one candidate-pair kind and writes the surviving
//  PP/PE/PT/EE pair into the temp arrays, with the all-(-1) sentinel elsewhere.
// ---------------------------------------------------------------------------
/// `vIs(offsets(k))` -- a *device-indexed* gather of an index vector.
///
/// LC cannot index a host array by a device value (RULE M2), so the port spells the
/// reference's `vIs(offsets(0))` as the same fixed-length select chain the ported
/// `distance::detail::active_offsets` uses to build its offsets.
template <int N>
[[nodiscard]] inline lcrt::Var<IndexT> gather_index(
    const lps::LVec<lcrt::Var<IndexT>, N>& v, const lcrt::Var<IndexT>& k) noexcept
{
    auto r = lcrt::Var<IndexT>{IndexT{-1}};
    for(int s = 0; s < N; ++s)
        r = lcrt::select(r, v(s), k == IndexT{s});
    return r;
}

/// `..._filter_active_k1_kernel`: AllP-CodimP -> (PP).
inline auto& filter_active_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector2i>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<Float>, lcrt::Buffer<LVector2i>,
        lcrt::Buffer<Float>, std::uint32_t>(
        [](lcrt::BufferVar<LVector3> positions, lcrt::BufferVar<LVector2i> PCodimP_pairs,
           lcrt::BufferVar<IndexT> surf_vertices, lcrt::BufferVar<IndexT> codim_vertices,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<LVector2i> temp_PPs,
           lcrt::BufferVar<Float> d_hats, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // default invalid
                temp_PPs->write(i, sentinel_pair2());

                const auto indices = load_vec2i(PCodimP_pairs->read(i));

                const auto P0 = surf_vertices->read(indices(0));
                const auto P1 = codim_vertices->read(indices(1));

                const auto V0 = load_vec3(positions->read(P0));
                const auto V1 = load_vec3(positions->read(P1));

                const auto thickness = PP_thickness(thicknesses->read(P0), thicknesses->read(P1));
                const auto d_hat     = PP_d_hat(d_hats->read(P0), d_hats->read(P1));

                const auto range = D_range(thickness, d_hat);

                auto D = zero_float();
                ldist::point_point_distance2(V0, V1, D);

                // the reference's `UIPC_KERNEL_ASSERT(D > range.x(), "Thickness
                // Violated! D(%f) should be > D_range.x(%f)", D, range.x())` -- LC's
                // device_assert carries a literal message (the two arguments are
                // quoted here instead of formatted into it)
                lcrt::device_assert(D > range(0),
                                    "Thickness Violated! D should be > D_range.x");

                $if(is_active_D(range, D))
                {
                    write_pair2(temp_PPs, i, P0, P1);
                };
            };
        });
    return k;
}

/// `..._filter_active_k2_kernel`: CodimP-AllE -> (PP, PE).
inline auto& filter_active_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector2i>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<LVector2i>, lcrt::Buffer<Float>, lcrt::Buffer<LVector2i>,
        lcrt::Buffer<LVector3i>, lcrt::Buffer<Float>, std::uint32_t>(
        [](lcrt::BufferVar<LVector3> positions,
           lcrt::BufferVar<LVector2i> CodimP_AllE_pairs,
           lcrt::BufferVar<IndexT> codim_veritces, lcrt::BufferVar<LVector2i> surf_edges,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<LVector2i> temp_PPs,
           lcrt::BufferVar<LVector3i> temp_PEs, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                temp_PPs->write(i, sentinel_pair2());
                temp_PEs->write(i, sentinel_pair3());

                const auto indices = load_vec2i(CodimP_AllE_pairs->read(i));
                const auto V       = codim_veritces->read(indices(0));
                const auto E       = load_vec2i(surf_edges->read(indices(1)));

                lps::LVec<lcrt::Var<IndexT>, 3> vIs;
                vIs(0) = V;
                vIs(1) = E(0);
                vIs(2) = E(1);

                const auto P0 = load_vec3(positions->read(vIs(0)));
                const auto P1 = load_vec3(positions->read(vIs(1)));
                const auto P2 = load_vec3(positions->read(vIs(2)));

                const auto thickness = PE_thickness(thicknesses->read(V),
                                                    thicknesses->read(E(0)),
                                                    thicknesses->read(E(1)));

                const auto d_hat = PE_d_hat(d_hats->read(V), d_hats->read(E(0)),
                                            d_hats->read(E(1)));

                const auto flag = ldist::point_edge_distance_flag(P0, P1, P2);

                const auto range = D_range(thickness, d_hat);

                auto D = zero_float();
                ldist::point_edge_distance2(flag, P0, P1, P2, D);

                lcrt::device_assert(D > range(0),
                                    "Thickness Violated! D should be > D_range.x");

                $if(is_active_D(range, D))
                {
                    lps::LVec<lcrt::Var<IndexT>, 3> offsets;
                    const auto dim = ldist::degenerate_point_edge(flag, offsets);

                    $if(dim == 2)
                    {
                        write_pair2(temp_PPs, i, gather_index(vIs, offsets(0)),
                                    gather_index(vIs, offsets(1)));
                    }
                    $elif(dim == 3)
                    {
                        write_pair3(temp_PEs, i, vIs(0), vIs(1), vIs(2));
                    }
                    $else
                    {
                        lcrt::device_assert(false, "unexpected degenerate case dim");
                    };
                };
            };
        });
    return k;
}

/// `..._filter_active_k3_kernel`: AllP-AllT -> (PP, PE, PT).
inline auto& filter_active_k3_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector2i>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<LVector3i>, lcrt::Buffer<Float>, lcrt::Buffer<LVector2i>,
        lcrt::Buffer<LVector3i>, lcrt::Buffer<LVector4i>, lcrt::Buffer<Float>,
        std::uint32_t>(
        [](lcrt::BufferVar<LVector3> positions, lcrt::BufferVar<LVector2i> PT_pairs,
           lcrt::BufferVar<IndexT> surf_vertices, lcrt::BufferVar<LVector3i> surf_triangles,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<LVector2i> temp_PPs,
           lcrt::BufferVar<LVector3i> temp_PEs, lcrt::BufferVar<LVector4i> temp_PTs,
           lcrt::BufferVar<Float> d_hats, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                temp_PPs->write(i, sentinel_pair2());
                temp_PEs->write(i, sentinel_pair3());
                temp_PTs->write(i, sentinel_pair4());

                const auto indices = load_vec2i(PT_pairs->read(i));
                const auto V       = surf_vertices->read(indices(0));
                const auto F       = load_vec3i(surf_triangles->read(indices(1)));

                lps::LVec<lcrt::Var<IndexT>, 4> vIs;
                vIs(0) = V;
                vIs(1) = F(0);
                vIs(2) = F(1);
                vIs(3) = F(2);

                const auto P0 = load_vec3(positions->read(vIs(0)));
                const auto P1 = load_vec3(positions->read(vIs(1)));
                const auto P2 = load_vec3(positions->read(vIs(2)));
                const auto P3 = load_vec3(positions->read(vIs(3)));

                const auto thickness = PT_thickness(thicknesses->read(V),
                                                    thicknesses->read(F(0)),
                                                    thicknesses->read(F(1)),
                                                    thicknesses->read(F(2)));

                const auto d_hat = PT_d_hat(d_hats->read(V), d_hats->read(F(0)),
                                            d_hats->read(F(1)), d_hats->read(F(2)));

                const auto flag = ldist::point_triangle_distance_flag(P0, P1, P2, P3);

                const auto range = D_range(thickness, d_hat);

                auto D = zero_float();
                ldist::point_triangle_distance2(flag, P0, P1, P2, P3, D);

                lcrt::device_assert(D > zero_float(), "D should be > 0");
                lcrt::device_assert(D > range(0),
                                    "Thickness Violated! D should be > D_range.x");

                $if(is_active_D(range, D))
                {
                    lps::LVec<lcrt::Var<IndexT>, 4> offsets;
                    const auto dim = ldist::degenerate_point_triangle(flag, offsets);

                    $if(dim == 2)
                    {
                        write_pair2(temp_PPs, i, gather_index(vIs, offsets(0)),
                                    gather_index(vIs, offsets(1)));
                    }
                    $elif(dim == 3)
                    {
                        write_pair3(temp_PEs, i, gather_index(vIs, offsets(0)),
                                    gather_index(vIs, offsets(1)),
                                    gather_index(vIs, offsets(2)));
                    }
                    $elif(dim == 4)
                    {
                        write_pair4(temp_PTs, i, vIs(0), vIs(1), vIs(2), vIs(3));
                    }
                    $else
                    {
                        lcrt::device_assert(false, "unexpected degenerate case dim");
                    };
                };
            };
        });
    return k;
}

/// `..._filter_active_k4_kernel`: AllE-AllE -> (PP, PE, EE) with the mollifier's
/// near-parallel escape (the reference classifies with `degenerate_edge_edge` only
/// when `need_mollify` is false).
inline auto& filter_active_k4_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3>, lcrt::Buffer<LVector2i>,
        lcrt::Buffer<LVector2i>, lcrt::Buffer<Float>, lcrt::Buffer<LVector2i>,
        lcrt::Buffer<LVector3i>, lcrt::Buffer<LVector4i>, lcrt::Buffer<Float>,
        std::uint32_t>(
        [](lcrt::BufferVar<LVector3> positions, lcrt::BufferVar<LVector3> rest_positions,
           lcrt::BufferVar<LVector2i> EE_pairs, lcrt::BufferVar<LVector2i> surf_edges,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<LVector2i> temp_PPs,
           lcrt::BufferVar<LVector3i> temp_PEs, lcrt::BufferVar<LVector4i> temp_EEs,
           lcrt::BufferVar<Float> d_hats, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                temp_PPs->write(i, sentinel_pair2());
                temp_PEs->write(i, sentinel_pair3());
                temp_EEs->write(i, sentinel_pair4());

                const auto indices = load_vec2i(EE_pairs->read(i));
                const auto E0      = load_vec2i(surf_edges->read(indices(0)));
                const auto E1      = load_vec2i(surf_edges->read(indices(1)));

                lps::LVec<lcrt::Var<IndexT>, 4> vIs;
                vIs(0) = E0(0);
                vIs(1) = E0(1);
                vIs(2) = E1(0);
                vIs(3) = E1(1);

                const auto P0 = load_vec3(positions->read(vIs(0)));
                const auto P1 = load_vec3(positions->read(vIs(1)));
                const auto P2 = load_vec3(positions->read(vIs(2)));
                const auto P3 = load_vec3(positions->read(vIs(3)));

                const auto thickness = EE_thickness(thicknesses->read(E0(0)),
                                                    thicknesses->read(E0(1)),
                                                    thicknesses->read(E1(0)),
                                                    thicknesses->read(E1(1)));

                const auto d_hat = EE_d_hat(d_hats->read(E0(0)), d_hats->read(E0(1)),
                                            d_hats->read(E1(0)), d_hats->read(E1(1)));

                const auto range = D_range(thickness, d_hat);

                const auto flag = ldist::edge_edge_distance_flag(P0, P1, P2, P3);

                auto D = zero_float();
                ldist::edge_edge_distance2(flag, P0, P1, P2, P3, D);

                lcrt::device_assert(D > range(0),
                                    "Thickness Violated! D should be > D_range.x");

                $if(is_active_D(range, D))
                {
                    auto eps_x = zero_float();
                    ldist::edge_edge_mollifier_threshold(
                        load_vec3(rest_positions->read(vIs(0))),
                        load_vec3(rest_positions->read(vIs(1))),
                        load_vec3(rest_positions->read(vIs(2))),
                        load_vec3(rest_positions->read(vIs(3))),
                        static_cast<Float>(1e-3),
                        eps_x);

                    $if(ldist::need_mollify(P0, P1, P2, P3, eps_x))
                    {
                        write_pair4(temp_EEs, i, vIs(0), vIs(1), vIs(2), vIs(3));
                    }
                    $else
                    {
                        // classify to EE/PE/PP
                        lps::LVec<lcrt::Var<IndexT>, 4> offsets;
                        const auto dim = ldist::degenerate_edge_edge(flag, offsets);

                        $if(dim == 2)
                        {
                            write_pair2(temp_PPs, i, gather_index(vIs, offsets(0)),
                                        gather_index(vIs, offsets(1)));
                        }
                        $elif(dim == 3)
                        {
                            write_pair3(temp_PEs, i, gather_index(vIs, offsets(0)),
                                        gather_index(vIs, offsets(1)),
                                        gather_index(vIs, offsets(2)));
                        }
                        $elif(dim == 4)
                        {
                            write_pair4(temp_EEs, i, vIs(0), vIs(1), vIs(2), vIs(3));
                        }
                        $else
                        {
                            lcrt::device_assert(false, "unexpected degenerate case dim");
                        };
                    };
                };
            };
        });
    return k;
}

// ---------------------------------------------------------------------------
//  CCD TOI (the reference's `filter_toi_k1..k4`, identical in all four variants;
//  the stackless copy passes `max_iter`/`large_enough_toi` as extra arguments --
//  the same two constants the shared kernel bakes in)
// ---------------------------------------------------------------------------
/// `..._filter_toi_k1_kernel`: AllP-CodimP point-point TOI.
inline auto& filter_toi_k1_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<Float>, lcrt::Buffer<LVector2i>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<IndexT>, lcrt::Buffer<Float>, lcrt::Buffer<LVector3>,
        lcrt::Buffer<LVector3>, lcrt::Buffer<Float>, Float, Float, std::uint32_t>(
        [](lcrt::BufferVar<Float> PP_tois, lcrt::BufferVar<LVector2i> PCodimP_pairs,
           lcrt::BufferVar<IndexT> codim_vertices, lcrt::BufferVar<IndexT> surf_vertices,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<LVector3> positions,
           lcrt::BufferVar<LVector3> dxs, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> eta, lcrt::Var<Float> alpha,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto indices = load_vec2i(PCodimP_pairs->read(i));
                const auto V0      = surf_vertices->read(indices(0));
                const auto V1      = codim_vertices->read(indices(1));

                const auto thickness = PP_thickness(thicknesses->read(V0), thicknesses->read(V1));
                const auto d_hat     = PP_d_hat(d_hats->read(V0), d_hats->read(V1));

                const auto VP0  = load_vec3(positions->read(V0));
                const auto VP1  = load_vec3(positions->read(V1));
                const auto dVP0 = alpha * load_vec3(dxs->read(V0));
                const auto dVP1 = alpha * load_vec3(dxs->read(V1));

                auto toi = lcrt::Var<Float>{kLargeEnoughToi};

                const auto faraway = !ldist::point_point_ccd_broadphase(VP0, VP1, dVP0, dVP1,
                                                                       d_hat + thickness);

                $if(!faraway)
                {
                    const auto hit = ldist::point_point_ccd(VP0, VP1, dVP0, dVP1, eta,
                                                            thickness, kToiMaxIter, toi);
                    $if(!hit)
                    {
                        toi = kLargeEnoughToi;
                    };
                };

                PP_tois->write(i, toi);
            };
        });
    return k;
}

/// `..._filter_toi_k2_kernel`: CodimP-AllE point-edge TOI.
inline auto& filter_toi_k2_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<Float>, lcrt::Buffer<LVector2i>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<Float>, lcrt::Buffer<LVector2i>, lcrt::Buffer<LVector3>,
        lcrt::Buffer<LVector3>, lcrt::Buffer<Float>, Float, Float, std::uint32_t>(
        [](lcrt::BufferVar<Float> PE_tois, lcrt::BufferVar<LVector2i> CodimP_AllE_pairs,
           lcrt::BufferVar<IndexT> codim_vertices, lcrt::BufferVar<Float> thicknesses,
           lcrt::BufferVar<LVector2i> surf_edges, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LVector3> dxs, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> eta, lcrt::Var<Float> alpha,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto indices = load_vec2i(CodimP_AllE_pairs->read(i));
                const auto V       = codim_vertices->read(indices(0));
                const auto E       = load_vec2i(surf_edges->read(indices(1)));

                const auto thickness = PE_thickness(thicknesses->read(V), thicknesses->read(E(0)),
                                                    thicknesses->read(E(1)));
                const auto d_hat = PE_d_hat(d_hats->read(V), d_hats->read(E(0)),
                                            d_hats->read(E(1)));

                const auto VP  = load_vec3(Ps->read(V));
                const auto dVP = alpha * load_vec3(dxs->read(V));

                const auto EP0  = load_vec3(Ps->read(E(0)));
                const auto EP1  = load_vec3(Ps->read(E(1)));
                const auto dEP0 = alpha * load_vec3(dxs->read(E(0)));
                const auto dEP1 = alpha * load_vec3(dxs->read(E(1)));

                auto toi = lcrt::Var<Float>{kLargeEnoughToi};

                const auto faraway = !ldist::point_edge_ccd_broadphase(
                    VP, EP0, EP1, dVP, dEP0, dEP1, d_hat + thickness);

                $if(!faraway)
                {
                    const auto hit = ldist::point_edge_ccd(VP, EP0, EP1, dVP, dEP0, dEP1,
                                                           eta, thickness, kToiMaxIter, toi);
                    $if(!hit)
                    {
                        toi = kLargeEnoughToi;
                    };
                };

                PE_tois->write(i, toi);
            };
        });
    return k;
}

/// `..._filter_toi_k3_kernel`: AllP-AllT point-triangle TOI.
inline auto& filter_toi_k3_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<Float>, lcrt::Buffer<LVector2i>, lcrt::Buffer<IndexT>,
        lcrt::Buffer<LVector3i>, lcrt::Buffer<Float>, lcrt::Buffer<LVector3>,
        lcrt::Buffer<LVector3>, lcrt::Buffer<Float>, Float, Float, std::uint32_t>(
        [](lcrt::BufferVar<Float> PT_tois, lcrt::BufferVar<LVector2i> PT_pairs,
           lcrt::BufferVar<IndexT> surf_vertices, lcrt::BufferVar<LVector3i> surf_triangles,
           lcrt::BufferVar<Float> thicknesses, lcrt::BufferVar<LVector3> Ps,
           lcrt::BufferVar<LVector3> dxs, lcrt::BufferVar<Float> d_hats,
           lcrt::Var<Float> eta, lcrt::Var<Float> alpha,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto indices = load_vec2i(PT_pairs->read(i));
                const auto V       = surf_vertices->read(indices(0));
                const auto F       = load_vec3i(surf_triangles->read(indices(1)));

                const auto thickness = PT_thickness(thicknesses->read(V),
                                                    thicknesses->read(F(0)),
                                                    thicknesses->read(F(1)),
                                                    thicknesses->read(F(2)));
                const auto d_hat = PT_d_hat(d_hats->read(V), d_hats->read(F(0)),
                                            d_hats->read(F(1)), d_hats->read(F(2)));

                const auto VP  = load_vec3(Ps->read(V));
                const auto dVP = alpha * load_vec3(dxs->read(V));

                const auto FP0 = load_vec3(Ps->read(F(0)));
                const auto FP1 = load_vec3(Ps->read(F(1)));
                const auto FP2 = load_vec3(Ps->read(F(2)));

                const auto dFP0 = alpha * load_vec3(dxs->read(F(0)));
                const auto dFP1 = alpha * load_vec3(dxs->read(F(1)));
                const auto dFP2 = alpha * load_vec3(dxs->read(F(2)));

                auto toi = lcrt::Var<Float>{kLargeEnoughToi};

                const auto faraway = !ldist::point_triangle_ccd_broadphase(
                    VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, d_hat + thickness);

                $if(!faraway)
                {
                    const auto hit = ldist::point_triangle_ccd(VP, FP0, FP1, FP2, dVP, dFP0,
                                                               dFP1, dFP2, eta, thickness,
                                                               kToiMaxIter, toi);
                    $if(!hit)
                    {
                        toi = kLargeEnoughToi;
                    };
                };

                PT_tois->write(i, toi);
            };
        });
    return k;
}

/// `..._filter_toi_k4_kernel`: AllE-AllE edge-edge TOI.
inline auto& filter_toi_k4_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<Float>, lcrt::Buffer<LVector2i>, lcrt::Buffer<LVector2i>,
        lcrt::Buffer<Float>, lcrt::Buffer<LVector3>, lcrt::Buffer<LVector3>,
        lcrt::Buffer<Float>, Float, Float, std::uint32_t>(
        [](lcrt::BufferVar<Float> EE_tois, lcrt::BufferVar<LVector2i> EE_pairs,
           lcrt::BufferVar<LVector2i> surf_edges, lcrt::BufferVar<Float> thicknesses,
           lcrt::BufferVar<LVector3> Ps, lcrt::BufferVar<LVector3> dxs,
           lcrt::BufferVar<Float> d_hats, lcrt::Var<Float> eta, lcrt::Var<Float> alpha,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(256u);
            const auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                const auto indices = load_vec2i(EE_pairs->read(i));
                const auto E0      = load_vec2i(surf_edges->read(indices(0)));
                const auto E1      = load_vec2i(surf_edges->read(indices(1)));

                const auto thickness = EE_thickness(thicknesses->read(E0(0)),
                                                    thicknesses->read(E0(1)),
                                                    thicknesses->read(E1(0)),
                                                    thicknesses->read(E1(1)));
                const auto d_hat = EE_d_hat(d_hats->read(E0(0)), d_hats->read(E0(1)),
                                            d_hats->read(E1(0)), d_hats->read(E1(1)));

                const auto EP0  = load_vec3(Ps->read(E0(0)));
                const auto EP1  = load_vec3(Ps->read(E0(1)));
                const auto dEP0 = alpha * load_vec3(dxs->read(E0(0)));
                const auto dEP1 = alpha * load_vec3(dxs->read(E0(1)));

                const auto EP2  = load_vec3(Ps->read(E1(0)));
                const auto EP3  = load_vec3(Ps->read(E1(1)));
                const auto dEP2 = alpha * load_vec3(dxs->read(E1(0)));
                const auto dEP3 = alpha * load_vec3(dxs->read(E1(1)));

                auto toi = lcrt::Var<Float>{kLargeEnoughToi};

                const auto faraway = !ldist::edge_edge_ccd_broadphase(
                    EP0, EP1, EP2, EP3, dEP0, dEP1, dEP2, dEP3, d_hat + thickness);

                $if(!faraway)
                {
                    const auto hit = ldist::edge_edge_ccd(EP0, EP1, EP2, EP3, dEP0, dEP1,
                                                          dEP2, dEP3, eta, thickness,
                                                          kToiMaxIter, toi);
                    $if(!hit)
                    {
                        toi = kLargeEnoughToi;
                    };
                };

                EE_tois->write(i, toi);
            };
        });
    return k;
}
// ---------------------------------------------------------------------------
//  The port's *reported-pair* holder (see the file note: the reference's
//  `ThisBVH::QueryBuffer` holds the predicate-selected pairs, the port's BVH holds the
//  raw AABB-overlap pairs and the selection pass compacts the survivors in here)
// ---------------------------------------------------------------------------
/// Same surface as `AtomicCountingLBVH::QueryBuffer` (`view`/`reserve`/`size`/`viewer`),
/// so the reference's filter bodies read `candidate_XX_pairs.size()`/`.view()` unchanged.
class SelectedPairs
{
  public:
    [[nodiscard]] auto view() const noexcept { return m_pairs.cview(0u, m_size); }
    /// The reference's `resize_discard`: the pass writes at most the raw count, so the
    /// buffer is sized to the raw count before the pass and to the count after.
    void reserve(size_t size)
    {
        m_pairs.resize(size);
        m_size = size;
    }
    [[nodiscard]] SizeT size() const noexcept { return m_size; }
    [[nodiscard]] auto  viewer() const noexcept { return view().viewer(); }

    /// The pass's output buffer (its capacity must cover the raw pair count).
    [[nodiscard]] lud::LDeviceBuffer<LVector2i>& buffer() noexcept { return m_pairs; }
    /// Publish the count the selection pass reported.
    void publish(SizeT count) noexcept { m_size = count; }

  private:
    lud::LDeviceBuffer<LVector2i> m_pairs;
    SizeT                         m_size = 0;
};

// ---------------------------------------------------------------------------
/// The info variants' keep-all node/leaf predicates (the port's stand-ins for the
/// reference's device functors; see the file note).  Namespace-scope types with a
/// *template* call operator (a local class cannot have template members, MSVC C2892,
/// `§7.32 rule 16`), and generic over their argument so that one type serves both info
/// structures (`InfoStacklessBVH::NodePredInfoT` and `InfoStacklessBVHV0`'s).
struct KeepAllNodes
{
    template <typename Info>
    auto operator()(const Info&) const noexcept
    {
        return true;
    }
};

// ---------------------------------------------------------------------------

struct KeepAllLeaves
{
    template <typename Info>
    auto operator()(const Info&) const noexcept
    {
        return true;
    }
    /// The v0 flavour's leaf predicate is called with the raw `(i, j)`
    /// (`§7.32 rule 16`: the port folds the reference's adapters into the kernel, so
    /// the predicate sees the ordered pair).
    template <typename A, typename B>
    auto operator()(const A&, const B&) const noexcept
    {
        return true;
    }
};

// ---------------------------------------------------------------------------
//  The host-side selection bookkeeping shared by the four variants
// ---------------------------------------------------------------------------
/// The per-kind scratch of one detect stage: the pass's flags array (grown to the raw
/// pair count) and the layer's stable-compaction scratch.
class SelectionWorkspace
{
  public:
    /// Record the flags pass and compact the survivors into `out`.
    ///
    /// @param record_flags a callable that records the pass kernel, writing `flags`
    template <typename RecordFlags>
    void record(lud::LStream& stream, SizeT raw_count, RecordFlags&& record_flags,
                const lud::LCBufferView<LVector2i>& raw,
                SelectedPairs& out, const lud::LBufferView<std::uint32_t>& out_count)
    {
        if(raw_count == 0u)
        {
            out.reserve(0u);
            out_count.fill(std::uint32_t{0});
            return;
        }
        if(m_flags.size() < raw_count)
            m_flags.resize(raw_count);
        auto flags = m_flags.view(0u, raw_count);

        record_flags(flags);

        // the pass can never report more pairs than the traversal did (it is a
        // selection), so the output buffer's capacity is exactly the raw count and the
        // reference's growth/retry cannot trigger here
        out.buffer().resize(raw_count);
        lla::compact_by_flags<LVector2i>(stream,
                                         raw.subview(0u, raw_count),
                                         flags,
                                         out.buffer().view(0u, raw_count),
                                         out_count,
                                         raw_count,
                                         &m_scratch);
    }

  private:
    lud::LDeviceBuffer<std::int32_t> m_flags;
    lla::StructSelectScratch         m_scratch;
};

/// The four reported candidate counts `[PP, PE, PT, EE]`, read back in one batch.
///
/// The reference reads its BVH's own `cpNum` (its predicate pruned inside the
/// traversal, so the BVH's count *is* the reported count); the port's counts come from
/// the selection passes, so they are batched here -- the mechanism the reference's
/// info_stackless_bvh variant uses for its four counts (`collect_query_counts_kernel`
/// + a single `query_counts.copy_to`).
[[nodiscard]] inline std::array<std::uint32_t, 4> read_candidate_counts(
    lud::LDeviceBuffer<std::uint32_t>& counts)
{
    frame_stream().flush();
    std::array<std::uint32_t, 4> h{};
    counts.cview().copy_to(span<std::uint32_t>{h});
    lud::main_stream().synchronize();
    return h;
}
}  // namespace simplex_trajectory_filter_detail

/// The four filters' short spelling of the shared namespace above.
namespace sftd = simplex_trajectory_filter_detail;
}  // namespace uipc::backend::luisa

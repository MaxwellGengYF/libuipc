#include <collision_detection/stackless_bvh.h>

#include <collision_detection/details/aabb_reduce.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>

#include <uipc/common/log.h>

#include <algorithm>
#include <cstdint>
#include <limits>

/*************************************************************************************************
 * T7.1a — StacklessBVH (port of
 * src/backends/cuda/collision_detection/{stackless_bvh.h,details/stackless_bvh.inl}).
 *
 * KERNEL MAP (reference __global__ -> here)
 *   StacklessBVH_initializeBuildState_kernel  -> initialize_build_state_kernel
 *   StacklessBVH_initializeQueryState_kernel  -> initialize_query_state_kernel
 *   StacklessBVH_calcMaxBVFromBox_kernel      -> detail::reduce_aabb_union
 *                                                (details/aabb_reduce.h: the struct
 *                                                element has no component atomics)
 *   StacklessBVH_calcMCsFromBox_kernel        -> calc_mcs_from_box_kernel
 *   StacklessBVH_calcInverseMapping_kernel    -> calc_inverse_mapping_kernel
 *   StacklessBVH_buildPrimitivesFromBox_kernel-> build_primitives_from_box_kernel
 *   StacklessBVH_calcExtNodeSplitMetrics_kernel -> calc_ext_node_split_metrics_kernel
 *   StacklessBVH_buildIntNodes_kernel         -> build_int_nodes_kernel
 *   StacklessBVH_calcIntNodeOrders_kernel     -> calc_int_node_orders_kernel
 *   StacklessBVH_updateBvhExtNodeLinks_kernel -> update_bvh_ext_node_links_kernel
 *   StacklessBVH_reorderNode_kernel           -> reorder_node_kernel
 *   (the two traversal kernels are in details/stackless_bvh.inl: they are templated on
 *    the record-time predicate)
 *
 * PORT RULES THAT THIS FILE INSTANCES
 *   * `__threadfence()` (an explicit one, or the ordering a cross-thread protocol
 *     needs) -> `volatile_read`/`volatile_write`, which LC's CUDA codegen implements as
 *     `__threadfence()` + the access.  The values are unchanged; the ordering is at
 *     least as strong as the reference's.
 *   * a buffer that the reference passes as a *zero-length* view (the internal-node
 *     arrays and `flags` when there is a single primitive) is padded to one element:
 *     LC rejects a zero-length buffer view as a kernel argument (cookbook §7.30
 *     rule 16).  The kernels' index guards use the explicit element counts, and the
 *     reference's `_flag.total_size() == 0` / `flags.size() == 0` early-outs become
 *     scalar tests on `num_internal_nodes` -- with the padding the buffer size is no
 *     longer a proxy for "there are no internal nodes".
 *   * whole-block dispatch + `$if(gid < n)` guards, and the block size is always a
 *     multiple of 32 (`K_THREADS` = 256).
 *************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;
using culbvh::K_THREADS;
// The kernel lambdas below live in this anonymous namespace, so the nested node type
// needs an alias (inside `StacklessBVH::Impl`'s member functions it is found by
// enclosing-class lookup; here it is not).
using Node = StacklessBVH::Node;

constexpr std::uint32_t kBlock = std::uint32_t(K_THREADS);

[[nodiscard]] constexpr std::uint32_t round_up(std::uint64_t n, std::uint32_t block) noexcept
{
    return static_cast<std::uint32_t>((n + block - 1u) / block * block);
}

/// The reference passes zero-length views of the internal-node arrays when there is
/// only one primitive; LC needs a valid (non-empty) argument, so they are padded.
[[nodiscard]] constexpr size_t padded(size_t n) noexcept
{
    return n == 0u ? 1u : n;
}

// ============================================================================
//  StacklessBVH_initializeBuildState_kernel
// ============================================================================
auto& initialize_build_state_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LAABB>, lcrt::Buffer<std::uint32_t>,
                                   lcrt::Buffer<std::uint32_t>, lcrt::Buffer<std::int32_t>,
                                   lcrt::Buffer<std::uint32_t>, lcrt::Buffer<std::uint32_t>,
                                   lcrt::Buffer<std::int32_t>, std::uint32_t>(
            [](lcrt::BufferVar<LAABB> scene_box, lcrt::BufferVar<std::uint32_t> flags,
               lcrt::BufferVar<std::uint32_t> ext_mark, lcrt::BufferVar<std::int32_t> ext_lca,
               lcrt::BufferVar<std::uint32_t> ext_par, lcrt::BufferVar<std::uint32_t> depths,
               lcrt::BufferVar<std::int32_t> unsorted_ids,
               lcrt::Var<std::uint32_t>      num_objs) noexcept
            {
                lcrt::set_block_size(kBlock);
                const auto idx = lcrt::dispatch_x();
                $if(idx == 0u)
                {
                    lcrt::Var<LAABB> box;
                    store_aabb(box, box_empty<lcrt::Var<float>>());
                    scene_box->write(0u, box);
                };
                $if(idx < num_objs - 1u)
                {
                    flags->write(idx, 0u);
                };
                $if(idx < num_objs)
                {
                    ext_mark->write(idx, 7u);
                    ext_lca->write(idx, 0);
                    ext_par->write(idx, 0u);
                    depths->write(idx, 0u);
                    unsorted_ids->write(idx, lcrt::cast<std::int32_t>(idx));
                };
                $if(idx == num_objs)
                {
                    ext_lca->write(idx, -1);
                };
            });
    return k;
}

// ============================================================================
//  StacklessBVH_initializeQueryState_kernel
// ============================================================================
auto& initialize_query_state_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LAABB>,
                                               lcrt::Buffer<std::int32_t>, std::uint32_t>(
        [](lcrt::BufferVar<LAABB> scene_box, lcrt::BufferVar<std::int32_t> unsorted_ids,
           lcrt::Var<std::uint32_t> num_objs) noexcept
        {
            lcrt::set_block_size(kBlock);
            const auto idx = lcrt::dispatch_x();
            $if(idx == 0u)
            {
                lcrt::Var<LAABB> box;
                store_aabb(box, box_empty<lcrt::Var<float>>());
                scene_box->write(0u, box);
            };
            $if(idx < num_objs)
            {
                unsorted_ids->write(idx, lcrt::cast<std::int32_t>(idx));
            };
        });
    return k;
}

// ============================================================================
//  StacklessBVH_calcMCsFromBox_kernel
// ============================================================================
auto& calc_mcs_from_box_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LAABB>, lcrt::Buffer<LAABB>,
                                   lcrt::Buffer<std::uint32_t>, std::uint32_t>(
            [](lcrt::BufferVar<LAABB> box, lcrt::BufferVar<LAABB> scene,
               lcrt::BufferVar<std::uint32_t> codes, lcrt::Var<std::uint32_t> n) noexcept
            {
                lcrt::set_block_size(kBlock);
                const auto idx = lcrt::dispatch_x();
                $if(idx < n)
                {
                    const auto bv     = load_aabb(box->read(idx));
                    const auto center = box_center(bv);
                    const auto scene_box = load_aabb(scene->read(0u));
                    const auto scene_min = scene_box.min;
                    const auto scene_size = box_sizes(scene_box);
                    lps::LVec<lcrt::Var<float>, 3> offset;
                    for(int c = 0; c < 3; ++c)
                        offset(c) = center(c) - scene_min(c);
                    codes->write(idx,
                                 culbvh::morton3D(offset(0) / scene_size(0),
                                                  offset(1) / scene_size(1),
                                                  offset(2) / scene_size(2)));
                };
            });
    return k;
}

// ============================================================================
//  StacklessBVH_calcInverseMapping_kernel
// ============================================================================
auto& calc_inverse_mapping_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,
                                               lcrt::Buffer<std::int32_t>, std::uint32_t>(
        [](lcrt::BufferVar<std::int32_t> map, lcrt::BufferVar<std::int32_t> inv_map,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            const auto idx = lcrt::dispatch_x();
            $if(idx < n)
            {
                // incoherent access, thus poor performance (the reference's comment)
                inv_map->write(map->read(idx), lcrt::cast<std::int32_t>(idx));
            };
        });
    return k;
}

// ============================================================================
//  StacklessBVH_buildPrimitivesFromBox_kernel
// ============================================================================
auto& build_primitives_from_box_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,
                                               lcrt::Buffer<LAABB>,
                                               lcrt::Buffer<std::int32_t>,
                                               lcrt::Buffer<LAABB>, std::uint32_t>(
        [](lcrt::BufferVar<std::int32_t> prim_idx, lcrt::BufferVar<LAABB> prim_box,
           lcrt::BufferVar<std::int32_t> prim_map, lcrt::BufferVar<LAABB> box,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            const auto idx = lcrt::dispatch_x();
            $if(idx < n)
            {
                const auto new_idx = prim_map->read(lcrt::cast<std::int32_t>(idx));
                prim_idx->write(new_idx, lcrt::cast<std::int32_t>(idx));
                prim_box->write(new_idx, box->read(idx));
            };
        });
    return k;
}

// ============================================================================
//  StacklessBVH_calcExtNodeSplitMetrics_kernel
// ============================================================================
auto& calc_ext_node_split_metrics_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::uint32_t>,
                                               lcrt::Buffer<std::int32_t>, std::uint32_t,
                                               std::uint32_t>(
        [](lcrt::BufferVar<std::uint32_t> codes, lcrt::BufferVar<std::int32_t> metrics,
           lcrt::Var<std::uint32_t> extsize, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            const auto idx = lcrt::dispatch_x();
            $if(idx < n)
            {
                lcrt::Var<std::int32_t> metric{33};
                // `idx != extsize - 1 ? 32 - __clz(codes(idx) ^ codes(idx + 1)) : 33`
                $if(idx != extsize - 1u)
                {
                    const auto x = codes->read(idx) ^ codes->read(idx + 1u);
                    metric = 32 - lcrt::cast<std::int32_t>(lcrt::clz(x));
                };
                metrics->write(idx, metric);
            };
        });
    return k;
}

// ============================================================================
//  StacklessBVH_buildIntNodes_kernel
// ============================================================================
// The atomic claim protocol: a leaf thread finds its parent (`cur`) from the
// `metric` comparison, registers itself as the parent's left or right child and then
// claims the parent with `atomicAdd(&flags(cur), 1)`.  The thread that sees `old == 1`
// merges the two children's boxes and walks on to the grandparent (the same shape as
// the linear BVH's internal-AABB pass, but over the culbvh leaf-pair metrics).
auto& build_int_nodes_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        // leaf nodes
        lcrt::Buffer<std::uint32_t>,      // depths (== count)
        lcrt::Buffer<std::int32_t>,       // lvs_lca
        lcrt::Buffer<std::int32_t>,       // lvs_metric
        lcrt::Buffer<std::uint32_t>,      // lvs_par
        lcrt::Buffer<std::uint32_t>,      // lvs_mark
        lcrt::Buffer<LAABB>,              // lvs_box
        // internal nodes
        lcrt::Buffer<std::int32_t>,       // tks_rc
        lcrt::Buffer<std::int32_t>,       // tks_lc
        lcrt::Buffer<std::int32_t>,       // tks_range_y
        lcrt::Buffer<std::int32_t>,       // tks_range_x
        lcrt::Buffer<std::uint32_t>,      // tks_mark
        lcrt::Buffer<LAABB>,              // tks_box
        lcrt::Buffer<std::uint32_t>,      // flag
        lcrt::Buffer<std::int32_t>,       // tks_par
        std::uint32_t>(
        [](lcrt::BufferVar<std::uint32_t> depths, lcrt::BufferVar<std::int32_t> lvs_lca,
           lcrt::BufferVar<std::int32_t> lvs_metric, lcrt::BufferVar<std::uint32_t> lvs_par,
           lcrt::BufferVar<std::uint32_t> lvs_mark, lcrt::BufferVar<LAABB> lvs_box,
           lcrt::BufferVar<std::int32_t> tks_rc, lcrt::BufferVar<std::int32_t> tks_lc,
           lcrt::BufferVar<std::int32_t> tks_range_y,
           lcrt::BufferVar<std::int32_t> tks_range_x, lcrt::BufferVar<std::uint32_t> tks_mark,
           lcrt::BufferVar<LAABB> tks_box, lcrt::BufferVar<std::uint32_t> flag,
           lcrt::BufferVar<std::int32_t> tks_par, lcrt::Var<std::uint32_t> size) noexcept
        {
            lcrt::set_block_size(kBlock);
            const auto gid = lcrt::dispatch_x();
            $if(gid < size)
            {
                const auto idx = gid;
                lvs_lca->write(idx, -1);
                depths->write(idx, 0u);

                // (l, r] -- DSK locals (an `auto` here would be an Expr, which LC's DSL
                // does not allow to be assigned twice)
                lcrt::Var<std::int32_t> l = lcrt::cast<std::int32_t>(idx) - 1;
                lcrt::Var<std::int32_t> r = lcrt::cast<std::int32_t>(idx);

                lcrt::Var<bool> mark{false};
                $if(l >= 0)
                {
                    mark = lvs_metric->read(l) < lvs_metric->read(r);  // determine direction
                };
                lcrt::Var<std::int32_t> cur{0};
                $if(mark)
                {
                    cur = l;
                }
                $else
                {
                    cur = r;
                };
                lvs_par->write(idx, lcrt::cast<std::uint32_t>(cur));

                // when we only have one external node there is no internal node to build
                $if(size > 1u)
                {
                    $if(mark)
                    {
                        tks_rc->volatile_write(cur, lcrt::cast<std::int32_t>(idx));
                        tks_range_y->write(cur, lcrt::cast<std::int32_t>(idx));
                        tks_mark.atomic(cur).fetch_or(0x00000002u);
                        lvs_mark->write(idx, 0x00000007u);
                    }
                    $else
                    {
                        tks_lc->volatile_write(cur, lcrt::cast<std::int32_t>(idx));
                        tks_range_x->write(cur, lcrt::cast<std::int32_t>(idx));
                        tks_mark.atomic(cur).fetch_or(0x00000001u);
                        lvs_mark->write(idx, 0x00000003u);
                    };

                    // `while(atomicAdd(&_flag(cur), 1) == 1) { ... }`
                    $while(true)
                    {
                        const auto old = flag.atomic(cur).fetch_add(1u);
                        $if(old != 1u)
                        {
                            $break;
                        };

                        const auto chl       = tks_lc->volatile_read(cur);
                        const auto chr       = tks_rc->volatile_read(cur);
                        const auto temp_mark = tks_mark->volatile_read(cur);

                        LBox<lcrt::Var<float>> box = box_empty<lcrt::Var<float>>();
                        $if((temp_mark & 1u) != 0u)
                        {
                            box = load_aabb(lvs_box->volatile_read(chl));
                        }
                        $else
                        {
                            box = load_aabb(tks_box->volatile_read(chl));
                        };
                        $if((temp_mark & 2u) != 0u)
                        {
                            box = box_merged(box, load_aabb(lvs_box->volatile_read(chr)));
                        }
                        $else
                        {
                            box = box_merged(box, load_aabb(tks_box->volatile_read(chr)));
                        };
                        lcrt::Var<LAABB> box_var;
                        store_aabb(box_var, box);
                        tks_box->volatile_write(cur, box_var);

                        tks_mark->write(cur, temp_mark & 0x00000007u);

                        l = tks_range_x->read(cur) - 1;
                        r = tks_range_y->read(cur);
                        lvs_lca->write(l + 1, cur);
                        depths->write(l + 1, depths->read(l + 1) + 1u);

                        $if(l >= 0)
                        {
                            mark = lvs_metric->read(l) < lvs_metric->read(r);
                        }
                        $else
                        {
                            mark = false;
                        };

                        // the root: the range covers every leaf
                        $if((l + 1 == 0) & (r == lcrt::cast<std::int32_t>(size) - 1))
                        {
                            tks_par->write(cur, -1);
                            tks_mark->write(cur,
                                            tks_mark->read(cur) & 0xFFFFFFFBu);
                            $break;
                        };

                        lcrt::Var<std::int32_t> par{0};
                        $if(mark)
                        {
                            par = l;
                        }
                        $else
                        {
                            par = r;
                        };
                        tks_par->write(cur, par);

                        $if(mark)
                        {
                            tks_rc->volatile_write(par, cur);
                            tks_range_y->write(par, r);
                            tks_mark.atomic(par).fetch_and(0xFFFFFFFDu);
                            tks_mark->write(cur, tks_mark->read(cur) | 0x00000004u);
                        }
                        $else
                        {
                            tks_lc->volatile_write(par, cur);
                            tks_range_x->write(par, l + 1);
                            tks_mark.atomic(par).fetch_and(0xFFFFFFFEu);
                            tks_mark->write(cur, tks_mark->read(cur) & 0xFFFFFFFBu);
                        };

                        cur = par;
                    };
                };
            };
        });
    return k;
}

// ============================================================================
//  StacklessBVH_calcIntNodeOrders_kernel
// ============================================================================
auto& calc_int_node_orders_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,
                                               lcrt::Buffer<std::int32_t>,
                                               lcrt::Buffer<std::uint32_t>,
                                               lcrt::Buffer<std::uint32_t>,
                                               lcrt::Buffer<std::int32_t>, std::uint32_t>(
        [](lcrt::BufferVar<std::int32_t> tks_lc, lcrt::BufferVar<std::int32_t> lcas,
           lcrt::BufferVar<std::uint32_t> depths, lcrt::BufferVar<std::uint32_t> offsets,
           lcrt::BufferVar<std::int32_t> tk_map, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            const auto gid = lcrt::dispatch_x();
            $if(gid < n)
            {
                const auto idx   = gid;
                const auto node0 = lcas->read(lcrt::cast<std::int32_t>(idx));
                const auto depth = depths->read(idx);
                const auto id0   = offsets->read(idx);

                $if(node0 != -1)
                {
                    // `for(; depth--; node = _tks_lc(node)) _tkMap(node) = id++;`
                    lcrt::Var<std::int32_t> d = lcrt::cast<std::int32_t>(depth);
                    lcrt::Var<std::int32_t> node = node0;
                    lcrt::Var<std::int32_t> id   = lcrt::cast<std::int32_t>(id0);
                    $while(true)
                    {
                        const auto old = d;
                        d              = d - 1;
                        $if(old == 0)
                        {
                            $break;
                        };
                        tk_map->write(node, id);
                        id   = id + 1;
                        node = tks_lc->read(node);
                    };
                };
            };
        });
    return k;
}

// ============================================================================
//  StacklessBVH_updateBvhExtNodeLinks_kernel
// ============================================================================
auto& update_bvh_ext_node_links_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>, lcrt::Buffer<std::int32_t>,
                                   lcrt::Buffer<std::uint32_t>, std::uint32_t>(
            [](lcrt::BufferVar<std::int32_t> map_table, lcrt::BufferVar<std::int32_t> lcas,
               lcrt::BufferVar<std::uint32_t> pars, lcrt::Var<std::uint32_t> n) noexcept
            {
                lcrt::set_block_size(kBlock);
                const auto gid = lcrt::dispatch_x();
                $if(gid < n)
                {
                    const auto idx = gid;
                    pars->write(idx,
                                lcrt::cast<std::uint32_t>(
                                    map_table->read(lcrt::cast<std::int32_t>(pars->read(idx)))));
                    const auto ori = lcas->read(lcrt::cast<std::int32_t>(idx));
                    $if(ori != -1)
                    {
                        lcas->write(idx, map_table->read(ori) << 1);
                    }
                    $else
                    {
                        lcas->write(idx,
                                    (lcrt::cast<std::int32_t>(idx) << 1) | 1);
                    };
                };
            });
    return k;
}

// ============================================================================
//  StacklessBVH_reorderNode_kernel
// ============================================================================
auto& reorder_node_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<std::int32_t>,   // lvs_lca
        lcrt::Buffer<LAABB>,          // lvs_box
        lcrt::Buffer<std::int32_t>,   // tkMap
        lcrt::Buffer<std::int32_t>,   // unorderedTks_lc
        lcrt::Buffer<std::uint32_t>,  // unorderedTks_mark
        lcrt::Buffer<std::int32_t>,   // unorderedTks_rangey
        lcrt::Buffer<LAABB>,          // unorderedTks_box
        lcrt::Buffer<Node>,           // nodes
        std::int32_t, std::uint32_t>(
        [](lcrt::BufferVar<std::int32_t> lvs_lca, lcrt::BufferVar<LAABB> lvs_box,
           lcrt::BufferVar<std::int32_t> tk_map, lcrt::BufferVar<std::int32_t> unordered_lc,
           lcrt::BufferVar<std::uint32_t> unordered_mark,
           lcrt::BufferVar<std::int32_t> unordered_rangey,
           lcrt::BufferVar<LAABB> unordered_box, lcrt::BufferVar<Node> nodes,
           lcrt::Var<std::int32_t> int_size, lcrt::Var<std::uint32_t> n) noexcept
        {
            lcrt::set_block_size(kBlock);
            const auto gid = lcrt::dispatch_x();
            $if(gid < n)
            {
                const auto idx = lcrt::cast<std::int32_t>(gid);

                lcrt::Var<Node> node;
                node.lc     = -1;
                auto escape = lvs_lca->read(idx + 1);

                $if(escape == -1)
                {
                    node.escape = -1;
                }
                $else
                {
                    const auto b_leaf = escape & 1;
                    const auto e      = escape >> 1;
                    node.escape       = e + lcrt::select(std::int32_t(0), int_size, b_leaf != 0);
                };
                store_aabb(node.bound, load_aabb(lvs_box->read(idx)));

                nodes->write(lcrt::cast<std::uint32_t>(idx) + lcrt::cast<std::uint32_t>(int_size),
                             node);

                $if(idx < int_size)
                {
                    lcrt::Var<Node> internal_node;
                    const auto      new_id = tk_map->read(idx);
                    const auto      mark   = unordered_mark->read(idx);

                    $if((mark & 1u) != 0u)
                    {
                        internal_node.lc =
                            unordered_lc->read(idx) + int_size;
                    }
                    $else
                    {
                        internal_node.lc = tk_map->read(unordered_lc->read(idx));
                    };
                    store_aabb(internal_node.bound, load_aabb(unordered_box->read(idx)));

                    const auto internal_escape = lvs_lca->read(unordered_rangey->read(idx) + 1);
                    $if(internal_escape == -1)
                    {
                        internal_node.escape = -1;
                    }
                    $else
                    {
                        const auto b_leaf = internal_escape & 1;
                        const auto e      = internal_escape >> 1;
                        internal_node.escape =
                            e + lcrt::select(std::int32_t(0), int_size, b_leaf != 0);
                    };

                    nodes->write(lcrt::cast<std::uint32_t>(new_id), internal_node);
                };
            };
        });
    return k;
}
}// namespace

// ===========================================================================
//  Impl: the build steps
// ===========================================================================
void StacklessBVH::Impl::calcMaxBVFromBox(lud::LCBufferView<LAABB> aabbs,
                                          lud::LBufferView<LAABB>  scene_box,
                                          lud::LStream&            stream,
                                          lud::LDeviceBuffer<LAABB>& partials)
{
    // The reference's `StacklessBVH_calcMaxBVFromBox_kernel` (a K_THREADS block that
    // reduces its slice with `__shfl_down_sync` trees and lets thread 0 `atomicMinf`/
    // `atomicMaxf` the 6 scalars into the global scene box) is exactly what
    // `detail::reduce_aabb_union` does for the whole buffer, minus the atomics (the
    // union of boxes is order-independent, so the value is identical).
    detail::reduce_aabb_union(stream, aabbs, scene_box, aabbs.size(),
                              partials);
}

void StacklessBVH::Impl::calcMCsFromBox(lud::LCBufferView<LAABB>        aabbs,
                                        lud::LCBufferView<LAABB>        scene_box,
                                        lud::LBufferView<std::uint32_t> codes,
                                        lud::LStream&                   stream)
{
    const auto n = static_cast<std::uint32_t>(aabbs.size());
    if(n == 0u)
        return;
    stream.record(calc_mcs_from_box_kernel()(aabbs, scene_box, codes, n)
                      .dispatch(round_up(n, kBlock)));
}

void StacklessBVH::Impl::calcInverseMapping(lud::LStream& stream)
{
    const auto n = static_cast<std::uint32_t>(sorted_id.size());
    if(n == 0u)
        return;
    stream.record(calc_inverse_mapping_kernel()(sorted_id.cview(), primMap.view(), n)
                      .dispatch(round_up(n, kBlock)));
}

void StacklessBVH::Impl::buildPrimitivesFromBox(lud::LCBufferView<LAABB> aabbs,
                                                lud::LStream&            stream)
{
    const auto n = static_cast<std::uint32_t>(aabbs.size());
    if(n == 0u)
        return;
    stream.record(build_primitives_from_box_kernel()(ext_idx.view(), ext_aabb.view(),
                                                     primMap.cview(), aabbs, n)
                      .dispatch(round_up(n, kBlock)));
}

void StacklessBVH::Impl::calcExtNodeSplitMetrics(lud::LStream& stream)
{
    const auto n = static_cast<std::uint32_t>(sorted_mtcode.size());
    if(n == 0u)
        return;
    stream.record(calc_ext_node_split_metrics_kernel()(sorted_mtcode.cview(),
                                                       metric.view(), n, n)
                      .dispatch(round_up(n, kBlock)));
}

void StacklessBVH::Impl::buildIntNodes(int size, lud::LStream& stream)
{
    const auto num_objs = static_cast<std::uint32_t>(size);
    if(num_objs == 0u)
        return;

    stream.record(build_int_nodes_kernel()(
                      // leaf nodes
                      count.view(), ext_lca.view(), metric.view(), ext_par.view(),
                      ext_mark.view(), ext_aabb.view(),
                      // internal nodes
                      int_rc.view(), int_lc.view(), int_range_y.view(),
                      int_range_x.view(), int_mark.view(), int_aabb.view(), flags.view(),
                      int_par.view(),
                      num_objs)
                      .dispatch(round_up(num_objs, kBlock)));
}

void StacklessBVH::Impl::calcIntNodeOrders(int size, lud::LStream& stream)
{
    const auto n = static_cast<std::uint32_t>(size);
    if(n == 0u)
        return;
    stream.record(calc_int_node_orders_kernel()(int_lc.view(), ext_lca.view(),
                                                count.view(), offsetTable.view(),
                                                tkMap.view(), n)
                      .dispatch(round_up(n, kBlock)));
}

void StacklessBVH::Impl::updateBvhExtNodeLinks(int size, lud::LStream& stream)
{
    // the reference: `if(flags.size() == 0) return;` -- a scalar test here, because the
    // port pads `flags` so that it stays a valid kernel argument (see the file note)
    if(size <= 1)
        return;

    const auto n = static_cast<std::uint32_t>(size);
    stream.record(update_bvh_ext_node_links_kernel()(tkMap.view(), ext_lca.view(),
                                                     ext_par.view(), n)
                      .dispatch(round_up(n, kBlock)));
}

void StacklessBVH::Impl::reorderNode(int intSize, lud::LStream& stream)
{
    const auto n = static_cast<std::uint32_t>(intSize + 1);
    stream.record(reorder_node_kernel()(
                      // leaf nodes
                      ext_lca.view(), ext_aabb.view(),
                      // internal nodes
                      tkMap.view(), int_lc.view(), int_mark.view(), int_range_y.view(),
                      int_aabb.view(),
                      // total nodes
                      nodes.view(), intSize, n)
                      .dispatch(round_up(n, kBlock)));
}

void StacklessBVH::Impl::build(lud::LCBufferView<LAABB> aabbs, lud::LStream& stream)
{
    objs         = aabbs;
    auto numObjs = aabbs.size();

    if(aabbs.size() == 0)
        return;

    const size_t numInternalNodes = numObjs - 1;  // total number of internal nodes
    const size_t numNodes         = numObjs * 2 - 1;  // total number of nodes

    mtcode.resize(numObjs);
    sorted_mtcode.resize(numObjs);
    sorted_id.resize(numObjs);
    primMap.resize(numObjs);
    ext_aabb.resize(numObjs);
    ext_idx.resize(numObjs);
    ext_lca.resize(numObjs + 1);
    ext_par.resize(numObjs);
    ext_mark.resize(numObjs);

    metric.resize(numObjs);
    tkMap.resize(numObjs);
    offsetTable.resize(numObjs);
    count.resize(numObjs);

    // PADDED: LC rejects a zero-length buffer view as a kernel argument (a single
    // primitive has no internal node at all), see the file note.
    flags.resize(padded(numInternalNodes));
    int_lc.resize(padded(numInternalNodes));
    int_rc.resize(padded(numInternalNodes));
    int_par.resize(padded(numInternalNodes));
    int_range_x.resize(padded(numInternalNodes));
    int_range_y.resize(padded(numInternalNodes));
    int_mark.resize(padded(numInternalNodes));
    int_aabb.resize(padded(numInternalNodes));

    nodes.resize(numNodes);

    const auto n = static_cast<std::uint32_t>(numObjs);
    stream.record(initialize_build_state_kernel()(scene_box.view(), flags.view(),
                                                  ext_mark.view(), ext_lca.view(),
                                                  ext_par.view(), count.view(),
                                                  primMap.view(), n)
                      .dispatch(round_up(uint64_t(n) + 1u, kBlock)));

    calcMaxBVFromBox(aabbs, scene_box.view(), stream, reduce_partials);

    calcMCsFromBox(aabbs, scene_box.cview(), mtcode.view(), stream);

    // CUB scratch is cached per stream by the port and grows only on demand.
    lla::sort_pairs_by_key<std::uint32_t, std::int32_t>(stream,
                                                        mtcode.cview(),
                                                        sorted_mtcode.view(),
                                                        primMap.cview(),
                                                        sorted_id.view(),
                                                        n,
                                                        32u);

    calcInverseMapping(stream);

    buildPrimitivesFromBox(aabbs, stream);

    calcExtNodeSplitMetrics(stream);

    buildIntNodes(static_cast<int>(numObjs), stream);

    lla::exclusive_scan<std::uint32_t>(stream, count.cview(), offsetTable.view(), n);

    calcIntNodeOrders(static_cast<int>(numObjs), stream);

    updateBvhExtNodeLinks(static_cast<int>(numObjs), stream);

    reorderNode(static_cast<int>(numInternalNodes), stream);
}

// ===========================================================================
//  The public entry points
// ===========================================================================
void StacklessBVH::build(lud::LCBufferView<LAABB> aabbs, lud::LStream* stream)
{
    auto& s = stream ? *stream : frame_stream();
    m_impl.build(aabbs, s);
}

void StacklessBVH::QueryBuffer::build(lud::LCBufferView<LAABB> aabbs, lud::LStream& stream)
{
    auto size = aabbs.size();
    if(size == 0u)
        return;
    m_queryMtCode.resize(size);
    m_querySortedMtCode.resize(size);
    m_queryId.resize(size);
    m_querySortedId.resize(size);

    const auto n = static_cast<std::uint32_t>(size);
    if(n > 0u)
    {
        stream.record(initialize_query_state_kernel()(m_querySceneBox.view(),
                                                      m_queryId.view(), n)
                          .dispatch(round_up(n, kBlock)));
    }

    Impl::calcMaxBVFromBox(aabbs, m_querySceneBox.view(), stream, m_queryReducePartials);
    Impl::calcMCsFromBox(aabbs, m_querySceneBox.cview(), m_queryMtCode.view(), stream);

    lla::sort_pairs_by_key<std::uint32_t, std::int32_t>(stream,
                                                        m_queryMtCode.cview(),
                                                        m_querySortedMtCode.view(),
                                                        m_queryId.cview(),
                                                        m_querySortedId.view(),
                                                        n,
                                                        32u);
}
}// namespace uipc::backend::luisa

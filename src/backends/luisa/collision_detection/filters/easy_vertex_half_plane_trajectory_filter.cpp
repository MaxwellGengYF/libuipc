#include <collision_detection/filters/easy_vertex_half_plane_trajectory_filter.h>
#include <frame_stream.h>
#include <utils/codim_thickness.h>
#include <pipeline/ipc_pipeline_flag.h>
#include <luisa_device/linear/algorithms.h>
#include <luisa_device/atomic.h>
#include <cstdint>
#include <iostream>
/*************************************************************************************************
 * T7.2a — EasyVertexHalfPlaneTrajectoryFilter (port of
 * src/backends/cuda/collision_detection/filters/easy_vertex_half_plane_trajectory_filter.cu,
 * 266 lines).
 *
 * WHAT IT COMPUTES (transcribed, both kernels kept as kernels -- there is no predicate
 * functor here, the reference passes the arrays as kernel arguments)
 * ----------------------------------------------------------------------------------
 * filter_active: one thread per surface vertex; over every half-plane instance j, the
 * pair survives iff BOTH masks allow it and the vertex's plane gap is inside the active
 * band:
 *
 *     subscene_mask(sL, sR) != 0   &&   contact_mask(L, R) != 0
 *     D = ((pos - plane_pos) . plane_normal)^2
 *     is_active_D(D_range(thickness, d_hat), D)  ==  thickness^2 < D < (thickness+d_hat)^2
 *
 * and the surviving pairs are appended through a single global atomic ticket, exactly
 * like the reference (the `last < max_count` guard drops the surplus, and the host
 * retries with a bigger buffer when `h_num_collisions > PHs.size()`).
 *
 * filter_toi: one thread per surface vertex; for every mask-allowed plane that the
 * vertex moves TOWARDS (`t = -N . (dx*alpha) > 0`) it computes
 * `toi = (-N . (P - x) - thickness) / t * (1 - eta)` with `eta = 0.1`, takes the
 * minimum, and the whole array is min-reduced into the manager's toi slot.
 *
 * HOST/KERNEL MAPPING (cookbook §3/§7.31)
 * --------------------------------------
 *  * the two `CDense2D<IndexT>` masks -> flat views + an explicit extent (`row * N + col`,
 *    the same layout the ported half-plane sanity checker uses).  The two `continue`s of
 *    the reference collapse into the `in_scope` conjunction the ported checker also uses:
 *    both writes in the body are guarded by it, so evaluating the body unconditionally is
 *    equivalent to the reference's early skips.
 *  * `half_plane_positions.total_size()` -> an explicit `n_planes` scalar argument (the
 *    reference's view carried its own size; the port's `LBufferView` is passed whole and
 *    its size must be a kernel scalar).
 *  * `cuda_tool::atomic_add(num.data(), 1)` -> `num.atomic(0u).fetch_add(1)`;
 *    `best_grid_dim/best_block_dim` -> `dispatch(n)`.
 *  * `info.toi().fill(1.1f)` happens *before* the reduce in the reference, so an empty
 *    surface leaves the slot at 1.1 (CUB's zero-length reduce does not launch); the port
 *    keeps that by guarding the ported reduce (which would otherwise write the identity),
 *    the T4.3 dytopo reporter's established shape.
 *  * `tois.resize_discard(n)` -> `tois.resize(n)` (the port's `resize_discard` mapping).
 *
 * There is NO `do_detect` body and NO predicate/callback: the reference's
 * `EasyVertexHalfPlaneTrajectoryFilter` is a plain kernel launcher, so the port's
 * "record-time predicate cannot read device arrays" limitation does not apply here (the
 * arrays arrive as kernel arguments).
 *************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

// TODO: just hard code the slackness for now
constexpr Float eta = 0.1;

/// `EasyVertexHalfPlaneTrajectoryFilter_filter_active_kernel`
auto& easy_vertex_half_plane_trajectory_filter_filter_active_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<std::int32_t>,  // num
        lcrt::Buffer<std::int32_t>,  // surf_vertices
        lcrt::Buffer<LVector3>,      // positions
        lcrt::Buffer<Float>,         // thicknesses
        lcrt::Buffer<std::int32_t>,  // contact_element_ids
        lcrt::Buffer<std::int32_t>,  // subscene_element_ids
        lcrt::Buffer<std::int32_t>,  // contact_mask_tabular (flat)
        lcrt::Buffer<std::int32_t>,  // subscene_mask_tabular (flat)
        lcrt::Buffer<LVector3>,      // half_plane_positions
        lcrt::Buffer<LVector3>,      // half_plane_normals
        lcrt::Buffer<LVector2i>,     // PHs
        lcrt::Buffer<Float>,         // d_hats
        std::int32_t,                // plane_vertex_offset
        std::int32_t,                // contact_mask_size
        std::int32_t,                // subscene_mask_size
        std::uint32_t,               // n_planes
        std::uint32_t,               // max_count
        std::uint32_t                // n
        >(
        [](lcrt::BufferVar<std::int32_t> num,
           lcrt::BufferVar<std::int32_t> surf_vertices,
           lcrt::BufferVar<LVector3> positions,
           lcrt::BufferVar<Float> thicknesses,
           lcrt::BufferVar<std::int32_t> contact_element_ids,
           lcrt::BufferVar<std::int32_t> subscene_element_ids,
           lcrt::BufferVar<std::int32_t> contact_mask_tabular,
           lcrt::BufferVar<std::int32_t> subscene_mask_tabular,
           lcrt::BufferVar<LVector3> half_plane_positions,
           lcrt::BufferVar<LVector3> half_plane_normals,
           lcrt::BufferVar<LVector2i> PHs,
           lcrt::BufferVar<Float> d_hats,
           lcrt::Var<std::int32_t> plane_vertex_offset,
           lcrt::Var<std::int32_t> contact_mask_size,
           lcrt::Var<std::int32_t> subscene_mask_size,
           lcrt::Var<std::uint32_t> n_planes,
           lcrt::Var<std::uint32_t> max_count,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                $for(j, 0u, n_planes)
                {
                    auto vI = surf_vertices->read(i);
                    auto vJ = plane_vertex_offset + lcrt::cast<std::int32_t>(j);

                    auto d_hat = d_hats->read(vI);

                    auto L = contact_element_ids->read(vI);
                    auto R = contact_element_ids->read(vJ);

                    auto sL = subscene_element_ids->read(vI);
                    auto sR = subscene_element_ids->read(vJ);

                    // the reference's two `continue`s (a && b)
                    auto in_scope =
                        (subscene_mask_tabular->read(sL * subscene_mask_size + sR) != 0)
                        & (contact_mask_tabular->read(L * contact_mask_size + R) != 0);

                    $if(in_scope)
                    {
                        auto pos = load_vec3(positions->read(vI));

                        auto plane_pos    = load_vec3(half_plane_positions->read(j));
                        auto plane_normal = load_vec3(half_plane_normals->read(j));

                        auto diff = pos - plane_pos;

                        auto dst = lps::dot(diff, plane_normal);

                        auto thickness = thicknesses->read(vI);

                        auto D = dst * dst;

                        auto range = D_range(thickness, d_hat);

                        $if(is_active_D(range, D))
                        {
                            auto last = num.atomic(0u).fetch_add(1);

                            $if(last < lcrt::cast<std::int32_t>(max_count))
                            {
                                lps::LVec<lcrt::Var<std::int32_t>, 2> pair;
                                pair(0) = vI;
                                pair(1) = lcrt::cast<std::int32_t>(j);
                                lcrt::Var<LVector2i> out;
                                store_vec2i(out, pair);
                                PHs->write(last, out);
                            };
                        };
                    };
                };
            };
        });
    return k;
}

/// `EasyVertexHalfPlaneTrajectoryFilter_filter_toi_kernel`
auto& easy_vertex_half_plane_trajectory_filter_filter_toi_kernel()
{
    static auto& k = lla::detail::cached_kernel<
        lcrt::Buffer<std::int32_t>,  // surf_vertices
        lcrt::Buffer<LVector3>,      // positions
        lcrt::Buffer<Float>,         // thicknesses
        lcrt::Buffer<std::int32_t>,  // contact_element_ids
        lcrt::Buffer<std::int32_t>,  // subscene_element_ids
        lcrt::Buffer<std::int32_t>,  // subscene_mask_tabular (flat)
        lcrt::Buffer<std::int32_t>,  // contact_mask_tabular (flat)
        lcrt::Buffer<LVector3>,      // displacements
        lcrt::Buffer<LVector3>,      // half_plane_positions
        lcrt::Buffer<LVector3>,      // half_plane_normals
        lcrt::Buffer<Float>,         // tois
        std::int32_t,                // plane_vertex_offset
        std::int32_t,                // contact_mask_size
        std::int32_t,                // subscene_mask_size
        std::uint32_t,               // n_planes
        Float,                       // alpha
        std::uint32_t                // n
        >(
        [](lcrt::BufferVar<std::int32_t> surf_vertices,
           lcrt::BufferVar<LVector3> positions,
           lcrt::BufferVar<Float> thicknesses,
           lcrt::BufferVar<std::int32_t> contact_element_ids,
           lcrt::BufferVar<std::int32_t> subscene_element_ids,
           lcrt::BufferVar<std::int32_t> subscene_mask_tabular,
           lcrt::BufferVar<std::int32_t> contact_mask_tabular,
           lcrt::BufferVar<LVector3> displacements,
           lcrt::BufferVar<LVector3> half_plane_positions,
           lcrt::BufferVar<LVector3> half_plane_normals,
           lcrt::BufferVar<Float> tois,
           lcrt::Var<std::int32_t> plane_vertex_offset,
           lcrt::Var<std::int32_t> contact_mask_size,
           lcrt::Var<std::int32_t> subscene_mask_size,
           lcrt::Var<std::uint32_t> n_planes,
           lcrt::Var<Float> alpha,
           lcrt::Var<std::uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // the reference's `Float min_toi = 1.1f;` -- the float literal's exact
                // double value, kept so an all-skipped vertex reports the same bits
                lcrt::Var<Float> min_toi{static_cast<Float>(1.1f)};  // large enough

                $for(j, 0u, n_planes)
                {
                    auto vI = surf_vertices->read(i);
                    auto vJ = plane_vertex_offset + lcrt::cast<std::int32_t>(j);

                    auto L = contact_element_ids->read(vI);
                    auto R = contact_element_ids->read(vJ);

                    auto sL = subscene_element_ids->read(vI);
                    auto sR = subscene_element_ids->read(vJ);

                    // the reference's two `continue`s (a && b)
                    auto in_scope =
                        (subscene_mask_tabular->read(sL * subscene_mask_size + sR) != 0)
                        & (contact_mask_tabular->read(L * contact_mask_size + R) != 0);

                    $if(in_scope)
                    {
                        auto x   = load_vec3(positions->read(vI));
                        auto dx  = load_vec3(displacements->read(vI)) * alpha;
                        auto x_t = x + dx;


                        auto P = load_vec3(half_plane_positions->read(j));
                        auto N = load_vec3(half_plane_normals->read(j));

                        auto thickness = thicknesses->read(vI);

                        auto t = -lps::dot(N, dx);
                        // moving away from the plane, no collision
                        $if(t > 0.0)
                        {
                            // t > 0, moving towards the plane
                            auto diff = P - x;
                            auto t0   = -lps::dot(N, diff) - thickness;

                            auto this_toi = t0 / t * (1.0 - eta);

                            min_toi = lcrt::min(min_toi, this_toi);
                        };
                    };
                };

                tois->write(i, min_toi);
            };
        });
    return k;
}
}// namespace

REGISTER_SIM_SYSTEM(EasyVertexHalfPlaneTrajectoryFilter);

constexpr bool PrintDebugInfo = false;

void EasyVertexHalfPlaneTrajectoryFilter::do_build(BuildInfo& info)
{
    require<IPCPipelineFlag>();
}

lud::LCBufferView<LVector2i> EasyVertexHalfPlaneTrajectoryFilter::candidate_PHs() const noexcept
{
    return m_impl.PHs.cview();
}

lud::LCBufferView<Float> EasyVertexHalfPlaneTrajectoryFilter::toi_PHs() const noexcept
{
    return m_impl.tois.cview();
}

void EasyVertexHalfPlaneTrajectoryFilter::do_detect(DetectInfo& info)
{
    // do nothing
}

void EasyVertexHalfPlaneTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info);
}

void EasyVertexHalfPlaneTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info);
}

void EasyVertexHalfPlaneTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info)
{
    auto& stream = frame_stream();

    // LC rejects a zero-length buffer view as a kernel argument (T7.1a rule 9) and the
    // reference's pair buffer starts empty.  Padding it to one element is exactly the
    // T7.1a `AtomicCountingLBVH::detect` fix: the kernel's own `last < max_count` guard
    // drops every surplus ticket, so nothing observable changes (with max_count == 0 the
    // reference writes nothing and the host retries with the grown buffer anyway).
    if(PHs.size() == 0u)
        PHs.resize(1u);

    auto query = [&]
    {
        num_collisions.copy_from(IndexT{0}, &stream);

        int n = static_cast<int>(info.surf_vertices().size());
        if(n > 0)
        {
            auto n_planes = static_cast<std::uint32_t>(info.plane_positions().size());
            stream.record(
                easy_vertex_half_plane_trajectory_filter_filter_active_kernel()(
                    num_collisions.view(),
                    info.surf_vertices(),
                    info.positions(),
                    info.thicknesses(),
                    info.contact_element_ids(),
                    info.subscene_element_ids(),
                    info.contact_mask_tabular(),
                    info.subscene_mask_tabular(),
                    info.plane_positions(),
                    info.plane_normals(),
                    PHs.view(),
                    info.d_hats(),
                    info.half_plane_vertex_offset(),
                    info.contact_mask_size(),
                    info.subscene_mask_size(),
                    n_planes,
                    static_cast<std::uint32_t>(PHs.size()),
                    static_cast<std::uint32_t>(n))
                    .dispatch(static_cast<std::uint32_t>(n)));
        }
    };

    query();

    h_num_collisions = lud::read_host_sync(stream, num_collisions.cview());

    if(h_num_collisions > static_cast<IndexT>(PHs.size()))
    {
        const auto new_size = static_cast<size_t>(static_cast<Float>(h_num_collisions) * reserve_ratio);
        PHs.reserve(new_size);
        PHs.resize(new_size);
        query();
    }

    info.PHs(PHs.cview(0, static_cast<size_t>(h_num_collisions)));

    if constexpr(PrintDebugInfo)
    {
        std::vector<LVector2i> phs(static_cast<size_t>(h_num_collisions));
        PHs.cview(0, static_cast<size_t>(h_num_collisions)).copy_to(phs);
        for(auto& ph : phs)
        {
            std::cout << "vI: " << ph(0) << ", pI: " << ph(1) << std::endl;
        }
    }
}

void EasyVertexHalfPlaneTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info)
{
    auto& stream = frame_stream();

    info.toi().fill(1.1f, &stream);
    tois.resize(info.surf_vertices().size());

    int n = static_cast<int>(info.surf_vertices().size());
    if(n > 0)
    {
        auto n_planes = static_cast<std::uint32_t>(info.plane_positions().size());
        stream.record(easy_vertex_half_plane_trajectory_filter_filter_toi_kernel()(
                          info.surf_vertices(),
                          info.positions(),
                          info.thicknesses(),
                          info.contact_element_ids(),
                          info.subscene_element_ids(),
                          info.subscene_mask_tabular(),
                          info.contact_mask_tabular(),
                          info.displacements(),
                          info.plane_positions(),
                          info.plane_normals(),
                          tois.view(),
                          info.half_plane_vertex_offset(),
                          info.contact_mask_size(),
                          info.subscene_mask_size(),
                          n_planes,
                          info.alpha(),
                          static_cast<std::uint32_t>(n))
                          .dispatch(static_cast<std::uint32_t>(n)));

        // the reference's CUB reduce does not launch for n == 0, so the slot keeps the
        // 1.1 fill above (the port's device_reduce_min would write the +inf identity)
        lla::device_reduce_min(stream, tois.cview(), info.toi(), tois.size());
    }
}
}// namespace uipc::backend::luisa

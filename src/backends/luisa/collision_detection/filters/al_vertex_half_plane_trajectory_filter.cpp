#include <collision_detection/filters/al_vertex_half_plane_trajectory_filter.h>
#include <frame_stream.h>
#include <pipeline/al_ipc_pipeline_flag.h>
#include <luisa_device/linear/algorithms.h>
#include <cstdint>
#include <iostream>
/*************************************************************************************************
 * T7.2a — ALVertexHalfPlaneTrajectoryFilter (port of
 * src/backends/cuda/collision_detection/filters/al_vertex_half_plane_trajectory_filter.cu,
 * 158 lines).
 *
 * Transcribed verbatim (see the header for the three Easy-vs-AL differences).  Mapping
 * notes are the Easy filter's, plus:
 *  * `PHs(pair_index) = Vector2i{vI, j}` is written BEFORE the two mask `continue`s in the
 *    reference, so the AL candidate grid contains every pair; in the port the mask checks
 *    only gate the *toi* half of the body, and the PAIR write stays unguarded -- the
 *    `continue`s collapse into the `in_scope` conjunction exactly as in the Easy kernel.
 *  * `tois(pair_index) = 1.1` (a double literal) and the caller's `tois.fill(1.1f)` (a
 *    float literal) both survive transcription: the reference mixes them in one buffer,
 *    and the min-reduce picks the smaller (`1.1`), which is why an all-skipped grid
 *    still reduces to 1.1.
 *  * `resize_discard(n)` -> `resize(n)`.
 *************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

// TODO: just hard code the slackness for now
constexpr Float eta = 0.001;

/// `ALVertexHalfPlaneTrajectoryFilter_filter_toi_kernel`
auto& al_vertex_half_plane_trajectory_filter_filter_toi_kernel()
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
        lcrt::Buffer<LVector2i>,     // PHs
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
           lcrt::BufferVar<LVector2i> PHs,
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
                $for(j, 0u, n_planes)
                {
                    auto vI         = surf_vertices->read(i);
                    auto vJ         = plane_vertex_offset + lcrt::cast<std::int32_t>(j);
                    auto pair_index = i * n_planes + j;

                    lps::LVec<lcrt::Var<std::int32_t>, 2> pair;
                    pair(0) = vI;
                    pair(1) = lcrt::cast<std::int32_t>(j);
                    lcrt::Var<LVector2i> out;
                    store_vec2i(out, pair);
                    PHs->write(pair_index, out);

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

                        $if(t <= 0.0)
                        {
                            // moving away from the plane, no collision
                            tois->write(pair_index, static_cast<Float>(1.1));
                        }
                        $else
                        {
                            // t > 0, moving towards the plane
                            auto diff = P - x;
                            auto t0   = -lps::dot(N, diff) - thickness;

                            auto this_toi = t0 / t * (1.0 - eta);

                            tois->write(pair_index, this_toi);
                        };
                    };
                };
            };
        });
    return k;
}
}// namespace

REGISTER_SIM_SYSTEM(ALVertexHalfPlaneTrajectoryFilter);

constexpr bool PrintDebugInfo = false;

void ALVertexHalfPlaneTrajectoryFilter::do_build(BuildInfo& info)
{
    require<ALIPCPipelineFlag>();
}

lud::LCBufferView<LVector2i> ALVertexHalfPlaneTrajectoryFilter::candidate_PHs() const noexcept
{
    return m_impl.PHs.cview();
}

lud::LCBufferView<Float> ALVertexHalfPlaneTrajectoryFilter::toi_PHs() const noexcept
{
    return m_impl.tois.cview();
}

void ALVertexHalfPlaneTrajectoryFilter::do_detect(DetectInfo& info)
{
    // do nothing
}

void ALVertexHalfPlaneTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info);
}

void ALVertexHalfPlaneTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info);
}

void ALVertexHalfPlaneTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info)
{
    // do nothing
}

void ALVertexHalfPlaneTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info)
{
    auto& stream = frame_stream();

    info.toi().fill(1.1f, &stream);
    tois.resize(info.surf_vertices().size() * info.plane_positions().size());
    tois.view().fill(1.1f, &stream);
    PHs.resize(info.surf_vertices().size() * info.plane_positions().size());

    int n = static_cast<int>(info.surf_vertices().size());
    auto n_planes = static_cast<std::uint32_t>(info.plane_positions().size());
    // The reference dispatches whenever `n > 0`; with no half-plane instance the kernel's
    // loop body never runs, so skipping the dispatch is equivalent -- and it avoids
    // handing LC a zero-length pair/toi view (T7.1a rule 9), which the grid would be in
    // that case.  The reduce then keeps the caller's `info.toi()` fill, exactly like the
    // reference's `if(tois.size() > 0)` guard (CUB's empty reduce does not launch).
    if(n > 0 && n_planes > 0)
    {
        stream.record(al_vertex_half_plane_trajectory_filter_filter_toi_kernel()(
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
                          PHs.view(),
                          tois.view(),
                          info.half_plane_vertex_offset(),
                          info.contact_mask_size(),
                          info.subscene_mask_size(),
                          n_planes,
                          info.alpha(),
                          static_cast<std::uint32_t>(n))
                          .dispatch(static_cast<std::uint32_t>(n)));

        if(tois.size() > 0)
            lla::device_reduce_min(stream, tois.cview(), info.toi(), tois.size());
    }
}
}// namespace uipc::backend::luisa

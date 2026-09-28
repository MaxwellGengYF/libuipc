/********************************************************************
 * @file half_plane_vertex_distance_check.cpp
 * @brief T4.3 — half-plane vs. surface-vertex distance sanity check
 *        (port of src/backends/cuda/sanity_check/half_plane_vertex_distance_check.cu).
 *
 * This is the one checker of the reference's three that does *not* need the
 * collision-detection BVH: it uses the scene surface the CPU sanity-check module
 * flattens (ISanityCheckContext::scene_simplicial_surface()), the contact /
 * subscene tabulars and the half-plane instances. The other two
 * (simplicial_surface_distance_check / simplicial_surface_intersection_check) are
 * deferred to T7.1 — see sanity_check/backend_sanity_checker.h.
 *
 * Port notes (cookbook §7.13 / §7.14)
 * ----------------------------------
 *  * `__global__ void k(...)` -> one LC DSL kernel compiled once and cached
 *    through `linalg::detail::cached_kernel` (deliberately leaked, §7.10 trap 1);
 *    the reference's `best_grid_dim/best_block_dim` launch is the plain
 *    `.dispatch(n)` with the same `if (i >= n) return;` early-out as `$if (i < n)`.
 *  * `cuda_tool::CDense2D<IndexT>` (the contact/subscene enable masks) becomes a
 *    flat `LDeviceBuffer<IndexT>` indexed `row * dim + col`, matching the host-side
 *    `h_mask[i * N + j]` fill order the reference uses. LC has no device 2D buffer
 *    element type and a flat buffer has the identical element sequence.
 *  * `cuda_tool::Dense<IndexT> has_violation` (a device scalar) becomes a
 *    one-element `LDeviceVar<IndexT>`; `atomicMax(has_violation.data(), 1)` becomes
 *    `atomic(0).fetch_max(1)`.
 *  * The reference's inner `for(h)` uses `continue` (mask mismatch) and `return`
 *    (violation found). Both are pure control-flow optimisations: the flag write
 *    and the `fetch_max(1)` are idempotent, so evaluating the remaining iterations
 *    is observationally identical. The port keeps the masks as *predicates*
 *    (`in_scope &`) and drops the early return, which avoids a device `break` and
 *    keeps the kernel branch-free. RULE M2: no device value drives host control
 *    flow anywhere in this file; the only host branches are on read-back values.
 *  * A device read-back is a mid-frame synchronisation (cookbook §7.13 rule 3):
 *    `read_back_*` commits the frame stream, waits, copies on the main stream and
 *    re-opens the frame if the caller had one.
 *  * Formatting of the reported message is byte-identical to the reference's
 *    (same format strings, same `close_geo_ids` map semantics — see Vec2iLess).
 *********************************************************************/
#include <type_define.h>
#include <uipc/core/object.h>
#include <uipc/core/contact_tabular.h>
#include <uipc/core/subscene_tabular.h>
#include <uipc/geometry/geometry_slot.h>
#include <sanity_check/backend_sanity_checker.h>
#include <uipc/backend/visitors/sanity_check_message_visitor.h>
#include <uipc/core/contact_model.h>
#include <uipc/common/range.h>
#include <uipc/geometry/simplicial_complex.h>
#include <uipc/io/simplicial_complex_io.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/builtin/geometry_type.h>
#include <uipc/common/map.h>

#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>

#include <string>
#include <vector>

namespace uipc::backend::luisa
{
namespace
{
    namespace lla = luisa_device::linalg;

    /**
     * @brief The reference's `std::less<uipc::Vector2i>` (`lhs[0] < rhs[0] ||
     *        (lhs[0] == rhs[0] && lhs[1] < rhs[1])`), spelled as a named comparator.
     *
     * `uipc::Vector2i` is an Eigen type; specialising `std::less` for it (what the
     * reference does) is not permitted by the standard, and the *order* is what
     * matters here — it fixes the order of the reported geometry pairs. A named
     * comparator gives that same lexicographic order without touching namespace std.
     */
    struct Vec2iLess
    {
        bool operator()(const Vector2i& lhs, const Vector2i& rhs) const noexcept
        {
            return lhs[0] < rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
        }
    };

    /// The checker kernel (reference:
    /// HalfPlaneVertexDistanceCheck_do_check_kernel). One thread per scene-surface
    /// vertex; the inner loop over the half-plane instances is device-bounded
    /// (cookbook §7.10 rule 5).
    auto& half_plane_vertex_distance_check_kernel()
    {
        static auto& k = lla::detail::cached_kernel<
            lcrt::Buffer<LVector3>,       // positions
            lcrt::Buffer<double>,         // thickness
            lcrt::Buffer<std::int32_t>,   // contact_ids
            lcrt::Buffer<std::int32_t>,   // subscene_ids
            lcrt::Buffer<std::int32_t>,   // contact_mask  (flat CN x CN)
            lcrt::Buffer<std::int32_t>,   // subscene_mask (flat SN x SN)
            lcrt::Buffer<LVector3>,       // hp_normals
            lcrt::Buffer<LVector3>,       // hp_origins
            lcrt::Buffer<std::int32_t>,   // hp_contact_ids
            lcrt::Buffer<std::int32_t>,   // hp_subscene_ids
            lcrt::Buffer<std::int32_t>,   // vertex_too_close (out)
            lcrt::Buffer<std::int32_t>,   // has_violation (out, 1 element)
            std::uint32_t,                // contact_mask dim (CN)
            std::uint32_t,                // subscene_mask dim (SN)
            std::uint32_t,                // num_hp
            std::uint32_t                 // n
            >(
            [](lcrt::BufferVar<LVector3> positions,
               lcrt::BufferVar<double>   thickness,
               lcrt::BufferVar<std::int32_t> contact_ids,
               lcrt::BufferVar<std::int32_t> subscene_ids,
               lcrt::BufferVar<std::int32_t> contact_mask,
               lcrt::BufferVar<std::int32_t> subscene_mask,
               lcrt::BufferVar<LVector3> hp_normals,
               lcrt::BufferVar<LVector3> hp_origins,
               lcrt::BufferVar<std::int32_t> hp_contact_ids,
               lcrt::BufferVar<std::int32_t> hp_subscene_ids,
               lcrt::BufferVar<std::int32_t> vertex_too_close,
               lcrt::BufferVar<std::int32_t> has_violation,
               lcrt::Var<std::uint32_t>      contact_mask_dim,
               lcrt::Var<std::uint32_t>      subscene_mask_dim,
               lcrt::Var<std::uint32_t>      num_hp,
               lcrt::Var<std::uint32_t>      n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    auto v_contact_id  = contact_ids->read(i);
                    auto v_subscene_id = subscene_ids->read(i);
                    auto v_thickness   = thickness->read(i);
                    auto v_position    = load_vec3(positions->read(i));

                    // The reference: `for(h) { if(!mask) continue; if(!mask)
                    // continue; d = ...; if(d <= 0) { set; atomicMax; return; } }`.
                    // Evaluating every h is equivalent (both writes are idempotent).
                    $for(h, 0u, num_hp)
                    {
                        auto hp_contact_id  = hp_contact_ids->read(h);
                        auto hp_subscene_id = hp_subscene_ids->read(h);

                        auto in_scope =
                            (subscene_mask->read(hp_subscene_id * subscene_mask_dim + v_subscene_id) != 0)
                            & (contact_mask->read(hp_contact_id * contact_mask_dim + v_contact_id) != 0);

                        auto d = lps::dot(v_position - load_vec3(hp_origins->read(h)),
                                          load_vec3(hp_normals->read(h)))
                                 - v_thickness;

                        $if(in_scope & (d <= 0.0))
                        {
                            vertex_too_close->write(i, std::int32_t{1});
                            has_violation->atomic(0u).fetch_max(std::int32_t{1});
                        };
                    };
                };
            });
        return k;
    }

    /// cuda_tool parity: a device read-back synchronises, and the work still sitting
    /// in the frame command list must be committed and completed first (§7.13 rule 3).
    template <typename T>
    T read_back(lud::LStream& stream, const lud::LBufferView<T>& one)
    {
        const bool was_open = stream.frame_open();
        stream.synchronize();
        T host{};
        stream.stream() << one.native().copy_to(lcrt::span{&host, 1u});
        stream.stream().synchronize();
        if(was_open)
        {
            stream.begin();
        }
        return host;
    }

    template <typename T>
    void read_back_into(lud::LStream& stream,
                        const lud::LBufferView<T>& one,
                        std::span<T>               dst)
    {
        const bool was_open = stream.frame_open();
        stream.synchronize();
        if(!dst.empty())
        {
            stream.stream() << one.native().copy_to(dst);
            stream.stream().synchronize();
        }
        if(was_open)
        {
            stream.begin();
        }
    }
}// namespace

class HalfPlaneVertexDistanceCheck final : public BackendSanityChecker
{
  public:
    constexpr static U64 SanityCheckerUID = 2;
    constexpr static U64 HalfPlaneUID     = 1;  // ImplicitGeometryUID = 1
    using BackendSanityChecker::BackendSanityChecker;

    vector<geometry::Geometry*> halfplanes;

    virtual void build() override
    {
        auto enable_contact = context().config().find<IndexT>("contact/enable");
        if(!enable_contact->view()[0])
        {
            throw BackendSanityCheckerException("Contact is not enabled");
        }
    }

    virtual U64 get_id() const noexcept override { return SanityCheckerUID; }

    void collect_halfplanes()
    {
        halfplanes.clear();
        auto geo_slots = context().geometries();
        for(auto& slot : geo_slots)
        {
            auto& geo = slot->geometry();
            if(geo.type() == builtin::ImplicitGeometry)
            {
                auto uid = geo.meta().find<U64>(builtin::implicit_geometry_uid);
                UIPC_ASSERT(uid, "ImplicitGeometryUID not found, why can it happen?");

                if(uid->view()[0] == HalfPlaneUID)
                {
                    halfplanes.push_back(&geo);
                }
            }
        }
    }

    static geometry::SimplicialComplex create_close_mesh(const geometry::SimplicialComplex& scene_surface,
                                                        const vector<IndexT>& vertex_is_too_close)
    {
        geometry::SimplicialComplex mesh;

        vector<SizeT> close_vertices;
        {
            close_vertices.reserve(vertex_is_too_close.size());
            for(auto&& [I, is_close] : enumerate(vertex_is_too_close))
            {
                if(is_close)
                    close_vertices.push_back(I);
            }

            mesh.vertices().resize(close_vertices.size());
            mesh.vertices().copy_from(scene_surface.vertices(),
                                      geometry::AttributeCopy::pull(close_vertices));
        }

        {
            auto src_Es = scene_surface.edges().topo().view();
            auto src_Fs = scene_surface.triangles().topo().view();

            vector<SizeT> close_edges;
            close_edges.reserve(src_Es.size());
            for(auto&& [I, E] : enumerate(src_Es))
            {
                if(vertex_is_too_close[E[0]] && vertex_is_too_close[E[1]])
                    close_edges.push_back(I);
            }
            mesh.edges().resize(close_edges.size());
            mesh.edges().copy_from(scene_surface.edges(),
                                   geometry::AttributeCopy::pull(close_edges));

            vector<SizeT> close_triangles;
            close_triangles.reserve(src_Fs.size());
            for(auto&& [I, F] : enumerate(src_Fs))
            {
                if(vertex_is_too_close[F[0]] && vertex_is_too_close[F[1]]
                   && vertex_is_too_close[F[2]])
                    close_triangles.push_back(I);
            }
            mesh.triangles().resize(close_triangles.size());
            mesh.triangles().copy_from(scene_surface.triangles(),
                                       geometry::AttributeCopy::pull(close_triangles));
        }

        {
            vector<SizeT> vert_remap(scene_surface.vertices().size(), -1);
            for(auto&& [I, V] : enumerate(close_vertices))
                vert_remap[V] = I;

            auto Map = [&]<IndexT N>(const Eigen::Vector<IndexT, N>& V) -> Eigen::Vector<IndexT, N>
            {
                auto ret = V;
                for(auto& v : ret)
                    v = vert_remap[v];
                return ret;
            };

            auto edge_topo_view = view(mesh.edges().topo());
            std::ranges::transform(edge_topo_view, edge_topo_view.begin(), Map);

            auto tri_topo_view = view(mesh.triangles().topo());
            std::ranges::transform(tri_topo_view, tri_topo_view.begin(), Map);
        }

        return mesh;
    }

    virtual SanityCheckResult do_check(core::SanityCheckMessage& msg) override
    {
        collect_halfplanes();

        if(halfplanes.empty())
            return SanityCheckResult::Success;

        auto& ctx = context();

        const geometry::SimplicialComplex& scene_surface = ctx.scene_simplicial_surface();

        auto& contact_tabular  = ctx.contact_tabular();
        auto& subscene_tabular = ctx.subscene_tabular();

        auto Vs = scene_surface.vertices().size() ? scene_surface.positions().view() :
                                                    span<const Vector3>{};
        auto Es = scene_surface.edges().size() ? scene_surface.edges().topo().view() :
                                                 span<const Vector2i>{};
        auto Fs = scene_surface.triangles().size() ?
                      scene_surface.triangles().topo().view() :
                      span<const Vector3i>{};

        if(Vs.size() == 0)
            return SanityCheckResult::Success;

        auto attr_cids =
            scene_surface.vertices().find<IndexT>("sanity_check/contact_element_id");
        UIPC_ASSERT(attr_cids, "`sanity_check/contact_element_id` is not found in scene surface");
        auto CIds = attr_cids->view();

        auto attr_scids = scene_surface.vertices().find<IndexT>(
            "sanity_check/subscene_contact_element_id");
        UIPC_ASSERT(attr_scids, "`sanity_check/subscene_contact_element_id` is not found in scene surface");
        auto SCIds = attr_scids->view();

        auto attr_v_geo_ids =
            scene_surface.vertices().find<IndexT>("sanity_check/geometry_id");
        UIPC_ASSERT(attr_v_geo_ids, "`sanity_check/geometry_id` is not found in scene surface");
        auto VGeoIds = attr_v_geo_ids->view();

        auto attr_v_object_id =
            scene_surface.vertices().find<IndexT>("sanity_check/object_id");
        UIPC_ASSERT(attr_v_object_id, "`sanity_check/object_id` is not found in scene surface");
        auto VObjectIds = attr_v_object_id->view();

        auto attr_thickness = scene_surface.vertices().find<Float>(builtin::thickness);
        span<const Float> VThickness =
            attr_thickness ? attr_thickness->view() : span<const Float>{};

        SizeT num_vertices = Vs.size();

        vector<Float> h_thickness(num_vertices, 0.0);
        if(VThickness.size() > 0)
        {
            for(SizeT i = 0; i < num_vertices; ++i)
                h_thickness[i] = VThickness[i];
        }

        SizeT          CN = contact_tabular.element_count();
        vector<IndexT> h_contact_mask(CN * CN);
        for(IndexT i = 0; i < (IndexT)CN; ++i)
            for(IndexT j = 0; j < (IndexT)CN; ++j)
                h_contact_mask[i * CN + j] =
                    contact_tabular.at(i, j).is_enabled() ? 1 : 0;

        SizeT          SN = subscene_tabular.element_count();
        vector<IndexT> h_subscene_mask(SN * SN);
        for(IndexT i = 0; i < (IndexT)SN; ++i)
            for(IndexT j = 0; j < (IndexT)SN; ++j)
                h_subscene_mask[i * SN + j] =
                    subscene_tabular.at(i, j).is_enabled() ? 1 : 0;

        // ---- upload (host Eigen Vector3 -> the device element type) -----------
        vector<LVector3> h_positions(num_vertices);
        for(SizeT i = 0; i < num_vertices; ++i)
            h_positions[i] = to_luisa(Vs[i]);

        lud::LDeviceBuffer<LVector3>     positions(h_positions);
        lud::LDeviceBuffer<Float>        thickness(h_thickness);
        lud::LDeviceBuffer<IndexT>       contact_ids(num_vertices);
        lud::LDeviceBuffer<IndexT>       subscene_ids(num_vertices);
        lud::LDeviceBuffer<IndexT>       contact_mask(h_contact_mask);
        lud::LDeviceBuffer<IndexT>       subscene_mask(h_subscene_mask);

        {
            vector<IndexT> h_contact_ids(CIds.begin(), CIds.end());
            vector<IndexT> h_subscene_ids(SCIds.begin(), SCIds.end());
            contact_ids.copy_from(span<const IndexT>{h_contact_ids});
            subscene_ids.copy_from(span<const IndexT>{h_subscene_ids});
        }

        struct HPInstance
        {
            Vector3 N, P;
            IndexT  cid, scid, geo_id, obj_id;
        };

        vector<HPInstance> hp_instances;
        for(auto& halfplane : halfplanes)
        {
            auto instance_count = halfplane->instances().size();
            auto Ns             = halfplane->instances().find<Vector3>("N");
            UIPC_ASSERT(Ns, "Normal vector `N` not found in half-plane");
            auto Ps = halfplane->instances().find<Vector3>("P");
            UIPC_ASSERT(Ps, "Origin point `P` not found in half-plane");
            auto HGeoIds = halfplane->instances().find<IndexT>("sanity_check/geometry_id");
            UIPC_ASSERT(HGeoIds, "`sanity_check/geometry_id` not found in half-plane");
            auto HObjIds = halfplane->instances().find<IndexT>("sanity_check/object_id");
            UIPC_ASSERT(HObjIds, "`sanity_check/object_id` not found in half-plane");

            auto   attr_cid  = halfplane->meta().find<IndexT>(builtin::contact_element_id);
            auto   attr_scid = halfplane->meta().find<IndexT>(builtin::subscene_element_id);
            IndexT HCid      = attr_cid ? attr_cid->view()[0] : IndexT(0);
            IndexT HSCid     = attr_scid ? attr_scid->view()[0] : IndexT(0);

            for(SizeT I = 0; I < instance_count; ++I)
                hp_instances.push_back({Ns->view()[I],
                                        Ps->view()[I],
                                        HCid,
                                        HSCid,
                                        HGeoIds->view()[I],
                                        HObjIds->view()[I]});
        }

        if(hp_instances.empty())
            return SanityCheckResult::Success;

        SizeT num_hp = hp_instances.size();

        vector<LVector3> h_hp_N(num_hp), h_hp_P(num_hp);
        vector<IndexT>   h_hp_cid(num_hp), h_hp_scid(num_hp);
        for(SizeT h = 0; h < num_hp; ++h)
        {
            h_hp_N[h]    = to_luisa(hp_instances[h].N);
            h_hp_P[h]    = to_luisa(hp_instances[h].P);
            h_hp_cid[h]  = hp_instances[h].cid;
            h_hp_scid[h] = hp_instances[h].scid;
        }

        lud::LDeviceBuffer<LVector3> hp_normals(h_hp_N);
        lud::LDeviceBuffer<LVector3> hp_origins(h_hp_P);
        lud::LDeviceBuffer<IndexT>   hp_contact_ids(h_hp_cid);
        lud::LDeviceBuffer<IndexT>   hp_subscene_ids(h_hp_scid);

        lud::LDeviceBuffer<IndexT> vertex_too_close(num_vertices);
        vertex_too_close.fill(0);
        lud::LDeviceVar<IndexT> has_violation;
        has_violation.copy_from(IndexT{0});

        auto& stream = frame_stream();

        int n = (int)num_vertices;
        if(n > 0)
        {
            auto k = &half_plane_vertex_distance_check_kernel();
            stream.record((*k)(positions.cview(),
                               thickness.cview(),
                               contact_ids.cview(),
                               subscene_ids.cview(),
                               contact_mask.cview(),
                               subscene_mask.cview(),
                               hp_normals.cview(),
                               hp_origins.cview(),
                               hp_contact_ids.cview(),
                               hp_subscene_ids.cview(),
                               vertex_too_close.view(),
                               has_violation.view(),
                               static_cast<std::uint32_t>(CN),
                               static_cast<std::uint32_t>(SN),
                               static_cast<std::uint32_t>(num_hp),
                               static_cast<std::uint32_t>(n))
                             .dispatch(static_cast<std::uint32_t>(n)));
        }

        IndexT h_has_violation = read_back(stream, has_violation.view());
        if(h_has_violation == 0)
            return SanityCheckResult::Success;

        vector<IndexT> h_vertex_too_close(num_vertices, 0);
        read_back_into(stream,
                       vertex_too_close.cview(),
                       span<IndexT>{h_vertex_too_close});

        uipc::map<Vector2i, Vector2i, Vec2iLess> close_geo_ids;
        for(SizeT vI = 0; vI < num_vertices; ++vI)
        {
            if(!h_vertex_too_close[vI])
                continue;
            for(auto& hp : hp_instances)
            {
                Float d = (Vs[vI] - hp.P).dot(hp.N) - h_thickness[vI];
                if(d <= 0.0)
                {
                    close_geo_ids[{(IndexT)VGeoIds[vI], hp.geo_id}] = {
                        (IndexT)VObjectIds[vI], hp.obj_id};
                }
            }
        }

        ::uipc::backend::SanityCheckMessageVisitor scmv{msg};
        auto&                                      buffer = scmv.message();

        for(auto& [GeoIds, ObjIds] : close_geo_ids)
        {
            auto obj_0 = find_object(ObjIds[0]);
            auto obj_1 = find_object(ObjIds[1]);

            UIPC_ASSERT(obj_0 != nullptr, "Object[{}] not found", ObjIds[0]);
            UIPC_ASSERT(obj_1 != nullptr, "Object[{}] not found", ObjIds[1]);

            std::string name_0{obj_0->name()};
            std::string name_1{obj_1->name()};

            fmt::format_to(std::back_inserter(buffer),
                           "Geometry({}) in Object[{}({})] is too close (distance <= 0) to HalfPlane({}) in "
                           "Object[{}({})]\n",
                           GeoIds[0],
                           name_0,
                           obj_0->id(),
                           GeoIds[1],
                           name_1,
                           obj_1->id());
        }

        auto close_mesh = create_close_mesh(scene_surface, h_vertex_too_close);

        fmt::format_to(std::back_inserter(buffer),
                       "Close mesh has {} vertices, {} edges, {} triangles.\n",
                       close_mesh.vertices().size(),
                       close_mesh.edges().size(),
                       close_mesh.triangles().size());

        std::string name = "close_mesh";

        auto sanity_check_mode = ctx.config().find<std::string>("sanity_check/mode");
        if(sanity_check_mode->view()[0] == "normal")
        {
            auto output_path = this_output_path();
            namespace fs     = std::filesystem;
            fs::path path{output_path};
            path /= fmt::format("{}.obj", name);
            auto path_str = path.string();

            geometry::SimplicialComplexIO io;
            io.write(path_str, close_mesh);
            fmt::format_to(std::back_inserter(buffer), "Close mesh is saved at {}.\n", path_str);
        }

        fmt::format_to(std::back_inserter(buffer),
                       "Create mesh [{}<{}>] for post-processing.",
                       name,
                       close_mesh.type());

        scmv.geometries()[name] =
            uipc::make_shared<geometry::SimplicialComplex>(std::move(close_mesh));

        return SanityCheckResult::Error;
    }
};

REGISTER_BACKEND_SANITY_CHECKER(HalfPlaneVertexDistanceCheck);
}// namespace uipc::backend::luisa

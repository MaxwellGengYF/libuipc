#include <affine_body/affine_body_dynamics.h>
/********************************************************************
 * @file affine_body/affine_body_dynamics.cpp
 * @brief T6.1a — the ABD SimSystem (port of
 * src/backends/cuda/affine_body/affine_body_dynamics.cu).
 *
 * Structure is the reference's: the SimSystemCreator gates on the scene having an
 * `AffineBody` constitution type; `Impl::init` runs
 *   _init_dof_info -> _build_constitutions -> _build_geo_infos ->
 *   _setup_geometry_attributes -> _build_geometry_on_host ->
 *   _build_geometry_on_device -> _distribute_geo_infos -> _init_diff_reporters
 * and `write_scene` writes `q` back into the geometry transforms (registered with
 * `on_write_scene` in `do_build`).
 *
 * The reference's cuda_tool crossings map as follows:
 *   BufferLaunch().resize + copy(dst.view(), src.data())  -> upload_* helpers here
 *   BufferLaunch().copy<Vector12>(dst, src)               -> copy_device (deep copy)
 *   BufferLaunch().fill(buf.view(), value)                -> resize_fill
 *   cuda_tool::wait_stream(nullptr)                       -> dropped: LC records on
 *                                                            one stream, so the
 *                                                            ordering is the
 *                                                            stream's (cookbook
 *                                                            §7.16 rule 4c)
 * `x_temps = xs;`-shaped idioms stay explicit: an LDeviceBuffer assignment *shares*
 * the allocation (cookbook §7.16 rule 4b).  The host build-up itself is Eigen,
 * exactly like the reference's.
 *
 * Deviations (reported in the T6.1a row):
 *  * `_build_geo_infos`' vertex-count transform casts to IndexT explicitly
 *    (cookbook §7.16 rule 10: C4244 otherwise; value-identical);
 *  * the inner `gravity_attr` of step 6 is renamed `body_gravity_attr` (the
 *    reference's inner redeclaration hides the outer one; the values are the
 *    reference's);
 *  * `add_reporter` keeps the reference's `UIPC_ASSERT(false, "NOT IMPL IN THIS
 *    VERSION")` verbatim (the diff reporters are M9's).
 *********************************************************************/
#include <affine_body/utils.h>
#include <affine_body/affine_body_constitution.h>
// `add_kinetic()` instantiates SimSystemSlot<AffineBodyKinetic>::register_sim_system,
// so the kinetic interface must be a complete type here (the same reason the FEM
// method includes finite_element_elastics.h).
#include <affine_body/affine_body_kinetic.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/common/enumerate.h>
// `Matrix12x12::inverse()` and the 3x3 `determinant()` of a block expression are
// declared by <Eigen/Core> but *defined* by the LU module; without it the calls
// compile and then fail to link (measured LNK2019).
#include <Eigen/LU>
#include <uipc/common/range.h>
#include <uipc/common/zip.h>
#include <uipc/geometry/simplicial_complex.h>
#include <utils/offset_count_collection.h>
// The test target's include path puts src/backends/none first, so the shell header
// must be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>

#include <algorithm>
#include <ranges>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<luisa::AffineBodyDynamics>
{
  public:
    static U<luisa::AffineBodyDynamics> create(SimEngine& engine)
    {
        auto  scene = dynamic_cast<luisa::SimEngine&>(engine).world().scene();
        auto& types = scene.constitution_tabular().types();
        if(types.find(std::string{builtin::AffineBody}) == types.end())
        {
            return nullptr;
        }
        return uipc::make_unique<luisa::AffineBodyDynamics>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::luisa
{
namespace
{
// ---------------------------------------------------------------------------
//  Host <-> device crossings (the reference's `async_copy`/`async_transfer`/
//  `async_resize` lambdas in _build_geometry_on_device; see the file comment)
// ---------------------------------------------------------------------------
void upload_vec12(lud::LDeviceBuffer<LVector12>& dst, span<const Vector12> src)
{
    dst.resize(src.size());
    if(src.empty())
    {
        return;
    }
    std::vector<LVector12> tmp(src.size());
    for(size_t i = 0; i < src.size(); ++i)
    {
        tmp[i] = to_luisa(src[i]);
    }
    dst.view().copy_from(span<const LVector12>{tmp});
}

void upload_mat12x12(lud::LDeviceBuffer<LMatrix12x12>& dst, span<const Matrix12x12> src)
{
    dst.resize(src.size());
    if(src.empty())
    {
        return;
    }
    std::vector<LMatrix12x12> tmp(src.size());
    for(size_t i = 0; i < src.size(); ++i)
    {
        tmp[i] = to_luisa(src[i]);
    }
    dst.view().copy_from(span<const LMatrix12x12>{tmp});
}

/// Same element type on both sides (the reference's `async_copy` for the ABD
/// attributes whose host and device element types coincide).
template <typename T>
void upload_copy(lud::LDeviceBuffer<T>& dst, span<const T> src)
{
    dst.resize(src.size());
    if(!src.empty())
    {
        dst.view().copy_from(src);
    }
}

/// `cuda_tool::DeviceBuffer::operator=`-shaped deep copy (cookbook §7.16 rule 4b).
template <typename T>
void copy_device(lud::LDeviceBuffer<T>& dst, const lud::LDeviceBuffer<T>& src)
{
    dst.resize(src.size());
    if(src.size() != 0u)
    {
        dst.view().copy_from(src.cview());
    }
}

template <typename T>
void resize_fill(lud::LDeviceBuffer<T>& dst, SizeT size, const T& value)
{
    dst.resize(size);
    if(size != 0u)
    {
        dst.view().fill(value);
    }
}

template <typename T>
void download_copy(span<T> dst, lud::LCBufferView<T> src)
{
    if(dst.empty())
    {
        return;
    }
    src.copy_to(dst);
    lud::main_stream().synchronize();
}

void download_vec12(span<Vector12> dst, lud::LCBufferView<LVector12> src)
{
    if(dst.empty())
    {
        return;
    }
    std::vector<LVector12> tmp(src.size());
    src.copy_to(span<LVector12>{tmp});
    lud::main_stream().synchronize();
    for(size_t i = 0; i < dst.size(); ++i)
    {
        dst[i] = to_eigen(tmp[i]);
    }
}
}  // namespace

REGISTER_SIM_SYSTEM(AffineBodyDynamics);

void AffineBodyDynamics::do_build()
{
    auto& config         = world().scene().config();
    auto  d_hat_attr     = config.find<Float>("contact/d_hat");
    m_impl.default_d_hat = d_hat_attr->view()[0];

    // Register the action to write the scene
    on_write_scene([this] { m_impl.write_scene(world()); });
}

bool AffineBodyDynamics::do_dump(DumpInfo& info)
{
    return m_impl.dump(info);
}

bool AffineBodyDynamics::do_try_recover(RecoverInfo& info)
{
    return m_impl.try_recover(info);
}

void AffineBodyDynamics::do_apply_recover(RecoverInfo& info)
{
    m_impl.apply_recover(info);
}

void AffineBodyDynamics::do_clear_recover(RecoverInfo& info)
{
    m_impl.clear_recover(info);
}

IndexT AffineBodyDynamics::dof_offset(SizeT frame) const
{
    return m_impl.dof_offset(frame);
}

IndexT AffineBodyDynamics::dof_count(SizeT frame) const
{
    return m_impl.dof_count(frame);
}

void AffineBodyDynamics::add_constitution(AffineBodyConstitution* constitution)
{
    check_state(SimEngineState::BuildSystems, "add_constitution()");
    // set the temp index, later we will sort constitution by uid
    // and reset the index
    m_impl.constitutions.register_sim_system(*constitution);
}

void AffineBodyDynamics::add_kinetic(AffineBodyKinetic* kinetic)
{
    check_state(SimEngineState::BuildSystems, "add_kinetic()");
    m_impl.kinetic.register_sim_system(*kinetic);
}

void AffineBodyDynamics::add_reporter(AffineBodyKineticDiffParmReporter* reporter)
{
    UIPC_ASSERT(false, "NOT IMPL IN THIS VERSION");

    //check_state(SimEngineState::BuildSystems, "add_reporter()");
    //m_impl.diff_parm_reporter.register_sim_system(*reporter);
}

void AffineBodyDynamics::init()
{
    m_impl.init(world());
}

void AffineBodyDynamics::Impl::init(WorldVisitor& world)
{
    _init_dof_info();

    _build_constitutions(world);
    _build_geo_infos(world);
    _setup_geometry_attributes(world);
    _build_geometry_on_host(world);
    _build_geometry_on_device(world);

    _distribute_geo_infos();
    _init_diff_reporters();
}

void AffineBodyDynamics::Impl::_build_constitutions(WorldVisitor& world)
{
    // 1) Sort the constitutions by uid
    auto constitution_view = constitutions.view();
    std::ranges::sort(constitution_view,
                      [](const AffineBodyConstitution* a, AffineBodyConstitution* b)
                      { return a->uid() < b->uid(); });
    for(auto&& [i, C] : enumerate(constitution_view))  // reset the index
        C->m_index = i;

    vector<U64> filter_uids(constitution_view.size());
    std::ranges::transform(constitution_view,
                           filter_uids.begin(),
                           [](const AffineBodyConstitution* filter)
                           { return filter->uid(); });

    for(auto&& [i, filter] : enumerate(constitution_view))
        constitution_uid_to_index[filter->uid()] = i;

    constitution_infos.resize(constitution_view.size());

    auto geo_slots = world.scene().geometries();

    // 2) Build the geo_infos and count the geometries, bodies and vertices per constitution
    OffsetCountCollection<IndexT> constitution_geo_offsets_counts;
    constitution_geo_offsets_counts.resize(constitution_view.size());

    OffsetCountCollection<IndexT> constitution_body_offsets_counts;
    constitution_body_offsets_counts.resize(constitution_view.size());

    OffsetCountCollection<IndexT> constitution_vertex_offsets_counts;
    constitution_vertex_offsets_counts.resize(constitution_view.size());

    span<IndexT> constitution_geo_counts    = constitution_geo_offsets_counts.counts();
    span<IndexT> constitution_body_counts   = constitution_body_offsets_counts.counts();
    span<IndexT> constitution_vertex_counts = constitution_vertex_offsets_counts.counts();

    list<GeoInfo> geo_info_list;
    for(auto&& [i, geo_slot] : enumerate(geo_slots))
    {
        auto& geo  = geo_slot->geometry();
        auto  cuid = geo.meta().find<U64>(builtin::constitution_uid);
        if(cuid)  // if has constitution uid
        {
            auto uid            = cuid->view()[0];
            auto instance_count = geo.instances().size();
            if(std::binary_search(filter_uids.begin(), filter_uids.end(), uid))
            {
                auto* sc = geo.as<geometry::SimplicialComplex>();
                UIPC_ASSERT(sc,
                            "The geometry is not a simplicial complex (it's {}). Why can it happen?",
                            geo.type());

                auto vert_count = sc->vertices().size();

                auto constitution_index = constitution_uid_to_index[uid];

                GeoInfo info;
                info.geo_id             = geo_slot->id();
                info.body_count         = instance_count;
                info.vertex_count       = vert_count * instance_count;  // total vertex count
                info.geo_slot_index     = i;
                info.constitution_uid   = uid;
                info.constitution_index = constitution_index;
                geo_info_list.push_back(std::move(info));

                constitution_geo_counts[constitution_index] += 1;  // count the geometry
                constitution_body_counts[constitution_index] +=
                    static_cast<IndexT>(info.body_count);  // count the body
                constitution_vertex_counts[constitution_index] +=
                    static_cast<IndexT>(info.vertex_count);  // count the vertex
            }
        }
    }

    // 3) Setup the offsets
    constitution_geo_offsets_counts.scan();
    constitution_body_offsets_counts.scan();
    constitution_vertex_offsets_counts.scan();

    span<const IndexT> constitution_geo_offsets = constitution_geo_offsets_counts.offsets();
    span<const IndexT> constitution_body_offsets = constitution_body_offsets_counts.offsets();
    span<const IndexT> constitution_vertex_offsets =
        constitution_vertex_offsets_counts.offsets();


    for(auto&& [i, info] : enumerate(constitution_infos))
    {
        info.geo_offset    = constitution_geo_offsets[i];
        info.geo_count     = constitution_geo_counts[i];
        info.body_offset   = constitution_body_offsets[i];
        info.body_count    = constitution_body_counts[i];
        info.vertex_offset = constitution_vertex_offsets[i];
        info.vertex_count  = constitution_vertex_counts[i];
    }

    // 4) set the total count
    abd_geo_count    = constitution_geo_offsets_counts.total_count();
    abd_body_count   = constitution_body_offsets_counts.total_count();
    abd_vertex_count = constitution_vertex_offsets_counts.total_count();

    // 4) copy to the geo_infos
    geo_infos.resize(geo_info_list.size());
    std::ranges::move(geo_info_list, geo_infos.begin());
}

void AffineBodyDynamics::Impl::_build_geo_infos(WorldVisitor& world)
{
    auto scene     = world.scene();
    auto geo_slots = scene.geometries();

    // 1) Sort the geo_infos by constitution uid
    std::ranges::stable_sort(geo_infos,
                             [](const GeoInfo& a, const GeoInfo& b)
                             { return a.constitution_uid < b.constitution_uid; });

    for(auto&& [i, info] : enumerate(geo_infos))
    {
        geo_id_to_geo_info_index[info.geo_id] = i;  // build map for geo_id to geo_info index
    }

    // 2) Setup the body offset and body count
    OffsetCountCollection<IndexT> geo_body_offsets_counts;
    geo_body_offsets_counts.resize(geo_infos.size());

    span<IndexT>       geo_body_counts  = geo_body_offsets_counts.counts();
    span<const IndexT> geo_body_offsets = geo_body_offsets_counts.offsets();

    std::ranges::transform(geo_infos,
                           geo_body_counts.begin(),
                           [](const GeoInfo& info) -> IndexT
                           { return static_cast<IndexT>(info.body_count); });

    geo_body_offsets_counts.scan();

    abd_body_count = geo_body_offsets_counts.total_count();

    for(auto&& [i, info] : enumerate(geo_infos))
    {
        info.body_offset = geo_body_offsets[i];
    }

    // 3) Setup the vertex offset and vertex count
    OffsetCountCollection<IndexT> geo_vertex_offsets_counts;
    geo_vertex_offsets_counts.resize(geo_infos.size());

    span<IndexT> geo_vertex_counts = geo_vertex_offsets_counts.counts();

    // NOTE: the reference's lambda returns SizeT into a span<IndexT> (an implicit
    // narrowing conversion).  The cast is explicit here for warning hygiene.
    std::ranges::transform(geo_infos,
                           geo_vertex_counts.begin(),
                           [](const GeoInfo& info) -> IndexT
                           { return static_cast<IndexT>(info.vertex_count); });

    geo_vertex_offsets_counts.scan();
    span<const IndexT> geo_vertex_offsets = geo_vertex_offsets_counts.offsets();

    for(auto&& [i, info] : enumerate(geo_infos))
    {
        info.vertex_offset = geo_vertex_offsets[i];
    }
}

void AffineBodyDynamics::Impl::_setup_geometry_attributes(WorldVisitor& world)
{
    // set the `backend_abd_body_offset` attribute in simplicial complex geometry
    auto geo_slots = world.scene().geometries();
    for_each(geo_slots,
             [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
             {
                 auto geoI = I.global_index();

                 auto abd_body_offset =
                     sc.meta().find<IndexT>(builtin::backend_abd_body_offset);
                 if(!abd_body_offset)
                     abd_body_offset =
                         sc.meta().create<IndexT>(builtin::backend_abd_body_offset);

                 auto body_offset_view    = geometry::view(*abd_body_offset);
                 body_offset_view.front() = geo_infos[geoI].body_offset;
             });
}

void AffineBodyDynamics::Impl::_build_geometry_on_host(WorldVisitor& world)
{
    auto scene             = world.scene();
    auto geo_slots         = scene.geometries();
    auto constitution_view = constitutions.view();

    // 1) Setup `q` and `q_v` for every affine body
    {
        h_body_id_to_q.resize(abd_body_count);
        h_body_id_to_q_v.resize(abd_body_count, Vector12::Zero());

        for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc)
            { return sc.transforms().view(); },
            [&](const ForEachInfo& I, const Matrix4x4& trans)
            {
                auto  bodyI = I.global_index();
                Float D     = trans.block<3, 3>(0, 0).determinant();
                UIPC_ASSERT(D >= 0,
                            "determinant of the transform matrix is non-positive, why can it happen?");
                Vector12& q = h_body_id_to_q[bodyI];
                q           = transform_to_q(trans);
            });

        for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc)
            {
                auto vel = sc.instances().find<Matrix4x4>(builtin::velocity);
                return vel ? vel->view() : span<const Matrix4x4>{};
            },
            [&](const ForEachInfo& I, const Matrix4x4& velocity)
            {
                auto&     geo_info = I.geo_info();
                auto      bodyI    = geo_info.body_offset + I.local_index();
                Vector12& q_v      = h_body_id_to_q_v[bodyI];
                q_v                = transform_to_q(velocity);
            });
    }


    // 2) Setup `J` for every vertex
    {
        h_vertex_id_to_J.resize(abd_vertex_count);

        span Js = h_vertex_id_to_J;

        for_each(geo_slots,
                 [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
                 {
                     auto geoI = I.global_index();

                     auto pos_view = sc.positions().view();

                     auto body_count  = sc.instances().size();
                     auto vert_count  = sc.vertices().size();
                     auto vert_offset = geo_infos[geoI].vertex_offset;

                     for(auto i : range(body_count))
                     {
                         auto body_vert_offset = vert_offset + i * vert_count;
                         auto sub_Js = Js.subspan(body_vert_offset, vert_count);

                         std::ranges::transform(pos_view,
                                                sub_Js.begin(),
                                                [](const Vector3& pos) -> ABDJacobi
                                                { return to_abd_jacobi(pos); });
                     }
                 });
    }


    // 3) Setup:
    // - `vertex_id_to_body_id`
    // - `vertex_id_to_contact_element_id`
    // - `vertex_id_to_d_hat`
    {
        h_vertex_id_to_body_id.resize(abd_vertex_count);
        h_vertex_id_to_contact_element_id.resize(abd_vertex_count);
        h_vertex_id_to_subscene_contact_element_id.resize(abd_vertex_count);
        h_vertex_id_to_d_hat.resize(abd_vertex_count);

        span v2b     = h_vertex_id_to_body_id;
        span v2c     = h_vertex_id_to_contact_element_id;
        span v2sc    = h_vertex_id_to_subscene_contact_element_id;
        span v2d_hat = h_vertex_id_to_d_hat;

        for_each(
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                auto geoI = I.global_index();

                auto body_count  = sc.instances().size();
                auto body_offset = geo_infos[geoI].body_offset;
                auto vert_count  = sc.vertices().size();
                auto vert_offset = geo_infos[geoI].vertex_offset;

                auto vert_contact_element_id =
                    sc.vertices().find<IndexT>(builtin::contact_element_id);

                auto vert_contact_subscene_element_id =
                    sc.vertices().find<IndexT>(builtin::subscene_element_id);

                auto contact_element_id = sc.meta().find<IndexT>(builtin::contact_element_id);

                auto subscene_element_id =
                    sc.meta().find<IndexT>(builtin::subscene_element_id);

                for(auto i : range(body_count))
                {
                    auto body_vert_offset = vert_offset + i * vert_count;
                    // cast for warning hygiene (cookbook §7.16 rule 10: C4267 on
                    // std::ranges::fill over a span<IndexT> otherwise)
                    auto body_id = static_cast<IndexT>(body_offset + i);

                    auto v2b_span     = v2b.subspan(body_vert_offset, vert_count);
                    auto v2c_span     = v2c.subspan(body_vert_offset, vert_count);
                    auto v2sc_span    = v2sc.subspan(body_vert_offset, vert_count);
                    auto v2d_hat_span = v2d_hat.subspan(body_vert_offset, vert_count);

                    std::ranges::fill(v2b_span, body_id);

                    if(vert_contact_element_id)
                    {
                        auto vert_contact_element_id_view = vert_contact_element_id->view();

                        UIPC_ASSERT(vert_contact_element_id_view.size() == vert_count,
                                    "The size of the contact_element_id attribute is not equal to the vertex count ({} != {}).",
                                    vert_contact_element_id_view.size(),
                                    vert_count);

                        std::ranges::copy(vert_contact_element_id_view, v2c_span.begin());
                    }
                    else
                    {
                        if(contact_element_id)
                        {
                            auto contact_element_id_view = contact_element_id->view();
                            std::ranges::fill(v2c_span, contact_element_id_view.front());
                        }
                        else
                        {
                            std::ranges::fill(v2c_span, 0);  // default 0
                        }
                    }

                    if(vert_contact_subscene_element_id)
                    {
                        auto vert_contact_subscene_element_id_view =
                            vert_contact_subscene_element_id->view();

                        UIPC_ASSERT(vert_contact_subscene_element_id_view.size() == vert_count,
                                    "The size of the contact_element_id attribute is not equal to the vertex count ({} != {}).",
                                    vert_contact_subscene_element_id_view.size(),
                                    vert_count);

                        std::ranges::copy(vert_contact_subscene_element_id_view,
                                          v2sc_span.begin());
                    }
                    else
                    {
                        if(subscene_element_id)
                        {
                            auto contact_subscene_element_id_view = subscene_element_id->view();
                            std::ranges::fill(v2sc_span, contact_subscene_element_id_view.front());
                        }
                        else
                        {
                            std::ranges::fill(v2sc_span, 0);  // default 0
                        }
                    }

                    auto meta_d_hat = sc.meta().find<Float>(builtin::d_hat);

                    if(meta_d_hat)
                    {
                        auto vert_d_hat_view = meta_d_hat->view();
                        std::ranges::fill(v2d_hat_span,
                                          vert_d_hat_view.front());  // use the meta d_hat value
                    }
                    else
                    {
                        std::ranges::fill(v2d_hat_span, default_d_hat);  // default d_hat
                    }
                }
            });
    }

    // 4) Setup:
    // - `body_abd_mass`
    // - `body_id_to_volume`
    // - `body_id_to_dim`
    // - `body_id_to_self_collision`
    {
        h_body_id_to_abd_mass.resize(abd_body_count);
        h_body_id_to_volume.resize(abd_body_count);
        h_body_id_to_dim.resize(abd_body_count);
        h_body_id_to_self_collision.resize(abd_body_count, 0);  // default 0

        span Js = h_vertex_id_to_J;

        for_each(
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                auto geoI = I.global_index();

                auto vert_count  = sc.vertices().size();
                auto vert_offset = geo_infos[geoI].vertex_offset;
                auto body_count  = sc.instances().size();
                auto body_offset = geo_infos[geoI].body_offset;

                // The reference computes the per-geometry J subspan here and never
                // reads it (its only consumer, the per-vertex mass, is commented out).
                auto sub_Js = Js.subspan(vert_offset, vert_count);
                (void)sub_Js;

                ABDJacobiDyadicMass geo_mass;
                {
                    auto mass_attr   = sc.meta().find<Float>(builtin::abd_mass);
                    auto mx_attr     = sc.meta().find<Vector3>(builtin::abd_mass_x_bar);
                    auto mxx_attr    = sc.meta().find<Matrix3x3>(builtin::abd_mass_x_bar_x_bar);
                    UIPC_ASSERT(mass_attr && mx_attr && mxx_attr,
                                "abd_mass/abd_mass_x_bar/abd_mass_x_bar_x_bar not found on affine body geometry");
                    auto m             = mass_attr->view()[0];
                    auto m_x_bar       = mx_attr->view()[0];
                    auto m_x_bar_x_bar = mxx_attr->view()[0];
                    geo_mass = to_abd_dyadic_mass(m, m_x_bar, m_x_bar_x_bar);
                }

                auto volume = sc.meta().find<Float>(builtin::volume);
                UIPC_ASSERT(volume,
                            "The `volume` attribute is not found in the affine body meta, why can it happen?");

                auto volume_v = volume->view().front();

                auto self_collision = sc.meta().find<IndexT>(builtin::self_collision);
                UIPC_ASSERT(self_collision,
                            "The `self_collision` attribute is not found in the affine body `meta`, why can it happen?");

                IndexT self_collision_value = self_collision->view()[0];

                for(auto i : range(body_count))
                {
                    auto body_id = body_offset + i;
                    // mass
                    h_body_id_to_abd_mass[body_id] = geo_mass;
                    // volume
                    h_body_id_to_volume[body_id] = volume_v;
                    // dim
                    h_body_id_to_dim[body_id] = sc.dim();
                    // self_collision
                    h_body_id_to_self_collision[body_id] = self_collision_value;
                }
            });
    }


    // 5) Compute the inverse of the mass matrix
    h_body_id_to_abd_mass_inv.resize(abd_body_count);
    std::ranges::transform(h_body_id_to_abd_mass,
                           h_body_id_to_abd_mass_inv.begin(),
                           [](const ABDJacobiDyadicMass& mass) -> Matrix12x12
                           { return abd_dyadic_mass_to_mat_eigen(mass).inverse(); });

    // 6) Setup the affine body gravity
    {
        span Js          = h_vertex_id_to_J;
        auto gravity_attr = scene.config().find<Vector3>("gravity");
        Vector3 gravity   = gravity_attr->view()[0];
        h_body_id_to_abd_gravity.resize(abd_body_count, Vector12::Zero());
        body_id_to_external_force.resize(abd_body_count);
        body_id_to_external_force_acc.resize(abd_body_count);
        for_each(geo_slots,
                 [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
                 {
                     auto geoI = I.global_index();

                     auto vert_count  = sc.vertices().size();
                     auto vert_offset = geo_infos[geoI].vertex_offset;
                     auto body_count  = sc.instances().size();
                     auto body_offset = geo_infos[geoI].body_offset;

                     auto sub_Js = Js.subspan(vert_offset, vert_count);
                     (void)sub_Js;  // unused in the reference as well

                     // NOTE: the reference re-declares `gravity_attr` here (C4456
                     // shadowing); renamed, values unchanged.
                     auto body_gravity_attr =
                         sc.instances().find<Vector3>(builtin::gravity);
                     auto gravity_view = body_gravity_attr ? body_gravity_attr->view() :
                                                             span<const Vector3>{};

                     auto mass_attr = sc.meta().find<Float>(builtin::abd_mass);
                     auto mx_attr   = sc.meta().find<Vector3>(builtin::abd_mass_x_bar);
                     UIPC_ASSERT(mass_attr && mx_attr,
                                 "abd_mass/abd_mass_x_bar not found on affine body geometry");
                     Float   total_m = mass_attr->view()[0];
                     Vector3 mx_bar  = mx_attr->view()[0];

                     for(auto i : range(body_count))
                     {
                         Vector3 local_gravity = body_gravity_attr ? gravity_view[i] : gravity;

                         Vector12 G;
                         G.segment<3>(0) = total_m * local_gravity;
                         G.segment<3>(3) = local_gravity(0) * mx_bar;
                         G.segment<3>(6) = local_gravity(1) * mx_bar;
                         G.segment<3>(9) = local_gravity(2) * mx_bar;

                         Matrix12x12 abd_body_mass_inv = h_body_id_to_abd_mass_inv[body_offset];
                         Vector12    g                 = abd_body_mass_inv * G;

                         auto body_id = body_offset + i;

                         h_body_id_to_abd_gravity[body_id] = g;
                     }
                 });
    }


    // 7) Setup the boundary type
    {
        h_body_id_to_is_fixed.resize(abd_body_count, 0);
        h_body_id_to_is_dynamic.resize(abd_body_count, 1);
        h_body_id_to_external_kinetic.resize(abd_body_count, 0);
        for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc)
            {
                auto is_fixed = sc.instances().find<IndexT>(builtin::is_fixed);
                auto is_dynamic = sc.instances().find<IndexT>(builtin::is_dynamic);
                auto external_kinetic = sc.instances().find<IndexT>(builtin::external_kinetic);

                UIPC_ASSERT(is_fixed,
                            "The is_fixed attribute is not found in the affine body geometry, why can it happen?");
                UIPC_ASSERT(is_dynamic,
                            "The is_dynamic attribute is not found in the affine body geometry, why can it happen?");
                UIPC_ASSERT(external_kinetic,
                            "The is_external_kinetic attribute is not found in the affine body geometry, why can it happen?");
                return zip(is_fixed->view(), is_dynamic->view(), external_kinetic->view());
            },
            [&](const ForEachInfo& I, auto&& data)
            {
                auto&& [fixed, dynamic, external_kinetic] = data;

                auto bodyI = I.global_index();

                h_body_id_to_is_fixed[bodyI]           = fixed;
                h_body_id_to_is_dynamic[bodyI]         = dynamic;
                h_body_id_to_external_kinetic[bodyI]   = external_kinetic;
            });
    }
}

void AffineBodyDynamics::Impl::_build_geometry_on_device(WorldVisitor& world)
{
    (void)world;
    upload_vec12(body_id_to_q, span<const Vector12>{h_body_id_to_q});
    upload_vec12(body_id_to_q_v, span<const Vector12>{h_body_id_to_q_v});
    upload_copy(body_id_to_dim, span<const IndexT>{h_body_id_to_dim});
    upload_copy(vertex_id_to_J, span<const ABDJacobi>{h_vertex_id_to_J});
    upload_copy(vertex_id_to_body_id, span<const IndexT>{h_vertex_id_to_body_id});
    upload_copy(body_id_to_abd_mass,
                span<const ABDJacobiDyadicMass>{h_body_id_to_abd_mass});
    upload_copy(body_id_to_volume, span<const Float>{h_body_id_to_volume});
    upload_mat12x12(body_id_to_abd_mass_inv,
                    span<const Matrix12x12>{h_body_id_to_abd_mass_inv});
    upload_vec12(body_id_to_abd_gravity, span<const Vector12>{h_body_id_to_abd_gravity});
    upload_copy(body_id_to_is_fixed, span<const IndexT>{h_body_id_to_is_fixed});
    upload_copy(body_id_to_is_dynamic, span<const IndexT>{h_body_id_to_is_dynamic});
    upload_copy(body_id_to_external_kinetic,
                span<const IndexT>{h_body_id_to_external_kinetic});

    // The reference's `x_temps = xs;`-shaped transfers (deep copies here).
    copy_device(body_id_to_q_temp, body_id_to_q);
    copy_device(body_id_to_q_tilde, body_id_to_q);
    copy_device(body_id_to_q_prev, body_id_to_q);

    resize_fill(body_id_to_dq, abd_body_count, LVector12{});
}

void AffineBodyDynamics::Impl::_distribute_geo_infos()
{
    // _distribute the body infos to each constitution
    for(auto&& [I, constitution] : enumerate(constitutions.view()))
    {
        FilteredInfo info{this, I};
        constitution->init(info);
    }
}

void AffineBodyDynamics::Impl::set_dof_info(SizeT frame, IndexT dof_offset, IndexT dof_count)
{
    UIPC_ASSERT(frame > 0, "frame 0 is not used");
    if(frame_to_dof_count.size() <= frame)
    {
        frame_to_dof_count.resize(frame + 1, -1);
        frame_to_dof_offset.resize(frame + 1, -1);
    }
    frame_to_dof_count[frame]  = dof_count;
    frame_to_dof_offset[frame] = dof_offset;
}

void AffineBodyDynamics::Impl::_init_dof_info()
{
    frame_to_dof_count.reserve(1024);
    frame_to_dof_offset.reserve(1024);
    // frame 0 is not used, just fill 0
    frame_to_dof_offset.push_back(0);
    frame_to_dof_count.push_back(0);
}

void AffineBodyDynamics::Impl::_init_diff_reporters()
{
    //for(auto&& reporter : diff_parm_reporter.view())
    //{
    // reporter->init();
    //}
}

void AffineBodyDynamics::Impl::_download_geometry_to_host()
{
    download_vec12(span<Vector12>{h_body_id_to_q}, body_id_to_q.cview());
}

void AffineBodyDynamics::Impl::write_scene(WorldVisitor& world)
{
    // 1) download from device to host
    _download_geometry_to_host();

    auto scene     = world.scene();
    auto geo_slots = scene.geometries();

    // 2) transfer from affine_body_dynamics qs to transforms
    for_each(
        geo_slots,
        [](geometry::SimplicialComplex& sc) { return view(sc.transforms()); },
        [&](const ForEachInfo& I, Matrix4x4& trans)
        {
            auto bodyI = I.global_index();

            Vector12& q = h_body_id_to_q[bodyI];

            trans = q_to_transform(q);
        });
}

IndexT AffineBodyDynamics::Impl::dof_offset(SizeT frame) const
{
    UIPC_ASSERT(frame > 0, "frame 0 is not used");
    return frame_to_dof_offset[frame];
}

IndexT AffineBodyDynamics::Impl::dof_count(SizeT frame) const
{
    UIPC_ASSERT(frame > 0, "frame 0 is not used");
    return frame_to_dof_count[frame];
}

// Dump & Recover:

bool AffineBodyDynamics::Impl::dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_q.dump(fmt::format("{}q.{}", path, frame), body_id_to_q)           //
           && dump_q_v.dump(fmt::format("{}q_v.{}", path, frame), body_id_to_q_v)  //
           && dump_q_prev.dump(fmt::format("{}q_prev.{}", path, frame), body_id_to_q_prev);  //
}

bool AffineBodyDynamics::Impl::try_recover(RecoverInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_q.load(fmt::format("{}q.{}", path, frame))            //
           && dump_q_v.load(fmt::format("{}q_v.{}", path, frame))     //
           && dump_q_prev.load(fmt::format("{}q_prev.{}", path, frame));  //
}

void AffineBodyDynamics::Impl::apply_recover(RecoverInfo& info)
{
    (void)info;
    dump_q.apply_to(body_id_to_q);
    dump_q_v.apply_to(body_id_to_q_v);
    dump_q_prev.apply_to(body_id_to_q_prev);
}

void AffineBodyDynamics::Impl::clear_recover(RecoverInfo& info)
{
    (void)info;
    dump_q.clean_up();
    dump_q_v.clean_up();
    dump_q_prev.clean_up();
}

AffineBodyDynamics::FilteredInfo::FilteredInfo(Impl* impl, SizeT constitution_index) noexcept
    : m_impl(impl)
    , m_constitution_index(constitution_index)
{
}

auto AffineBodyDynamics::FilteredInfo::geo_infos() const noexcept -> span<const GeoInfo>
{
    auto& constitution_info = m_impl->constitution_infos[m_constitution_index];
    return span{m_impl->geo_infos}.subspan(constitution_info.geo_offset,
                                           constitution_info.geo_count);
}

auto AffineBodyDynamics::FilteredInfo::constitution_info() const noexcept
    -> const ConstitutionInfo&
{
    return m_impl->constitution_infos[m_constitution_index];
}

SizeT AffineBodyDynamics::FilteredInfo::body_count() const noexcept
{
    return constitution_info().body_count;
}

SizeT AffineBodyDynamics::FilteredInfo::vertex_count() const noexcept
{
    return constitution_info().vertex_count;
}

void AffineBodyDynamics::overwrite_qs(lud::LCBufferView<LVector12> qs)
{
    m_impl.body_id_to_q.view().copy_from(qs);
}
}  // namespace uipc::backend::luisa

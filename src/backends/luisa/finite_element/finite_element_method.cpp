#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/fem_utils.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/simplicial_complex.h>
#include <uipc/common/map.h>
#include <uipc/common/set.h>
#include <uipc/common/vector.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/zip.h>
#include <uipc/common/algorithm/run_length_encode.h>
#include <uipc/common/json_eigen.h>
#include <ranges>
#include <backends/luisa/sim_engine.h>
#include <frame_stream.h>
#include <utils/offset_count_collection.h>

// elastics (T5.3's system; the T5.1 part of it is ported, see its header)
#include <finite_element/finite_element_elastics.h>
// constitutions (the dimension bases are ported as interfaces in T5.1, the
// concrete constitutions are T5.2's)
#include <finite_element/fem_3d_constitution.h>
#include <finite_element/codim_2d_constitution.h>
#include <finite_element/codim_1d_constitution.h>
#include <finite_element/codim_0d_constitution.h>
#include <uipc/builtin/constitution_type.h>
// diff parm reporters (T5.5): the reference includes
// finite_element_diff_parm_reporter.h here, but nothing in this file uses the
// type beyond a forward declaration (the `add_reporter`/`add_kinetic_reporter`
// bodies are commented out in the reference too), and <diff_sim/diff_parm_reporter.h>
// is not ported yet. T5.5 adds the include back when it ports the reporters.
// features
#include <finite_element/finite_element_state_accessor_feature.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<luisa::FiniteElementMethod>
{
  public:
    static U<luisa::FiniteElementMethod> create(SimEngine& engine)
    {
        auto  scene = dynamic_cast<luisa::SimEngine&>(engine).world().scene();
        auto& types = scene.constitution_tabular().types();
        if(types.find(std::string{builtin::FiniteElement}) == types.end())
        {
            return nullptr;
        }
        return uipc::make_unique<luisa::FiniteElementMethod>(engine);
    }
};
}  // namespace uipc::backend


bool operator<(const uipc::backend::luisa::FiniteElementMethod::DimUID& a,
               const uipc::backend::luisa::FiniteElementMethod::DimUID& b)
{
    return a.dim < b.dim || (a.dim == b.dim && a.uid < b.uid);
}

namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

// ---------------------------------------------------------------------------
//  Host <-> device conversions
//
//  The FEM method's *host* data is Eigen (`Vector3`, `Vector2i`, ...), exactly
//  like the reference's; only the device buffers change element type
//  (`Vector3` -> `LVector3`, cookbook §7.13 rule 1). Both are byte-identical
//  layouts (static_assert in type_define.h), but they are unrelated types, so the
//  crossing is an explicit per-element conversion.
// ---------------------------------------------------------------------------

/// H2D: host Eigen vector array -> device buffer. cuda_tool's `view().copy_from`
/// synchronises; an LC copy is asynchronous, so the callers below (which are all
/// inside init()/write_scene(), i.e. outside any frame) keep the same ordering by
/// relying on the stream order, exactly like the reference.
void upload_vec3(lud::LDeviceBuffer<LVector3>& dst, span<const Vector3> src)
{
    dst.resize(src.size());
    if(src.empty())
    {
        return;
    }
    std::vector<LVector3> tmp(src.size());
    for(size_t i = 0; i < src.size(); ++i)
    {
        tmp[i] = to_luisa(src[i]);
    }
    dst.view().copy_from(span<const LVector3>{tmp});
}

template <typename T>
void upload_scalar(lud::LDeviceBuffer<T>& dst, span<const T> src)
{
    dst.resize(src.size());
    if(!src.empty())
    {
        dst.view().copy_from(src);
    }
}

/// H2D with a per-element conversion (Eigen Vector2i/3i/4i -> LVector*).
template <typename DeviceT, typename HostT>
void upload_converted(lud::LDeviceBuffer<DeviceT>& dst, span<const HostT> src)
{
    dst.resize(src.size());
    if(src.empty())
    {
        return;
    }
    std::vector<DeviceT> tmp(src.size());
    for(size_t i = 0; i < src.size(); ++i)
    {
        tmp[i] = to_luisa(src[i]);
    }
    dst.view().copy_from(span<const DeviceT>{tmp});
}

/// D2H: device buffer -> host Eigen vector array (cuda_tool's `copy_to`
/// synchronises, so the read-back waits explicitly).
void download_vec3(span<Vector3> dst, lud::LCBufferView<LVector3> src)
{
    if(dst.empty())
    {
        return;
    }
    std::vector<LVector3> tmp(src.size());
    src.copy_to(span<LVector3>{tmp});
    lud::main_stream().synchronize();
    for(size_t i = 0; i < dst.size(); ++i)
    {
        dst[i] = to_eigen(tmp[i]);
    }
}

/// A device buffer copy. cuda_tool's `DeviceBuffer::operator=` deep-copies
/// (`copy_from`), so `x_temps = xs;` must not be a shared handle here: an LC
/// LDeviceBuffer's implicit copy assignment shares the underlying buffer.
template <typename T>
void copy_device(lud::LDeviceBuffer<T>& dst, const lud::LDeviceBuffer<T>& src)
{
    dst.resize(src.size());
    if(src.size() != 0u)
    {
        dst.view().copy_from(src.cview());
    }
}

// ---------------------------------------------------------------------------
//  Kernel caches (compile once, never destroyed -- cookbook §7.10 trap 1)
// ---------------------------------------------------------------------------

/// k1: rest lengths of the codim-1d elements.
auto& build_on_device_k1()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,
                                                lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<double>,
                                                uint32_t>(
        [](lcrt::BufferVar<LVector2i> codim_1ds,
           lcrt::BufferVar<LVector3>  x_bars,
           lcrt::BufferVar<double>    rest_lengths,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto edge = load_vec2i(codim_1ds->read(i));
                auto x0   = load_vec3(x_bars->read(edge(0)));
                auto x1   = load_vec3(x_bars->read(edge(1)));

                rest_lengths->write(i, lps::norm(x1 - x0));
            };
        });
    return k;
}

/// k2: rest areas of the codim-2d elements.
auto& build_on_device_k2()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3i>,
                                                lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<double>,
                                                uint32_t>(
        [](lcrt::BufferVar<LVector3i> codim_2ds,
           lcrt::BufferVar<LVector3>  x_bars,
           lcrt::BufferVar<double>    rest_areas,
           lcrt::Var<uint32_t>        n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto tri = load_vec3i(codim_2ds->read(i));
                auto x0  = load_vec3(x_bars->read(tri(0)));
                auto x1  = load_vec3(x_bars->read(tri(1)));
                auto x2  = load_vec3(x_bars->read(tri(2)));

                auto E01 = x1 - x0;
                auto E02 = x2 - x0;

                rest_areas->write(
                    i,
                    lps::detail::mul(lps::norm(lps::cross(E01, E02)), 0.5));
            };
        });
    return k;
}

/// k3: the FEM3D material basis (`Dm^-1`) and the rest volumes, with the
/// reference's negative-volume check.
///
/// `UIPC_KERNEL_ASSERT` -> `lcrt::device_assert`: LuisaCompute has a real device
/// assert (luisa/dsl/builtin.h). The reference formats the four tet indices into
/// the message, which needs a *device* formatted string; LC takes a fixed message
/// only, so the port keeps the check and drops the indices from the text (the
/// failing dispatch is still reported by LC).
auto& build_on_device_k3()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector4i>,
                                                lcrt::Buffer<LVector3>,
                                                lcrt::Buffer<LMatrix3x3>,
                                                lcrt::Buffer<double>,
                                                uint32_t>(
        [](lcrt::BufferVar<LVector4i>  tets,
           lcrt::BufferVar<LVector3>   x_bars,
           lcrt::BufferVar<LMatrix3x3> Dm3x3_inv,
           lcrt::BufferVar<double>     rest_volumes,
           lcrt::Var<uint32_t>         n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto tet = load_vec4i(tets->read(i));
                auto x0  = load_vec3(x_bars->read(tet(0)));
                auto x1  = load_vec3(x_bars->read(tet(1)));
                auto x2  = load_vec3(x_bars->read(tet(2)));
                auto x3  = load_vec3(x_bars->read(tet(3)));

                auto Dm = fem::Ds(x0, x1, x2, x3);
                lcrt::Var<LMatrix3x3> dm_inv;
                store_mat3x3(dm_inv, lps::inverse(Dm));
                Dm3x3_inv->write(i, dm_inv);

                auto V = lps::detail::div(lps::determinant(Dm), 6.0);
                lcrt::device_assert(V > 0.0, "Negative volume tetrahedron");
                rest_volumes->write(i, V);
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FiniteElementMethod);

void FiniteElementMethod::do_build()
{
    const auto& scene = world().scene();

    m_impl.default_gravity = scene.config().find<Vector3>("gravity")->view()[0];
    m_impl.default_d_hat = scene.config().find<Float>("contact/d_hat")->view()[0];
    m_impl.global_vertex_manager = &require<GlobalVertexManager>();

    // Register the action to write the scene
    on_write_scene([this] { m_impl.write_scene(world()); });
}

IndexT FiniteElementMethod::dof_offset(SizeT frame) const noexcept
{
    return m_impl.dof_offset(frame);
}

IndexT FiniteElementMethod::dof_count(SizeT frame) const noexcept
{
    return m_impl.dof_count(frame);
}

void FiniteElementMethod::add_elastics(FiniteElementElastics* elastics)
{
    check_state(SimEngineState::BuildSystems, "add_elastics()");
    m_impl.elastics.register_sim_system(*elastics);
}

void FiniteElementMethod::add_constitution(FiniteElementConstitution* constitution)
{
    check_state(SimEngineState::BuildSystems, "add_constitution()");
    m_impl.constitutions.register_sim_system(*constitution);
}

void FiniteElementMethod::add_constitution(FiniteElementExtraConstitution* constitution)
{
    check_state(SimEngineState::BuildSystems, "add_constitution()");
    m_impl.extra_constitutions.register_sim_system(*constitution);
}

void FiniteElementMethod::add_reporter(FiniteElementConstitutionDiffParmReporter* reporter)
{
    // The reference has this body commented out as well (T5.5 wires the diff
    // reporters up); kept verbatim.
    //check_state(SimEngineState::BuildSystems, "add_reporter()");
    //m_impl.constitution_diff_parm_reporters.register_sim_system(*reporter);
}

void FiniteElementMethod::add_reporter(FiniteElementExtraConstitutionDiffParmReporter* reporter)
{
    //check_state(SimEngineState::BuildSystems, "add_reporter()");
    //m_impl.extra_constitution_diff_parm_reporters.register_sim_system(*reporter);
}

void FiniteElementMethod::add_kinetic_reporter(FiniteElementDiffParmReporter* reporter)
{
    //check_state(SimEngineState::BuildSystems, "add_reporter()");
    //m_impl.kinetic_diff_parm_reporter.register_sim_system(*reporter);
}

void FiniteElementMethod::init()
{
    m_impl.init(world());
}

bool FiniteElementMethod::do_dump(DumpInfo& info)
{
    return m_impl.dump(info);
}

bool FiniteElementMethod::do_try_recover(RecoverInfo& info)
{
    return m_impl.try_recover(info);
}

void FiniteElementMethod::do_apply_recover(RecoverInfo& info)
{
    m_impl.apply_recover(info);
}

void FiniteElementMethod::do_clear_recover(RecoverInfo& info)
{
    m_impl.clear_recover(info);
}

void FiniteElementMethod::Impl::init(WorldVisitor& world)
{
    _init_dof_info();

    // 1) setup base constitution infos
    _classify_base_constitutions();
    _build_geo_infos(world);
    _build_base_constitution_infos();
    _build_on_host(world);
    _build_on_device();

    // 2) Init event
    _init_base_constitutions();
    _init_extra_constitutions();
    elastics->init();

    _init_diff_reporters();
}

void FiniteElementMethod::Impl::_init_dof_info()
{
    frame_to_dof_count.reserve(1024);
    frame_to_dof_offset.reserve(1024);
    // frame 0 is not used
    frame_to_dof_offset.push_back(0);
    frame_to_dof_count.push_back(0);
}

void FiniteElementMethod::Impl::_classify_base_constitutions()
{
    auto constitution_view = constitutions.view();

    // 1) sort the constitutions by (dim, uid)
    std::sort(constitution_view.begin(),
              constitution_view.end(),
              [](const FiniteElementConstitution* a, const FiniteElementConstitution* b)
              {
                  auto   uida = a->uid();
                  auto   uidb = b->uid();
                  auto   dima = a->dim();
                  auto   dimb = b->dim();
                  DimUID uid_dim_a{dima, uida};
                  DimUID uid_dim_b{dimb, uidb};
                  return uid_dim_a < uid_dim_b;
              });

    for(auto&& [i, c] : enumerate(constitution_view))
        c->m_index = i;

    // 2) classify the constitutions
    codim_0d_constitutions.reserve(constitution_view.size());
    codim_1d_constitutions.reserve(constitution_view.size());
    codim_2d_constitutions.reserve(constitution_view.size());
    fem_3d_constitutions.reserve(constitution_view.size());

    for(auto&& constitution : constitution_view)
    {
        auto dim = constitution->dim();
        switch(dim)
        {
            case 0: {
                auto derived = dynamic_cast<Codim0DConstitution*>(constitution);
                UIPC_ASSERT(derived, "The constitution is not a Codim0DConstitution, its dim = {}", dim);
                derived->m_index_in_dim = codim_0d_constitutions.size();
                codim_0d_constitutions.push_back(derived);
                codim_0d_uid_to_index.insert({derived->uid(), derived->m_index_in_dim});
            }
            break;
            case 1: {
                auto derived = dynamic_cast<Codim1DConstitution*>(constitution);
                UIPC_ASSERT(derived, "The constitution is not a Codim1DConstitution, its dim = {}", dim);
                derived->m_index_in_dim = codim_1d_constitutions.size();
                codim_1d_constitutions.push_back(derived);
                codim_1d_uid_to_index.insert({derived->uid(), derived->m_index_in_dim});
            }
            break;
            case 2: {
                auto derived = dynamic_cast<Codim2DConstitution*>(constitution);
                UIPC_ASSERT(derived, "The constitution is not a Codim2DConstitution, its dim = {}", dim);
                derived->m_index_in_dim = codim_2d_constitutions.size();
                codim_2d_constitutions.push_back(derived);
                codim_2d_uid_to_index.insert({derived->uid(), derived->m_index_in_dim});
            }
            break;
            case 3: {
                auto derived = dynamic_cast<FEM3DConstitution*>(constitution);
                UIPC_ASSERT(derived, "The constitution is not a FEM3DConstitution, its dim = {}", dim);
                derived->m_index_in_dim = fem_3d_constitutions.size();
                fem_3d_constitutions.push_back(derived);
                fem_3d_uid_to_index.insert({derived->uid(), derived->m_index_in_dim});
            }
            break;
            default:
                break;
        }
    }
}

void FiniteElementMethod::Impl::_init_diff_reporters()
{
    // TODO: Placeholder for diff sim system initialization (T5.5)
}

void FiniteElementMethod::Impl::_build_geo_infos(WorldVisitor& world)
{
    set<U64> filter_uids;

    for(auto&& filter : constitutions.view())
        filter_uids.insert(filter->uid());

    // 1) find all the finite element constitutions
    auto geo_slots = world.scene().geometries();
    geo_infos.reserve(geo_slots.size());

    for(auto&& [i, geo_slot] : enumerate(geo_slots))
    {
        auto& geo  = geo_slot->geometry();
        auto  cuid = geo.meta().find<U64>(builtin::constitution_uid);
        if(cuid)
        {
            auto uid = cuid->view()[0];
            if(filter_uids.find(uid) != filter_uids.end())  // if exists
            {
                auto* sc = geo.as<geometry::SimplicialComplex>();
                UIPC_ASSERT(sc,
                            "The geometry is not a simplicial complex (it's {}). Why can it happen?",
                            geo.type());

                GeoInfo info;
                info.geo_slot_index = i;
                info.vertex_count   = sc->vertices().size();
                info.dim_uid.dim    = sc->dim();
                info.dim_uid.uid    = uid;

                switch(sc->dim())
                {
                    case 0:
                        info.primitive_count = sc->vertices().size();
                        break;
                    case 1:
                        info.primitive_count = sc->edges().size();
                        break;
                    case 2:
                        info.primitive_count = sc->triangles().size();
                        break;
                    case 3:
                        info.primitive_count = sc->tetrahedra().size();
                        break;
                    default:
                        break;
                }

                geo_infos.push_back(info);
            }
        }
    }

    // 2) sort geometry by (dim, uid)
    std::sort(geo_infos.begin(),
              geo_infos.end(),
              [](const GeoInfo& a, const GeoInfo& b)
              { return a.dim_uid < b.dim_uid; });


    // 3) setup vertex offsets and primitive offsets
    {
        OffsetCountCollection<IndexT> vertex_offsets_counts;
        vertex_offsets_counts.resize(geo_infos.size());
        OffsetCountCollection<IndexT> primitive_offsets_counts;
        primitive_offsets_counts.resize(geo_infos.size());

        span<IndexT> vertex_counts    = vertex_offsets_counts.counts();
        span<IndexT> primitive_counts = primitive_offsets_counts.counts();

        std::transform(geo_infos.begin(),
                       geo_infos.end(),
                       vertex_counts.begin(),
                       [](const GeoInfo& info)
                       { return static_cast<IndexT>(info.vertex_count); });

        std::transform(geo_infos.begin(),
                       geo_infos.end(),
                       primitive_counts.begin(),
                       [](const GeoInfo& info)
                       { return static_cast<IndexT>(info.primitive_count); });

        vertex_offsets_counts.scan();

        // we don't calculate the primitive offset here
        // the primitive offset is related to the dimension, every dim starts from 0

        span<const IndexT> vertex_offsets = vertex_offsets_counts.offsets();

        for(auto&& [i, info] : enumerate(geo_infos))
        {
            info.vertex_offset = vertex_offsets[i];
        }

        h_positions.resize(vertex_offsets_counts.total_count());
    }


    // 4) setup dim infos
    {
        std::array<SizeT, 4> dim_geo_counts;
        std::array<SizeT, 4> dim_geo_offsets;
        dim_geo_counts.fill(0);


        vector<SizeT> offsets;
        offsets.reserve(dim_geo_counts.size());
        vector<SizeT> counts;
        counts.reserve(dim_geo_counts.size());

        // encode the dimension
        encode_offset_count(geo_infos.begin(),
                            geo_infos.end(),
                            std::back_inserter(offsets),
                            std::back_inserter(counts),
                            [](const GeoInfo& current, const GeoInfo& value)
                            { return current.dim_uid.dim == value.dim_uid.dim; });

        for(auto&& [offset, count] : zip(offsets, counts))
        {
            auto& info                       = geo_infos[offset];
            dim_geo_counts[info.dim_uid.dim] = count;
        }

        std::exclusive_scan(dim_geo_counts.begin(),
                            dim_geo_counts.end(),
                            dim_geo_offsets.begin(),
                            SizeT{0});

        for(auto&& [i, dim_info] : enumerate(dim_infos))
        {
            dim_info.geo_info_offset = dim_geo_offsets[i];
            dim_info.geo_info_count  = dim_geo_counts[i];
        }
    }


    // 4) setup dim_info vertex and primitive
    vector<SizeT> dim_primitive_counts(dim_infos.size(), 0);
    vector<SizeT> dim_vertex_counts(dim_infos.size(), 0);
    vector<SizeT> dim_vertex_offsets(dim_infos.size(), 0);

    for(auto&& [i, dim_info] : enumerate(dim_infos))
    {
        auto it = std::find_if(geo_infos.begin(),
                               geo_infos.end(),
                               [i](const GeoInfo& info)
                               { return info.dim_uid.dim == i; });

        if(it == geo_infos.end())
            continue;

        OffsetCountCollection<IndexT> primitive_offsets_counts;
        primitive_offsets_counts.resize(dim_info.geo_info_count);

        span<IndexT>   primitive_counts = primitive_offsets_counts.counts();
        vector<IndexT> vertex_counts(dim_info.geo_info_count);

        auto geo_span =
            span{geo_infos}.subspan(dim_info.geo_info_offset, dim_info.geo_info_count);

        std::ranges::transform(geo_span,
                               primitive_counts.begin(),
                               [](const GeoInfo& info)
                               { return static_cast<IndexT>(info.primitive_count); });

        // exclusive scan the primitive counts
        primitive_offsets_counts.scan();

        span<const IndexT> primitive_offsets = primitive_offsets_counts.offsets();

        for(auto&& [j, info] : enumerate(geo_span))
        {
            info.primitive_offset = primitive_offsets[j];
        }

        dim_primitive_counts[i] = primitive_offsets_counts.total_count();

        UIPC_ASSERT(geo_span.size() == vertex_counts.size(),
                    "Size mismatching in geo_span({}) and vertex_counts({}), why can it happen?",
                    geo_span.size(),
                    vertex_counts.size());

        std::ranges::transform(geo_span,
                               vertex_counts.begin(),
                               [](const GeoInfo& info)
                               { return static_cast<IndexT>(info.vertex_count); });


        dim_vertex_counts[i] =
            std::accumulate(vertex_counts.begin(), vertex_counts.end(), 0);
    }

    std::exclusive_scan(dim_vertex_counts.begin(),
                        dim_vertex_counts.end(),
                        dim_vertex_offsets.begin(),
                        SizeT{0});

    for(auto&& [i, dim_info] : enumerate(dim_infos))
    {
        dim_info.vertex_count     = dim_vertex_counts[i];
        dim_info.vertex_offset    = dim_vertex_offsets[i];
        dim_info.primitive_offset = 0;  // always 0
        dim_info.primitive_count  = dim_primitive_counts[i];
    }


    h_codim_0ds.resize(dim_infos[0].primitive_count);
    h_codim_1ds.resize(dim_infos[1].primitive_count);
    h_codim_2ds.resize(dim_infos[2].primitive_count);
    h_tets.resize(dim_infos[3].primitive_count);
}

void FiniteElementMethod::Impl::_build_base_constitution_infos()
{
    auto build_infos = [&]<std::derived_from<FiniteElementConstitution> ConstitutionT>(
                           vector<ConstitutionInfo>& infos,
                           span<ConstitutionT*>      constitutions,
                           IndexT                    dim,
                           unordered_map<U64, SizeT> uid_to_index)
    {
        infos.resize(constitutions.size());

        OffsetCountCollection<SizeT> vertex_offsets_counts;
        vertex_offsets_counts.resize(infos.size());
        OffsetCountCollection<SizeT> primitive_offsets_counts;
        primitive_offsets_counts.resize(infos.size());
        OffsetCountCollection<SizeT> geometry_offsets_counts;
        geometry_offsets_counts.resize(infos.size());

        auto vertex_counts    = vertex_offsets_counts.counts();
        auto primitive_counts = primitive_offsets_counts.counts();
        auto geometry_counts  = geometry_offsets_counts.counts();

        const auto& dim_info = dim_infos[dim];

        auto geo_info_subspan =
            span{geo_infos}.subspan(dim_info.geo_info_offset, dim_info.geo_info_count);

        for(auto&& geo_info : geo_info_subspan)
        {
            auto index = uid_to_index[geo_info.dim_uid.uid];
            geometry_counts[index]++;
            vertex_counts[index] += geo_info.vertex_count;
            primitive_counts[index] += geo_info.primitive_count;
        }

        SizeT dim_geo_offset    = dim_info.geo_info_offset;
        SizeT dim_vertex_offset = dim_info.vertex_offset;

        vertex_offsets_counts.scan(dim_vertex_offset);
        primitive_offsets_counts.scan(0);
        geometry_offsets_counts.scan(dim_geo_offset);

        auto vertex_offsets    = vertex_offsets_counts.offsets();
        auto primitive_offsets = primitive_offsets_counts.offsets();
        auto geometry_offsets  = geometry_offsets_counts.offsets();

        for(auto&& [i, info] : enumerate(infos))
        {
            info.vertex_count     = vertex_counts[i];
            info.vertex_offset    = vertex_offsets[i];
            info.primitive_count  = primitive_counts[i];
            info.primitive_offset = primitive_offsets[i];
            info.geo_info_count   = geometry_counts[i];
            info.geo_info_offset  = geometry_offsets[i];
        }
        return 0;
    };


    build_infos(codim_0d_constitution_infos, span{codim_0d_constitutions}, 0, codim_0d_uid_to_index);
    build_infos(codim_1d_constitution_infos, span{codim_1d_constitutions}, 1, codim_1d_uid_to_index);
    build_infos(codim_2d_constitution_infos, span{codim_2d_constitutions}, 2, codim_2d_uid_to_index);
    build_infos(fem_3d_constitution_infos, span{fem_3d_constitutions}, 3, fem_3d_uid_to_index);
}

void FiniteElementMethod::Impl::_build_on_host(WorldVisitor& world)
{
    auto geo_slots      = world.scene().geometries();
    auto rest_geo_slots = world.scene().rest_geometries();

    // 1. Vertex Attributes
    {
        h_gravities.resize(h_positions.size(), default_gravity);
        h_rest_positions.resize(h_positions.size());
        h_velocities.resize(h_positions.size(), Vector3::Zero());  // fill 0 for default
        h_thicknesses.resize(h_positions.size(), 0);  // fill 0 for default
        h_dimensions.resize(h_positions.size(), 3);   // fill 3(D) for default
        h_masses.resize(h_positions.size());
        h_vertex_contact_element_ids.resize(h_positions.size(), 0);  // fill 0 for default
        h_vertex_subscene_contact_element_ids.resize(h_positions.size(), 0);  // fill 0 for default
        h_vertex_d_hat.resize(h_positions.size(), default_d_hat);  // fill default d_hat
        h_vertex_is_fixed.resize(h_positions.size(), 0);  // fill 0 for default, default non-fixed
        h_vertex_is_dynamic.resize(h_positions.size(), 1);  // fill 1 for default, default dynamic
        h_vertex_body_id.resize(h_positions.size(), -1);  // fill -1 for default, invalid body id


        for(auto&& [i, info] : enumerate(geo_infos))
        {
            auto& geo_slot      = geo_slots[info.geo_slot_index];
            auto& rest_geo_slot = rest_geo_slots[info.geo_slot_index];
            auto& geo           = geo_slot->geometry();
            auto& rest_geo      = rest_geo_slot->geometry();
            auto* sc            = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc,
                        "The geometry is not a simplicial complex (it's {}). Why can it happen?",
                        geo.type());
            auto* rest_sc = rest_geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(rest_sc,
                        "The geometry is not a simplicial complex (it's {}). Why can it happen?",
                        rest_geo.type());
            UIPC_ASSERT(sc->instances().size() == 1,
                        "Finite element geometry only supports one instance. But yours {} (Geo ID={})",
                        sc->instances().size(),
                        geo_slot->id());

            // 1) setup primitives
            switch(sc->dim())
            {
                case 0: {
                    auto dst_codim_0d_span =
                        span{h_codim_0ds}.subspan(info.primitive_offset, info.primitive_count);
                    std::iota(dst_codim_0d_span.begin(),
                              dst_codim_0d_span.end(),
                              static_cast<IndexT>(info.vertex_offset));
                }
                break;
                case 1: {
                    auto dst_codim_1d_span =
                        span{h_codim_1ds}.subspan(info.primitive_offset, info.primitive_count);

                    auto edge_view = sc->edges().topo().view();
                    UIPC_ASSERT(edge_view.size() == dst_codim_1d_span.size(),
                                "edge size mismatching");

                    std::transform(edge_view.begin(),
                                   edge_view.end(),
                                   dst_codim_1d_span.begin(),
                                   [&](const Vector2i& edge) -> Vector2i
                                   { return edge.array() + info.vertex_offset; });
                }
                break;
                case 2: {
                    auto dst_codim_2d_span =
                        span{h_codim_2ds}.subspan(info.primitive_offset, info.primitive_count);

                    auto tri_view = sc->triangles().topo().view();
                    UIPC_ASSERT(tri_view.size() == dst_codim_2d_span.size(),
                                "triangle size mismatching");

                    std::transform(tri_view.begin(),
                                   tri_view.end(),
                                   dst_codim_2d_span.begin(),
                                   [&](const Vector3i& tri) -> Vector3i
                                   { return tri.array() + info.vertex_offset; });
                }
                break;
                case 3: {
                    auto dst_tet_span =
                        span{h_tets}.subspan(info.primitive_offset, info.primitive_count);

                    auto tet_view = sc->tetrahedra().topo().view();
                    UIPC_ASSERT(tet_view.size() == dst_tet_span.size(),
                                "tetrahedra size mismatching");

                    std::transform(tet_view.begin(),
                                   tet_view.end(),
                                   dst_tet_span.begin(),
                                   [&](const Vector4i& tet) -> Vector4i
                                   { return tet.array() + info.vertex_offset; });
                }
                break;
                default:
                    break;
            }

            {  // 2) fill backend_fem_vertex_offset in geometry
                auto vertex_offset =
                    sc->meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                if(!vertex_offset)
                    vertex_offset =
                        sc->meta().create<IndexT>(builtin::backend_fem_vertex_offset, -1);
                auto vertex_offset_view = geometry::view(*vertex_offset);
                std::ranges::fill(vertex_offset_view,
                                  static_cast<IndexT>(info.vertex_offset));
            }

            {  // 3) setup positions and velocities
                auto ctransview = sc->transforms().view();
                if(!ctransview[0].isIdentity())
                {
                    auto transview = view(sc->transforms());
                    Transform t(transview[0]);
                    auto posview = view(sc->positions());
                    std::ranges::transform(posview,
                                           posview.begin(),
                                           [&](const Vector3& p) -> Vector3
                                           { return t * p; });

                    transview[0].setIdentity();

                    logger::warn(R"(FEM Geometry ID={} has non-identity transform. The transform is applied to the positions and reset to identity.
To avoid this warning, please apply the transform to the positions mannally. https://github.com/spiriMirror/libuipc/issues/152)",
                                 geo_slot->id());
                }

                auto pos_view = sc->positions().view();
                auto dst_pos_span =
                    span{h_positions}.subspan(info.vertex_offset, info.vertex_count);
                UIPC_ASSERT(pos_view.size() == dst_pos_span.size(), "position size mismatching");
                std::copy(pos_view.begin(), pos_view.end(), dst_pos_span.begin());

                auto rest_pos_view = rest_sc->positions().view();
                auto dst_rest_pos_span =
                    span{h_rest_positions}.subspan(info.vertex_offset, info.vertex_count);
                UIPC_ASSERT(rest_pos_view.size() == dst_rest_pos_span.size(),
                            "rest position size mismatching");
                std::ranges::copy(rest_pos_view, dst_rest_pos_span.begin());

                auto vel = sc->vertices().find<Vector3>(builtin::velocity);
                if(vel)  // if user set the velocity
                {
                    auto vel_view = vel->view();
                    auto dst_vel_span =
                        span{h_velocities}.subspan(info.vertex_offset, info.vertex_count);
                    UIPC_ASSERT(vel_view.size() == dst_vel_span.size(),
                                "velocity size mismatching");
                    std::ranges::copy(vel_view, dst_vel_span.begin());
                }
                // else, keep the default value (0)
            }

            {  // 4) setup mass
                auto volume      = rest_sc->vertices().find<Float>(builtin::volume);
                auto volume_view = volume->view();

                auto meta_mass_density   = sc->meta().find<Float>(builtin::mass_density);
                auto vertex_mass_density = sc->vertices().find<Float>(builtin::mass_density);
                UIPC_ASSERT(meta_mass_density || vertex_mass_density,
                            "mass density is not found in the geometry");
                auto mass_density_view = vertex_mass_density ?
                                             vertex_mass_density->view() :
                                             meta_mass_density->view();

                auto vert_mass = sc->vertices().find<Float>("mass");
                auto mass_view = vert_mass ? vert_mass->view() : span<const Float>{};


                auto dst_mass_span =
                    span{h_masses}.subspan(info.vertex_offset, info.vertex_count);
                UIPC_ASSERT(volume_view.size() == dst_mass_span.size(), "mass size mismatching");
                std::ranges::copy(volume_view, dst_mass_span.begin());

                for(auto&& [i, dst_vert_mass] : enumerate(dst_mass_span))
                {
                    if(vert_mass)
                    {
                        dst_vert_mass = mass_view[i];  // high priority
                    }
                    else
                    {
                        auto density = vertex_mass_density ? mass_density_view[i] :
                                                             mass_density_view[0];
                        dst_vert_mass = density * volume_view[i];
                    }
                }
            }

            {  // 5) setup thickness
                auto thickness = sc->vertices().find<Float>(builtin::thickness);
                auto dst_thickness_span =
                    span{h_thicknesses}.subspan(info.vertex_offset, info.vertex_count);

                if(thickness)
                {
                    auto thickness_view = thickness->view();
                    UIPC_ASSERT(thickness_view.size() == dst_thickness_span.size(),
                                "thickness size mismatching");
                    std::ranges::copy(thickness_view, dst_thickness_span.begin());
                }
            }

            {  // 6) setup vertex contact element id and d_hat

                auto dst_eid_span = span{h_vertex_contact_element_ids}.subspan(
                    info.vertex_offset, info.vertex_count);

                auto dst_subscene_eid_span =
                    span{h_vertex_subscene_contact_element_ids}.subspan(
                        info.vertex_offset, info.vertex_count);

                auto vert_ceid = sc->vertices().find<IndexT>(builtin::contact_element_id);

                if(vert_ceid)
                {
                    auto ceid_view = vert_ceid->view();
                    UIPC_ASSERT(ceid_view.size() == dst_eid_span.size(),
                                "contact element id size mismatching");

                    std::ranges::copy(ceid_view, dst_eid_span.begin());
                }
                else
                {
                    auto ceid = sc->meta().find<IndexT>(builtin::contact_element_id);

                    if(ceid)
                    {
                        auto eid = ceid->view()[0];
                        std::ranges::fill(dst_eid_span, eid);
                    }
                }

                auto vert_subscene_ceid =
                    sc->vertices().find<IndexT>(builtin::subscene_element_id);
                if(vert_subscene_ceid)
                {
                    auto subscene_ceid_view = vert_subscene_ceid->view();
                    UIPC_ASSERT(subscene_ceid_view.size()
                                    == dst_subscene_eid_span.size(),
                                "subscene contact element id size mismatching");

                    std::ranges::copy(subscene_ceid_view, dst_subscene_eid_span.begin());
                }
                else
                {
                    auto subscene_ceid =
                        sc->meta().find<IndexT>(builtin::subscene_element_id);

                    if(subscene_ceid)
                    {
                        auto eid = subscene_ceid->view()[0];
                        std::ranges::fill(dst_subscene_eid_span, eid);
                    }
                    else
                    {
                        std::ranges::fill(dst_subscene_eid_span, 0);
                    }
                }

                auto dst_d_hat_span =
                    span{h_vertex_d_hat}.subspan(info.vertex_offset, info.vertex_count);

                auto meta_d_hat = sc->meta().find<Float>(builtin::d_hat);
                if(meta_d_hat)
                {
                    auto d_hat_view = meta_d_hat->view();
                    std::ranges::fill(dst_d_hat_span, d_hat_view.front());
                }
                else
                {
                    std::ranges::fill(dst_d_hat_span, default_d_hat);
                }
            }

            {  // 7) setup vertex is_fixed

                auto is_fixed = sc->vertices().find<IndexT>(builtin::is_fixed);

                auto dst_is_fixed_span =
                    span{h_vertex_is_fixed}.subspan(info.vertex_offset, info.vertex_count);

                if(is_fixed)
                {
                    auto is_fixed_view = is_fixed->view();
                    UIPC_ASSERT(is_fixed_view.size() == dst_is_fixed_span.size(),
                                "is_fixed size mismatching");
                    std::ranges::copy(is_fixed_view, dst_is_fixed_span.begin());
                }
            }

            {  // 8) setup dimension
                auto dst_dim_span =
                    span{h_dimensions}.subspan(info.vertex_offset, info.vertex_count);
                std::ranges::fill(dst_dim_span, sc->dim());
            }

            {  // 9) setup vertex is_dynamic
                auto is_dynamic = sc->vertices().find<IndexT>(builtin::is_dynamic);
                auto dst_is_dynamic =
                    span{h_vertex_is_dynamic}.subspan(info.vertex_offset, info.vertex_count);

                if(is_dynamic)
                {
                    auto is_dynamic_view = is_dynamic->view();
                    UIPC_ASSERT(is_dynamic_view.size() == dst_is_dynamic.size(),
                                "is_kinematic size mismatching");
                    std::ranges::copy(is_dynamic_view, dst_is_dynamic.begin());
                }
            }

            {  // 10) setup vertex gravities

                auto gravity_attr = sc->vertices().find<Vector3>(builtin::gravity);
                auto dst_gravties =
                    span{h_gravities}.subspan(info.vertex_offset, info.vertex_count);

                if(gravity_attr)
                {
                    auto gravity_view = gravity_attr->view();
                    UIPC_ASSERT(gravity_view.size() == dst_gravties.size(),
                                "gravity size mismatching");
                    std::ranges::copy(gravity_view, dst_gravties.begin());
                }
            }

            {  // 11) setup vertex body id

                IndexT body_id = i;  // geo slot index is the body id
                auto dst_body_id_span =
                    span{h_vertex_body_id}.subspan(info.vertex_offset, info.vertex_count);
                std::ranges::fill(dst_body_id_span, body_id);
            }
        }
    }

    // 2. Body Attributes
    {
        h_body_self_collision.resize(geo_infos.size(), 1);  // fill 1 for default turn on self-collision

        for(auto&& [i, info] : enumerate(geo_infos))
        {
            auto& geo_slot = geo_slots[info.geo_slot_index];
            auto& geo      = geo_slot->geometry();
            auto* sc       = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT(sc,
                        "The geometry is not a simplicial complex (it's {}). Why can it happen?",
                        geo.type());

            {  // 1) setup body self-collision

                auto self_collision = sc->meta().find<IndexT>(builtin::self_collision);
                UIPC_ASSERT(self_collision, "self_collision is not found in finite element `meta`, why can it happen?");
                h_body_self_collision[i] = self_collision->view()[0];
            }
        }
    }
}

void FiniteElementMethod::Impl::_build_on_device()
{
    // 1) Vertex States
    upload_vec3(xs, span<const Vector3>{h_positions});
    upload_vec3(x_bars, span<const Vector3>{h_rest_positions});

    // The reference's `x_temps = xs;` deep-copies (cuda_tool::DeviceBuffer's copy
    // assignment calls copy_from); an LC LDeviceBuffer copy assignment would share
    // the allocation instead.
    copy_device(x_temps, xs);
    copy_device(x_tildes, xs);
    copy_device(x_prevs, xs);

    upload_scalar(is_fixed, span<const IndexT>{h_vertex_is_fixed});
    upload_scalar(is_dynamic, span<const IndexT>{h_vertex_is_dynamic});
    upload_vec3(gravities, span<const Vector3>{h_gravities});

    dxs.resize(xs.size());  // zero-initialised, like the reference's Vector3::Zero()
    upload_vec3(vs, span<const Vector3>{h_velocities});

    upload_scalar(masses, span<const Float>{h_masses});
    upload_scalar(thicknesses, span<const Float>{h_thicknesses});

    vertex_external_forces.resize(xs.size());      // zero
    vertex_external_force_accs.resize(xs.size());  // zero

    // 2) Elements
    upload_scalar(codim_0ds, span<const IndexT>{h_codim_0ds});

    upload_converted(codim_1ds, span<const Vector2i>{h_codim_1ds});
    rest_lengths.resize(codim_1ds.size());

    upload_converted(codim_2ds, span<const Vector3i>{h_codim_2ds});
    rest_areas.resize(codim_2ds.size());

    upload_converted(tets, span<const Vector4i>{h_tets});
    rest_volumes.resize(tets.size());

    // 3) Material Space Attribute
    auto& stream = frame_stream();

    // Rod
    {
        auto& k = build_on_device_k1();
        auto  n = static_cast<uint32_t>(codim_1ds.size());
        if(n > 0)
        {
            stream.record(k(codim_1ds.view(), x_bars.view(), rest_lengths.view(), n)
                              .dispatch(n));
        }
    }


    // Shell
    {
        auto& k = build_on_device_k2();
        auto  n = static_cast<uint32_t>(codim_2ds.size());
        if(n > 0)
        {
            stream.record(k(codim_2ds.view(), x_bars.view(), rest_areas.view(), n)
                              .dispatch(n));
        }
    }

    // FEM3D Material Basis
    Dm3x3_invs.resize(tets.size());
    {
        auto& k = build_on_device_k3();
        auto  n = static_cast<uint32_t>(tets.size());
        if(n > 0)
        {
            stream.record(k(tets.view(),
                            x_bars.view(),
                            Dm3x3_invs.view(),
                            rest_volumes.view(),
                            n)
                              .dispatch(n));
        }
    }
}

void FiniteElementMethod::Impl::_init_base_constitutions()
{
    for(auto&& [i, c] : enumerate(codim_0d_constitutions))
    {
        c->init();
    }

    for(auto&& [i, c] : enumerate(codim_1d_constitutions))
    {
        c->init();
    }

    for(auto&& [i, c] : enumerate(codim_2d_constitutions))
    {
        c->init();
    }

    for(auto&& [i, c] : enumerate(fem_3d_constitutions))
    {
        c->init();
    }
}

void FiniteElementMethod::Impl::_init_extra_constitutions()
{
    for(auto&& [i, c] : enumerate(extra_constitutions.view()))
    {
        c->init();
        auto uid = c->uid();
        extra_constitution_uid_to_index.insert({uid, i});
    }
}

void FiniteElementMethod::Impl::_download_geometry_to_host()
{
    download_vec3(span<Vector3>{h_positions}, xs.cview());
}

void FiniteElementMethod::Impl::write_scene(WorldVisitor& world)
{
    _download_geometry_to_host();

    auto geo_slots = world.scene().geometries();

    auto position_span = span{h_positions};

    for(auto&& [i, info] : enumerate(geo_infos))
    {
        auto& geo_slot = geo_slots[info.geo_slot_index];
        auto& geo      = geo_slot->geometry();
        auto* sc       = geo.as<geometry::SimplicialComplex>();
        UIPC_ASSERT(sc,
                    "The geometry is not a simplicial complex (it's {}). Why can it happen?",
                    geo.type());

        // 1) write positions back
        auto pos_view     = geometry::view(sc->positions());
        auto src_pos_span = position_span.subspan(info.vertex_offset, info.vertex_count);
        UIPC_ASSERT(pos_view.size() == src_pos_span.size(), "position size mismatching");
        std::copy(src_pos_span.begin(), src_pos_span.end(), pos_view.begin());

        // 2) write primitives back
        // TODO:
        // Now there is no topology modification, so no need to write back
        // In the future, we may need to write back the topology if the topology is modified
    }
}

// Dump & Recover:

bool FiniteElementMethod::Impl::dump(DumpInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_xs.dump(fmt::format("{}q.{}", path, frame), xs) //
           && dump_vs.dump(fmt::format("{}q_v.{}", path, frame), vs)  //
           && dump_x_prevs.dump(fmt::format("{}q_prev.{}", path, frame), x_prevs);  //
}

bool FiniteElementMethod::Impl::try_recover(RecoverInfo& info)
{
    auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
    auto frame = info.frame();

    return dump_xs.load(fmt::format("{}q.{}", path, frame)) //
           && dump_vs.load(fmt::format("{}q_v.{}", path, frame)) //
           && dump_x_prevs.load(fmt::format("{}q_prev.{}", path, frame));  //
}

void FiniteElementMethod::Impl::apply_recover(RecoverInfo& info)
{
    dump_xs.apply_to(xs);
    dump_vs.apply_to(vs);
    dump_x_prevs.apply_to(x_prevs);
}

void FiniteElementMethod::Impl::clear_recover(RecoverInfo& info)
{
    dump_xs.clean_up();
    dump_vs.clean_up();
    dump_x_prevs.clean_up();
}

void FiniteElementMethod::Impl::set_dof_info(SizeT frame, IndexT dof_offset, IndexT dof_count)
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

IndexT FiniteElementMethod::Impl::dof_offset(SizeT frame) const noexcept
{
    return frame_to_dof_offset[frame];
}

IndexT FiniteElementMethod::Impl::dof_count(SizeT frame) const noexcept
{
    return frame_to_dof_count[frame];
}


auto FiniteElementMethod::FilteredInfo::geo_infos() const noexcept -> span<const GeoInfo>
{
    auto info = this->constitution_info();
    return span{m_impl->geo_infos}.subspan(info.geo_info_offset, info.geo_info_count);
}

auto FiniteElementMethod::FilteredInfo::constitution_info() const noexcept -> const ConstitutionInfo&
{
    switch(m_dim)
    {
        case 0:
            return m_impl->codim_0d_constitution_infos[m_index_in_dim];
        case 1:
            return m_impl->codim_1d_constitution_infos[m_index_in_dim];
        case 2:
            return m_impl->codim_2d_constitution_infos[m_index_in_dim];
        case 3:
            return m_impl->fem_3d_constitution_infos[m_index_in_dim];
        default:
            UIPC_ASSERT(false, "Invalid dimension");
            return m_impl->codim_0d_constitution_infos[m_index_in_dim];  // dummy
    }
}

size_t FiniteElementMethod::FilteredInfo::vertex_count() const noexcept
{
    return constitution_info().vertex_count;
}

size_t FiniteElementMethod::FilteredInfo::primitive_count() const noexcept
{
    return constitution_info().primitive_count;
}

void FiniteElementMethod::overwrite_xs(lud::LCBufferView<LVector3> xs)
{
    if(xs.size() != 0u)
    {
        m_impl.xs.view().copy_from(xs);
    }
}
}  // namespace uipc::backend::luisa

#include <contact_system/adaptive_strategies/gipc_adaptive_parameter_strategy.h>
#include <uipc/builtin/constants.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <uipc/common/zip.h>
#include <uipc/common/enumerate.h>
#include <uipc/core/contact_tabular.h>
#include <contact_system/global_contact_manager.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>

/*************************************************************************************************
 * T8.1a — GIPCAdaptiveParameterStrategy (port of
 * src/backends/cuda/contact_system/adaptive_strategies/gipc_adaptive_parameter_strategy.cu,
 * 250 lines).  ref: https://github.com/KemengHuang/Stiff-GIPC
 *
 * Mapping notes (cookbook §3; see the header):
 *  * `do_init_kernel` / `do_compute_parameters_kernel` write one coefficient of a
 *    `DeviceBuffer2D<ContactCoeff>`; the port's tabular is *flat*, so each kernel takes
 *    the extent `N` as an extra scalar argument and indexes `x * N + y`.
 *  * `DeviceDenseVector::copy_to(std::vector&)` -> the port's explicit
 *    `read_back_vector` (the two host copies are dead in the reference too — they are
 *    transcribed because the reference has them, with the port's commit/synchronise
 *    discipline, §7.13 rule 3).
 *  * `linear_system->ctx().dot(...)` is `lla::LinearSystemContext::dot` (the same
 *    persistent reduction context, §7.12).
 *************************************************************************************************/
namespace uipc::backend::luisa
{
namespace
{
namespace lla = luisa_device::linalg;

/// The reference's `coefL.kappa = 1.0;` (mu preserved): read the element, overwrite one
/// member, write it back — LC has no in-place member store.
inline void write_coeff_kappa(lcrt::BufferVar<ContactCoeff>& buf,
                              const lcrt::Var<IndexT>&      idx,
                              const lcrt::Var<Float>&       kappa) noexcept
{
    lcrt::Var<ContactCoeff> c;
    c.kappa = kappa;
    c.mu    = buf->read(idx).mu;
    buf->write(idx, c);
}

/// `do_init_kernel` : the adaptive models' (and their symmetric) entries get kappa = 1.
auto& do_init_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,
                                   lcrt::Buffer<ContactCoeff>,
                                   std::int32_t,
                                   std::uint32_t>(
            [](lcrt::BufferVar<LVector2i> adaptive_topos,
               lcrt::BufferVar<ContactCoeff> contact_tabular,
               lcrt::Var<std::int32_t> N,
               lcrt::Var<std::uint32_t> n) noexcept
            {
                auto I = lcrt::dispatch_x();
                $if(I < n)
                {
                    auto topo = load_vec2i(adaptive_topos->read(I));
                    auto x    = topo(0);
                    auto y    = topo(1);
                    // reference: `coefL = contact_tabular(topo.x(), topo.y()); coefL.kappa = 1.0;`
                    write_coeff_kappa(contact_tabular, x * N + y, lcrt::Var<Float>{Float{1.0}});
                    // reference: `coefR = contact_tabular(topo.y(), topo.x()); coefR.kappa = 1.0;`
                    write_coeff_kappa(contact_tabular, y * N + x, lcrt::Var<Float>{Float{1.0}});
                };
            });
    return k;
}

/// `do_compute_parameters_kernel` : the adaptive models' entries get `new_kappa`.
auto& do_compute_parameters_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,
                                   lcrt::Buffer<ContactCoeff>,
                                   Float,
                                   std::int32_t,
                                   std::uint32_t>(
            [](lcrt::BufferVar<LVector2i> adaptive_topos,
               lcrt::BufferVar<ContactCoeff> contact_tabular,
               lcrt::Var<Float> new_kappa,
               lcrt::Var<std::int32_t> N,
               lcrt::Var<std::uint32_t> n) noexcept
            {
                auto I = lcrt::dispatch_x();
                $if(I < n)
                {
                    auto topo = load_vec2i(adaptive_topos->read(I));
                    auto x    = topo(0);
                    auto y    = topo(1);
                    write_coeff_kappa(contact_tabular, x * N + y, new_kappa);
                    write_coeff_kappa(contact_tabular, y * N + x, new_kappa);
                };
            });
    return k;
}

/// The reference's implicit synchronous device read (`copy_to(h_vector)`), with the
/// port's commit/synchronise discipline (cookbook §7.13 rule 3, global_vertex_manager's
/// `read_back`).
inline void read_back_vector(lud::LStream& stream, const lud::LCBufferView<Float>& src, std::vector<Float>& dst)
{
    dst.resize(src.size());
    if(dst.empty())
        return;
    const bool was_open = stream.frame_open();
    stream.synchronize();
    stream.stream() << src.native().copy_to(lcrt::span<Float>{dst.data(), dst.size()});
    stream.stream().synchronize();
    if(was_open)
    {
        stream.begin();
    }
}
}// namespace

REGISTER_SIM_SYSTEM(GIPCAdaptiveParameterStrategy);

void GIPCAdaptiveParameterStrategy::do_build(BuildInfo& info)
{
    auto scene = world().scene();
    {
        auto kappas         = scene.contact_tabular().contact_models().find<Float>("resistance");
        auto kappa_view     = kappas->view();
        auto found_min_kappa = std::ranges::min(kappa_view);
        if(found_min_kappa >= 0.0)
            throw SimSystemException{"No Adaptive Kappa is detected"};
    }

    contact_manager        = require<GlobalContactManager>();
    vertex_manager         = require<GlobalVertexManager>();
    surface_manager        = require<GlobalSimplicialSurfaceManager>();
    linear_system          = require<GlobalLinearSystem>();
    dytopo_effect_manager  = require<GlobalDyTopoEffectManager>();

    min_kappa  = scene.config().find<Float>("contact/adaptive/min_kappa")->view()[0];
    init_kappa = scene.config().find<Float>("contact/adaptive/init_kappa")->view()[0];
    max_kappa  = scene.config().find<Float>("contact/adaptive/max_kappa")->view()[0];

    on_write_scene([this] { write_scene(); });
}

void GIPCAdaptiveParameterStrategy::do_init(InitInfo& info)
{
    auto scene          = world().scene();
    auto N              = scene.contact_tabular().element_count();
    auto contact_models = scene.contact_tabular().contact_models();
    auto kappas         = contact_models.find<Float>("resistance");
    auto topos          = contact_models.find<Vector2i>("topo");

    auto kappa_view = kappas->view();
    auto topo_view  = topos->view();

    std::vector<LVector2i> h_adaptive_topos;
    h_adaptive_topos.reserve(topos->size());
    h_adaptive_kappa_index.reserve(topos->size());

    // copy if a contact model is adaptive
    for(auto&& [i, topo] : enumerate(topo_view))
    {
        if(kappa_view[i] < 0.0)
        {
            h_adaptive_topos.push_back(to_luisa(topo));
            h_adaptive_kappa_index.push_back(i);
        }
    }

    adaptive_topos.copy_from(span<const LVector2i>{h_adaptive_topos});

    // initialize test contact tabular
    // non-adaptive kappa to 0.0 (don't contribute)
    // adaptive kappa to 1.0 (contribute)
    test_contact_tabular = std::make_shared<lud::LDeviceBuffer<ContactCoeff>>();
    test_contact_tabular->resize(static_cast<size_t>(N) * static_cast<size_t>(N));
    test_contact_tabular->fill(ContactCoeff{0.0f});

    auto& stream = frame_stream();
    if(adaptive_topos.size() > 0)
    {
        auto k = &do_init_kernel();
        auto n = static_cast<std::uint32_t>(adaptive_topos.size());
        stream.record((*k)(adaptive_topos.view(),
                           test_contact_tabular->view(),
                           static_cast<std::int32_t>(N),
                           n)
                          .dispatch(n));
    }

    //auto d_hats = vertex_manager->d_hats();
    //DeviceVar<Float> min_d_hat;
    //DeviceReduce().Min(d_hats.data(), min_d_hat.data(), d_hats.size());
}

void GIPCAdaptiveParameterStrategy::compute_gradient(AdaptiveParameterInfo& info)
{
    // set a test contact tabular
    auto original_contact_tabular = info.exchange_contact_tabular(test_contact_tabular);

    auto _compute_gradient = [this](EnergyComponentFlags flags,
                                    lla::LDenseVectorView<Float> gradient)
    {
        // 1. Compute dytopo effect
        GlobalDyTopoEffectManager::ComputeDyTopoEffectInfo dytopo_effect_info;
        dytopo_effect_info.component_flags(flags);
        dytopo_effect_info.gradient_only(true);
        dytopo_effect_manager->compute_dytopo_effect(dytopo_effect_info);

        // 2. Compute the gradient
        GlobalLinearSystem::ComputeGradientInfo grad_info;
        grad_info.buffer_view(gradient);
        grad_info.flags(flags);
        linear_system->compute_gradient(grad_info);
    };

    contact_gradient.resize(linear_system->dof_count());
    non_contact_gradient.resize(linear_system->dof_count());

    // compute contact gradient
    _compute_gradient(EnergyComponentFlags::Contact, contact_gradient.view());
    // compute non-contact gradient
    _compute_gradient(EnergyComponentFlags::Complement, non_contact_gradient.view());

    // recover original contact tabular
    info.exchange_contact_tabular(original_contact_tabular);
}

void GIPCAdaptiveParameterStrategy::do_compute_parameters(AdaptiveParameterInfo& info)
{
    auto& stream = frame_stream();

    compute_gradient(info);

    std::vector<Float> h_contact_gradient;
    read_back_vector(stream, contact_gradient.cview().buffer_view(), h_contact_gradient);

    std::vector<Float> h_non_contact_gradient;
    read_back_vector(stream, non_contact_gradient.cview().buffer_view(), h_non_contact_gradient);

    auto& ctx = linear_system->ctx();

    Float contact2 = ctx.dot(contact_gradient.cview(), contact_gradient.cview());
    Float proj     = ctx.dot(contact_gradient.cview(), non_contact_gradient.cview());

    // 1. No contact contribution -> keep initial kappa
    Float proj_kappa = (contact2 == 0.0) ? init_kappa : -proj / contact2;

    // 2. Clamp to the scene-adaptive corridor when the manager computed
    // one (Stiff-GIPC suggestKappa/upperBoundKappa); config min/max
    // are only the fallback
    Float lo = contact_manager->kappa_lower_bound();
    Float hi = contact_manager->kappa_upper_bound();
    if(lo <= 0.0)
        lo = min_kappa;
    if(hi <= 0.0)
        hi = max_kappa;
    new_kappa = std::clamp(proj_kappa, lo, hi);

    logger::info(R"(Adaptive Contact Parameter: > Kappa = {:e}
* ProjKappa = {:e}, Contact^2 = {:e}, Contact.NonContact = {:e}
* LoBound = {:e}, InitKappa={:e}, HiBound = {:e})",
                 new_kappa,
                 proj_kappa,
                 contact2,
                 proj,
                 lo,
                 init_kappa,
                 hi);

    if(adaptive_topos.size() > 0)
    {
        auto k = &do_compute_parameters_kernel();
        auto n = static_cast<std::uint32_t>(adaptive_topos.size());
        stream.record((*k)(adaptive_topos.view(),
                           info.contact_tabular(),
                           new_kappa,
                           contact_manager->contact_tabular_size(),
                           n)
                          .dispatch(n));
    }
}

void GIPCAdaptiveParameterStrategy::write_scene()
{
    auto scene       = world().scene();
    auto kappa       = scene.contact_tabular().contact_models().find<Float>("resistance");
    auto kappa_view  = view(*kappa);
    for(auto index : h_adaptive_kappa_index)
        kappa_view[index] = new_kappa;
}
}  // namespace uipc::backend::luisa

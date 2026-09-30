#include <contact_system/global_contact_manager.h>

#include <contact_system/contact_reporter.h>

#include <contact_system/adaptive_contact_parameter_reporter.h>

#include <collision_detection/global_trajectory_filter.h>

#include <collision_detection/simplex_trajectory_filter.h>

#include <global_geometry/global_simplicial_surface_manager.h>

#include <finite_element/finite_element_method.h>

#include <affine_body/affine_body_dynamics.h>

#include <uipc/common/enumerate.h>

#include <uipc/common/zip.h>

#include <uipc/core/contact_tabular.h>

#include <uipc/core/subscene_tabular.h>

#include <utils/codim_thickness.h>

#include <luisa_device/math/distance/ccd.h>

#include <frame_stream.h>

#include <luisa_device/linear/algorithms.h>

// The test target's include path puts src/backends/none first, so the engine shell

// header must be qualified (cookbook §7.15 rule 7).

#include <backends/luisa/sim_engine.h>

#include <Eigen/Dense>

#include <algorithm>

#include <cmath>

#include <numeric>

#include <utility>

/*************************************************************************************************

 * T8.1a — GlobalContactManager, completed in place (the T7.2a M7 slice + everything the

 * eight `T8.1` scope markers deferred).

 *

 * Transcribed verbatim from src/backends/cuda/contact_system/global_contact_manager.cu:

 *  * the SimSystemCreator (gated on `contact/enable`), REGISTER_SIM_SYSTEM, do_build (now

 *    including the contact-coefficient default read);

 *  * Impl::init (the two scene-relative overrides, the scene-adaptive kappa corridor, the

 *    two tabulars, the vertex contact buffers and the reporter fan-out);

 *  * _build_contact_tabular (the mask half *and* the ContactCoeff table with its

 *    default-kappa policy and symmetric half) and _build_subscene_tabular;

 *  * the six device kernels (the two CFL norm kernels + the four feasible-step CCD

 *    kernels) and compute_d_hat / compute_adaptive_kappa / compute_cfl_condition /

 *    compute_feasible_step / compute_adaptive_parameters / the two add_reporter overloads

 *    / the accessors / AdaptiveParameterInfo.

 *

 * Type mappings (the batch's standard, see the header): the two DeviceBuffer2D become

 * flat LDeviceBuffer<T> + an extent; cub::DeviceReduce -> lla::device_reduce_min/max;

 * the implicit DeviceVar read-backs become lud::read_host_sync; distance::*_ccd ->

 * luisa_device::math::distance::*_ccd and *_thickness -> utils/codim_thickness.h.

 *************************************************************************************************/

namespace uipc::backend

{

template <>

class SimSystemCreator<luisa::GlobalContactManager>

{

  public:

    static U<luisa::GlobalContactManager> create(luisa::SimEngine& engine)

    {

        auto contact_enable_attr =

            engine.world().scene().config().find<IndexT>("contact/enable");

        bool contact_enable = contact_enable_attr->view()[0] != 0;

        if(contact_enable)

            return uipc::make_unique<luisa::GlobalContactManager>(engine);

        return nullptr;

    }

};

}// namespace uipc::backend



namespace uipc::backend::luisa

{

namespace

{

namespace lla  = luisa_device::linalg;

namespace ldist = luisa_device::math::distance;



/// The reference's four feasible-step kernels pass `max_iter = 1000`.  The port cannot

/// transcribe that literal: `luisa_device/math/distance/details/ccd.inl`'s `ccd_loop`

/// *unrolls* its iteration budget at AST-build time (the reference's `max_iter` is a

/// device value, the port's is a host constant), so 1000 would emit a 1000x unrolled

/// AST and blow the JIT up.  The port uses the same budget the ported trajectory

/// filters pass to the identical CCD call for the identical loop

/// (`details/simplex_trajectory_filter_kernels.h: kToiMaxIter = 16`); with eta = 0.2 the

/// loop's gap test fires within two iterations for every approaching pair, so the

/// returned `toc` is unchanged.  (Documented deviation, cookbook ### 7.35.)

constexpr int kFeasibleToiMaxIter = 16;



/// cuda_tool parity: a device read-back synchronises, and the recorded frame work must

/// be committed and completed first (cookbook §7.13 rule 3; global_vertex_manager's

/// `read_back`).

template <typename T>

void read_back_into(lud::LStream& stream, const lud::LBufferView<T>& src, std::span<T> dst)

{

    if(dst.empty())

        return;

    const bool was_open = stream.frame_open();

    stream.synchronize();

    stream.stream() << src.native().copy_to(dst);

    stream.stream().synchronize();

    if(was_open)

    {

        stream.begin();

    }

}



/// `compute_cfl_condition_kernel`: the norm of a *surface* vertex's displacement.

auto& compute_cfl_condition_kernel()

{

    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3>,

                                                lcrt::Buffer<Float>,

                                                lcrt::Buffer<IndexT>,

                                                std::uint32_t>(

        [](lcrt::BufferVar<LVector3> disps,

           lcrt::BufferVar<Float> disp_norms,

           lcrt::BufferVar<IndexT> surf_vertices,

           lcrt::Var<std::uint32_t> n) noexcept

        {

            auto i = lcrt::dispatch_x();

            $if(i < n)

            {

                // reference: `disp_norms(i) = disps(surf_vertices(i)).norm();`

                auto v  = load_vec3(disps->read(surf_vertices->read(i)));

                auto n2 = v(0) * v(0) + v(1) * v(1) + v(2) * v(2);

                disp_norms->write(i, lcrt::sqrt(n2));

            };

        });

    return k;

}



/// `compute_cfl_condition_all_kernel` (the no-surface-manager fallback).

auto& compute_cfl_condition_all_kernel()

{

    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3>,

                                                lcrt::Buffer<Float>,

                                                std::uint32_t>(

        [](lcrt::BufferVar<LVector3> disps,

           lcrt::BufferVar<Float> disp_norms,

           lcrt::Var<std::uint32_t> n) noexcept

        {

            auto i = lcrt::dispatch_x();

            $if(i < n)

            {

                auto v  = load_vec3(disps->read(i));

                auto n2 = v(0) * v(0) + v(1) * v(1) + v(2) * v(2);

                disp_norms->write(i, lcrt::sqrt(n2));

            };

        });

    return k;

}



/// `compute_feasible_step_PT_kernel`.

auto& compute_feasible_step_PT_kernel()

{

    static auto& k =

        lla::detail::cached_kernel<lcrt::Buffer<LVector4i>,

                                   lcrt::Buffer<LVector3>,

                                   lcrt::Buffer<LVector3>,

                                   lcrt::Buffer<Float>,

                                   lcrt::Buffer<Float>,

                                   Float,

                                   std::uint32_t>(

            [](lcrt::BufferVar<LVector4i> PTs,

               lcrt::BufferVar<LVector3> Ps,

               lcrt::BufferVar<LVector3> dxs,

               lcrt::BufferVar<Float> thicknesses,

               lcrt::BufferVar<Float> tois,

               lcrt::Var<Float> eta,

               lcrt::Var<std::uint32_t> n) noexcept

            {

                auto i = lcrt::dispatch_x();

                $if(i < n)

                {

                    auto PT = load_vec4i(PTs->read(i));

                    auto thickness = PT_thickness(thicknesses->read(PT(0)),

                                                  thicknesses->read(PT(1)),

                                                  thicknesses->read(PT(2)),

                                                  thicknesses->read(PT(3)));

                    lcrt::Var<Float> toi = lcrt::Var<Float>{Float{1.0}};

                    auto hit = ldist::point_triangle_ccd(load_vec3(Ps->read(PT(0))),

                                                         load_vec3(Ps->read(PT(1))),

                                                         load_vec3(Ps->read(PT(2))),

                                                         load_vec3(Ps->read(PT(3))),

                                                         load_vec3(dxs->read(PT(0))),

                                                         load_vec3(dxs->read(PT(1))),

                                                         load_vec3(dxs->read(PT(2))),

                                                         load_vec3(dxs->read(PT(3))),

                                                         eta,

                                                         thickness,

                                                         kFeasibleToiMaxIter,

                                                         toi);

                    tois->write(i, lcrt::select(Float{1.0}, toi, hit));

                };

            });

    return k;

}



/// `compute_feasible_step_EE_kernel`.

auto& compute_feasible_step_EE_kernel()

{

    static auto& k =

        lla::detail::cached_kernel<lcrt::Buffer<LVector4i>,

                                   lcrt::Buffer<LVector3>,

                                   lcrt::Buffer<LVector3>,

                                   lcrt::Buffer<Float>,

                                   lcrt::Buffer<Float>,

                                   Float,

                                   std::uint32_t>(

            [](lcrt::BufferVar<LVector4i> EEs,

               lcrt::BufferVar<LVector3> Ps,

               lcrt::BufferVar<LVector3> dxs,

               lcrt::BufferVar<Float> thicknesses,

               lcrt::BufferVar<Float> tois,

               lcrt::Var<Float> eta,

               lcrt::Var<std::uint32_t> n) noexcept

            {

                auto i = lcrt::dispatch_x();

                $if(i < n)

                {

                    auto EE = load_vec4i(EEs->read(i));

                    auto thickness = EE_thickness(thicknesses->read(EE(0)),

                                                  thicknesses->read(EE(1)),

                                                  thicknesses->read(EE(2)),

                                                  thicknesses->read(EE(3)));

                    lcrt::Var<Float> toi = lcrt::Var<Float>{Float{1.0}};

                    auto hit = ldist::edge_edge_ccd(load_vec3(Ps->read(EE(0))),

                                                    load_vec3(Ps->read(EE(1))),

                                                    load_vec3(Ps->read(EE(2))),

                                                    load_vec3(Ps->read(EE(3))),

                                                    load_vec3(dxs->read(EE(0))),

                                                    load_vec3(dxs->read(EE(1))),

                                                    load_vec3(dxs->read(EE(2))),

                                                    load_vec3(dxs->read(EE(3))),

                                                    eta,

                                                    thickness,

                                                    kFeasibleToiMaxIter,

                                                    toi);

                    tois->write(i, lcrt::select(Float{1.0}, toi, hit));

                };

            });

    return k;

}



/// `compute_feasible_step_PE_kernel`.

auto& compute_feasible_step_PE_kernel()

{

    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3i>,

                                                lcrt::Buffer<LVector3>,

                                                lcrt::Buffer<LVector3>,

                                                lcrt::Buffer<Float>,

                                                lcrt::Buffer<Float>,

                                                Float,

                                                std::uint32_t>(

        [](lcrt::BufferVar<LVector3i> PEs,

           lcrt::BufferVar<LVector3> Ps,

           lcrt::BufferVar<LVector3> dxs,

           lcrt::BufferVar<Float> thicknesses,

           lcrt::BufferVar<Float> tois,

           lcrt::Var<Float> eta,

           lcrt::Var<std::uint32_t> n) noexcept

        {

            auto i = lcrt::dispatch_x();

            $if(i < n)

            {

                auto PE = load_vec3i(PEs->read(i));

                auto thickness = PE_thickness(thicknesses->read(PE(0)),

                                              thicknesses->read(PE(1)),

                                              thicknesses->read(PE(2)));

                lcrt::Var<Float> toi = lcrt::Var<Float>{Float{1.0}};

                auto hit = ldist::point_edge_ccd(load_vec3(Ps->read(PE(0))),

                                                 load_vec3(Ps->read(PE(1))),

                                                 load_vec3(Ps->read(PE(2))),

                                                 load_vec3(dxs->read(PE(0))),

                                                 load_vec3(dxs->read(PE(1))),

                                                 load_vec3(dxs->read(PE(2))),

                                                 eta,

                                                 thickness,

                                                 kFeasibleToiMaxIter,

                                                 toi);

                tois->write(i, lcrt::select(Float{1.0}, toi, hit));

            };

        });

    return k;

}



/// `compute_feasible_step_PP_kernel`.

auto& compute_feasible_step_PP_kernel()

{

    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector2i>,

                                                lcrt::Buffer<LVector3>,

                                                lcrt::Buffer<LVector3>,

                                                lcrt::Buffer<Float>,

                                                lcrt::Buffer<Float>,

                                                Float,

                                                std::uint32_t>(

        [](lcrt::BufferVar<LVector2i> PPs,

           lcrt::BufferVar<LVector3> Ps,

           lcrt::BufferVar<LVector3> dxs,

           lcrt::BufferVar<Float> thicknesses,

           lcrt::BufferVar<Float> tois,

           lcrt::Var<Float> eta,

           lcrt::Var<std::uint32_t> n) noexcept

        {

            auto i = lcrt::dispatch_x();

            $if(i < n)

            {

                auto PP = load_vec2i(PPs->read(i));

                auto thickness = PP_thickness(thicknesses->read(PP(0)), thicknesses->read(PP(1)));

                lcrt::Var<Float> toi = lcrt::Var<Float>{Float{1.0}};

                auto hit = ldist::point_point_ccd(load_vec3(Ps->read(PP(0))),

                                                  load_vec3(Ps->read(PP(1))),

                                                  load_vec3(dxs->read(PP(0))),

                                                  load_vec3(dxs->read(PP(1))),

                                                  eta,

                                                  thickness,

                                                  kFeasibleToiMaxIter,

                                                  toi);

                tois->write(i, lcrt::select(Float{1.0}, toi, hit));

            };

        });

    return k;

}

}// namespace



REGISTER_SIM_SYSTEM(GlobalContactManager);



void GlobalContactManager::do_build()

{

    const auto& config = world().scene().config();



    m_impl.global_vertex_manager   = require<GlobalVertexManager>();

    m_impl.global_trajectory_filter = find<GlobalTrajectoryFilter>();

    m_impl.global_surface_manager  = find<GlobalSimplicialSurfaceManager>();

    m_impl.finite_element_method   = find<FiniteElementMethod>();

    m_impl.affine_body_dynamics    = find<AffineBodyDynamics>();



    auto d_hat_attr = config.find<Float>("contact/d_hat");

    m_impl.d_hat    = d_hat_attr->view()[0];



    m_impl.dt_attr = config.find<Float>("dt");

    UIPC_ASSERT(m_impl.dt_attr, "Scene config must have a 'dt' attribute.");



    auto eps_velocity_attr = config.find<Float>("contact/eps_velocity");

    m_impl.eps_velocity    = eps_velocity_attr->view()[0];



    auto cfl_enable_attr = config.find<IndexT>("cfl/enable");

    m_impl.cfl_enabled   = cfl_enable_attr->view()[0] != 0;



    m_impl.kappa = world().scene().contact_tabular().default_model().resistance();

}



lud::LCBufferView<ContactCoeff> GlobalContactManager::contact_tabular() const noexcept

{

    return m_impl.contact_tabular->cview();

}



lud::LCBufferView<IndexT> GlobalContactManager::contact_mask_tabular() const noexcept

{

    return m_impl.contact_mask_tabular.cview();

}



lud::LCBufferView<IndexT> GlobalContactManager::subscene_mask_tabular() const noexcept

{

    return m_impl.subscene_mask_tabular.cview();

}



IndexT GlobalContactManager::contact_tabular_size() const noexcept

{

    return m_impl.contact_tabular_size;

}



IndexT GlobalContactManager::contact_mask_size() const noexcept

{

    return m_impl.contact_mask_size;

}



IndexT GlobalContactManager::subscene_mask_size() const noexcept

{

    return m_impl.subscene_mask_size;

}



void GlobalContactManager::init()

{

    m_impl.init(world());

}



namespace

{

/// The reference's `using MaskMatrix = Eigen::Matrix<IndexT, Dynamic, Dynamic, RowMajor>;`

/// (the flat mask layout is row-major `row * N + col`).

using MaskMatrix = Eigen::Matrix<IndexT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}// namespace



void GlobalContactManager::Impl::init(WorldVisitor& world)

{

    // 0) scene-relative d_hat override: if contact/d_hat_relative > 0,

    // the gap is relative_dhat * scene diagonal (Stiff-GIPC convention)

    {

        auto  rel_attr = world.scene().config().find<Float>("contact/d_hat_relative");

        Float rel      = rel_attr ? rel_attr->view()[0] : 0.0;

        if(rel > 0.0)

        {

            Float diag = global_vertex_manager->scene_diagonal();

            d_hat      = rel * diag;

            logger::info("Contact d_hat (relative): {} = {} x scene_diagonal({})",

                         d_hat,

                         rel,

                         diag);

        }

    }



    // 0.1) scene-relative friction eps_velocity override:

    // eps_velocity = relative * scene diagonal (Stiff-GIPC convention:

    // their per-step slip threshold is sqrt(fDhat)*dt = 1e-2*diag*dt,

    // and our eps_vh = eps_velocity*dt, so eps_velocity = 1e-2*diag)

    {

        auto  rel_attr = world.scene().config().find<Float>("contact/eps_velocity_relative");

        Float rel      = rel_attr ? rel_attr->view()[0] : 0.0;

        if(rel > 0.0)

        {

            Float diag   = global_vertex_manager->scene_diagonal();

            eps_velocity = rel * diag;

            logger::info("Contact eps_velocity (relative): {} = {} x scene_diagonal({})",

                         eps_velocity,

                         rel,

                         diag);

        }

    }



    // 0.2) Scene-adaptive kappa corridor: port of Stiff-GIPC's

    // suggestKappa/upperBoundKappa (GIPC.cu:9850-9888). Their barrier

    // applies kappa raw while ours is scaled by dt^2, so the computed

    // values are divided by dt^2 to stay in our convention.

    {

        kappa_lower_bound = -1.0;

        kappa_upper_bound = -1.0;



        Float diag      = global_vertex_manager->scene_diagonal();

        SizeT vert_count = global_vertex_manager->positions().size();

        if(diag > 0.0 && vert_count > 0)

        {

            // meanMass = total scene mass / total vertex count (Stiff

            // convention: FEM vertex masses directly; ABD bodies contribute

            // their scalar body mass once)

            Float total_mass = 0.0;

            if(finite_element_method)

            {

                auto fem_masses = finite_element_method->masses();

                std::vector<Float> h_masses(fem_masses.size());

                read_back_into(frame_stream(), fem_masses, span<Float>{h_masses});

                total_mass = std::accumulate(h_masses.begin(), h_masses.end(), Float{0});

            }

            if(affine_body_dynamics)

            {

                auto body_masses = affine_body_dynamics->body_masses();

                std::vector<ABDJacobiDyadicMass> h_body_masses(body_masses.size());

                read_back_into(frame_stream(),

                               body_masses,

                               span<ABDJacobiDyadicMass>{h_body_masses});

                for(auto& bm : h_body_masses)

                    total_mass += static_cast<Float>(bm.m_mass);

            }

            Float mean_mass = total_mass / vert_count;



            Float scale = world.scene()

                              .config()

                              .find<Float>("contact/adaptive/kappa_eval_scale")

                              ->view()[0];

            constexpr Float min_kappa_coef = 1e11;  // Stiff's minKappaCoef

            Float dt                       = dt_attr->view()[0];

            Float d                        = scale * diag * diag;

            Float dHat                     = d_hat * d_hat;  // Stiff's dHat is a squared distance

            if(d > 0.0 && dHat > 0.0 && dt > 0.0)

            {

                Float t   = d - dHat;

                Float H_b = -2.0 * std::log(d / dHat) - 4.0 * t / d + (t * t) / (d * d);

                if(std::isfinite(H_b) && H_b > 0.0)

                {

                    Float base = min_kappa_coef / (4.0 * d * H_b);

                    if(mean_mass > 0.0)

                        base *= mean_mass;

                    kappa_lower_bound = base / (dt * dt);

                    kappa_upper_bound = 100.0 * kappa_lower_bound;

                }

            }

            logger::info(

                "Adaptive kappa corridor: [{:e}, {:e}] "

                "(meanMass={:e}, diag={}, d_hat={}, scale={:e}, dt={})",

                kappa_lower_bound,

                kappa_upper_bound,

                mean_mass,

                diag,

                d_hat,

                scale,

                dt);

        }

    }



    // 1) init tabular

    _build_contact_tabular(world);

    _build_subscene_tabular(world);



    // 2) vertex contact info

    vert_is_active_contact.resize(global_vertex_manager->positions().size());

    vert_is_active_contact.fill(0);

    vert_disp_norms.resize(global_vertex_manager->positions().size());

    vert_disp_norms.fill(0.0);



    // 3) reporters

    auto contact_reporter_view = contact_reporters.view();

    for(auto&& [i, R] : enumerate(contact_reporter_view))

        R->init();

    for(auto&& [i, R] : enumerate(contact_reporter_view))

        R->m_index = i;

    if(adaptive_contact_parameter_reporter)

    {

        adaptive_contact_parameter_reporter->init();

    }

}



void GlobalContactManager::Impl::_build_contact_tabular(WorldVisitor& world)

{

    // Specification:

    // https://spirimirror.github.io/libuipc-doc/specification/#contact-tabular



    auto contact_models = world.scene().contact_tabular().contact_models();



    auto attr_topo          = contact_models.find<Vector2i>("topo");

    auto attr_resistance    = contact_models.find<Float>("resistance");

    auto attr_friction_rate = contact_models.find<Float>("friction_rate");

    auto attr_enabled       = contact_models.find<IndexT>("is_enabled");



    UIPC_ASSERT(attr_topo != nullptr, "topo is not found in contact tabular");

    UIPC_ASSERT(attr_resistance != nullptr, "resistance is not found in contact tabular");

    UIPC_ASSERT(attr_friction_rate != nullptr, "friction_rate is not found in contact tabular");

    UIPC_ASSERT(attr_enabled != nullptr, "is_enabled is not found in contact tabular");



    auto topo_view          = attr_topo->view();

    auto resistance_view    = attr_resistance->view();

    auto friction_rate_view = attr_friction_rate->view();

    auto enabled_view       = attr_enabled->view();



    auto N = world.scene().contact_tabular().element_count();



    // if no contact model is defined, then take the default one

    // so it can be on or off, depending on the default_model setting

    h_contact_mask_tabular.resize(N * N, enabled_view[0]);



    auto mask_map = Eigen::Map<MaskMatrix>(h_contact_mask_tabular.data(), N, N);



    // Default contact stiffness policy:

    //  - the scene-adaptive corridor [kappa_lower_bound, kappa_upper_bound]

    // (Stiff-GIPC suggestKappa/upperBoundKappa, computed in init) rules;

    //  - when it is not computable for the scene, fall back to the

    // contact/adaptive/{min,max}_kappa config values;

    //  - negative kappa (adaptive-kappa opt-in) is never clamped

    auto& scene_cfg = world.scene().config();

    Float min_kappa = scene_cfg.find<Float>("contact/adaptive/min_kappa")->view()[0];

    Float max_kappa = scene_cfg.find<Float>("contact/adaptive/max_kappa")->view()[0];

    if(kappa_lower_bound > 0.0)

        min_kappa = kappa_lower_bound;

    if(kappa_upper_bound > 0.0)

        max_kappa = kappa_upper_bound;



    Float default_kappa = resistance_view[0];

    if(!world.scene().contact_tabular().default_model_is_user_set())

    {

        default_kappa = min_kappa;

        logger::info("Contact default kappa not set by user; using the kappa lower bound = {}",

                     min_kappa);

    }

    else if(default_kappa >= 0.0 && (default_kappa < min_kappa || default_kappa > max_kappa))

    {

        logger::warn("Contact default kappa {} is clamped to [{}] (valid range: [{}, {}])",

                     default_kappa,

                     std::clamp(default_kappa, min_kappa, max_kappa),

                     min_kappa,

                     max_kappa);

        default_kappa = std::clamp(default_kappa, min_kappa, max_kappa);

    }



    h_contact_tabular.resize(

        N * N, ContactCoeff{.kappa = default_kappa, .mu = friction_rate_view[0]});



    // set the defined contact model

    for(SizeT i = 0; i < topo_view.size(); ++i)

    {

        auto  ids        = topo_view[i];

        Float kappa      = resistance_view[i];

        Float mu         = friction_rate_view[i];

        auto& is_enabled = enabled_view[i];



        // entry 0 is the default model: use the policy-resolved default kappa

        Float kk = (i == 0) ? default_kappa : kappa;

        if(kk >= 0.0 && (kk < min_kappa || kk > max_kappa))

        {

            Float clamped = std::clamp(kk, min_kappa, max_kappa);

            if(i != 0)

                logger::warn(

                    "Contact kappa {} for model ({},{}) is clamped to {} "

                    "(valid range: [{}, {}])",

                    kk,

                    ids.x(),

                    ids.y(),

                    clamped,

                    min_kappa,

                    max_kappa);

            kk = clamped;

        }



        ContactCoeff coeff{.kappa = kk, .mu = mu};



        auto upper               = ids.x() * N + ids.y();

        h_contact_tabular[upper] = coeff;

        mask_map(ids.x(), ids.y()) = is_enabled;



        auto lower               = ids.y() * N + ids.x();

        h_contact_tabular[lower] = coeff;

        mask_map(ids.y(), ids.x()) = is_enabled;

    }



    contact_tabular = std::make_shared<lud::LDeviceBuffer<ContactCoeff>>();



    // the reference's `resize(Extent2D{N, N})` on a 2D buffer == N*N flat elements

    contact_tabular->resize(N * N);

    contact_tabular->view().copy_from(span<const ContactCoeff>{h_contact_tabular});

    contact_tabular_size = static_cast<IndexT>(N);



    contact_mask_size = static_cast<IndexT>(N);

    contact_mask_tabular.resize(N * N);

    contact_mask_tabular.view().copy_from(span<const IndexT>{h_contact_mask_tabular});

}



void GlobalContactManager::Impl::_build_subscene_tabular(WorldVisitor& world)

{

    // Specification:

    // https://spirimirror.github.io/libuipc-doc/specification/#subscene-tabular



    auto subscene_models = world.scene().subscene_tabular().subscene_models();



    auto topo       = subscene_models.find<Vector2i>("topo");

    auto is_enabled = subscene_models.find<IndexT>("is_enabled");



    UIPC_ASSERT(topo != nullptr, "subscene topo is not found in contact tabular");

    UIPC_ASSERT(is_enabled != nullptr, "subscene is_enabled is not found in contact tabular");



    auto topo_view    = topo->view();

    auto enabled_view = is_enabled->view();

    auto SN           = world.scene().subscene_tabular().element_count();



    h_subcene_mask_tabular.resize(SN * SN);

    auto mask_map = Eigen::Map<MaskMatrix>(h_subcene_mask_tabular.data(), SN, SN);



    // According to Specification:

    // 1. default turn off the contact between two different subscenes

    // 2. enable self-scene-contact

    mask_map.setIdentity();



    for(auto&& [ids, enabled] : zip(topo_view, enabled_view))

    {

        mask_map(ids.x(), ids.y()) = enabled;

        mask_map(ids.y(), ids.x()) = enabled;

    }



    subscene_mask_size = static_cast<IndexT>(SN);

    subscene_mask_tabular.resize(SN * SN);

    subscene_mask_tabular.view().copy_from(span<const IndexT>{h_subcene_mask_tabular});

}



void GlobalContactManager::Impl::compute_d_hat()

{

    // TODO: Now do nothing

}



void GlobalContactManager::Impl::compute_adaptive_kappa()

{

    // TODO: Now do nothing

}



Float GlobalContactManager::Impl::compute_cfl_condition()

{

    if(!cfl_enabled)  // if cfl is disabled, just return 1.0

        return 1.0;



    if(global_trajectory_filter)

    {

        auto displacements = global_vertex_manager->displacements();



        if(displacements.size() > 0)

        {

            auto& stream = frame_stream();

            if(global_surface_manager)

            {

                // Stiff-GIPC design: max |dx| over all *surface* vertices

                auto surf_verts = global_surface_manager->surf_vertices();

                auto n          = static_cast<std::uint32_t>(surf_verts.size());

                vert_disp_norms.resize(n);

                if(n > 0)

                {

                    auto k = &compute_cfl_condition_kernel();

                    stream.record((*k)(displacements.cview(),

                                       vert_disp_norms.view(),

                                       surf_verts,

                                       n)

                                      .dispatch(n));

                }

            }

            else

            {

                // fallback: max |dx| over all vertices

                auto n = static_cast<std::uint32_t>(displacements.size());

                vert_disp_norms.resize(n);

                auto k = &compute_cfl_condition_all_kernel();

                stream.record((*k)(displacements.cview(), vert_disp_norms.view(), n).dispatch(n));

            }

        }



        lla::device_reduce_max(frame_stream(),

                               vert_disp_norms.cview(),

                               max_disp_norm.view(),

                               vert_disp_norms.size());



        Float h_max_disp_norm = lud::read_host_sync(frame_stream(), max_disp_norm.view());

        return h_max_disp_norm == 0.0 ? 1.0 : std::min(0.5 * d_hat / h_max_disp_norm, 1.0);

    }

    else

    {

        return 1.0;

    }

}



Float GlobalContactManager::Impl::compute_feasible_step()

{

    if(!global_trajectory_filter)

        return 1.0;



    auto simplex = global_trajectory_filter->find<SimplexTrajectoryFilter>();

    if(!simplex)

        return 1.0;



    // Stiff-GIPC slackness: keep >= 20% of every active pair's current gap

    constexpr Float eta = 0.2;



    auto positions     = global_vertex_manager->positions();

    auto displacements = global_vertex_manager->displacements();

    auto thicknesses   = global_vertex_manager->thicknesses();



    auto PTs = simplex->PTs();

    auto EEs = simplex->EEs();

    auto PEs = simplex->PEs();

    auto PPs = simplex->PPs();



    SizeT n_pt = PTs.size(), n_ee = EEs.size(), n_pe = PEs.size(), n_pp = PPs.size();

    SizeT total = n_pt + n_ee + n_pe + n_pp;

    if(total == 0)

        return 1.0;



    auto& stream = frame_stream();

    feasible_tois.resize(total);

    SizeT off = 0;

    // PT

    if(n_pt > 0)

    {

        auto k = &compute_feasible_step_PT_kernel();

        auto n = static_cast<std::uint32_t>(n_pt);

        stream.record((*k)(PTs,

                           positions,

                           displacements,

                           thicknesses,

                           feasible_tois.view(off, n_pt),

                           eta,

                           n)

                          .dispatch(n));

    }

    off += n_pt;

    // EE

    if(n_ee > 0)

    {

        auto k = &compute_feasible_step_EE_kernel();

        auto n = static_cast<std::uint32_t>(n_ee);

        stream.record((*k)(EEs,

                           positions,

                           displacements,

                           thicknesses,

                           feasible_tois.view(off, n_ee),

                           eta,

                           n)

                          .dispatch(n));

    }

    off += n_ee;

    // PE

    if(n_pe > 0)

    {

        auto k = &compute_feasible_step_PE_kernel();

        auto n = static_cast<std::uint32_t>(n_pe);

        stream.record((*k)(PEs,

                           positions,

                           displacements,

                           thicknesses,

                           feasible_tois.view(off, n_pe),

                           eta,

                           n)

                          .dispatch(n));

    }

    off += n_pe;

    // PP

    if(n_pp > 0)

    {

        auto k = &compute_feasible_step_PP_kernel();

        auto n = static_cast<std::uint32_t>(n_pp);

        stream.record((*k)(PPs,

                           positions,

                           displacements,

                           thicknesses,

                           feasible_tois.view(off, n_pp),

                           eta,

                           n)

                          .dispatch(n));

    }



    lla::device_reduce_min(stream,

                           feasible_tois.cview(),

                           min_feasible_toi.view(),

                           total);



    Float h_min = lud::read_host_sync(stream, min_feasible_toi.view());

    return h_min < 1.0 ? h_min : 1.0;

}



Float GlobalContactManager::compute_cfl_condition()

{

    return m_impl.compute_cfl_condition();

}



Float GlobalContactManager::compute_feasible_step()

{

    return m_impl.compute_feasible_step();

}



void GlobalContactManager::compute_adaptive_parameters()

{

    if(!m_impl.adaptive_contact_parameter_reporter)

        return;



    auto info = AdaptiveParameterInfo(&m_impl);

    m_impl.adaptive_contact_parameter_reporter->compute_parameters(info);

}



Float GlobalContactManager::d_hat() const

{

    return m_impl.d_hat;

}



Float GlobalContactManager::eps_velocity() const

{

    return m_impl.eps_velocity;

}



Float GlobalContactManager::kappa_lower_bound() const

{

    return m_impl.kappa_lower_bound;

}



Float GlobalContactManager::kappa_upper_bound() const

{

    return m_impl.kappa_upper_bound;

}



bool GlobalContactManager::cfl_enabled() const

{

    return m_impl.cfl_enabled;

}



void GlobalContactManager::add_reporter(ContactReporter* reporter)

{

    check_state(SimEngineState::BuildSystems, "add_reporter()");

    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");

    m_impl.contact_reporters.register_sim_system(*reporter);

}



void GlobalContactManager::add_reporter(AdaptiveContactParameterReporter* reporter)

{

    check_state(SimEngineState::BuildSystems, "add_reporter()");

    UIPC_ASSERT(!m_impl.adaptive_contact_parameter_reporter,

                "AdaptiveContactParameterReporter is already registered, name: {}",

                m_impl.adaptive_contact_parameter_reporter->name());

    UIPC_ASSERT(reporter != nullptr, "reporter is nullptr");

    m_impl.adaptive_contact_parameter_reporter.register_sim_system(*reporter);

}



lud::LBufferView<ContactCoeff> GlobalContactManager::AdaptiveParameterInfo::contact_tabular() const noexcept

{

    return m_impl->contact_tabular->view();

}



S<lud::LDeviceBuffer<ContactCoeff>> GlobalContactManager::AdaptiveParameterInfo::exchange_contact_tabular(

    S<lud::LDeviceBuffer<ContactCoeff>> new_buffer) const noexcept

{

    return std::exchange(m_impl->contact_tabular, new_buffer);

}



}  // namespace uipc::backend::luisa


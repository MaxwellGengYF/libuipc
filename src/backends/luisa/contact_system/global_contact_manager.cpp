#include <contact_system/global_contact_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <finite_element/finite_element_method.h>
#include <affine_body/affine_body_dynamics.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/zip.h>
#include <uipc/core/contact_tabular.h>
#include <uipc/core/subscene_tabular.h>
// The test target's include path puts src/backends/none first, so the engine shell
// header must be qualified (cookbook §7.15 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
/*************************************************************************************************
 * T7.2a — GlobalContactManager, the M7 slice (see the header's SCOPE NOTE (T8.1)).
 *
 * Transcribed verbatim from src/backends/cuda/contact_system/global_contact_manager.cu:
 *  * the SimSystemCreator (gated on `contact/enable`) and REGISTER_SIM_SYSTEM;
 *  * `do_build()` minus the contact-coefficient read (`kappa =
 *    contact_tabular().default_model().resistance()`, a `ContactCoeff` quantity);
 *  * `Impl::init()`'s step 0 (the two scene-relative overrides), step 1 (the two
 *    tabulars) and step 2 (the vertex contact buffers) minus the kappa corridor
 *    (step 0.2, which only feeds the contact coefficients) and the reporter fan-out
 *    (step 3, which needs the un-ported ContactReporter/AdaptiveContactParameterReporter);
 *  * `_build_contact_tabular()`'s **mask half** (the `mask_map` writes) and
 *    `_build_subscene_tabular()` in full.
 *
 * The only non-mechanical translations are the two documented type mappings: the two
 * `DeviceBuffer2D<IndexT>` masks become flat `LDeviceBuffer<IndexT>` + an extent
 * (the port's device layer has no 2D buffer), and `resize(n, v)` becomes
 * `resize(n)` + `fill(v)`.
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
REGISTER_SIM_SYSTEM(GlobalContactManager);

void GlobalContactManager::do_build()
{
    const auto& config = world().scene().config();

    m_impl.global_vertex_manager    = require<GlobalVertexManager>();
    m_impl.global_trajectory_filter = find<GlobalTrajectoryFilter>();
    m_impl.global_surface_manager   = find<GlobalSimplicialSurfaceManager>();
    m_impl.finite_element_method    = find<FiniteElementMethod>();
    m_impl.affine_body_dynamics     = find<AffineBodyDynamics>();

    auto d_hat_attr = config.find<Float>("contact/d_hat");
    m_impl.d_hat    = d_hat_attr->view()[0];

    m_impl.dt_attr = config.find<Float>("dt");
    UIPC_ASSERT(m_impl.dt_attr, "Scene config must have a 'dt' attribute.");

    auto eps_velocity_attr = config.find<Float>("contact/eps_velocity");
    m_impl.eps_velocity    = eps_velocity_attr->view()[0];

    auto cfl_enable_attr = config.find<IndexT>("cfl/enable");
    m_impl.cfl_enabled   = cfl_enable_attr->view()[0] != 0;

    // SCOPE NOTE (T8.1): the reference also reads the contact-coefficient default
    // (`m_impl.kappa = world().scene().contact_tabular().default_model().resistance();`).
}

Float GlobalContactManager::d_hat() const
{
    return m_impl.d_hat;
}

Float GlobalContactManager::eps_velocity() const
{
    return m_impl.eps_velocity;
}

bool GlobalContactManager::cfl_enabled() const
{
    return m_impl.cfl_enabled;
}

lud::LCBufferView<IndexT> GlobalContactManager::contact_mask_tabular() const noexcept
{
    return m_impl.contact_mask_tabular.cview();
}

lud::LCBufferView<IndexT> GlobalContactManager::subscene_mask_tabular() const noexcept
{
    return m_impl.subscene_mask_tabular.cview();
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
    //    the gap is relative_dhat * scene diagonal (Stiff-GIPC convention)
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
    //    eps_velocity = relative * scene diagonal (Stiff-GIPC convention:
    //    their per-step slip threshold is sqrt(fDhat)*dt = 1e-2*diag*dt,
    //    and our eps_vh = eps_velocity*dt, so eps_velocity = 1e-2*diag)
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

    // SCOPE NOTE (T8.1): the reference's step 0.2 (the scene-adaptive kappa corridor,
    // Stiff-GIPC's suggestKappa/upperBoundKappa) belongs to the contact-coefficient
    // half and is not needed by the trajectory filters.

    // 1) init tabular
    _build_contact_tabular(world);
    _build_subscene_tabular(world);

    // 2) vertex contact info
    vert_is_active_contact.resize(global_vertex_manager->positions().size());
    vert_is_active_contact.fill(0);

    // SCOPE NOTE (T8.1): the reference's step 2 also resizes `vert_disp_norms`, and
    // step 3 fans out to the contact reporters / the adaptive parameter reporter.
    // Neither exists in the port yet (T8.1).
}

void GlobalContactManager::Impl::_build_contact_tabular(WorldVisitor& world)
{
    // Specification:
    // https://spirimirror.github.io/libuipc-doc/specification/#contact-tabular

    auto contact_models = world.scene().contact_tabular().contact_models();

    auto attr_topo    = contact_models.find<Vector2i>("topo");
    auto attr_enabled = contact_models.find<IndexT>("is_enabled");

    UIPC_ASSERT(attr_topo != nullptr, "topo is not found in contact tabular");
    UIPC_ASSERT(attr_enabled != nullptr, "is_enabled is not found in contact tabular");

    auto topo_view    = attr_topo->view();
    auto enabled_view = attr_enabled->view();

    auto N = world.scene().contact_tabular().element_count();

    // if no contact model is defined, then take the default one
    // so it can be on or off, depending on the default_model setting
    h_contact_mask_tabular.resize(N * N, enabled_view[0]);

    auto mask_map = Eigen::Map<MaskMatrix>(h_contact_mask_tabular.data(), N, N);

    // SCOPE NOTE (T8.1): the reference's remaining body of this function builds
    // `h_contact_tabular` / `contact_tabular` (the `ContactCoeff` kappa/mu table and
    // its default-kappa policy). It is the contact models' data, not the filters'.

    // set the defined contact model
    for(SizeT i = 0; i < topo_view.size(); ++i)
    {
        auto  ids        = topo_view[i];
        auto& is_enabled = enabled_view[i];

        mask_map(ids.x(), ids.y()) = is_enabled;

        // the reference writes h_contact_tabular[lower] here as well; that entry is
        // the symmetric ContactCoeff half (SCOPE NOTE (T8.1))
        mask_map(ids.y(), ids.x()) = is_enabled;
    }

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
}// namespace uipc::backend::luisa

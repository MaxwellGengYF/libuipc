#pragma once
#include <type_define.h>
#include <sstream>
#include <sim_engine_state.h>
#include <backends/common/sim_engine.h>
#include <sim_action_collection.h>

// LuisaCompute runtime headers
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>

namespace uipc::backend::luisa
{
// Forward declarations for system classes
class GlobalVertexManager;
class GlobalSimplicialSurfaceManager;
class GlobalBodyManager;
class GlobalContactManager;
class GlobalDyTopoEffectManager;
class GlobalTrajectoryFilter;

class TimeIntegratorManager;
class LineSearcher;
class GlobalLinearSystem;
class GlobalAnimator;
class GlobalExternalForceManager;
class GlobalDiffSimManager;
class AffineBodyDynamics;
class FiniteElementMethod;
class InterAffineBodyConstitutionManager;
class NewtonToleranceManager;

/**
 * @brief Main simulation engine for the LuisaCompute backend
 * 
 * This engine uses LuisaCompute (a high-performance compute framework) 
 * for GPU acceleration of physics simulations.
 */
class SimEngine final : public backend::SimEngine
{
    friend class SimSystem;

  public:
    SimEngine(EngineCreateInfo* info);
    virtual ~SimEngine();

    SimEngine(const SimEngine&)            = delete;
    SimEngine& operator=(const SimEngine&) = delete;

    /**
     * @brief Get the current simulation state
     */
    SimEngineState state() const noexcept;

    /**
     * @brief Get the current Newton iteration count
     */
    SizeT newton_iter() const noexcept;

    /**
     * @brief Get the current line search iteration count
     */
    SizeT line_search_iter() const noexcept;

    /**
     * @brief Get the LuisaCompute context
     */
    luisa::compute::Context& luisa_context() noexcept;

    /**
     * @brief Get the LuisaCompute device
     */
    luisa::compute::Device& luisa_device() noexcept;

    /**
     * @brief Get the compute stream for command submission
     */
    luisa::compute::Stream& compute_stream() noexcept;

  private:
    // Core engine interface implementations
    virtual void  do_init(InitInfo& info) override;
    virtual void  do_advance() override;
    virtual void  do_sync() override;
    virtual void  do_retrieve() override;
    virtual SizeT get_frame() const override;

    // Checkpoint/Recovery
    virtual bool do_dump(DumpInfo&) override;
    virtual bool do_try_recover(RecoverInfo&) override;
    virtual void do_apply_recover(RecoverInfo&) override;
    virtual void do_clear_recover(RecoverInfo&) override;

    // Internal methods
    void build();
    void init_scene();
    void dump_global_surface();

    // LuisaCompute runtime objects
    std::unique_ptr<luisa::compute::Context> m_luisa_context;
    luisa::compute::Device                   m_luisa_device;
    luisa::compute::Stream                   m_compute_stream;
    std::string                              m_backend_name = "cuda";  // Default to cuda backend

    // Engine state (exposed to friend SimSystem)
    std::stringstream m_string_stream;
    SimEngineState    m_state = SimEngineState::None;

    // Event collections
    SimActionCollection<void()> m_on_init_scene;
    void                        event_init_scene();
    SimActionCollection<void()> m_on_rebuild_scene;
    void                        event_rebuild_scene();
    SimActionCollection<void()> m_on_write_scene;
    void                        event_write_scene();

    // Aware Top Systems (pointers filled during build)
    GlobalVertexManager*                m_global_vertex_manager                = nullptr;
    GlobalSimplicialSurfaceManager*     m_global_simplicial_surface_manager    = nullptr;
    GlobalBodyManager*                  m_global_body_manager                  = nullptr;
    GlobalContactManager*               m_global_contact_manager               = nullptr;
    GlobalDyTopoEffectManager*          m_global_dytopo_effect_manager         = nullptr;
    GlobalTrajectoryFilter*             m_global_trajectory_filter             = nullptr;

    // Newton Solver Systems
    TimeIntegratorManager*  m_time_integrator_manager  = nullptr;
    LineSearcher*           m_line_searcher            = nullptr;
    GlobalLinearSystem*     m_global_linear_system     = nullptr;
    NewtonToleranceManager* m_newton_tolerance_manager = nullptr;

    GlobalAnimator*             m_global_animator               = nullptr;
    GlobalExternalForceManager* m_global_external_force_manager = nullptr;
    GlobalDiffSimManager*       m_global_diff_sim_manager       = nullptr;
    AffineBodyDynamics*         m_affine_body_dynamics          = nullptr;
    InterAffineBodyConstitutionManager* m_inter_affine_body_constitution_manager = nullptr;
    FiniteElementMethod*        m_finite_element_method         = nullptr;

    // Simulation counters
    SizeT m_current_frame    = 0;
    SizeT m_newton_iter      = 0;
    SizeT m_line_search_iter = 0;

    // Simulation parameters
    bool  m_semi_implicit_enabled  = true;
    Float m_semi_implicit_beta_tol = 1e-3f;
    Float m_newton_scene_tol       = 0.01f;
    bool  m_friction_enabled       = false;

    // Attribute slots for configuration
    template <typename T>
    using CAS = S<const geometry::AttributeSlot<T>>;

    CAS<Float>  m_newton_velocity_tol;
    CAS<IndexT> m_newton_max_iter;
    CAS<IndexT> m_newton_min_iter;
    CAS<IndexT> m_strict_mode;
    CAS<Float>  m_ccd_tol;
    CAS<IndexT> m_dump_surface;
};
}  // namespace uipc::backend::luisa

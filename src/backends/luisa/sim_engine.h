#pragma once
/********************************************************************
 * @file sim_engine.h
 * @brief LuisaCompute backend engine.
 *
 * T1.4 delivered the shell (state machine, event collections, system graph and
 * LC runtime bring-up). T4.1 added the engine-aware global geometry managers and
 * T4.2 adds the pipeline flag resolution and the per-frame lifecycle of the
 * systems ported in T4.2 (animator, external forces, time integrators). The full
 * advance pipeline (newton / line search / linear solve) is still assembled in
 * Phase 8 (T8.4); `do_advance()` here runs the reference frame prologue and
 * epilogue around the ported systems and advances the frame counter.
 *********************************************************************/
#include <type_define.h>
#include <sstream>
#include <sim_engine_state.h>
#include <backends/common/sim_engine.h>
#include <sim_action_collection.h>

namespace uipc::backend::luisa
{
class GlobalVertexManager;
class GlobalBodyManager;
class GlobalSimplicialSurfaceManager;
class TimeIntegratorManager;
class GlobalAnimator;
class GlobalExternalForceManager;
class GlobalDyTopoEffectManager;
class LineSearcher;
// T5.1: the FEM method is engine-aware (the reference engine owns a
// `find<FiniteElementMethod>()` pointer and calls its init() in the
// "before common scene initialization" step).
class FiniteElementMethod;
// T6.1a: the ABD dynamics is engine-aware in the same way (the reference resolves
// it *before* the FEM method and calls its init() first).
class AffineBodyDynamics;
// T6.1b: so is the inter-affine-body constitution manager (reference step 1 order:
// affine_body_dynamics -> inter_affine_body_constitution_manager ->
// finite_element_method -> global_body_manager).
class InterAffineBodyConstitutionManager;

class SimEngine final : public backend::SimEngine
{
    friend class SimSystem;

  public:
    enum class PipelineType
    {
        Basic,
        AugmentedLagrangian
    };

    SimEngine(EngineCreateInfo* info);
    ~SimEngine() override;

    SimEngine(const SimEngine&)            = delete;
    SimEngine& operator=(const SimEngine&) = delete;

    [[nodiscard]] SimEngineState state() const noexcept { return m_state; }
    [[nodiscard]] SizeT          newton_iter() const noexcept { return m_newton_iter; }
    [[nodiscard]] SizeT          line_search_iter() const noexcept { return m_line_search_iter; }

    /// T4.2: which pipeline the scene config selected (`contact/constitution`).
    /// Mirrors the reference's `m_pipeline_type` + `set_pipeline_type()` check.
    [[nodiscard]] PipelineType pipeline_type() const noexcept { return m_pipeline_type; }

  private:
    void  do_init(InitInfo& info) override;
    bool  do_dump(DumpInfo&) override;
    bool  do_try_recover(RecoverInfo&) override;
    void  do_apply_recover(RecoverInfo& info) override;
    void  do_clear_recover(RecoverInfo& info) override;
    void  do_advance() override;
    void  do_sync() override;
    void  do_retrieve() override;
    SizeT get_frame() const override;
    Json  do_frame_stats() const override;

    void build_systems();
    // T4.1/T4.2: resolve the engine-aware systems (the reference's
    // `SimEngine::build()` does the same require<>/find<> calls right after
    // build_systems()). They must be resolved while the state is BuildSystems.
    void build_engine_systems();
    void init_scene();
    void set_pipeline_type();
    void advance();
    void step_animation_and_external_forces();
    void event_init_scene();
    void event_rebuild_scene();
    void event_write_scene();

    // ---- LC runtime bring-up ---------------------------------------------
    void init_device(const Json& config);

    std::stringstream m_string_stream;
    SimEngineState    m_state = SimEngineState::None;

    // Event collections (populated during do_build()).
    SimActionCollection<void()> m_on_init_scene;
    SimActionCollection<void()> m_on_rebuild_scene;
    SimActionCollection<void()> m_on_write_scene;

    // Engine-aware topo systems (T4.1); the reference SimEngine owns the same
    // three pointers and calls their init() in init_scene().
    GlobalVertexManager*            m_global_vertex_manager            = nullptr;
    GlobalBodyManager*              m_global_body_manager              = nullptr;
    GlobalSimplicialSurfaceManager* m_global_simplicial_surface_manager = nullptr;

    // T4.2 systems the reference engine also owns. `TimeIntegratorManager` is
    // required (the reference `require<>`s it); the animator and the external
    // force manager are optional (`find<>`: the animator only exists when the
    // scene has a Constraint constitution, and the force manager only has work
    // when a reporter registered).
    TimeIntegratorManager*      m_time_integrator_manager      = nullptr;
    GlobalAnimator*             m_global_animator              = nullptr;
    GlobalExternalForceManager* m_global_external_force_manager = nullptr;
    // T4.3: the reference `find<>`s the dytopo effect manager (it only exists
    // when the scene enables contact or uses an InterPrimitive constitution)
    // and `require<>`s the LineSearcher, which is always registered.
    GlobalDyTopoEffectManager* m_global_dytopo_effect_manager = nullptr;
    LineSearcher* m_line_searcher = nullptr;
    // T5.1: optional (`find<>`) — the FEM method only exists when the scene uses a
    // FiniteElement constitution type.
    FiniteElementMethod* m_finite_element_method = nullptr;
    // T6.1a: optional (`find<>`) — the ABD dynamics only exists when the scene uses
    // an AffineBody constitution type.
    AffineBodyDynamics* m_affine_body_dynamics = nullptr;
    // T6.1b: optional (`find<>`) — the inter-affine-body constitution manager only
    // exists when the scene uses an `InterAffineBody` constitution type (a joint or
    // an articulation).
    InterAffineBodyConstitutionManager* m_inter_affine_body_constitution_manager = nullptr;

    PipelineType m_pipeline_type = PipelineType::Basic;

    SizeT m_current_frame = 0;
    SizeT m_newton_iter   = 0;
    SizeT m_line_search_iter = 0;
};
}// namespace uipc::backend::luisa

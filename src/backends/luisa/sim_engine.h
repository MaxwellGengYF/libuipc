#pragma once
/********************************************************************
 * @file sim_engine.h
 * @brief LuisaCompute backend engine.
 *
 * T1.4 delivers the shell (state machine, event collections, system graph and
 * LC runtime bring-up). The full advance pipeline (newton / line search /
 * linear solve) is assembled in Phase 8 (T8.4); `do_advance()` here only runs
 * the registered scene events and advances the frame counter.
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

class SimEngine final : public backend::SimEngine
{
    friend class SimSystem;

  public:
    SimEngine(EngineCreateInfo* info);
    ~SimEngine() override;

    SimEngine(const SimEngine&)            = delete;
    SimEngine& operator=(const SimEngine&) = delete;

    [[nodiscard]] SimEngineState state() const noexcept { return m_state; }
    [[nodiscard]] SizeT          newton_iter() const noexcept { return m_newton_iter; }
    [[nodiscard]] SizeT          line_search_iter() const noexcept { return m_line_search_iter; }

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
    // T4.1: resolve the engine-aware geometry managers (the reference's
    // `SimEngine::build()` does the same require<>/find<> calls right after
    // build_systems()). They must be resolved while the state is BuildSystems.
    void build_geometry_systems();
    void init_scene();
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

    SizeT m_current_frame = 0;
    SizeT m_newton_iter   = 0;
    SizeT m_line_search_iter = 0;
};
}// namespace uipc::backend::luisa

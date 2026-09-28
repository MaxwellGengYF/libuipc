#pragma once
/********************************************************************
 * @file backend_sanity_checker.h
 * @brief Backend sanity-checker base (port of
 *        src/backends/cuda/sanity_check/backend_sanity_checker.h).
 *
 * The reference derives its checkers from `core::ISanityChecker` (not from the
 * CPU `uipc_sanity_check` module's SanityChecker), takes the shared
 * `core::ISanityCheckContext` (built by that CPU module) and self-registers with
 * REGISTER_BACKEND_SANITY_CHECKER; the frontend then lets the backend *replace* a
 * CPU checker with the same id (SanityCheckerCollection::insert). This port keeps
 * that mechanism verbatim — only the namespace changes, so a checker's
 * `msg->name()` is the demangled `uipc::backend::luisa::…` name.
 *
 * ---------------------------------------------------------------------------
 * NOT PORTED (deliberate, see the T4.3 report)
 * ---------------------------------------------------------------------------
 * Two of the reference's three checkers are absent from this directory:
 *
 *   simplicial_surface_distance_check.cu
 *   simplicial_surface_intersection_check.cu
 *
 * Both build and query the collision-detection BVH
 * (src/backends/cuda/collision_detection/info_stackless_bvh.h + its .cu:
 * InfoStacklessBVH::build()/query()), which is T7.1 of the plan and has no
 * LuisaCompute counterpart yet (src/backends/luisa/collision_detection/ holds
 * only aabb.h from T4.1). They must not be stubbed: a checker that silently
 * reports "no violation" is worse than no checker at all. Their port belongs to
 * T7.1 (BVH structures), which the plan already schedules before T9's
 * verification harness.
 *
 * The one checker that does not need the BVH is ported here:
 * half_plane_vertex_distance_check.cpp.
 *********************************************************************/
#include <uipc/core/i_sanity_checker.h>
#include <uipc/common/demangle.h>
#include <backends/common/sanity_checker_auto_register.h>
#include <filesystem>

namespace uipc::backend::luisa
{
using uipc::core::SanityCheckResult;

class BackendSanityChecker : public core::ISanityChecker
{
  public:
    explicit BackendSanityChecker(core::ISanityCheckContext& ctx) noexcept
        : m_ctx(ctx)
    {
    }

    std::string_view workspace() const noexcept { return m_ctx.workspace(); }

    std::string this_output_path() const noexcept
    {
        namespace fs = std::filesystem;
        fs::path path{workspace()};
        path /= fmt::format("{}", get_id());
        fs::exists(path) || fs::create_directories(path);
        return path.string();
    }

    S<const core::Object> find_object(IndexT id) const noexcept
    {
        return m_ctx.find_object(id);
    }

  protected:
    virtual std::string get_name() const noexcept override
    {
        return uipc::demangle(typeid(*this).name());
    }

    core::ISanityCheckContext&       context() noexcept { return m_ctx; }
    const core::ISanityCheckContext& context() const noexcept { return m_ctx; }

  private:
    core::ISanityCheckContext& m_ctx;
};

class BackendSanityCheckerException : public Exception
{
  public:
    using Exception::Exception;
};
}// namespace uipc::backend::luisa

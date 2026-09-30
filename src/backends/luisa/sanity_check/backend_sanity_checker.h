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
 * THE THREE CHECKERS
 * ---------------------------------------------------------------------------
 * half_plane_vertex_distance_check.cpp (uid 2, since T4.3), and — since T7.1b, which
 * ported the collision-detection BVH they query — simplicial_surface_distance_check.cpp
 * (uid 3) and simplicial_surface_intersection_check.cpp (uid 1).  The latter two build
 * and query `collision_detection::InfoStacklessBVH` (`build()` with the per-primitive
 * bid/cid spans, then `query()` / `detect()`); see their file headers for the one port
 * deviation the DSL forces (the reference's device predicates become host filters over
 * the reported candidate pairs).
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

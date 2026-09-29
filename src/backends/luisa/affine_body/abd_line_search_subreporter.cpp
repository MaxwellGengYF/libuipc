#include <affine_body/abd_line_search_subreporter.h>
/*************************************************************************************************
* T6.1b (pulled forward from T6.4) — ABDLineSearchSubreporter's four forwarding
* hooks (port of src/backends/cuda/affine_body/abd_line_search_subreporter.cu).
*
* WHY THIS FILE EXISTS IN T6.1b: `AffineBodyAnimatorLineSearchSubreporter` (and
* the inter-affine-body manager's own line-search subreporter, registered at the
* bottom of inter_affine_body_constitution_manager.cpp) derive from this base, and
* a *vtable* needs the base's only virtual override (`do_build()`) plus the
* forwarding hooks the derived classes' `do_*` are reached through.  Leaving them
* undefined was a measured link error, not a "not implemented" signal -- the exact
* precedent T5.3 recorded for `fem_line_search_subreporter.cpp` (pulled forward
* from T5.5).
*
* WHAT IS STILL T6.4's: `ABDLineSearchReporter`'s own bodies (its `Impl::{init,
* record_start_point, step_forward, compute_energy}` kernels, the five `do_*`
* hooks and its `REGISTER_SIM_SYSTEM`) and, in *this* file's class, nothing --
* the four bodies below are the whole of it.  Consequence: with no
* `ABDLineSearchReporter` registered, `require<ABDLineSearchReporter>()` throws
* `SimSystemException` and every ABD line-search subreporter is invalidated, i.e.
* the port leaves the explicit not-implemented signal in place.
*************************************************************************************************/
namespace uipc::backend::luisa
{
void ABDLineSearchSubreporter::do_build()
{
    auto& abd_line_search = require<ABDLineSearchReporter>();

    // let subclass do the actual build
    BuildInfo info;
    do_build(info);

    abd_line_search.add_reporter(this);
}

void ABDLineSearchSubreporter::init()
{
    InitInfo info;
    do_init(info);
}

void ABDLineSearchSubreporter::report_extent(ReportExtentInfo& info)
{
    do_report_extent(info);
}

void ABDLineSearchSubreporter::compute_energy(ComputeEnergyInfo& info)
{
    do_compute_energy(info);
}
}  // namespace uipc::backend::luisa

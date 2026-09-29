#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/abd_line_search_subreporter.h>
/*************************************************************************************************
* T6.1b (pulled forward from T6.4) — the *one* body of ABDLineSearchReporter this
* batch needs: `add_reporter` (port of the matching body in
* src/backends/cuda/affine_body/abd_line_search_reporter.cu:209-214).
*
* WHY: `ABDLineSearchSubreporter::do_build()` (pulled forward with this batch,
* see abd_line_search_subreporter.cpp) calls `abd_line_search.add_reporter(this)`,
* and that call site is compiled -- so the symbol must exist or the link fails.
*
* WHAT IS STILL T6.4's (declared in abd_line_search_reporter.h, deliberately left
* undefined here -- cookbook §7.16 rule 13): `Impl::{init, record_start_point,
* step_forward, compute_energy}` (the reference's three kernels + the
* `kinetic->compute_energy` / `constitution->compute_energy` fan-outs, which need
* T6.2's kinetic and T6.3's constitutions), the five `do_*` hooks and
* `REGISTER_SIM_SYSTEM(ABDLineSearchReporter)`.
*************************************************************************************************/
namespace uipc::backend::luisa
{
void ABDLineSearchReporter::add_reporter(ABDLineSearchSubreporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter is null");
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    m_impl.reporters.register_sim_system(*reporter);
}
}  // namespace uipc::backend::luisa

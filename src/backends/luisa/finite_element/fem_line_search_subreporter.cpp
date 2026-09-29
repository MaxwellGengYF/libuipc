#include <finite_element/fem_line_search_subreporter.h>

/***************************************************************
 * T5.3 (pulled forward from T5.5) — FEMLineSearchSubreporter
 * (reference: src/backends/cuda/finite_element/fem_line_search_subreporter.cu).
 * The four forwarding hooks are the reference's, unchanged.
 ***************************************************************/
namespace uipc::backend::luisa
{
void FEMLineSearchSubreporter::do_build()
{
    auto& fem_line_search = require<FEMLineSearchReporter>();

    // let subclass do the actual build
    BuildInfo info;
    do_build(info);

    fem_line_search.add_reporter(this);
}

void FEMLineSearchSubreporter::init()
{
    InitInfo info;
    do_init(info);
}

void FEMLineSearchSubreporter::report_extent(ReportExtentInfo& info)
{
    do_report_extent(info);
}

void FEMLineSearchSubreporter::compute_energy(ComputeEnergyInfo& info)
{
    do_compute_energy(info);
}
}  // namespace uipc::backend::luisa

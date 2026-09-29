#include <finite_element/finite_element_external_force_reporter.h>
#include <finite_element/finite_element_external_force_manager.h>

/***************************************************************
 * T5.3 — FiniteElementExternalForceReporter's four forwarding hooks (reference:
 * src/backends/cuda/finite_element/finite_element_external_force_reporter.cu).
 *
 * Note the reference's order: the subclass's `do_build(info)` runs *before*
 * `require<FEMExternalForceManager>()`, i.e. a reporter that throws in its own
 * build hook never registers -- preserved verbatim.
 ***************************************************************/
namespace uipc::backend::luisa
{
void FiniteElementExternalForceReporter::do_build()
{
    BuildInfo info;
    do_build(info);

    auto& manager = require<FEMExternalForceManager>();
    manager.register_reporter(this);
}

U64 FiniteElementExternalForceReporter::uid() const noexcept
{
    return get_uid();
}

void FiniteElementExternalForceReporter::init()
{
    do_init();
}

void FiniteElementExternalForceReporter::step(ExternalForceInfo& info)
{
    do_step(info);
}
}  // namespace uipc::backend::luisa

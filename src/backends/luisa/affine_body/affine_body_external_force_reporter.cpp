#include <affine_body/affine_body_external_force_reporter.h>
#include <affine_body/affine_body_external_force_manager.h>

/********************************************************************
 * T6.1a — AffineBodyExternalForceReporter's four forwarding hooks (reference:
 * src/backends/cuda/affine_body/affine_body_external_force_reporter.cu).
 *
 * Note the reference's order: the subclass's `do_build(info)` runs *before*
 * `require<AffineBodyExternalForceManager>()`, i.e. a reporter that throws in its
 * own build hook never registers -- preserved verbatim.
 *********************************************************************/
namespace uipc::backend::luisa
{
void AffineBodyExternalForceReporter::do_build()
{
    BuildInfo info;
    do_build(info);

    // Register with AffineBodyExternalForceManager
    auto& manager = require<AffineBodyExternalForceManager>();
    manager.register_reporter(this);
}

U64 AffineBodyExternalForceReporter::uid() const noexcept
{
    return get_uid();
}

void AffineBodyExternalForceReporter::init()
{
    do_init();
}

void AffineBodyExternalForceReporter::step(ExternalForceInfo& info)
{
    do_step(info);
}
}  // namespace uipc::backend::luisa

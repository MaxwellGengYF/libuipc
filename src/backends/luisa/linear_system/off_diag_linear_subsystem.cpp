#include <linear_system/off_diag_linear_subsystem.h>
#include <uipc/common/log.h>

/*************************************************************************************************
* T8.3a-1 — OffDiagLinearSubsystem (port of
* src/backends/cuda/linear_system/off_diag_linear_subsystem.cu).
*
* The only changes besides the namespace are the doc comment and the
* `#include <uipc/common/log.h>` spelling — every body is the reference's.
*
* ORDER IS LOAD-BEARING: `do_build()` runs the derived `do_build(BuildInfo&)`
* *first* (that is where a concrete subsystem calls `connect(l, r)`), asserts both
* ends are set, stores `m_l`/`m_r` and only then registers with the
* `GlobalLinearSystem`.  That is why `uid()` — which
* `GlobalLinearSystem::Impl::init()`'s sort calls on every registered off-diag
* subsystem — can never see a null end: a registered instance always went through
* the assert above.
*************************************************************************************************/
namespace uipc::backend::luisa
{
void OffDiagLinearSubsystem::do_build()
{
    auto&     global_linear_system = require<GlobalLinearSystem>();
    BuildInfo info;
    do_build(info);

    UIPC_ASSERT(info.m_diag_l != nullptr && info.m_diag_r != nullptr,
                "Did you forget to call BuildInfo::connect() in {}'s do_build()?",
                this->name());

    m_l = info.m_diag_l;
    m_r = info.m_diag_r;

    global_linear_system.add_subsystem(this);
}

void OffDiagLinearSubsystem::init()
{
    InitInfo info;
    do_init(info);
}

void OffDiagLinearSubsystem::do_init(InitInfo& info) {}

std::tuple<U64, U64> OffDiagLinearSubsystem::uid() const noexcept
{
    return std::make_tuple(m_l->uid(), m_r->uid());
}
}  // namespace uipc::backend::luisa

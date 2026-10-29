#pragma once
/********************************************************************
 * @file bench_app.h
 * @brief The shared `main` of every `uipc_bench_<tier>_<backend>` binary (§6.1).
 *
 * A tier binary is: its case list + its device.  Everything else -- option parsing,
 * the environment header, the record stream, the report and the cold/warm cache census
 * ("P6: never compare a cold and a warm run") -- is here, so a Tier-0 row and a Tier-1
 * row are written by the same code and are therefore comparable in `summary.csv`.
 *********************************************************************/
#include <functional>
#include <string_view>
#include <vector>

#include <common/bench_case.h>
#include <common/bench_device.h>

namespace uipc::benchmark
{
/// The tier's device, as the shared main needs it.  Both `BenchDevice` (Tier 0) and
/// `BenchKernels` (Tier 1) satisfy it; the tier's `main` binds the lambdas.
struct AppDeviceHooks
{
    std::function<void(std::string_view program_path)> init;
    std::function<void()>                              shutdown;
    std::function<const DeviceInfo&()>                 info;
    /// Address handed to each case as `CaseContext::device` (see bench_case.h).
    std::function<void*()>                             object;
};

/// Run the selected cases of `cases`, write the records into `--out`, return the exit
/// code (0 = at least one case ran, 1 = nothing matched, 2 = bad option).
[[nodiscard]] int bench_app_main(int                        argc,
                                 char**                     argv,
                                 const std::vector<CaseDef>& cases,
                                 const AppDeviceHooks&       hooks);
}  // namespace uipc::benchmark

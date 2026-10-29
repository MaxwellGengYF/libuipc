/********************************************************************
 * @file t1_main.cpp
 * @brief Entry point of the Tier-1 binaries (`uipc_bench_t1_<backend>`, §6.1).
 *
 * Same skeleton as Tier 0 (common/bench_app.cpp): the only differences are the device
 * (`BenchKernels` instead of `BenchDevice`) and the case list (`t1_registry()`).
 *
 * One binary per backend, one *process* per case: B3's "a second Engine in one process
 * breaks the LC stream" applies to the kernel bench too (it drives the same
 * `luisa_device` stream), which is why the harness runs one case per invocation.
 *********************************************************************/
#include <common/bench_app.h>
#include <common/bench_case.h>
#include <common/bench_kernels.h>

#include <string_view>
#include <vector>

namespace uipc::benchmark
{
/// Defined in tier1/t1_cases.cpp; linked only into the Tier-1 binaries.
std::vector<CaseDef> const& t1_registry();
}  // namespace uipc::benchmark

int main(int argc, char** argv)
{
    uipc::benchmark::BenchKernels kernels;
    const uipc::benchmark::AppDeviceHooks hooks{
        [&](std::string_view program_path) { kernels.init(program_path); },
        [&] { kernels.shutdown(); },
        [&]() -> const uipc::benchmark::DeviceInfo& { return kernels.info(); },
        [&]() -> void* { return &kernels; },
    };
    return uipc::benchmark::bench_app_main(argc, argv, uipc::benchmark::t1_registry(), hooks);
}

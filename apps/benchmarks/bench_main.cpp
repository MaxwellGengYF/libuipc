/********************************************************************
 * @file bench_main.cpp
 * @brief Entry point of the Tier-0 binaries (`uipc_bench_t0_<backend>`, §6.1).
 *
 * Everything except "which device and which cases" lives in common/bench_app.cpp,
 * so Tier 1 (`t1_main.cpp`) and every later tier share one implementation of the
 * argument parsing, the environment header and the report.
 *********************************************************************/
#include <common/bench_app.h>
#include <common/bench_case.h>
#include <common/bench_device.h>

int main(int argc, char** argv)
{
    uipc::benchmark::BenchDevice device;
    const uipc::benchmark::AppDeviceHooks hooks{
        [&](std::string_view program_path) { device.init(program_path); },
        [&] { device.shutdown(); },
        [&]() -> const uipc::benchmark::DeviceInfo& { return device.info(); },
        [&]() -> void* { return &device; },
    };
    return uipc::benchmark::bench_app_main(
        argc, argv, uipc::benchmark::case_registry(), hooks);
}

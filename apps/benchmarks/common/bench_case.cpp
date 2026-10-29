/********************************************************************
 * @file bench_case.cpp
 * @brief Shared case helpers (checksums, the §4.1 graph-economics line).
 *********************************************************************/
#include <common/bench_case.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace uipc::benchmark
{
void check_all_equal(BenchDevice& device, BufferHandle buffer, std::size_t count, float expected, CaseRecord& record)
{
    std::vector<float> host(count);
    device.download(buffer, host.data(), count);
    // Exact comparison is correct here: the kernels only ever store a constant
    // (fill) or add 1.0f to small integers (increment), so every operation is
    // exact in float. A tolerance would hide exactly the mis-dispatch this check
    // exists to catch (§4.1: dispatch(n) takes a thread count, not a block count).
    std::size_t bad = 0u;
    for(std::size_t i = 0; i < count; ++i)
    {
        if(host[i] != expected)
        {
            if(bad < 4u)
                record.checksum_note += " mismatch[" + std::to_string(i)
                                        + "]=" + std::to_string(host[i]);
            ++bad;
        }
    }
    record.checksum = static_cast<double>(host.empty() ? 0.0f : host.front());
    record.expected_checksum = static_cast<double>(expected);
    record.checksum_ok       = (bad == 0u);
    if(bad != 0u)
        record.checksum_note = "expected " + std::to_string(expected)
                               + " everywhere, " + std::to_string(bad)
                               + " elements differ;" + record.checksum_note;
}

double sum_host(BenchDevice& device, BufferHandle buffer, std::size_t count)
{
    std::vector<float> host(count);
    device.download(buffer, host.data(), count);
    double sum = 0.0;
    for(float v : host)
        sum += static_cast<double>(v);
    return sum;
}

void print_graph_economics(const char*   case_id,
                           double        us_per_dispatch,
                           double        us_per_graph_dispatch,
                           std::uint32_t dispatches_per_frame)
{
    // §4.1: graph_gain_per_frame ~= dispatches_per_frame x (us_per_statement -
    // us_per_graph_dispatch). The comparison that decides the graph workstream
    // (optimize_plan.md OP-A5/G) is this number against the command-list gain.
    const double plain_ms =
        us_per_dispatch * static_cast<double>(dispatches_per_frame) / 1000.0;
    const double graph_ms =
        us_per_graph_dispatch * static_cast<double>(dispatches_per_frame) / 1000.0;
    std::printf(
        "[economics] %s  dispatches/frame=%u  plain=%.3f ms/frame  graph=%.3f ms/frame  "
        "ceiling gain=%.3f ms/frame (%.1f%%)\n",
        case_id,
        dispatches_per_frame,
        plain_ms,
        graph_ms,
        plain_ms - graph_ms,
        plain_ms > 0.0 ? (plain_ms - graph_ms) / plain_ms * 100.0 : 0.0);
    std::fflush(stdout);
}
}// namespace uipc::benchmark

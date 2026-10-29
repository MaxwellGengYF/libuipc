#pragma once
/********************************************************************
 * @file bench_harness.h
 * @brief The measurement loop of bench_plan.md §6.2.
 *
 * Contract (copied from lc_optimize §24.3 and the existing `time_it` precedent in
 * apps/tests/backends/luisa/luisa_device_spmv_tests.cpp:1023):
 *
 *   - warm-up (compile + first launch) is discarded;
 *   - `reps` repetitions are enqueued back to back with ONE sync at the end;
 *   - the measured window is calibrated to >= kTargetWindowMs;
 *   - a case that needs a per-rep sync reports it as `sync_per_rep`;
 *   - every round is reported (median, P95, min/max, CoV) -- never one number.
 *
 * The host reference checksum is asserted by the caller before timing (§6.2:
 * "a wrong kernel that runs in 0 ms must fail loudly").
 *********************************************************************/
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <common/bench_stats.h>

namespace uipc::benchmark
{
inline constexpr double kTargetWindowMs = 100.0;

/// One measured quantity: `reps` enqueues of a repetition, `rounds` times.
struct Timing
{
    std::uint64_t       reps         = 0;
    std::uint32_t       rounds       = 0;
    std::uint32_t       warmup       = 0;
    bool                sync_per_rep = false;
    std::vector<double> round_ms;         ///< per-round total, ms
    double              median_ms = 0.0;  ///< median of round_ms
    double              p95_ms    = 0.0;
    double              min_ms    = 0.0;
    double              max_ms    = 0.0;
    double              cov_pct   = 0.0;  ///< agreement of the rounds

    /// Median cost of one rep, and of one work unit if `units_per_rep` is known.
    [[nodiscard]] double per_rep_us() const noexcept
    {
        return reps == 0u ? 0.0 : median_ms * 1000.0 / static_cast<double>(reps);
    }
    [[nodiscard]] double per_unit_us(double units_per_rep) const noexcept
    {
        const double units = static_cast<double>(reps) * units_per_rep;
        return units == 0.0 ? 0.0 : median_ms * 1000.0 / units;
    }
    [[nodiscard]] bool stable() const noexcept { return cov_pct <= 3.0; }
};

/**
 * @brief Run `enqueue` (`reps` times per round, `rounds` rounds) and time it.
 *
 * `enqueue` must only *submit*; `sync` is called by the harness once at the end
 * of each round (or after every rep when sync_per_rep is requested).
 */
template <typename Enqueue>
Timing time_reps(Enqueue&&             enqueue,
                 std::uint64_t         reps,
                 std::uint32_t         rounds       = 3u,
                 std::uint32_t         warmup       = 1u,
                 bool                  sync_per_rep = false,
                 std::function<void()> sync         = {})
{
    Timing t;
    t.reps         = reps;
    t.rounds       = rounds;
    t.warmup       = warmup;
    t.sync_per_rep = sync_per_rep;

    auto once = [&]
    {
        if(sync_per_rep)
        {
            const auto t0 = std::chrono::steady_clock::now();
            for(std::uint64_t i = 0; i < reps; ++i)
            {
                enqueue(i);
                if(sync)
                    sync();
            }
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                .count();
        }
        const auto t0 = std::chrono::steady_clock::now();
        for(std::uint64_t i = 0; i < reps; ++i)
            enqueue(i);
        if(sync)
            sync();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    };

    for(std::uint32_t i = 0; i < warmup; ++i)
        once();

    t.round_ms.reserve(rounds);
    for(std::uint32_t i = 0; i < rounds; ++i)
        t.round_ms.push_back(once());

    t.median_ms   = median(t.round_ms);
    t.p95_ms      = percentile(t.round_ms, 95.0);
    const auto mm = std::minmax_element(t.round_ms.begin(), t.round_ms.end());
    t.min_ms      = *mm.first;
    t.max_ms      = *mm.second;
    t.cov_pct     = cov_percent(t.round_ms);
    return t;
}

/**
 * @brief §6.2 "N calibrated so that the measured window is >= 100 ms".
 *
 * Times a single rep once (after one discarded warm-up), then scales `reps` so
 * that a round lands in [kTargetWindowMs, 20 * kTargetWindowMs]. The calibration
 * pass itself is never part of a reported number.
 */
template <typename Enqueue>
std::uint64_t calibrate_reps(Enqueue&&             enqueue,
                             std::uint64_t         min_reps = 10u,
                             std::uint64_t         max_reps = 1000000u,
                             std::function<void()> sync     = {})
{
    enqueue(0u);
    if(sync)
        sync();

    const auto t0 = std::chrono::steady_clock::now();
    enqueue(0u);
    if(sync)
        sync();
    const double one_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();

    if(one_ms <= 0.0)
        return max_reps;

    auto reps = static_cast<std::uint64_t>(kTargetWindowMs / one_ms);
    if(reps < min_reps)
        reps = min_reps;
    if(reps > max_reps)
        reps = max_reps;
    return reps;
}
}// namespace uipc::benchmark

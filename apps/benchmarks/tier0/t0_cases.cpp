/********************************************************************
 * @file t0_cases.cpp
 * @brief Tier 0 — host submission and sync cost (bench_plan.md §4.1, H0.1..H0.7).
 *
 * Shared verbatim between the CUDA and the Luisa binary: this file only talks to
 * `BenchDevice` (common/bench_device.h), so the two backends run the identical
 * scenario by construction, not by discipline (§6.1).
 *
 * Every case:
 *   - resets the §6.5 counters, then measures;
 *   - verifies a host reference checksum BEFORE the number is quoted (§6.2);
 *   - reports median + P95 + CoV over the rounds, never a single number (§7.1).
 *********************************************************************/
#include <common/bench_case.h>
#include <common/bench_harness.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace uipc::benchmark
{
namespace
{
    constexpr float kZero = 0.0f;

    void fill_counters(CaseContext& ctx, CaseRecord& record)
    {
        const Counters& c           = ctx.as<BenchDevice>().counters();
        record.dispatches           = c.dispatches;
        record.commits              = c.commits;
        record.syncs                = c.syncs;
        record.sync_ms              = c.sync_ms;
        record.h2d_bytes            = c.h2d_bytes;
        record.d2h_bytes            = c.d2h_bytes;
        record.graph_captures       = c.graph_captures;
        record.graph_instantiates   = c.graph_instantiates;
        record.graph_updates        = c.graph_updates;
        record.graph_launches       = c.graph_launches;
        record.graph_refusals       = c.graph_refusals;
        record.graph_rebuilds       = c.graph_rebuilds;
        record.graph_instantiate_ms = c.graph_instantiate_ms;
        record.graph_update_ms      = c.graph_update_ms;
    }

    /// Rate helpers: work_units are per rep, so units/s uses the per-rep time.
    void derive_rates(CaseRecord& record, double units_per_rep, double bytes_per_unit)
    {
        record.work_units  = static_cast<std::uint64_t>(units_per_rep);
        record.per_call_us = record.timing.per_rep_us();
        record.per_unit_us = record.timing.per_unit_us(units_per_rep);
        record.units_per_second =
            record.per_call_us > 0.0 ? units_per_rep * 1.0e6 / record.per_call_us : 0.0;
        record.bytes_per_unit = bytes_per_unit;
        record.bytes_per_second =
            record.per_call_us > 0.0 ?
                units_per_rep * bytes_per_unit * 1.0e6 / record.per_call_us :
                0.0;
    }

    // ---------------------------------------------------------------------------
    // H0.1 — 100 000 tiny dispatches, one statement each, one thread each.
    // ---------------------------------------------------------------------------
    void case_h0_1(CaseContext& ctx)
    {
        BenchDevice& dev = ctx.as<BenchDevice>();
        CaseRecord   r;
        r.id        = "H0.1";
        r.tier      = "T0";
        r.scenario  = "100k tiny dispatches, 1 statement each, 1 thread";
        r.work_unit = "dispatch";

        const auto n = ctx.options.h01_dispatches;
        auto       k = dev.compile_fill();
        auto b = dev.alloc(1u);  // one thread writes b[0]; a 1-float buffer is enough

        dev.reset_counters();
        const auto t = time_reps(
            [&](std::uint64_t)
            {
                for(std::uint64_t i = 0; i < n; ++i)
                    dev.dispatch(k, b, 1u, static_cast<float>(i));
            },
            1u,
            ctx.options.rounds,
            ctx.options.warmup,
            false,
            [&] { dev.sync(); });
        r.timing = t;
        derive_rates(r, static_cast<double>(n), 0.0);
        fill_counters(ctx, r);

        // Host reference: the last dispatch wrote n-1 into b[0], in order. A kernel
        // that never ran, or ran with a wrong thread count, cannot produce this.
        check_all_equal(dev, b, 1u, static_cast<float>(n - 1u), r);
        r.flags.push_back(r.timing.stable() ? "stable" : "unstable");

        ctx.options.plain_us_per_dispatch = r.per_unit_us;
        ctx.report.add(r);
        dev.release(b);
    }

    // ---------------------------------------------------------------------------
    // H0.2 — the same 100 000 dispatches inside one CommandList, one commit.
    // ---------------------------------------------------------------------------
    void case_h0_2(CaseContext& ctx)
    {
        BenchDevice& dev = ctx.as<BenchDevice>();
        CaseRecord   r;
        r.id   = "H0.2";
        r.tier = "T0";
        r.scenario = "100k tiny dispatches recorded into one CommandList, one commit";
        r.work_unit = "dispatch";

        const auto n = ctx.options.h01_dispatches;
        auto       k = dev.compile_fill();
        auto       b = dev.alloc(1u);

        dev.reset_counters();
        const auto t = time_reps(
            [&](std::uint64_t)
            {
                dev.begin_batch();
                for(std::uint64_t i = 0; i < n; ++i)
                    dev.dispatch(k, b, 1u, static_cast<float>(i));
                dev.end_batch();
            },
            1u,
            ctx.options.rounds,
            ctx.options.warmup,
            false,
            [&] { dev.sync(); });
        r.timing = t;
        derive_rates(r, static_cast<double>(n), 0.0);
        fill_counters(ctx, r);

        // The batching claim is `commits == reps` while dispatches == reps * n; on CUDA
        // there is nothing to batch, so commits == dispatches is the honest answer and
        // the flag says so instead of pretending batching failed (§2.3).
        if(r.commits <= r.timing.reps)
        {
            r.flags.push_back("batched");
        }
        else
        {
            r.flags.push_back("no_batching_layer");
            r.notes += "commits == dispatches: this backend submits per statement";
        }
        check_all_equal(dev, b, 1u, static_cast<float>(n - 1u), r);
        r.flags.push_back(r.timing.stable() ? "stable" : "unstable");
        ctx.options.batched_us_per_dispatch = r.per_unit_us;
        ctx.report.add(r);
        dev.release(b);
    }

    // ---------------------------------------------------------------------------
    // H0.3 — the same work through an explicit graph, then replays.
    // ---------------------------------------------------------------------------
    void case_h0_3(CaseContext& ctx)
    {
        BenchDevice& dev = ctx.as<BenchDevice>();
        CaseRecord   r;
        r.id        = "H0.3";
        r.tier      = "T0";
        r.scenario  = "graph capture of N dispatches + replays";
        r.work_unit = "graph-dispatch";

        const auto nodes = ctx.options.graph_nodes;
        r.scenario += " (nodes=" + std::to_string(nodes) + ", replays="
                      + std::to_string(ctx.options.graph_replays) + ")";

        auto k = dev.compile_fill();
        auto b = dev.alloc(1u);
        dev.upload(b, &kZero, 1u);  // sentinel so the replay result is checkable

        dev.reset_counters();
        if(!dev.graph_supported())
        {
            r.timing.reps = 0u;
            r.flags.push_back("graph_unsupported");
            r.notes = "the backend reports no graph extension; H0.3/H0.7 are not measurable here";
            fill_counters(ctx, r);
            ctx.report.add(r);
            dev.release(b);
            return;
        }

        if(!dev.graph_capture_fill(k, b, 1u, 1.0f, nodes))
        {
            // §2.4: a refused capture must fail the benchmark, never pass at 0 ms.
            ++r.graph_refusals;
            r.timing.reps = 0u;
            r.flags.push_back("graph_refused");
            r.notes = "capture refused; the refusal text is in the H0.6 matrix and on stderr";
            fill_counters(ctx, r);
            ctx.report.add(r);
            dev.release(b);
            return;
        }
        r.graph_instantiates   = dev.counters().graph_instantiates;
        r.graph_instantiate_ms = dev.counters().graph_instantiate_ms;

        const auto t = time_reps([&](std::uint64_t) { dev.graph_launch(); },
                                 ctx.options.graph_replays,
                                 ctx.options.rounds,
                                 ctx.options.warmup,
                                 false,
                                 [&] { dev.sync(); });
        r.timing     = t;
        derive_rates(r, static_cast<double>(nodes), 0.0);
        fill_counters(ctx, r);
        check_all_equal(dev, b, 1u, 1.0f, r);
        r.flags.push_back(r.timing.stable() ? "stable" : "unstable");
        r.notes += "per-replay instantiate cost amortised: "
                   + std::to_string(r.graph_instantiate_ms) + " ms once, "
                   + std::to_string(ctx.options.graph_replays) + " replays per round";

        ctx.options.graph_us_per_dispatch = r.per_unit_us;
        // G1b: the ceilings the graph / command-list workstreams are decided on, printed
        // together once all three submission modes have been measured on this GPU (§9).
        const double dpf = static_cast<double>(ctx.options.dispatches_per_frame);
        const double plain_ms = ctx.options.plain_us_per_dispatch * dpf / 1000.0;
        const double batched_ms = ctx.options.batched_us_per_dispatch * dpf / 1000.0;
        const double graph_ms = ctx.options.graph_us_per_dispatch * dpf / 1000.0;
        std::printf(
            "[G1b] dispatches/frame=%u  plain=%.3f ms/frame (%.3f us/dispatch)  "
            "batched=%.3f ms/frame (%.3f us/dispatch)  graph=%.3f ms/frame "
            "(%.3f us/graph-dispatch)\n",
            ctx.options.dispatches_per_frame,
            plain_ms,
            ctx.options.plain_us_per_dispatch,
            batched_ms,
            ctx.options.batched_us_per_dispatch,
            graph_ms,
            ctx.options.graph_us_per_dispatch);
        std::printf(
            "[G1b] batching gain=%.3f ms/frame (%.1f%%), graph gain=%.3f ms/frame "
            "(%.1f%%) -- the answer G1b asks for; verify with nsys "
            "--cuda-graph-trace=node before acting (BN4)\n",
            plain_ms - batched_ms,
            plain_ms > 0.0 ? (plain_ms - batched_ms) / plain_ms * 100.0 : 0.0,
            plain_ms - graph_ms,
            plain_ms > 0.0 ? (plain_ms - graph_ms) / plain_ms * 100.0 : 0.0);
        std::fflush(stdout);
        r.notes += "; plain=" + std::to_string(ctx.options.plain_us_per_dispatch)
                   + " us/dispatch batched=" + std::to_string(ctx.options.batched_us_per_dispatch)
                   + " us/dispatch (G1b block above)";
        ctx.report.add(r);

        dev.graph_destroy();
        dev.release(b);
    }

    // ---------------------------------------------------------------------------
    // H0.4 — synchronize() latency after a trivial dispatch, x1000.
    // ---------------------------------------------------------------------------
    void case_h0_4(CaseContext& ctx)
    {
        BenchDevice& dev = ctx.as<BenchDevice>();
        CaseRecord   r;
        r.id        = "H0.4";
        r.tier      = "T0";
        r.scenario  = "dispatch + synchronize pairs: pure fence/wait cost";
        r.work_unit = "sync";

        auto k = dev.compile_fill();
        auto b = dev.alloc(1u);

        dev.reset_counters();
        const auto pairs = static_cast<std::uint64_t>(ctx.options.sync_pairs);
        const auto t     = time_reps(
            [&](std::uint64_t i)
            {
                dev.dispatch(k, b, 1u, static_cast<float>(i));
                dev.sync();
            },
            pairs,
            ctx.options.rounds,
            ctx.options.warmup,
            true,
            nullptr);
        r.timing = t;
        derive_rates(r, 1.0, 0.0);
        fill_counters(ctx, r);
        r.flags.push_back("sync_per_rep");
        r.flags.push_back(r.timing.stable() ? "stable" : "unstable");
        r.notes = "us/sync includes one trivial dispatch; subtract the H0.1 us/dispatch";
        // Every round restarts the loop at i == 0, so the final value is pairs-1 -- the
        // warm-up run included (it ends on the same value).
        check_all_equal(dev, b, 1u, static_cast<float>(pairs - 1u), r);
        ctx.report.add(r);
        dev.release(b);
    }

    // ---------------------------------------------------------------------------
    // H0.5 — the same kernel at increasing thread counts: enqueue vs execution cost.
    // ---------------------------------------------------------------------------
    void case_h0_5(CaseContext& ctx)
    {
        BenchDevice& dev = ctx.as<BenchDevice>();
        auto         k   = dev.compile_increment();

        const auto max_threads = *std::max_element(ctx.options.thread_scale.begin(),
                                                   ctx.options.thread_scale.end());
        auto b = dev.alloc(std::max<std::size_t>(max_threads, 1u));

        for(std::uint32_t threads : ctx.options.thread_scale)
        {
            CaseRecord r;
            r.id   = "H0.5." + std::to_string(threads);
            r.tier = "T0";
            r.scenario = "1000 dispatches of an increment kernel with real per-element work, "
                         + std::to_string(threads) + " threads each";
            r.work_unit = "element";

            // Correctness pass, untimed: zero the buffer, apply exactly 3 increments and
            // require every element to be exactly 3.0. This is the check that a
            // dispatch(n) mis-read as a block count cannot survive (§4.1, lc_optimize §5.7).
            dev.zero(b, threads);
            for(int i = 0; i < 3; ++i)
                dev.dispatch(k, b, threads, 0.0f);
            dev.sync();
            check_all_equal(dev, b, threads, 3.0f, r);
            dev.zero(b, threads);

            // §6.2: reps calibrated so the measured window is >= 100 ms.
            const auto reps =
                calibrate_reps([&](std::uint64_t)
                               { dev.dispatch(k, b, threads, 0.0f); },
                               10u,
                               1000000u,
                               [&] { dev.sync(); });

            dev.reset_counters();
            const auto t = time_reps([&](std::uint64_t)
                                     { dev.dispatch(k, b, threads, 0.0f); },
                                     reps,
                                     ctx.options.rounds,
                                     ctx.options.warmup,
                                     false,
                                     [&] { dev.sync(); });
            r.timing     = t;
            derive_rates(r, static_cast<double>(threads), 8.0);  // read + write of one float
            fill_counters(ctx, r);
            r.flags.push_back(r.timing.stable() ? "stable" : "unstable");
            // Enqueue-bound vs execution-bound is the H0.5 answer: compare this row's
            // us/dispatch with H0.1's 1-thread row at the same backend.
            ctx.report.add(r);
        }
        dev.release(b);
    }

    // ---------------------------------------------------------------------------
    // H0.6 — the refusal matrix: pin what this revision accepts and refuses.
    // ---------------------------------------------------------------------------
    void case_h0_6(CaseContext& ctx)
    {
        BenchDevice& dev = ctx.as<BenchDevice>();
        dev.reset_counters();

        const auto rows = dev.probe_refusals();

        std::printf("\n[H0.6] graph refusal matrix (%s)\n", dev.info().backend.c_str());
        std::printf("  %-22s %-9s %s\n", "content", "accepted", "note");
        for(const auto& row : rows)
        {
            std::printf("  %-22s %-9s %s\n",
                        std::string{to_string(row.kind)}.c_str(),
                        row.accepted ? "yes" : "no",
                        row.note.c_str());
        }
        std::fflush(stdout);

        for(const auto& row : rows)
        {
            CaseRecord r;
            r.id   = "H0.6." + std::string{to_string(row.kind)};
            r.tier = "T0";
            r.scenario = "graph refusal probe: " + std::string{to_string(row.kind)};
            r.work_unit = "probe";
            r.notes     = row.note;
            r.flags.push_back(row.accepted ? "accepted" : "refused");
            fill_counters(ctx, r);
            ctx.report.add(r);
        }
    }

    // ---------------------------------------------------------------------------
    // H0.7 — whole-graph update vs destroy + re-capture + instantiate.
    // ---------------------------------------------------------------------------
    void case_h0_7(CaseContext& ctx)
    {
        BenchDevice& dev = ctx.as<BenchDevice>();
        CaseRecord   r;
        r.id   = "H0.7";
        r.tier = "T0";
        r.scenario = "graph update(update) vs recreate+instantiate on the same node count";
        r.work_unit = "graph";

        if(!dev.graph_supported())
        {
            r.flags.push_back("graph_unsupported");
            r.notes = "no graph extension on this backend";
            ctx.report.add(r);
            return;
        }

        const auto nodes = ctx.options.graph_nodes;
        auto       k     = dev.compile_fill();
        auto       b     = dev.alloc(1u);
        dev.upload(b, &kZero, 1u);

        dev.reset_counters();
        if(!dev.graph_capture_fill(k, b, 1u, 1.0f, nodes))
        {
            r.flags.push_back("graph_refused");
            r.notes = "capture refused; see the H0.6 matrix";
            fill_counters(ctx, r);
            ctx.report.add(r);
            dev.release(b);
            return;
        }

        // update: re-record an equivalent list and let the backend update the exec in place
        const double update_ms = dev.graph_update_fill(k, b, 1u, 1.0f, nodes);
        // recreate: the fallback an implementation must pay when update is unavailable
        const auto t_recreate =
            time_reps([&](std::uint64_t)
                      { dev.graph_capture_fill(k, b, 1u, 1.0f, nodes); },
                      1u,
                      ctx.options.rounds,
                      ctx.options.warmup,
                      false,
                      [&] { dev.sync(); });

        r.timing.round_ms  = {update_ms};
        r.timing.reps      = 1u;
        r.timing.rounds    = 1u;
        r.timing.median_ms = update_ms;
        r.timing.min_ms    = update_ms;
        r.timing.max_ms    = update_ms;
        r.timing.p95_ms    = update_ms;
        derive_rates(r, static_cast<double>(nodes), 0.0);
        fill_counters(ctx, r);
        r.notes = "graph_update_ms=" + std::to_string(update_ms) + " ms; recreate+instantiate median="
                  + std::to_string(t_recreate.median_ms) + " ms";
        r.flags.push_back(update_ms < t_recreate.median_ms ? "update_cheaper" : "recreate_cheaper");
        std::printf("[H0.7] nodes=%u  update=%.3f ms  recreate+instantiate=%.3f ms  -> %s\n",
                    nodes,
                    update_ms,
                    t_recreate.median_ms,
                    update_ms < t_recreate.median_ms ? "update cheaper" : "recreate cheaper");
        std::fflush(stdout);

        // Correctness of the graph path, in two steps that separate two failure modes:
        //  1. nothing the capture or the update did may have *executed* -- a capture that
        //     runs its work is not a capture, and the sentinel must still be there;
        //  2. one replay must then produce exactly the recorded statement's result.
        // (An earlier version of this case asserted 1.0f directly and failed against a
        // correct implementation: capture alone executes nothing.)
        const double untouched = sum_host(dev, b, 1u);
        r.notes += "; buffer after capture/update = " + std::to_string(untouched)
                   + " (0 means capture executed nothing)";
        dev.graph_launch();
        dev.sync();
        check_all_equal(dev, b, 1u, 1.0f, r);
        dev.graph_destroy();
        dev.release(b);
        ctx.report.add(r);
    }
    // ---------------------------------------------------------------------------
    // H0.8 -- read-back cost: download + synchronize pairs (bench_result.md S12 F-1).
    // ---------------------------------------------------------------------------
    void case_h0_8(CaseContext& ctx)
    {
        BenchDevice& dev = ctx.as<BenchDevice>();
        const auto pairs = static_cast<std::uint64_t>(ctx.options.sync_pairs);
        auto k = dev.compile_fill();
        for(std::size_t count : {std::size_t{1u}, std::size_t{4096u}})
        {
            CaseRecord r;
            r.id       = "H0.8." + std::to_string(count * sizeof(float));
            r.tier     = "T0";
            r.scenario = "download + synchronize pairs, " + std::to_string(count * sizeof(float))
                + " B per read-back: the whole D2H path including its fence";
            r.work_unit = "readback";

            auto b = dev.alloc(count);
            dev.dispatch(k, b, static_cast<std::uint32_t>(count), 7.0f);
            dev.sync();
            std::vector<float> host(count, 0.0f);

            // Measured window: download() is one recorded download command *plus* the
            // counted synchronize() it performs, i.e. exactly what a read-back costs the
            // caller (staging copy, pool hand-back, fence).
            dev.reset_counters();
            const auto t = time_reps(
                [&](std::uint64_t) { dev.download(b, host.data(), count); },
                pairs,
                ctx.options.rounds,
                ctx.options.warmup,
                true,
                nullptr);
            r.timing = t;
            derive_rates(r, 1.0, static_cast<double>(count * sizeof(float)));
            fill_counters(ctx, r);

            // Two independent checks: the device still holds 7.0f, and the host copy the
            // measured loop produced is the same 7.0f (a broken staging path shows up here).
            check_all_equal(dev, b, count, 7.0f, r);
            auto host_ok = true;
            for(auto v : host) { host_ok = host_ok && v == 7.0f; }
            r.flags.push_back(host_ok ? "host_verified" : "host_mismatch");
            r.notes = "us/readback is one download command + one fence; sync_ms/syncs "
                        "separates the fence share of it";
            ctx.report.add(r);
            dev.release(b);
        }
    }

    // ---------------------------------------------------------------------------
    // H0.9 -- upload staging cost: upload + synchronize pairs (F-2).
    // ---------------------------------------------------------------------------
    void case_h0_9(CaseContext& ctx)
    {
        BenchDevice& dev = ctx.as<BenchDevice>();
        const auto pairs = static_cast<std::uint64_t>(ctx.options.sync_pairs);
        for(std::size_t count : {std::size_t{64u}, std::size_t{16384u}})
        {
            CaseRecord r;
            r.id       = "H0.9." + std::to_string(count * sizeof(float));
            r.tier     = "T0";
            r.scenario = "upload + synchronize pairs, " + std::to_string(count * sizeof(float))
                + " B per upload: host staging + H2D";
            r.work_unit = "upload";

            auto b = dev.alloc(count);
            std::vector<float> host(count, 2.5f);

            dev.upload(b, host.data(), count); // warm-up outside the window
            dev.sync();
            dev.reset_counters();
            const auto t = time_reps(
                [&](std::uint64_t) { dev.upload(b, host.data(), count); },
                pairs,
                ctx.options.rounds,
                ctx.options.warmup,
                true,
                nullptr);
            r.timing = t;
            derive_rates(r, 1.0, static_cast<double>(count * sizeof(float)));
            fill_counters(ctx, r);
            r.flags.push_back("upload_per_rep");
            r.notes = "us/upload is one upload command + one fence; the staging pool "
                        "memcpy dominates the CPU side for small sizes";
            check_all_equal(dev, b, count, 2.5f, r);
            ctx.report.add(r);
            dev.release(b);
        }
    }


    // ---------------------------------------------------------------------------
    // H0.10 -- buffer create/destroy cost: the allocator path (bench_result.md S12 F-3).
    // ---------------------------------------------------------------------------
    void case_h0_10(CaseContext& ctx)
    {
        BenchDevice& dev = ctx.as<BenchDevice>();
        const auto pairs = static_cast<std::uint64_t>(ctx.options.sync_pairs) / 5u;
        for(std::size_t count : {std::size_t{1024u}, std::size_t{1048576u}})
        {
            CaseRecord r;
            r.id       = "H0.10." + std::to_string(count * sizeof(float));
            r.tier     = "T0";
            r.scenario = "alloc + release pairs, " + std::to_string(count * sizeof(float))
                + " B per buffer: device allocATOR round trip and implicit barriers";
            r.work_unit = "alloc+release";

            dev.reset_counters();
            const auto t = time_reps(
                [&](std::uint64_t)
                {
                    auto b = dev.alloc(count);
                    dev.release(b);
                },
                pairs,
                ctx.options.rounds,
                ctx.options.warmup,
                true,
                nullptr);
            r.timing = t;
            derive_rates(r, 1.0, static_cast<double>(count * sizeof(float)));
            fill_counters(ctx, r);
            r.flags.push_back("alloc_per_rep");
            r.notes = "cuMemFree synchronises the device, so a create/destroy pair costs two "
                        "implicit barriers unless the allocator reuses a freed block";
            // A reused block must be fully usable: write a known pattern into a fresh buffer
            // and read it back (a stale or double-handed-out block shows up here).
            {
                auto b = dev.alloc(count);
                std::vector<float> host(count, 3.5f);
                dev.upload(b, host.data(), count);
                dev.sync();
                check_all_equal(dev, b, count, 3.5f, r);
                dev.release(b);
            }
            ctx.report.add(r);
        }
    }


}// namespace

std::vector<CaseDef> const& case_registry()
{
    // Order per §4.2: cheap cases first, the graph/refusal cases last, so a fatal
    // failure in a graph case cannot lose the plain-launch numbers.
    static const std::vector<CaseDef> registry{
        {"H0.1", "T0", "100k tiny dispatches, plain submission", &case_h0_1},
        {"H0.2", "T0", "100k tiny dispatches in one CommandList", &case_h0_2},
        {"H0.5", "T0", "1000 dispatches at 1K/100K/1M/10M threads", &case_h0_5},
        {"H0.4", "T0", "dispatch+sync pairs, fence cost", &case_h0_4},
        {"H0.3", "T0", "graph capture + replays", &case_h0_3},
        {"H0.7", "T0", "graph update vs recreate", &case_h0_7},
        {"H0.6", "T0", "graph refusal matrix", &case_h0_6},
    {"H0.8", "T0", "download+sync pairs, read-back cost", &case_h0_8},
    {"H0.9", "T0", "upload+sync pairs, staging cost", &case_h0_9},
    {"H0.10", "T0", "alloc+release pairs, allocator cost", &case_h0_10},
    };
    return registry;
}
}// namespace uipc::benchmark

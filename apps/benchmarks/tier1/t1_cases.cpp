/********************************************************************
 * @file t1_cases.cpp
 * @brief Tier 1 — kernel / primitive microbenchmarks (bench_plan.md §4.2, K1..K10).
 *
 * Cases are written once against `common/bench_kernels.h`; the per-backend TU supplies
 * the kernels.  Every case reports
 *
 *   * work units per rep (triplets, elements, calls) -> `per_unit_us`, `units/s`,
 *   * bytes per unit                       -> `bytes_per_second` (the GB/s roofline),
 *   * a checksum over a *probe window*      -> the correctness gate (§6.2: correctness
 *     before timing; a full read-back inside the window would measure the copy).
 *
 * Implemented here (§4.2 K0..K2, the kernels a PCG iteration spends its time in):
 *
 * | id | case | work unit | what a change is expected to move |
 * |---|---|---|---|
 * | K0.fill / K0.copy | 64 MiB of float | element | the achievable GB/s band on this GPU |
 * | K1.spmv.sym / .rbk / .rbk_sym / .rbk_sym_dot | 1000 rows x 3 dofs, >= 1e5 block entries | triplet | the solver's operator apply, per variant |
 * | K2.dot / .nrm2 / .axpy / .fill @ 1e6 and 3000 | dense n | element | the inner products (and the anomalous small n) |
 *
 * K3..K10 (BVH build/traverse, trajectory filter, FEM/ABD constitutions, preconditioner,
 * DyTopo) are the next step: they need a problem builder per subsystem, and the two
 * backends' builders differ.  K5's *budget* dimension is covered today by
 * `scripts/measure_ccd_bound.py` (see agent_docs/performance/2026-10-28-op-c3-*).
 *********************************************************************/
#include <common/bench_case.h>
#include <common/bench_harness.h>
#include <common/bench_kernels.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace uipc::benchmark
{
namespace
{
/// Rate helpers (§4.2): `units_per_rep` is the work of one measured rep.
void derive_rates(CaseRecord& record, double units_per_rep, double bytes_per_unit)
{
    record.work_units  = static_cast<std::uint64_t>(units_per_rep);
    record.per_call_us = record.timing.per_rep_us();
    record.per_unit_us = record.timing.per_unit_us(units_per_rep);
    record.units_per_second =
        record.per_call_us > 0.0 ? units_per_rep * 1.0e6 / record.per_call_us : 0.0;
    record.bytes_per_unit = bytes_per_unit;
    record.bytes_per_second =
        record.per_call_us > 0.0 ? units_per_rep * bytes_per_unit * 1.0e6 / record.per_call_us : 0.0;
}

/// Tier 1's counters: `dispatches` is the number of kernel calls in the window (Tier 0
/// reads these from the backend; the kernel bench counts what it issued).
void fill_kernel_counters(CaseRecord& record, std::uint64_t calls_per_rep)
{
    record.dispatches = calls_per_rep;
    record.commits    = calls_per_rep;
    record.syncs      = 1u;
}

/// The block size a sweep needs to be self-describing (OP-C1).  It is part of the
/// record's notes *and* a flag, so `summary.csv` groups by it.
void note_block_dim(BenchKernels& kernels, CaseRecord& r)
{
    r.notes += "block_dim=" + std::to_string(kernels.block_dim()) + "; ";
    r.flags.push_back("block_dim_" + std::to_string(kernels.block_dim()));
}

// ---------------------------------------------------------------------------
//  K0 — the bandwidth roofline
// ---------------------------------------------------------------------------
void run_bandwidth(CaseContext& ctx, BenchKernels& kernels, BenchKernels::BwOp op, std::uint64_t floats)
{
    CaseRecord r;
    r.id       = std::string{"K0."} + BenchKernels::name(op);
    r.tier     = "T1";
    r.work_unit = "element";
    r.scenario = std::string{BenchKernels::name(op)} + " over " + std::to_string(floats) + " floats";

    kernels.make_bandwidth(floats, 0x51u);
    const double bytes_per_unit = static_cast<double>(kernels.bandwidth_bytes_per_call(op))
                                  / static_cast<double>(kernels.bandwidth_floats());

    // One rep = 8 calls: a single 64 MiB pass at ~500 GB/s is 0.13 ms, too short to time
    // reliably against the clock granularity; 8 keeps a rep near a millisecond.
    constexpr int kCallsPerRep = 8;
    const double  checksum = kernels.bandwidth_checksum();
    (void)checksum;

    const auto t = time_reps(
        [&](std::uint64_t)
        {
            for(int c = 0; c < kCallsPerRep; ++c)
                kernels.bandwidth(op);
        },
        1u,
        ctx.options.rounds,
        ctx.options.warmup,
        false,
        [&] { kernels.synchronize(); });
    r.timing = t;
    derive_rates(r,
                 static_cast<double>(kernels.bandwidth_floats()) * kCallsPerRep,
                 bytes_per_unit);
    fill_kernel_counters(r, kCallsPerRep);

    // Correctness gate: fill must be exactly 1.0, copy must reproduce the source's
    // probe window (the same probe is used for both, so a wrong dispatch shows up).
    const double after = kernels.bandwidth_checksum();
    r.checksum          = after;
    r.checksum_ok       = std::isfinite(after);
    if(op == BenchKernels::BwOp::Fill)
    {
        const double expected = 1.0 * static_cast<double>(std::min<std::uint64_t>(64u, floats));
        r.expected_checksum    = expected;
        r.checksum_ok          = std::abs(after - expected) <= 1e-6 * std::max(1.0, expected);
        if(!r.checksum_ok)
            r.checksum_note = "fill probe window is not all 1.0";
    }
    note_block_dim(kernels, r);
    r.flags.push_back(t.stable() ? "stable" : "unstable");
    ctx.report.add(r);
}

// ---------------------------------------------------------------------------
//  K1 — SpMV
// ---------------------------------------------------------------------------
void run_spmv(CaseContext& ctx, BenchKernels& kernels, BenchKernels::SpmvVariant v,
              std::uint64_t rows, double entries_per_row)
{
    CaseRecord r;
    r.id        = std::string{"K1.spmv."} + BenchKernels::name(v);
    r.tier      = "T1";
    r.work_unit = "triplet";
    r.scenario  = std::string{BenchKernels::name(v)} + ": symmetric block BCOO, "
                  + std::to_string(rows) + " rows x 3 dofs";

    const auto setup = kernels.make_spmv(rows, entries_per_row, 0x4b31u);
    const double bytes_per_unit = static_cast<double>(setup.bytes_per_call)
                                  / static_cast<double>(std::max<std::uint64_t>(1u, setup.triplets));

    // A rep is 20 applies: one apply of ~1e5 triplets is a few microseconds on this GPU,
    // so a single call would sit inside the clock and launch noise.
    constexpr int kCallsPerRep = 20;
    kernels.synchronize();
    const double before = kernels.spmv_checksum();

    const auto t = time_reps(
        [&](std::uint64_t)
        {
            for(int c = 0; c < kCallsPerRep; ++c)
                kernels.spmv(v);
        },
        1u,
        ctx.options.rounds,
        ctx.options.warmup,
        false,
        [&] { kernels.synchronize(); });
    r.timing = t;
    derive_rates(r,
                 static_cast<double>(setup.triplets) * kCallsPerRep,
                 bytes_per_unit);
    fill_kernel_counters(r, kCallsPerRep);

    const double after = kernels.spmv_checksum();
    r.checksum          = after;
    r.expected_checksum = before;
    // `y = 1.0 * A * x + 1.0 * y` accumulates over the calls, so the probe window is
    // *expected to grow*; the gate is that it stays finite and actually moved (a kernel
    // that silently stopped writing would freeze the value).
    r.checksum_ok = std::isfinite(after) && std::abs(after) > 0.0;
    if(!r.checksum_ok)
        r.checksum_note = "the result probe window did not move";
    r.notes += "triplets=" + std::to_string(setup.triplets)
               + " dofs=" + std::to_string(setup.dofs) + "; ";
    note_block_dim(kernels, r);
    r.flags.push_back(t.stable() ? "stable" : "unstable");
    ctx.report.add(r);
}

// ---------------------------------------------------------------------------
//  K2 — dense BLAS / reduction
// ---------------------------------------------------------------------------
void run_blas(CaseContext& ctx, BenchKernels& kernels, BenchKernels::BlasOp op, std::uint64_t n)
{
    CaseRecord r;
    r.id        = std::string{"K2."} + BenchKernels::name(op) + "." + std::to_string(n);
    r.tier      = "T1";
    r.work_unit = "element";
    r.scenario  = std::string{BenchKernels::name(op)} + " at n = " + std::to_string(n);

    kernels.make_dense(n, 0x4b32u);
    const double bytes_per_unit = n == 0u ? 0.0
                                          : static_cast<double>(kernels.dense_bytes_per_call(int(op)))
                                                / static_cast<double>(n);

    // dot/nrm2 already contain a device wait (they return a host value), so a rep of 20
    // would measure 20 round trips plus 20 launches -- which is exactly what the solver
    // pays.  axpy/fill are pure launches, so they get more calls per rep.
    const int calls = (op == BenchKernels::BlasOp::Dot || op == BenchKernels::BlasOp::Nrm2)
                          ? 20
                          : 50;
    kernels.synchronize();

    const auto t = time_reps(
        [&](std::uint64_t)
        {
            for(int c = 0; c < calls; ++c)
                kernels.blas(op);
        },
        1u,
        ctx.options.rounds,
        ctx.options.warmup,
        false,
        [&] { kernels.synchronize(); });
    r.timing = t;
    derive_rates(r, static_cast<double>(n) * calls, bytes_per_unit);
    fill_kernel_counters(r, static_cast<std::uint64_t>(calls));

    const double after = kernels.blas_checksum();
    r.checksum          = after;
    r.checksum_ok       = std::isfinite(after);
    note_block_dim(kernels, r);
    r.flags.push_back(t.stable() ? "stable" : "unstable");
    if(op == BenchKernels::BlasOp::Dot || op == BenchKernels::BlasOp::Nrm2)
        r.flags.push_back("sync_per_rep");
    ctx.report.add(r);
}

// ---------------------------------------------------------------------------
//  the registry
// ---------------------------------------------------------------------------
}  // namespace

std::vector<CaseDef> const& t1_registry()
{
    static const std::vector<CaseDef> registry{
        // K0 first: cheapest, and it establishes the bandwidth band of this GPU.
        {"K0.fill", "T1", "64 MiB fill (write-only bandwidth)", [](CaseContext& ctx)
         {
             run_bandwidth(ctx, ctx.as<BenchKernels>(), BenchKernels::BwOp::Fill, 16u * 1024u * 1024u);
         }},
        {"K0.copy", "T1", "64 MiB device-to-device copy", [](CaseContext& ctx)
         {
             run_bandwidth(ctx, ctx.as<BenchKernels>(), BenchKernels::BwOp::Copy, 16u * 1024u * 1024u);
         }},
        // K1: the §4.2 shape (1000 rows, 0.2 density, >= 1e5 block entries).
        {"K1.spmv.sym", "T1", "SpMV sym_spmv, symmetric block BCOO", [](CaseContext& ctx)
         {
             run_spmv(ctx, ctx.as<BenchKernels>(), BenchKernels::SpmvVariant::Sym, 1000u, 0.2);
         }},
        {"K1.spmv.rbk", "T1", "SpMV rbk_spmv (general row-blocked)", [](CaseContext& ctx)
         {
             run_spmv(ctx, ctx.as<BenchKernels>(), BenchKernels::SpmvVariant::Rbk, 1000u, 0.2);
         }},
        {"K1.spmv.rbk_sym", "T1", "SpMV rbk_sym_spmv", [](CaseContext& ctx)
         {
             run_spmv(ctx, ctx.as<BenchKernels>(), BenchKernels::SpmvVariant::RbkSym, 1000u, 0.2);
         }},
        {"K1.spmv.rbk_sym_dot", "T1", "SpMV + fused x^T(Ax)", [](CaseContext& ctx)
         {
             run_spmv(ctx, ctx.as<BenchKernels>(), BenchKernels::SpmvVariant::RbkSymDot, 1000u, 0.2);
         }},
        // K2: the PCG inner products, at the large size and at the anomalous 3000.
        {"K2.dot.1000000", "T1", "dot at n = 1e6", [](CaseContext& ctx)
         {
             run_blas(ctx, ctx.as<BenchKernels>(), BenchKernels::BlasOp::Dot, 1000000u);
         }},
        {"K2.dot.3000", "T1", "dot at n = 3000 (the anomalous case)", [](CaseContext& ctx)
         {
             run_blas(ctx, ctx.as<BenchKernels>(), BenchKernels::BlasOp::Dot, 3000u);
         }},
        {"K2.nrm2.1000000", "T1", "nrm2 at n = 1e6", [](CaseContext& ctx)
         {
             run_blas(ctx, ctx.as<BenchKernels>(), BenchKernels::BlasOp::Nrm2, 1000000u);
         }},
        {"K2.axpy.1000000", "T1", "axpy at n = 1e6", [](CaseContext& ctx)
         {
             run_blas(ctx, ctx.as<BenchKernels>(), BenchKernels::BlasOp::Axpy, 1000000u);
         }},
        {"K2.fill.1000000", "T1", "fill at n = 1e6", [](CaseContext& ctx)
         {
             run_blas(ctx, ctx.as<BenchKernels>(), BenchKernels::BlasOp::Fill, 1000000u);
         }},
    };
    return registry;
}
}  // namespace uipc::benchmark

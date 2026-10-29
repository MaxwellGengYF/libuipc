/********************************************************************
 * @file luisa_frame_stream_test.cpp
 * @brief Regression suite for the frame command list facade
 *        (luisa_device/stream.h: LStream::record / record_upload /
 *        commit / synchronize / flush and `LStreamCounters`).
 *
 * Every case drives a *local* `lud::LStream` over `lud::main_stream()` so the
 * counters start at zero and a case cannot disturb the next one. The contracts
 * under test (optimize_plan.md OP-A1/OP-A2):
 *
 *  1. `record()` batches into the open frame `lcrt::CommandList`; without an
 *     open list it submits immediately (standalone code keeps working).
 *  2. `commit()` submits the list once and closes it; `frame_opens` counts only
 *     real opens, so `begin()` stays idempotent.
 *  3. `synchronize()` / `flush()` are *barriers*, not mode switches: they
 *     commit, wait, and re-open the list, so the rest of the frame keeps
 *     batching.
 *  4. Ordering no longer follows statement order: a read-back, a device-to-
 *     device copy or a fill issued while a list is open must be *recorded* to
 *     run after the kernels already recorded (the live `z = r` bug).
 *  5. `record_upload()` consumes the host pointer inside the statement, so an
 *     upload never outlives the caller's stack buffer.
 *
 * NOTE: no `zzz_` prefix — the runtime teardown must stay in the last TU.
 *********************************************************************/
#include "luisa_test_runtime.h"

#include <luisa_device/luisa_device.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace lud  = uipc::backend::luisa::luisa_device;
namespace lcrt = ::luisa::compute;

using Float = double;  // uipc::Float is double

namespace
{
// ---------------------------------------------------------------------------
//  kernels (compiled once, in a function-local static)
// ---------------------------------------------------------------------------
/// The trivial kernel every batching case records: write `value` into every
/// element with index < n. Nothing but device-resident arguments, so it is a
/// recordable command.
auto& write_value_kernel()
{
    static auto k =
        lud::compile1<lcrt::Buffer<Float>, Float, uint32_t>(
            [](lcrt::BufferVar<Float> v,
               lcrt::Var<Float>       value,
               lcrt::Var<uint32_t>    n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if (i < n)
                {
                    v->write(i, value);
                };
            });
    return k;
}

// ---------------------------------------------------------------------------
//  host helpers
// ---------------------------------------------------------------------------
/// Read a buffer back on the host through `stream`: the download is *recorded*
/// (so it is ordered after whatever the frame recorded) and then the barrier
/// commits and waits.
[[nodiscard]] std::vector<Float> read_back(lud::LStream& stream,
                                           lud::LDeviceBuffer<Float>& buf)
{
    std::vector<Float> host(buf.size(), -1.0);
    buf.view().copy_to(std::span<Float>{host}, &stream);
    stream.synchronize();
    return host;
}

/// The lifetime contract of OP-A2 (test 8): the only source of the values is a
/// *local* vector, clobbered before this function returns. A deferred
/// (recorded) upload would read the clobbered memory at commit.
void upload_from_temporary(lud::LStream& stream, lud::LBufferView<Float> dst)
{
    std::vector<Float> local{11.0, 22.0, 33.0, 44.0};
    dst.copy_from(std::span<const Float>{local}, &stream);
    std::fill(local.begin(), local.end(), -1.0);  // clobber, then return
}
}// namespace

// ---------------------------------------------------------------------------
//  1. batching: N records -> one commit
// ---------------------------------------------------------------------------
TEST_CASE("record batches into a single commit", "[luisa][frame-stream]")
{
    test_runtime();

    lud::LStream               stream{lud::main_stream()};
    lud::LDeviceBuffer<Float> buf;
    buf.resize(4u);

    const auto base = stream.counters();

    stream.begin();
    for(int i = 0; i < 5; ++i)
    {
        stream.record(write_value_kernel()(buf.view(), 1.0, 4u).dispatch(4u));
    }
    stream.commit();

    REQUIRE(stream.counters().commands == 5u);
    REQUIRE(stream.counters().commits == 1u);
    REQUIRE(stream.counters().frame_opens == 1u);
    REQUIRE(stream.counters().direct_submits == 0u);
    REQUIRE(!stream.frame_open());

    // `LStreamCounters operator-` (the engine samples a frame as a delta).
    const auto delta = stream.counters() - base;
    REQUIRE(delta.commands == 5u);
    REQUIRE(delta.commits == 1u);
    REQUIRE(delta.frame_opens == 1u);
    REQUIRE(delta.direct_submits == 0u);
    REQUIRE(delta.uploads == 0u);
}

// ---------------------------------------------------------------------------
//  2. no frame open -> per-statement submission (the old behaviour)
// ---------------------------------------------------------------------------
TEST_CASE("record without an open frame submits immediately",
          "[luisa][frame-stream]")
{
    test_runtime();

    lud::LStream               stream{lud::main_stream()};
    lud::LDeviceBuffer<Float> buf;
    buf.resize(4u);

    // no begin(): the buffer is written *now*, not at a later commit
    for(int i = 0; i < 3; ++i)
    {
        stream.record(write_value_kernel()(buf.view(), 5.0, 4u).dispatch(4u));
    }

    REQUIRE(stream.counters().commands == 0u);
    REQUIRE(stream.counters().direct_submits == 3u);
    REQUIRE(stream.counters().commits == 0u);
    REQUIRE(stream.counters().frame_opens == 0u);
    REQUIRE(!stream.frame_open());

    const auto got = read_back(stream, buf);
    REQUIRE(got.size() == 4u);
    for(const auto v : got)
    {
        REQUIRE(v == 5.0);  // the three direct submits really executed
    }
    // synchronize() must not invent a commit for a stream that never batched.
    REQUIRE(stream.counters().commits == 0u);
    REQUIRE(stream.counters().commands == 0u);
}

// ---------------------------------------------------------------------------
//  3. synchronize() re-opens the frame (OP-A1: a barrier, not a mode switch)
// ---------------------------------------------------------------------------
TEST_CASE("synchronize re-opens an open frame", "[luisa][frame-stream]")
{
    test_runtime();

    lud::LStream               stream{lud::main_stream()};
    lud::LDeviceBuffer<Float> buf;
    buf.resize(4u);

    stream.begin();
    stream.record(write_value_kernel()(buf.view(), 2.0, 4u).dispatch(4u));
    stream.synchronize();

    // re-opened, so batching survives the barrier
    REQUIRE(stream.frame_open());
    REQUIRE(stream.counters().commits == 1u);
    REQUIRE(stream.counters().syncs == 1u);

    stream.record(write_value_kernel()(buf.view(), 3.0, 4u).dispatch(4u));
    REQUIRE(stream.counters().direct_submits == 0u);  // batched, not submitted
    stream.commit();

    REQUIRE(stream.counters().commands == 2u);
    REQUIRE(stream.counters().commits == 2u);
    REQUIRE(!stream.frame_open());

    const auto got = read_back(stream, buf);
    for(const auto v : got)
    {
        REQUIRE(v == 3.0);
    }
}

// ---------------------------------------------------------------------------
//  4. flush() == synchronize(): the regression this file exists for
// ---------------------------------------------------------------------------
TEST_CASE("flush keeps batching alive", "[luisa][frame-stream]")
{
    test_runtime();

    lud::LStream               stream{lud::main_stream()};
    lud::LDeviceBuffer<Float> buf;
    buf.resize(4u);

    stream.begin();
    stream.record(write_value_kernel()(buf.view(), 2.0, 4u).dispatch(4u));
    stream.flush();  // commit + wait, and keep the frame list open

    REQUIRE(stream.frame_open());
    REQUIRE(stream.counters().commits == 1u);
    REQUIRE(stream.counters().syncs == 1u);  // flush() is synchronize()

    // The regression: a read-back used to close the list for the rest of the
    // frame, so this second dispatch degraded to a per-statement submit.
    stream.record(write_value_kernel()(buf.view(), 4.0, 4u).dispatch(4u));
    stream.flush();

    REQUIRE(stream.counters().commands == 2u);
    REQUIRE(stream.counters().commits == 2u);
    REQUIRE(stream.counters().direct_submits == 0u);
    REQUIRE(stream.counters().syncs == 2u);
    REQUIRE(stream.frame_open());

    const auto got = read_back(stream, buf);
    for(const auto v : got)
    {
        REQUIRE(v == 4.0);
    }
}

// ---------------------------------------------------------------------------
//  5. a read-back inside the frame sees the recorded write
// ---------------------------------------------------------------------------
TEST_CASE("a recorded write is visible to a read-back in the same frame",
          "[luisa][frame-stream]")
{
    test_runtime();

    lud::LStream               stream{lud::main_stream()};
    lud::LDeviceBuffer<Float> buf;
    buf.resize(1u);

    stream.begin();
    stream.record(write_value_kernel()(buf.view(), 42.0, 1u).dispatch(1u));

    // No explicit flush: the read-back itself is stream-aware. It records the
    // download behind the pending dispatch and then barriers. An implementation
    // that submitted the read on the *raw* stream would let it jump ahead of
    // the recorded dispatch and return the pre-frame value 0 here.
    const Float v = lud::read_host_sync(stream, buf.cview());
    REQUIRE(v == 42.0);

    REQUIRE(stream.counters().direct_submits == 0u);  // read was recorded
    REQUIRE(stream.counters().commands == 2u);        // dispatch + download
    REQUIRE(stream.counters().commits == 1u);
}

// ---------------------------------------------------------------------------
//  6. device-to-device copy ordering (the live `z = r` bug)
// ---------------------------------------------------------------------------
TEST_CASE("a device-to-device copy is ordered after the recorded kernel",
          "[luisa][frame-stream]")
{
    test_runtime();

    lud::LStream               stream{lud::main_stream()};
    lud::LDeviceBuffer<Float> a;
    lud::LDeviceBuffer<Float> b;
    a.resize(1u);
    b.resize(1u);  // value-initialised: b holds 0 until the kernel writes it

    stream.begin();
    stream.record(write_value_kernel()(b.view(), 7.0, 1u).dispatch(1u));
    // Must be recorded: a copy submitted on the raw stream would run *before*
    // the recorded kernel and copy b's old contents (0) into a.
    a.view().copy_from(b.cview(), &stream);
    REQUIRE(stream.counters().commands == 2u);
    REQUIRE(stream.counters().direct_submits == 0u);
    stream.commit();
    stream.synchronize();

    const auto got = read_back(stream, a);
    REQUIRE(got.size() == 1u);
    REQUIRE(got[0] == 7.0);
}

// ---------------------------------------------------------------------------
//  7. an upload inside a frame: ordered, accounted, and batching survives
// ---------------------------------------------------------------------------
TEST_CASE("an upload inside a frame is ordered and accounted",
          "[luisa][frame-stream]")
{
    test_runtime();

    lud::LStream               stream{lud::main_stream()};
    lud::LDeviceBuffer<Float> dst;
    dst.resize(4u);

    std::vector<Float> host_values{1.0, 2.0, 3.0, 4.0};

    stream.begin();
    stream.record(write_value_kernel()(dst.view(), 9.0, 4u).dispatch(4u));
    // The upload captures `host_values.data()`, so record_upload() must submit
    // the pending dispatch first (order) and then the upload itself.
    dst.view().copy_from(std::span<const Float>{host_values}, &stream);

    REQUIRE(stream.counters().uploads == 1u);
    REQUIRE(stream.counters().commands == 1u);        // only the dispatch
    REQUIRE(stream.counters().commits == 1u);         // the upload committed it
    REQUIRE(stream.counters().direct_submits == 1u);  // the upload itself
    REQUIRE(stream.frame_open());                     // re-opened afterwards

    stream.commit();
    stream.synchronize();

    const auto got = read_back(stream, dst);
    REQUIRE(got.size() == 4u);
    REQUIRE(got == host_values);  // the upload landed after the 9.0 write
}

// ---------------------------------------------------------------------------
//  8. the upload lifetime contract (deferred memcpy would read dead memory)
// ---------------------------------------------------------------------------
TEST_CASE("an upload does not outlive its host buffer", "[luisa][frame-stream]")
{
    test_runtime();

    lud::LStream               stream{lud::main_stream()};
    lud::LDeviceBuffer<Float> dst;
    dst.resize(4u);

    stream.begin();
    upload_from_temporary(stream, dst.view());  // clobbers its vector on return
    stream.commit();
    stream.synchronize();

    const std::vector<Float> want{11.0, 22.0, 33.0, 44.0};
    const auto               got = read_back(stream, dst);
    REQUIRE(got.size() == 4u);
    // -1.0 here means the upload was deferred (recorded) instead of submitted
    // by record_upload(): LC's BufferUploadCommand stores the source pointer.
    REQUIRE(got == want);
}

// ---------------------------------------------------------------------------
//  9. fill inside a frame is ordered after the recorded write
// ---------------------------------------------------------------------------
TEST_CASE("fill inside a frame is ordered", "[luisa][frame-stream]")
{
    test_runtime();

    lud::LStream               stream{lud::main_stream()};
    lud::LDeviceBuffer<Float> buf;
    buf.resize(4u);

    stream.begin();
    stream.record(write_value_kernel()(buf.view(), 1.0, 4u).dispatch(4u));
    buf.view().fill(3.0, &stream);  // a dispatch: recordable, not direct

    REQUIRE(stream.counters().commands == 2u);
    REQUIRE(stream.counters().direct_submits == 0u);

    stream.commit();
    stream.synchronize();

    const auto got = read_back(stream, buf);
    REQUIRE(got.size() == 4u);
    // 1.0 means the fill jumped ahead of the recorded write; -1.0/0.0 means it
    // was skipped entirely.
    for(const auto v : got)
    {
        REQUIRE(v == 3.0);
    }
}

// ---------------------------------------------------------------------------
// 10. frame_opens counts only real opens (begin() is idempotent)
// ---------------------------------------------------------------------------
TEST_CASE("frame_opens counts only real opens", "[luisa][frame-stream]")
{
    test_runtime();

    lud::LStream stream{lud::main_stream()};

    stream.begin();
    stream.begin();
    stream.begin();
    REQUIRE(stream.frame_open());
    REQUIRE(stream.counters().frame_opens == 1u);

    stream.commit();
    REQUIRE(!stream.frame_open());

    stream.begin();
    REQUIRE(stream.frame_open());
    REQUIRE(stream.counters().frame_opens == 2u);

    stream.commit();

    // reset_counters() zeroes the sample; the counters are a plain snapshot
    // (the frame list was closed by the commit above).
    stream.reset_counters();
    REQUIRE(stream.counters().frame_opens == 0u);
    REQUIRE(stream.counters().commands == 0u);
    REQUIRE(stream.counters().commits == 0u);
    REQUIRE(!stream.frame_open());
}

//  11. OP-C1/OP-C3: the trace-time knobs (block size, CCD budget)
TEST_CASE("tuning knobs are process-wide and name-keyed",
          "[luisa][frame-stream]")
{
    auto& t = lud::tuning();
    const auto saved_dim  = t.block_dim;
    const auto saved_iter = t.ccd_max_iter;

    // The historical default is LC's own 256 (luisa_device/kernel.h).
    REQUIRE(lud::kDefaultBlockDim == 256u);

    // Unnamed lookup == the global knob; a named lookup falls back to it.
    t.block_dim = 128u;
    REQUIRE(lud::block_dim_for() == 128u);
    REQUIRE(lud::block_dim_for("not-registered") == 128u);

    // A registered name wins, and re-registering patches instead of appending.
    lud::detail::set_named_block_dim("k.named", 64u);
    REQUIRE(lud::block_dim_for("k.named") == 64u);
    lud::detail::set_named_block_dim("k.named", 512u);
    REQUIRE(lud::block_dim_for("k.named") == 512u);
    REQUIRE(lud::block_dim_for("trajectory_filter.toi") == 128u);

    // OP-C3: the CCD budget the TOI kernels trace into their `$while`.
    t.ccd_max_iter = 16;
    REQUIRE(lud::ccd_max_iter() == 16);
    t.ccd_max_iter = lud::kDefaultCcdMaxIter;
    REQUIRE(lud::ccd_max_iter() == 64);

    t.block_dim    = saved_dim;
    t.ccd_max_iter = saved_iter;
}

//  12. OP-A4: `compiles` counts shader creation, never dispatch
TEST_CASE("compile counter counts shader creation, not dispatch",
          "[luisa][frame-stream]")
{
    test_runtime();
    const auto before = lud::compile_stats::total();

    // Two *calls* = two shader creations, even though LC's disk cache makes the
    // second one cheap (this is the metric that must be 0 per frame).
    for(int r = 0; r < 2; ++r)
    {
        auto shader = lud::compile1<lcrt::Buffer<Float>, Float, uint32_t>(
            [](lcrt::BufferVar<Float> v,
               lcrt::Var<Float> value,
               lcrt::Var<uint32_t> n) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i < n)
                {
                    v->write(i, value);
                };
            });
    }
    REQUIRE(lud::compile_stats::total() - before == 2u);

    // Dispatching an already-compiled kernel is free of compiles: the engine's
    // "compiles per frame == 0 after frame 1" acceptance (optimize_plan.md A4).
    lud::LDeviceBuffer<Float> buf;
    buf.resize(8u);
    const auto after_compile = lud::compile_stats::total();
    lud::main_stream() << write_value_kernel()(buf.view(), 7.0, 8u).dispatch(8u);
    lud::main_stream().synchronize();
    REQUIRE(lud::compile_stats::total() == after_compile);
}

/* What these cases protect against (read this before changing stream.h):
 *
 *  - Ordering. `record()` is the only legal way to issue a command while a
 *    frame list is open. A read-back, a d2d copy or a fill submitted on the raw
 *    lcrt::Stream jumps ahead of everything recorded: test 5 would read 0,
 *    test 6 would copy b's previous contents (`z = r`), test 9 would fill
 *    before the recorded write. Tests 5/6/9 fail loudly if a call site or a
 *    container helper is switched back to `stream.stream() << cmd`.
 *  - Deferred-upload lifetime. LC's BufferUploadCommand stores the *host*
 *    pointer it was given. `record_upload()` therefore never defers: it commits
 *    what is pending (so order is preserved) and submits the upload now, then
 *    re-opens the frame. Test 8 passes the address of a stack vector that is
 *    clobbered before the frame is committed; a recorded upload would upload
 *    -1.0 (silently, no error). Test 7 checks the accounting of that path.
 *  - Batching survived a barrier. `synchronize()`/`flush()` must commit, wait
 *    and re-open (OP-A1). Tests 3 and 4 assert both the re-open and the counter
 *    deltas; without the re-open the rest of the frame degrades to one
 *    submission per statement.
 *  - Counter hygiene. Tests 1/2/10 pin `commands` vs `direct_submits` vs
 *    `commits` vs `frame_opens`, i.e. that the numbers the engine samples as a
 *    per-frame delta still mean "what a CommandList absorbed" and "what
 *    escaped it".
 */

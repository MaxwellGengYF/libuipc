#include <luisa_device/stream.h>

// NVTX ranges for the frame command-list lifecycle (BN1 I2, bench_plan.md 13.3): they
// make the per-frame commit and sync cost visible on the profiler timeline, not only in
// a counter. Names are the fixed `LC.*` vocabulary from that section. With
// `UIPC_NVTX=0` every call below is an empty inline stub.
#include <backends/common/nvtx_range.h>

namespace uipc::backend::luisa::luisa_device
{
LStream& LStream::current()
{
    // Leaked on purpose (see the header): the cached objects and the LC resources
    // they own must not be destroyed by static destruction after Runtime::shutdown().
    static auto* s = new LStream{main_stream()};
    return *s;
}

void LStream::begin()
{
    uipc::nvtx::range guard{"LC.begin"};
    if(!m_list.has_value())
    {
        m_list.emplace(lcrt::CommandList::create());
        ++m_counters.frame_opens;
    }
}

void LStream::commit()
{
    if(!m_list.has_value())
    {
        return;
    }

    if(!m_dirty)
    {
        // An open but empty list: drop it without submitting. An empty list is legal
        // to destroy, and the counters must describe real submissions only.
        m_list.reset();
        return;
    }

    // Only a commit that has something to submit is a *submission*: the range is
    // inside the branch so that `nvtx_pushpop_sum` on `LC.commit` counts real
    // submits, not the no-op calls made from synchronize()/flush() below.
    uipc::nvtx::range guard{"LC.commit"};

    (*m_stream) << m_list->commit();
    m_list.reset();
    m_dirty = false;
    ++m_counters.commits;
}

void LStream::flush()
{
    // Identical to synchronize() on purpose: commit, wait, and keep the frame list
    // open. `flush` is the older name for "commit and wait"; it must not have the
    // side effect of turning batching off for the rest of the frame.
    uipc::nvtx::range guard{"LC.flush"};
    synchronize();
}

void LStream::synchronize()
{
    uipc::nvtx::range guard{"LC.sync"};

    // Anything recorded but not submitted must be submitted first, otherwise a
    // following synchronize() would not observe the recorded work.
    const bool reopen = m_list.has_value();
    commit();

    const auto t0 = std::chrono::steady_clock::now();
    m_stream->synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    m_counters.sync_ms +=
        std::chrono::duration<double, std::milli>{t1 - t0}.count();
    ++m_counters.syncs;

    // A barrier, not a mode switch (OP-A1/R-4): without this every read-back in a
    // frame leaves the list closed, and the rest of that frame silently falls back to
    // one submission per statement. Ordering is unaffected -- the device is idle here.
    if(reopen)
    {
        begin();
    }
}

LStream::~LStream()
{
    // The cached objects own device buffers, and destroying a buffer after
    // Runtime::shutdown() is fatal in the LC CUDA backend (it fast-fails the
    // process). A long-lived stream (a static one, or one owned by a system that
    // outlives the runtime teardown) therefore releases them only while the runtime
    // is still alive; afterwards it leaks them, which is harmless at exit.
    if(!Runtime::instance().initialized())
        return;

    // Submit whatever is still recorded before the cached objects disappear: a
    // pending list holds raw handles into them, and LC asserts on destroying a
    // non-empty uncommitted list ("Destructing non-empty command list. Did you forget
    // to commit?").
    commit();

    for(auto& o : m_cached)
    {
        if(o.ptr)
            o.destroy(o.ptr);
    }
}

void* LStream::cached_object(const void* key, void* (*create)(), void (*destroy)(void*))
{
    for(auto& o : m_cached)
    {
        if(o.key == key)
            return o.ptr;
    }
    auto* ptr = create();
    m_cached.push_back(CachedObject{key, ptr, destroy});
    return ptr;
}
}// namespace uipc::backend::luisa::luisa_device

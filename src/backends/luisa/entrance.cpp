#include <sim_engine.h>

#include <backends/common/module.h>

// The backend module entry points are declared with C linkage in
// backends/common/module.h, so they must be defined at global scope (a
// namespace would give them C++ linkage and the loader would not find them).
UIPC_BACKEND_API EngineInterface* uipc_create_engine(EngineCreateInfo* info)
{
    return new uipc::backend::luisa::SimEngine(info);
}

UIPC_BACKEND_API void uipc_destroy_engine(EngineInterface* engine)
{
    delete engine;
}

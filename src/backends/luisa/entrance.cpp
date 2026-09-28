#include <sim_engine.h>

#include <backends/common/module.h>

namespace uipc::backend::luisa
{
// Backend module entry points (see backends/common/module.h).
UIPC_BACKEND_API EngineInterface* uipc_create_engine(EngineCreateInfo* info)
{
    return new SimEngine(info);
}

UIPC_BACKEND_API void uipc_destroy_engine(EngineInterface* engine)
{
    delete engine;
}
}// namespace uipc::backend::luisa

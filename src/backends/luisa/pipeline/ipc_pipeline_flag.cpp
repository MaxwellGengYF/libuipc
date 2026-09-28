#include <pipeline/ipc_pipeline_flag.h>
#include <sim_engine.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<luisa::IPCPipelineFlag>
{
  public:
    static U<luisa::IPCPipelineFlag> create(SimEngine& engine)
    {
        auto scene      = dynamic_cast<luisa::SimEngine&>(engine).world().scene();
        auto ctype_attr = scene.config().find<std::string>("contact/constitution");

        if(ctype_attr->view()[0] != "ipc")
        {
            return nullptr;  // Not an IPC pipeline
        }
        return uipc::make_unique<luisa::IPCPipelineFlag>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(IPCPipelineFlag);
void IPCPipelineFlag::do_build() {}
}  // namespace uipc::backend::luisa

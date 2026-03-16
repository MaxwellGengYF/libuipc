#include <time_integrator/bdf1_flag.h>
#include <sim_engine.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<luisa::BDF1Flag>
{
  public:
    static U<luisa::BDF1Flag> create(SimEngine& engine)
    {
        auto scene = dynamic_cast<luisa::SimEngine&>(engine).world().scene();
        auto itype_attr = scene.config().find<std::string>("integrator/type");

        if(itype_attr->view()[0] != "bdf1")
        {
            return nullptr;  // Not a BDF1 integrator
        }
        return uipc::make_unique<luisa::BDF1Flag>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(BDF1Flag);
void BDF1Flag::do_build() {}
}  // namespace uipc::backend::luisa

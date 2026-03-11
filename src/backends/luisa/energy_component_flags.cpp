#include <energy_component_flags.h>

namespace uipc::backend::luisa
{
std::string enum_flags_name(EnergyComponentFlags flags)
{
    return std::string{magic_enum::enum_flags_name(flags)};
}
}  // namespace uipc::backend::luisa

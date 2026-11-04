#include <energy_component_flags.h>

namespace uipc::backend::luisa
{
std::string enum_flags_name(EnergyComponentFlags flags)
{
    auto name = magic_enum::enum_flags_name(flags);
    return std::string{name.data(), name.size()};
}
}// namespace uipc::backend::luisa

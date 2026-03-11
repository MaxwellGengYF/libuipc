#pragma once
#include <type_define.h>
#include <uipc/common/flag.h>

namespace uipc::backend::luisa
{
/**
 * @brief Flags for energy components in the simulation
 * 
 * Used to distinguish between contact and non-contact (complement) energy
 * components in the IPC solver.
 */
enum class EnergyComponentFlags : uint32_t
{
    None = 0,
    // Contact Part
    Contact = 1,
    // NonContact Part
    Complement = 1 << 1,
    All        = Contact | Complement
};

/**
 * @brief Get the string representation of EnergyComponentFlags
 */
std::string enum_flags_name(EnergyComponentFlags flags);
}  // namespace uipc::backend::luisa

namespace magic_enum::customize
{
template <>
struct enum_range<uipc::backend::luisa::EnergyComponentFlags>
{
    static constexpr bool is_flags = true;
};
}  // namespace magic_enum::customize

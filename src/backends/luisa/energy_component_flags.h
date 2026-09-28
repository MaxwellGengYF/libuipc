#pragma once
/********************************************************************
 * @file energy_component_flags.h
 * @brief Energy component flags (port of
 *        src/backends/cuda/energy_component_flags.h — pure host code, namespace
 *        change only).
 *
 * A dytopo-effect reporter declares which part of the energy it contributes to
 * (contact / complement / all); the manager partitions its reporters with this
 * flag. `magic_enum::enum_flags_name` gives the readable log spelling.
 *********************************************************************/
#include <type_define.h>
#include <uipc/common/flag.h>

namespace uipc::backend::luisa
{
enum class EnergyComponentFlags : uint32_t
{
    None = 0,
    // Contact Part
    Contact = 1,
    // NonContact Part
    Complement = 1 << 1,
    All = Contact | Complement
};

/**
 * @brief Get the string representation of EnergyComponentFlags
 *
 * A wrapper around magic_enum::enum_flags_name (the reference could not call it
 * directly under NVCC; kept as a non-template function so the call sites and the
 * log text are identical).
 */
std::string enum_flags_name(EnergyComponentFlags flags);
}// namespace uipc::backend::luisa

namespace magic_enum::customize
{
template <>
struct enum_range<uipc::backend::luisa::EnergyComponentFlags>
{
    static constexpr bool is_flags = true;
};
}// namespace magic_enum::customize

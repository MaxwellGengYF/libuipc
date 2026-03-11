#pragma once
#include <uipc/common/type_define.h>
#include <uipc/common/smart_pointer.h>

namespace uipc::backend::luisa
{
class SimSystem;

/**
 * @brief A slot for storing a SimSystem pointer with type information
 */
class SimSystemSlot
{
  public:
    SimSystemSlot(SimSystem* system = nullptr)
        : m_system(system)
    {}

    SimSystem* get() const noexcept { return m_system; }
    void       set(SimSystem* system) noexcept { m_system = system; }

    explicit operator bool() const noexcept { return m_system != nullptr; }

  private:
    SimSystem* m_system = nullptr;
};
}  // namespace uipc::backend::luisa

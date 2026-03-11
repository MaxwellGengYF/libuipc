#pragma once
#include <vector>
#include <functional>
#include <uipc/common/type_define.h>
#include <sim_system.h>

namespace uipc::backend::luisa
{
/**
 * @brief A collection of actions that can be registered and invoked
 */
template <typename Signature>
class SimActionCollection
{
  public:
    using Action = std::function<Signature>;

    void operator()() const
    {
        for(const auto& action : m_actions)
        {
            action.callback();
        }
    }

    void clear() noexcept { m_actions.clear(); }

    void insert(std::function<Signature>&& action) noexcept
    {
        m_actions.push_back(ActionSlot{nullptr, std::move(action)});
    }

    void register_action(SimSystem& system, std::function<Signature>&& action) noexcept
    {
        m_actions.push_back(ActionSlot{&system, std::move(action)});
    }

  private:
    struct ActionSlot
    {
        SimSystem*          system;
        std::function<void()> callback;
    };
    std::vector<ActionSlot> m_actions;
};
}  // namespace uipc::backend::luisa

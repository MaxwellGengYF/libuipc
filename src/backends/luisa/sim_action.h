#pragma once
#include <functional>
#include <uipc/common/smart_pointer.h>

namespace uipc::backend::luisa
{
/**
 * @brief A simple wrapper for std::function to allow for comparison
 */
template <typename T>
class SimAction
{
  public:
    SimAction(std::function<T>&& f)
        : m_id(reinterpret_cast<uint64_t>(&f))
        , m_func(std::move(f))
    {}

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) const
    {
        return m_func(std::forward<Args>(args)...);
    }

    uint64_t id() const noexcept { return m_id; }

  private:
    uint64_t            m_id;
    std::function<void()> m_func;
};
}  // namespace uipc::backend::luisa

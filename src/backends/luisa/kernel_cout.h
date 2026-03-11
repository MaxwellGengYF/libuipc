#pragma once
#include <luisa/dsl/stmt.h>

/*****************************************************************/ /**
 * \file   kernel_cout.h
 * \brief  Kernel-side console output utility using LuisaCompute
 * 
 * To use: `cout << xxx` in kernel, you should now
 * 
 * @code
 * 
 * Kernel1D kernel = [&](...) {
 *     ...
 *     KernelCout::print("Value at {}: {}\n", i, data[i]);
 *     ...
 * };
 * 
 * @endcode
 * 
 * Or use device_log directly:
 * @code
 * device_log("Value at {}: {}\n", i, data[i]);
 * @endcode
 * 
 * \author Lenovo
 * \date   April 2025
 *********************************************************************/

namespace uipc::backend::luisa
{
class KernelCout
{
  public:
    /**
     * @brief Print formatted output from device/kernel code
     * 
     * @tparam Args Variadic template for argument types
     * @param fmt Format string (uses {} placeholders like fmt::format)
     * @param args Arguments to format
     * 
     * Example:
     * @code
     * KernelCout::print("Value: {}, Index: {}\n", value, index);
     * @endcode
     */
    template<typename... Args>
    static void print(luisa::string_view fmt, Args&&... args) noexcept
    {
        ::luisa::compute::device_log(fmt, std::forward<Args>(args)...);
    }

    /**
     * @brief Print a simple string from device/kernel code
     * 
     * @param str String to print
     * 
     * Example:
     * @code
     * KernelCout::print("Hello from kernel\n");
     * @endcode
     */
    static void print(luisa::string_view str) noexcept
    {
        ::luisa::compute::device_log(str);
    }

  private:
    KernelCout() = delete;
    ~KernelCout() = delete;
};
}  // namespace uipc::backend::luisa

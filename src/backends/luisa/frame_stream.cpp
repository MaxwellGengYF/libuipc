#include <frame_stream.h>

namespace uipc::backend::luisa
{
lud::LStream& frame_stream()
{
    // Deliberately leaked: a long-lived LStream must not be destroyed after
    // Runtime::shutdown() (releasing an LC resource whose device is gone aborts
    // the process -- cookbook §7.10 trap 1).
    static auto* s = new lud::LStream{lud::main_stream()};
    return *s;
}
}  // namespace uipc::backend::luisa

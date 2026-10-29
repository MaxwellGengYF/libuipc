#include <frame_stream.h>

namespace uipc::backend::luisa
{
lud::LStream& frame_stream()
{
    // The singleton lives in the device layer (`LStream::current()`) so that the low
    // level containers (LBufferView / LDeviceBuffer / LDeviceVar) can default to it
    // without including this backend-level header. It is deliberately leaked: a
    // long-lived LStream must not be destroyed after Runtime::shutdown() (releasing
    // an LC resource whose device is gone aborts the process -- cookbook §7.10 trap 1).
    return lud::LStream::current();
}
}  // namespace uipc::backend::luisa

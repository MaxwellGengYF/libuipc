#include <utils/offset_count_collection.h>
#include <type_define.h>

namespace uipc::backend::luisa
{
// Explicit template instantiations for common integer types
// These are the same types used in the CUDA backend

template class OffsetCountCollection<I32>;
template class OffsetCountCollection<I64>;
template class OffsetCountCollection<U64>;
template class OffsetCountCollection<U32>;

}  // namespace uipc::backend::luisa

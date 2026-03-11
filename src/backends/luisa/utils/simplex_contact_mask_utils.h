#pragma once
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
using namespace luisa;
using namespace luisa::compute;

/// Helper function to read from a 2D table stored as 1D buffer
/// table(i, j) = table[i * width + j]
inline auto table_read(BufferView<IndexT> table, UInt width, UInt i, UInt j) noexcept
{
    return table->read(i * width + j);
}

inline auto allow_PT_contact(BufferView<IndexT> table, UInt width, Int4 cids) noexcept
{
    return def<bool>(table_read(table, width, cids.x, cids.y)
                     && table_read(table, width, cids.x, cids.z)
                     && table_read(table, width, cids.x, cids.w));
}

inline auto allow_EE_contact(BufferView<IndexT> table, UInt width, Int4 cids) noexcept
{
    return def<bool>(table_read(table, width, cids.x, cids.z)
                     && table_read(table, width, cids.x, cids.w)
                     && table_read(table, width, cids.y, cids.z)
                     && table_read(table, width, cids.y, cids.w));
}

inline auto allow_PE_contact(BufferView<IndexT> table, UInt width, Int3 cids) noexcept
{
    return def<bool>(table_read(table, width, cids.x, cids.y)
                     && table_read(table, width, cids.x, cids.z));
}

inline auto allow_PP_contact(BufferView<IndexT> table, UInt width, Int2 cids) noexcept
{
    return def<bool>(table_read(table, width, cids.x, cids.y));
}
}  // namespace uipc::backend::luisa

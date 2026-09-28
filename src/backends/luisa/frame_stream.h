#pragma once
/********************************************************************
 * @file frame_stream.h
 * @brief The backend's frame command-list stream (the cuda_tool
 *        `default_stream()` analogue).
 *
 * Reference code never passes a stream: cuda_tool submits every launch/copy on
 * the default CUDA stream, so a ported system has no stream parameter to thread
 * through. The LuisaCompute port needs an explicit LStream for that, and every
 * ported system must see the *same* one, otherwise a frame opened by the engine
 * would only batch some of the work (each LStream owns its own CommandList).
 *
 * Phase 8 opens/commits this stream once per frame; until then
 * `LStream::frame_open()` is false and every command is submitted immediately,
 * which is exactly cuda_tool's behaviour.
 *********************************************************************/
#include <type_define.h>

namespace uipc::backend::luisa
{
/// The process-wide frame stream. Deliberately leaked (cookbook §7.10 trap 1):
/// destroying a long-lived LStream after Runtime::shutdown() releases cached
/// device buffers whose device is already gone, which aborts the process.
[[nodiscard]] lud::LStream& frame_stream();
}  // namespace uipc::backend::luisa

#pragma once
/********************************************************************
 * @file compat.h
 * @brief Namespace bridge between libuipc's `luisa` backend namespace and
 *        LuisaCompute's global `luisa` namespace.
 *
 * The backend lives in `uipc::backend::luisa`, so a plain `luisa::compute::...`
 * reference inside it would resolve to the *enclosing* `uipc::backend::luisa`
 * rather than LuisaCompute. Every luisa_device header therefore refers to the
 * engine through the `lcrt` (LuisaCompute runTime) alias defined here.
 *********************************************************************/
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa::luisa_device
{
// LuisaCompute's global namespace, reached explicitly so it never collides with
// the enclosing backend namespace.
namespace lcrt = ::luisa::compute;
}// namespace uipc::backend::luisa::luisa_device

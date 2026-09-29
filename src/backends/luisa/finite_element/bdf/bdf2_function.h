#pragma once
/********************************************************************
 * @file bdf2_function.h
 * @brief T5.3 — the BDF2 helper functions (port of
 * src/backends/cuda/finite_element/bdf/bdf2_function.h).
 *
 * Identical shape: one namespace `sym::fem_bdf2` that includes `sym/bdf2.inl`
 * inside it.  The includes are the backend's own (`<type_define.h>` here carries
 * the numerics layer's `lps` alias as well, which is what the transcribed `.inl`
 * spells its vector type with).
 *********************************************************************/
#include <type_define.h>

namespace uipc::backend::luisa
{
namespace sym::fem_bdf2
{
#include "sym/bdf2.inl"
}  // namespace sym::fem_bdf2
}  // namespace uipc::backend::luisa

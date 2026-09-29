#pragma once
/********************************************************************
 * @file affine_body/bdf/bdf2_function.h
 * @brief T6.2 — the ABD BDF2 helper functions (port of
 * src/backends/cuda/affine_body/bdf/bdf2_function.h).
 *
 * The FEM twin is `finite_element/bdf/bdf2_function.h` (T5.3, cookbook §7.18):
 * identical shape — one namespace `sym::abd_bdf2` that includes `sym/bdf2.inl`
 * inside it, so the transcribed formulas land in exactly the namespace the
 * integrators call them from.
 *
 * The include is the *ABD* type layer rather than the backend's global one: that is
 * where `LVector12` (the 12-dof state container) and the `lps` alias the transcribed
 * `.inl` spells its vector type with both live (cookbook §7.21 rule 3).
 *********************************************************************/
#include <affine_body/type_define.h>

namespace uipc::backend::luisa
{
namespace sym::abd_bdf2
{
#include "sym/bdf2.inl"
}  // namespace sym::abd_bdf2
}  // namespace uipc::backend::luisa

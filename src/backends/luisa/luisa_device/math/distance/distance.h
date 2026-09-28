#pragma once
/********************************************************************
 * @file distance.h
 * @brief T2.3 — umbrella header for the ported distance / CCD primitives.
 *
 * Mirrors `src/backends/cuda/utils/distance/` (the directory the reference
 * backend includes as a whole):
 *
 *   cuda:                     luisa:
 *   point_point.h             math/distance/point_point.h
 *   point_edge.h              math/distance/point_edge.h
 *   point_triangle.h          math/distance/point_triangle.h
 *   edge_edge.h               math/distance/edge_edge.h
 *   edge_edge_mollifier.h     math/distance/mollifier.h
 *   ccd.h                     math/distance/ccd.h
 *   distance_flagged.h        math/distance/flagged.h
 *
 * All primitives are templated on the scalar type: `T = double` evaluates on the
 * host, `T = lcrt::Var<double>` records device IR (cookbook §7).
 *********************************************************************/
#include <luisa_device/math/distance/point_point.h>
#include <luisa_device/math/distance/point_edge.h>
#include <luisa_device/math/distance/point_triangle.h>
#include <luisa_device/math/distance/edge_edge.h>
#include <luisa_device/math/distance/mollifier.h>
#include <luisa_device/math/distance/ccd.h>
#include <luisa_device/math/distance/flagged.h>

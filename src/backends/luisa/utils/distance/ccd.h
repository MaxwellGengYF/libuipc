#pragma once
#include "type_define.h"
#include <luisa/core/basic_types.h>
#include <luisa/core/mathematics.h>

// ref: https://github.com/ipc-sim/Codim-IPC/tree/main/Library/Math/Distance
namespace uipc::backend::luisa::distance {

using namespace ::luisa;

// Type aliases for compatibility
template <typename T>
using Vector3 = std::conditional_t<std::is_same_v<T, float>, float3, double3>;

template <typename T>
using Vector2 = std::conditional_t<std::is_same_v<T, float>, float2, double2>;

// 2D CCD broadphase
LC_GPU_CALLABLE inline bool point_edge_ccd_broadphase(const float2& p,
                                                    const float2& e0,
                                                    const float2& e1,
                                                    const float2& dp,
                                                    const float2& de0,
                                                    const float2& de1,
                                                    float dist);

LC_GPU_CALLABLE inline bool point_edge_ccd_broadphase(const double2& p,
                                                    const double2& e0,
                                                    const double2& e1,
                                                    const double2& dp,
                                                    const double2& de0,
                                                    const double2& de1,
                                                    double dist);

// 3D CD broadphase (continuous detection)
LC_GPU_CALLABLE inline bool point_edge_cd_broadphase(const float3& x0,
                                                   const float3& x1,
                                                   const float3& x2,
                                                   float dist);

LC_GPU_CALLABLE inline bool point_edge_cd_broadphase(const double3& x0,
                                                   const double3& x1,
                                                   const double3& x2,
                                                   double dist);

LC_GPU_CALLABLE inline bool point_triangle_cd_broadphase(const float3& p,
                                                       const float3& t0,
                                                       const float3& t1,
                                                       const float3& t2,
                                                       float dist);

LC_GPU_CALLABLE inline bool point_triangle_cd_broadphase(const double3& p,
                                                       const double3& t0,
                                                       const double3& t1,
                                                       const double3& t2,
                                                       double dist);

LC_GPU_CALLABLE inline bool edge_edge_cd_broadphase(const float3& ea0,
                                                  const float3& ea1,
                                                  const float3& eb0,
                                                  const float3& eb1,
                                                  float dist);

LC_GPU_CALLABLE inline bool edge_edge_cd_broadphase(const double3& ea0,
                                                  const double3& ea1,
                                                  const double3& eb0,
                                                  const double3& eb1,
                                                  double dist);

// 3D CCD broadphase
LC_GPU_CALLABLE inline bool point_triangle_ccd_broadphase(const float3& p,
                                                        const float3& t0,
                                                        const float3& t1,
                                                        const float3& t2,
                                                        const float3& dp,
                                                        const float3& dt0,
                                                        const float3& dt1,
                                                        const float3& dt2,
                                                        float dist);

LC_GPU_CALLABLE inline bool point_triangle_ccd_broadphase(const double3& p,
                                                        const double3& t0,
                                                        const double3& t1,
                                                        const double3& t2,
                                                        const double3& dp,
                                                        const double3& dt0,
                                                        const double3& dt1,
                                                        const double3& dt2,
                                                        double dist);

LC_GPU_CALLABLE inline bool edge_edge_ccd_broadphase(const float3& ea0,
                                                   const float3& ea1,
                                                   const float3& eb0,
                                                   const float3& eb1,
                                                   const float3& dea0,
                                                   const float3& dea1,
                                                   const float3& deb0,
                                                   const float3& deb1,
                                                   float dist);

LC_GPU_CALLABLE inline bool edge_edge_ccd_broadphase(const double3& ea0,
                                                   const double3& ea1,
                                                   const double3& eb0,
                                                   const double3& eb1,
                                                   const double3& dea0,
                                                   const double3& dea1,
                                                   const double3& deb0,
                                                   const double3& deb1,
                                                   double dist);

LC_GPU_CALLABLE inline bool point_edge_ccd_broadphase(const float3& p,
                                                    const float3& e0,
                                                    const float3& e1,
                                                    const float3& dp,
                                                    const float3& de0,
                                                    const float3& de1,
                                                    float dist);

LC_GPU_CALLABLE inline bool point_edge_ccd_broadphase(const double3& p,
                                                    const double3& e0,
                                                    const double3& e1,
                                                    const double3& dp,
                                                    const double3& de0,
                                                    const double3& de1,
                                                    double dist);

LC_GPU_CALLABLE inline bool point_point_ccd_broadphase(const float3& p0,
                                                     const float3& p1,
                                                     const float3& dp0,
                                                     const float3& dp1,
                                                     float dist);

LC_GPU_CALLABLE inline bool point_point_ccd_broadphase(const double3& p0,
                                                     const double3& p1,
                                                     const double3& dp0,
                                                     const double3& dp1,
                                                     double dist);

// CCD narrowphase
LC_GPU_CALLABLE inline bool point_triangle_ccd(float3 p,
                                             float3 t0,
                                             float3 t1,
                                             float3 t2,
                                             float3 dp,
                                             float3 dt0,
                                             float3 dt1,
                                             float3 dt2,
                                             float eta,
                                             float thickness,
                                             int max_iter,
                                             float& toc);

LC_GPU_CALLABLE inline bool point_triangle_ccd(double3 p,
                                             double3 t0,
                                             double3 t1,
                                             double3 t2,
                                             double3 dp,
                                             double3 dt0,
                                             double3 dt1,
                                             double3 dt2,
                                             double eta,
                                             double thickness,
                                             int max_iter,
                                             double& toc);

LC_GPU_CALLABLE inline bool edge_edge_ccd(float3 ea0,
                                        float3 ea1,
                                        float3 eb0,
                                        float3 eb1,
                                        float3 dea0,
                                        float3 dea1,
                                        float3 deb0,
                                        float3 deb1,
                                        float eta,
                                        float thickness,
                                        int max_iter,
                                        float& toc);

LC_GPU_CALLABLE inline bool edge_edge_ccd(double3 ea0,
                                        double3 ea1,
                                        double3 eb0,
                                        double3 eb1,
                                        double3 dea0,
                                        double3 dea1,
                                        double3 deb0,
                                        double3 deb1,
                                        double eta,
                                        double thickness,
                                        int max_iter,
                                        double& toc);

LC_GPU_CALLABLE inline bool point_edge_ccd(float3 p,
                                         float3 e0,
                                         float3 e1,
                                         float3 dp,
                                         float3 de0,
                                         float3 de1,
                                         float eta,
                                         float thickness,
                                         int max_iter,
                                         float& toc);

LC_GPU_CALLABLE inline bool point_edge_ccd(double3 p,
                                         double3 e0,
                                         double3 e1,
                                         double3 dp,
                                         double3 de0,
                                         double3 de1,
                                         double eta,
                                         double thickness,
                                         int max_iter,
                                         double& toc);

LC_GPU_CALLABLE inline bool point_point_ccd(float3 p0,
                                          float3 p1,
                                          float3 dp0,
                                          float3 dp1,
                                          float eta,
                                          float thickness,
                                          int max_iter,
                                          float& toc);

LC_GPU_CALLABLE inline bool point_point_ccd(double3 p0,
                                          double3 p1,
                                          double3 dp0,
                                          double3 dp1,
                                          double eta,
                                          double thickness,
                                          int max_iter,
                                          double& toc);

}  // namespace uipc::backend::luisa::distance

#include "details/ccd.inl"

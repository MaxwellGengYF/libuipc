// ref: https://github.com/ipc-sim/Codim-IPC/tree/main/Library/Math/Distance
// Refactored for LuisaCompute

namespace uipc::backend::luisa::distance
{
// Helper functions for component-wise operations
template <typename T>
LC_GPU_CALLABLE inline auto cwise_max(const T& a, const T& b)
{
    return luisa::max(a, b);
}

template <typename T>
LC_GPU_CALLABLE inline auto cwise_max(const T& a, const T& b, const T& c)
{
    return luisa::max(luisa::max(a, b), c);
}

template <typename T>
LC_GPU_CALLABLE inline auto cwise_max(const T& a, const T& b, const T& c, const T& d)
{
    return luisa::max(luisa::max(a, b), luisa::max(c, d));
}

template <typename T>
LC_GPU_CALLABLE inline auto cwise_max(const T& a, const T& b, const T& c, const T& d, const T& e, const T& f)
{
    return luisa::max(luisa::max(luisa::max(a, b), luisa::max(c, d)), luisa::max(e, f));
}

template <typename T>
LC_GPU_CALLABLE inline auto cwise_min(const T& a, const T& b)
{
    return luisa::min(a, b);
}

template <typename T>
LC_GPU_CALLABLE inline auto cwise_min(const T& a, const T& b, const T& c)
{
    return luisa::min(luisa::min(a, b), c);
}

template <typename T>
LC_GPU_CALLABLE inline auto cwise_min(const T& a, const T& b, const T& c, const T& d)
{
    return luisa::min(luisa::min(a, b), luisa::min(c, d));
}

template <typename T>
LC_GPU_CALLABLE inline auto cwise_min(const T& a, const T& b, const T& c, const T& d, const T& e, const T& f)
{
    return luisa::min(luisa::min(luisa::min(a, b), luisa::min(c, d)), luisa::min(e, f));
}

template <typename T>
LC_GPU_CALLABLE inline bool any_component_greater(const T& a, const T& b)
{
    return luisa::any(a > b);
}

template <typename T>
LC_GPU_CALLABLE inline T max_component(const T& v)
{
    if constexpr(std::is_same_v<T, float2> || std::is_same_v<T, double2>)
    {
        return luisa::max(v.x, v.y);
    }
    else if constexpr(std::is_same_v<T, float3> || std::is_same_v<T, double3>)
    {
        return luisa::max(luisa::max(v.x, v.y), v.z);
    }
    else if constexpr(std::is_same_v<T, float4> || std::is_same_v<T, double4>)
    {
        return luisa::max(luisa::max(v.x, v.y), luisa::max(v.z, v.w));
    }
    return v;
}

template <typename T>
LC_GPU_CALLABLE inline T min_component(const T& v)
{
    if constexpr(std::is_same_v<T, float2> || std::is_same_v<T, double2>)
    {
        return luisa::min(v.x, v.y);
    }
    else if constexpr(std::is_same_v<T, float3> || std::is_same_v<T, double3>)
    {
        return luisa::min(luisa::min(v.x, v.y), v.z);
    }
    else if constexpr(std::is_same_v<T, float4> || std::is_same_v<T, double4>)
    {
        return luisa::min(luisa::min(v.x, v.y), luisa::min(v.z, v.w));
    }
    return v;
}

// Helper to get maximum of 4 scalar values
template <typename T>
LC_GPU_CALLABLE inline T max_of_4(T a, T b, T c, T d)
{
    return luisa::max(luisa::max(a, b), luisa::max(c, d));
}

template <typename T>
LC_GPU_CALLABLE inline T min_of_4(T a, T b, T c, T d)
{
    return luisa::min(luisa::min(a, b), luisa::min(c, d));
}

// ==================== float2 implementations ====================

LC_GPU_CALLABLE inline bool point_edge_ccd_broadphase(const float2& p,
                                                    const float2& e0,
                                                    const float2& e1,
                                                    const float2& dp,
                                                    const float2& de0,
                                                    const float2& de1,
                                                    float dist)
{
    const float2 max_p = cwise_max(p, p + dp);
    const float2 min_p = cwise_min(p, p + dp);
    const float2 max_e = cwise_max(e0, e1, e0 + de0, e1 + de1);
    const float2 min_e = cwise_min(e0, e1, e0 + de0, e1 + de1);
    return !(any_component_greater(min_p, max_e + make_float2(dist)) ||
             any_component_greater(min_e, max_p + make_float2(dist)));
}

LC_GPU_CALLABLE inline bool point_edge_ccd_broadphase(const double2& p,
                                                    const double2& e0,
                                                    const double2& e1,
                                                    const double2& dp,
                                                    const double2& de0,
                                                    const double2& de1,
                                                    double dist)
{
    const double2 max_p = cwise_max(p, p + dp);
    const double2 min_p = cwise_min(p, p + dp);
    const double2 max_e = cwise_max(e0, e1, e0 + de0, e1 + de1);
    const double2 min_e = cwise_min(e0, e1, e0 + de0, e1 + de1);
    return !(any_component_greater(min_p, max_e + make_double2(dist)) ||
             any_component_greater(min_e, max_p + make_double2(dist)));
}

// ==================== float3 implementations ====================

LC_GPU_CALLABLE inline bool point_edge_cd_broadphase(const float3& x0,
                                                   const float3& x1,
                                                   const float3& x2,
                                                   float dist)
{
    const float3 max_e = cwise_max(x1, x2);
    const float3 min_e = cwise_min(x1, x2);
    return !(any_component_greater(x0, max_e + make_float3(dist)) ||
             any_component_greater(min_e, x0 + make_float3(dist)));
}

LC_GPU_CALLABLE inline bool point_edge_cd_broadphase(const double3& x0,
                                                   const double3& x1,
                                                   const double3& x2,
                                                   double dist)
{
    const double3 max_e = cwise_max(x1, x2);
    const double3 min_e = cwise_min(x1, x2);
    return !(any_component_greater(x0, max_e + make_double3(dist)) ||
             any_component_greater(min_e, x0 + make_double3(dist)));
}

LC_GPU_CALLABLE inline bool point_triangle_cd_broadphase(const float3& p,
                                                       const float3& t0,
                                                       const float3& t1,
                                                       const float3& t2,
                                                       float dist)
{
    const float3 max_tri = cwise_max(t0, t1, t2);
    const float3 min_tri = cwise_min(t0, t1, t2);
    return !(any_component_greater(p, max_tri + make_float3(dist)) ||
             any_component_greater(min_tri, p + make_float3(dist)));
}

LC_GPU_CALLABLE inline bool point_triangle_cd_broadphase(const double3& p,
                                                       const double3& t0,
                                                       const double3& t1,
                                                       const double3& t2,
                                                       double dist)
{
    const double3 max_tri = cwise_max(t0, t1, t2);
    const double3 min_tri = cwise_min(t0, t1, t2);
    return !(any_component_greater(p, max_tri + make_double3(dist)) ||
             any_component_greater(min_tri, p + make_double3(dist)));
}

LC_GPU_CALLABLE inline bool edge_edge_cd_broadphase(const float3& ea0,
                                                  const float3& ea1,
                                                  const float3& eb0,
                                                  const float3& eb1,
                                                  float dist)
{
    const float3 max_a = cwise_max(ea0, ea1);
    const float3 min_a = cwise_min(ea0, ea1);
    const float3 max_b = cwise_max(eb0, eb1);
    const float3 min_b = cwise_min(eb0, eb1);
    return !(any_component_greater(min_a, max_b + make_float3(dist)) ||
             any_component_greater(min_b, max_a + make_float3(dist)));
}

LC_GPU_CALLABLE inline bool edge_edge_cd_broadphase(const double3& ea0,
                                                  const double3& ea1,
                                                  const double3& eb0,
                                                  const double3& eb1,
                                                  double dist)
{
    const double3 max_a = cwise_max(ea0, ea1);
    const double3 min_a = cwise_min(ea0, ea1);
    const double3 max_b = cwise_max(eb0, eb1);
    const double3 min_b = cwise_min(eb0, eb1);
    return !(any_component_greater(min_a, max_b + make_double3(dist)) ||
             any_component_greater(min_b, max_a + make_double3(dist)));
}

LC_GPU_CALLABLE inline bool point_triangle_ccd_broadphase(const float3& p,
                                                        const float3& t0,
                                                        const float3& t1,
                                                        const float3& t2,
                                                        const float3& dp,
                                                        const float3& dt0,
                                                        const float3& dt1,
                                                        const float3& dt2,
                                                        float dist)
{
    const float3 max_p = cwise_max(p, p + dp);
    const float3 min_p = cwise_min(p, p + dp);
    const float3 max_tri = cwise_max(t0, t1, t2, t0 + dt0, t1 + dt1, t2 + dt2);
    const float3 min_tri = cwise_min(t0, t1, t2, t0 + dt0, t1 + dt1, t2 + dt2);
    return !(any_component_greater(min_p, max_tri + make_float3(dist)) ||
             any_component_greater(min_tri, max_p + make_float3(dist)));
}

LC_GPU_CALLABLE inline bool point_triangle_ccd_broadphase(const double3& p,
                                                        const double3& t0,
                                                        const double3& t1,
                                                        const double3& t2,
                                                        const double3& dp,
                                                        const double3& dt0,
                                                        const double3& dt1,
                                                        const double3& dt2,
                                                        double dist)
{
    const double3 max_p = cwise_max(p, p + dp);
    const double3 min_p = cwise_min(p, p + dp);
    const double3 max_tri = cwise_max(t0, t1, t2, t0 + dt0, t1 + dt1, t2 + dt2);
    const double3 min_tri = cwise_min(t0, t1, t2, t0 + dt0, t1 + dt1, t2 + dt2);
    return !(any_component_greater(min_p, max_tri + make_double3(dist)) ||
             any_component_greater(min_tri, max_p + make_double3(dist)));
}

LC_GPU_CALLABLE inline bool edge_edge_ccd_broadphase(const float3& ea0,
                                                   const float3& ea1,
                                                   const float3& eb0,
                                                   const float3& eb1,
                                                   const float3& dea0,
                                                   const float3& dea1,
                                                   const float3& deb0,
                                                   const float3& deb1,
                                                   float dist)
{
    const float3 max_a = cwise_max(ea0, ea1, ea0 + dea0, ea1 + dea1);
    const float3 min_a = cwise_min(ea0, ea1, ea0 + dea0, ea1 + dea1);
    const float3 max_b = cwise_max(eb0, eb1, eb0 + deb0, eb1 + deb1);
    const float3 min_b = cwise_min(eb0, eb1, eb0 + deb0, eb1 + deb1);
    return !(any_component_greater(min_a, max_b + make_float3(dist)) ||
             any_component_greater(min_b, max_a + make_float3(dist)));
}

LC_GPU_CALLABLE inline bool edge_edge_ccd_broadphase(const double3& ea0,
                                                   const double3& ea1,
                                                   const double3& eb0,
                                                   const double3& eb1,
                                                   const double3& dea0,
                                                   const double3& dea1,
                                                   const double3& deb0,
                                                   const double3& deb1,
                                                   double dist)
{
    const double3 max_a = cwise_max(ea0, ea1, ea0 + dea0, ea1 + dea1);
    const double3 min_a = cwise_min(ea0, ea1, ea0 + dea0, ea1 + dea1);
    const double3 max_b = cwise_max(eb0, eb1, eb0 + deb0, eb1 + deb1);
    const double3 min_b = cwise_min(eb0, eb1, eb0 + deb0, eb1 + deb1);
    return !(any_component_greater(min_a, max_b + make_double3(dist)) ||
             any_component_greater(min_b, max_a + make_double3(dist)));
}

LC_GPU_CALLABLE inline bool point_edge_ccd_broadphase(const float3& p,
                                                    const float3& e0,
                                                    const float3& e1,
                                                    const float3& dp,
                                                    const float3& de0,
                                                    const float3& de1,
                                                    float dist)
{
    const float3 max_p = cwise_max(p, p + dp);
    const float3 min_p = cwise_min(p, p + dp);
    const float3 max_e = cwise_max(e0, e1, e0 + de0, e1 + de1);
    const float3 min_e = cwise_min(e0, e1, e0 + de0, e1 + de1);
    return !(any_component_greater(min_p, max_e + make_float3(dist)) ||
             any_component_greater(min_e, max_p + make_float3(dist)));
}

LC_GPU_CALLABLE inline bool point_edge_ccd_broadphase(const double3& p,
                                                    const double3& e0,
                                                    const double3& e1,
                                                    const double3& dp,
                                                    const double3& de0,
                                                    const double3& de1,
                                                    double dist)
{
    const double3 max_p = cwise_max(p, p + dp);
    const double3 min_p = cwise_min(p, p + dp);
    const double3 max_e = cwise_max(e0, e1, e0 + de0, e1 + de1);
    const double3 min_e = cwise_min(e0, e1, e0 + de0, e1 + de1);
    return !(any_component_greater(min_p, max_e + make_double3(dist)) ||
             any_component_greater(min_e, max_p + make_double3(dist)));
}

LC_GPU_CALLABLE inline bool point_point_ccd_broadphase(const float3& p0,
                                                     const float3& p1,
                                                     const float3& dp0,
                                                     const float3& dp1,
                                                     float dist)
{
    const float3 max_p0 = cwise_max(p0, p0 + dp0);
    const float3 min_p0 = cwise_min(p0, p0 + dp0);
    const float3 max_p1 = cwise_max(p1, p1 + dp1);
    const float3 min_p1 = cwise_min(p1, p1 + dp1);
    return !(any_component_greater(min_p0, max_p1 + make_float3(dist)) ||
             any_component_greater(min_p1, max_p0 + make_float3(dist)));
}

LC_GPU_CALLABLE inline bool point_point_ccd_broadphase(const double3& p0,
                                                     const double3& p1,
                                                     const double3& dp0,
                                                     const double3& dp1,
                                                     double dist)
{
    const double3 max_p0 = cwise_max(p0, p0 + dp0);
    const double3 min_p0 = cwise_min(p0, p0 + dp0);
    const double3 max_p1 = cwise_max(p1, p1 + dp1);
    const double3 min_p1 = cwise_min(p1, p1 + dp1);
    return !(any_component_greater(min_p0, max_p1 + make_double3(dist)) ||
             any_component_greater(min_p1, max_p0 + make_double3(dist)));
}

// ==================== CCD Narrowphase (requires distance functions) ====================

// Forward declarations for distance functions (should be defined in distance.h)
template <typename T>
LC_GPU_CALLABLE int point_triangle_distance_flag(const T& p, const T& t0, const T& t1, const T& t2);

template <typename T>
LC_GPU_CALLABLE void point_triangle_distance2(int flag, const T& p, const T& t0, const T& t1, const T& t2, float& dist2);

template <typename T>
LC_GPU_CALLABLE void point_triangle_distance2(int flag, const T& p, const T& t0, const T& t1, const T& t2, double& dist2);

template <typename T>
LC_GPU_CALLABLE int edge_edge_distance_flag(const T& ea0, const T& ea1, const T& eb0, const T& eb1);

template <typename T>
LC_GPU_CALLABLE void edge_edge_distance2(int flag, const T& ea0, const T& ea1, const T& eb0, const T& eb1, float& dist2);

template <typename T>
LC_GPU_CALLABLE void edge_edge_distance2(int flag, const T& ea0, const T& ea1, const T& eb0, const T& eb1, double& dist2);

template <typename T>
LC_GPU_CALLABLE int point_edge_distance_flag(const T& p, const T& e0, const T& e1);

template <typename T>
LC_GPU_CALLABLE void point_edge_distance2(int flag, const T& p, const T& e0, const T& e1, float& dist2);

template <typename T>
LC_GPU_CALLABLE void point_edge_distance2(int flag, const T& p, const T& e0, const T& e1, double& dist2);

template <typename T>
LC_GPU_CALLABLE int point_point_distance_flag(const T& p0, const T& p1);

template <typename T>
LC_GPU_CALLABLE void point_point_distance2(int flag, const T& p0, const T& p1, float& dist2);

template <typename T>
LC_GPU_CALLABLE void point_point_distance2(int flag, const T& p0, const T& p1, double& dist2);

// point_triangle_ccd for float
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
                                             float& toc)
{
    float3 mov = (dt0 + dt1 + dt2 + dp) * 0.25f;
    dt0 -= mov;
    dt1 -= mov;
    dt2 -= mov;
    dp -= mov;
    
    float dispMag0 = luisa::length_squared(dt0);
    float dispMag1 = luisa::length_squared(dt1);
    float dispMag2 = luisa::length_squared(dt2);
    float maxDispMag = luisa::length(dp) + luisa::sqrt(max_of_4(dispMag0, dispMag1, dispMag2, 0.0f));

    if(maxDispMag <= 0.0f)
    {
        return false;
    }

    float dist2_cur;
    int flag = point_triangle_distance_flag(p, t0, t1, t2);
    point_triangle_distance2(flag, p, t0, t1, t2, dist2_cur);
    float dist_cur = luisa::sqrt(dist2_cur);
    float gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    float toc_prev = toc;
    toc = 0.0f;
    
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        float tocLowerBound = (1.0f - eta) * (dist2_cur - thickness * thickness)
                              / ((dist_cur + thickness) * maxDispMag);

        p += tocLowerBound * dp;
        t0 += tocLowerBound * dt0;
        t1 += tocLowerBound * dt1;
        t2 += tocLowerBound * dt2;
        flag = point_triangle_distance_flag(p, t0, t1, t2);
        point_triangle_distance2(flag, p, t0, t1, t2, dist2_cur);
        dist_cur = luisa::sqrt(dist2_cur);
        if(toc > 0.0f && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

// point_triangle_ccd for double
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
                                             double& toc)
{
    double3 mov = (dt0 + dt1 + dt2 + dp) * 0.25;
    dt0 -= mov;
    dt1 -= mov;
    dt2 -= mov;
    dp -= mov;
    
    double dispMag0 = luisa::length_squared(dt0);
    double dispMag1 = luisa::length_squared(dt1);
    double dispMag2 = luisa::length_squared(dt2);
    double maxDispMag = luisa::length(dp) + luisa::sqrt(max_of_4(dispMag0, dispMag1, dispMag2, 0.0));

    if(maxDispMag <= 0.0)
    {
        return false;
    }

    double dist2_cur;
    int flag = point_triangle_distance_flag(p, t0, t1, t2);
    point_triangle_distance2(flag, p, t0, t1, t2, dist2_cur);
    double dist_cur = luisa::sqrt(dist2_cur);
    double gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    double toc_prev = toc;
    toc = 0.0;
    
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        double tocLowerBound = (1.0 - eta) * (dist2_cur - thickness * thickness)
                               / ((dist_cur + thickness) * maxDispMag);

        p += tocLowerBound * dp;
        t0 += tocLowerBound * dt0;
        t1 += tocLowerBound * dt1;
        t2 += tocLowerBound * dt2;
        flag = point_triangle_distance_flag(p, t0, t1, t2);
        point_triangle_distance2(flag, p, t0, t1, t2, dist2_cur);
        dist_cur = luisa::sqrt(dist2_cur);
        if(toc > 0.0 && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

// edge_edge_ccd for float
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
                                        float& toc)
{
    float3 mov = (dea0 + dea1 + deb0 + deb1) * 0.25f;
    dea0 -= mov;
    dea1 -= mov;
    deb0 -= mov;
    deb1 -= mov;
    
    float maxDispMag = luisa::sqrt(max_of_4(
                               luisa::length_squared(dea0),
                               luisa::length_squared(dea1), 0.0f, 0.0f))
                       + luisa::sqrt(max_of_4(
                               luisa::length_squared(deb0),
                               luisa::length_squared(deb1), 0.0f, 0.0f));
                               
    if(maxDispMag == 0.0f)
    {
        return false;
    }

    float dist2_cur;
    int flag = edge_edge_distance_flag(ea0, ea1, eb0, eb1);
    edge_edge_distance2(flag, ea0, ea1, eb0, eb1, dist2_cur);
    float dFunc = dist2_cur - thickness * thickness;
    if(dFunc <= 0.0f)
    {
        float dists[4] = {
            luisa::length_squared(ea0 - eb0),
            luisa::length_squared(ea0 - eb1),
            luisa::length_squared(ea1 - eb0),
            luisa::length_squared(ea1 - eb1)
        };
        dist2_cur = min_of_4(dists[0], dists[1], dists[2], dists[3]);
        dFunc = dist2_cur - thickness * thickness;
    }
    float dist_cur = luisa::sqrt(dist2_cur);
    float gap = eta * dFunc / (dist_cur + thickness);
    float toc_prev = toc;
    toc = 0.0f;
    
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        float tocLowerBound = (1.0f - eta) * dFunc / ((dist_cur + thickness) * maxDispMag);

        ea0 += tocLowerBound * dea0;
        ea1 += tocLowerBound * dea1;
        eb0 += tocLowerBound * deb0;
        eb1 += tocLowerBound * deb1;
        flag = edge_edge_distance_flag(ea0, ea1, eb0, eb1);
        edge_edge_distance2(flag, ea0, ea1, eb0, eb1, dist2_cur);
        dFunc = dist2_cur - thickness * thickness;
        if(dFunc <= 0.0f)
        {
            float dists[4] = {
                luisa::length_squared(ea0 - eb0),
                luisa::length_squared(ea0 - eb1),
                luisa::length_squared(ea1 - eb0),
                luisa::length_squared(ea1 - eb1)
            };
            dist2_cur = min_of_4(dists[0], dists[1], dists[2], dists[3]);
            dFunc = dist2_cur - thickness * thickness;
        }
        dist_cur = luisa::sqrt(dist2_cur);
        if(toc > 0.0f && (dFunc / (dist_cur + thickness) < gap))
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

// edge_edge_ccd for double
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
                                        double& toc)
{
    double3 mov = (dea0 + dea1 + deb0 + deb1) * 0.25;
    dea0 -= mov;
    dea1 -= mov;
    deb0 -= mov;
    deb1 -= mov;
    
    double maxDispMag = luisa::sqrt(max_of_4(
                                luisa::length_squared(dea0),
                                luisa::length_squared(dea1), 0.0, 0.0))
                        + luisa::sqrt(max_of_4(
                                luisa::length_squared(deb0),
                                luisa::length_squared(deb1), 0.0, 0.0));
                                
    if(maxDispMag == 0.0)
    {
        return false;
    }

    double dist2_cur;
    int flag = edge_edge_distance_flag(ea0, ea1, eb0, eb1);
    edge_edge_distance2(flag, ea0, ea1, eb0, eb1, dist2_cur);
    double dFunc = dist2_cur - thickness * thickness;
    if(dFunc <= 0.0)
    {
        double dists[4] = {
            luisa::length_squared(ea0 - eb0),
            luisa::length_squared(ea0 - eb1),
            luisa::length_squared(ea1 - eb0),
            luisa::length_squared(ea1 - eb1)
        };
        dist2_cur = min_of_4(dists[0], dists[1], dists[2], dists[3]);
        dFunc = dist2_cur - thickness * thickness;
    }
    double dist_cur = luisa::sqrt(dist2_cur);
    double gap = eta * dFunc / (dist_cur + thickness);
    double toc_prev = toc;
    toc = 0.0;
    
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        double tocLowerBound = (1.0 - eta) * dFunc / ((dist_cur + thickness) * maxDispMag);

        ea0 += tocLowerBound * dea0;
        ea1 += tocLowerBound * dea1;
        eb0 += tocLowerBound * deb0;
        eb1 += tocLowerBound * deb1;
        flag = edge_edge_distance_flag(ea0, ea1, eb0, eb1);
        edge_edge_distance2(flag, ea0, ea1, eb0, eb1, dist2_cur);
        dFunc = dist2_cur - thickness * thickness;
        if(dFunc <= 0.0)
        {
            double dists[4] = {
                luisa::length_squared(ea0 - eb0),
                luisa::length_squared(ea0 - eb1),
                luisa::length_squared(ea1 - eb0),
                luisa::length_squared(ea1 - eb1)
            };
            dist2_cur = min_of_4(dists[0], dists[1], dists[2], dists[3]);
            dFunc = dist2_cur - thickness * thickness;
        }
        dist_cur = luisa::sqrt(dist2_cur);
        if(toc > 0.0 && (dFunc / (dist_cur + thickness) < gap))
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

// point_edge_ccd for float
LC_GPU_CALLABLE inline bool point_edge_ccd(float3 p,
                                         float3 e0,
                                         float3 e1,
                                         float3 dp,
                                         float3 de0,
                                         float3 de1,
                                         float eta,
                                         float thickness,
                                         int max_iter,
                                         float& toc)
{
    float3 mov = (dp + de0 + de1) * (1.0f / 3.0f);
    de0 -= mov;
    de1 -= mov;
    dp -= mov;
    
    float maxDispMag = luisa::length(dp) 
                       + luisa::sqrt(max_of_4(
                               luisa::length_squared(de0),
                               luisa::length_squared(de1), 0.0f, 0.0f));
                               
    if(maxDispMag == 0.0f)
    {
        return false;
    }

    float dist2_cur;
    int flag = point_edge_distance_flag(p, e0, e1);
    point_edge_distance2(flag, p, e0, e1, dist2_cur);
    float dist_cur = luisa::sqrt(dist2_cur);
    float gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    float toc_prev = toc;
    toc = 0.0f;
    
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        float tocLowerBound = (1.0f - eta) * (dist2_cur - thickness * thickness)
                              / ((dist_cur + thickness) * maxDispMag);

        p += tocLowerBound * dp;
        e0 += tocLowerBound * de0;
        e1 += tocLowerBound * de1;
        flag = point_edge_distance_flag(p, e0, e1);
        point_edge_distance2(flag, p, e0, e1, dist2_cur);
        dist_cur = luisa::sqrt(dist2_cur);
        if(toc > 0.0f && (dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

// point_edge_ccd for double
LC_GPU_CALLABLE inline bool point_edge_ccd(double3 p,
                                         double3 e0,
                                         double3 e1,
                                         double3 dp,
                                         double3 de0,
                                         double3 de1,
                                         double eta,
                                         double thickness,
                                         int max_iter,
                                         double& toc)
{
    double3 mov = (dp + de0 + de1) * (1.0 / 3.0);
    de0 -= mov;
    de1 -= mov;
    dp -= mov;
    
    double maxDispMag = luisa::length(dp) 
                        + luisa::sqrt(max_of_4(
                                luisa::length_squared(de0),
                                luisa::length_squared(de1), 0.0, 0.0));
                                
    if(maxDispMag == 0.0)
    {
        return false;
    }

    double dist2_cur;
    int flag = point_edge_distance_flag(p, e0, e1);
    point_edge_distance2(flag, p, e0, e1, dist2_cur);
    double dist_cur = luisa::sqrt(dist2_cur);
    double gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    double toc_prev = toc;
    toc = 0.0;
    
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        double tocLowerBound = (1.0 - eta) * (dist2_cur - thickness * thickness)
                               / ((dist_cur + thickness) * maxDispMag);

        p += tocLowerBound * dp;
        e0 += tocLowerBound * de0;
        e1 += tocLowerBound * de1;
        flag = point_edge_distance_flag(p, e0, e1);
        point_edge_distance2(flag, p, e0, e1, dist2_cur);
        dist_cur = luisa::sqrt(dist2_cur);
        if(toc > 0.0 && (dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

// point_point_ccd for float
LC_GPU_CALLABLE inline bool point_point_ccd(float3 p0,
                                          float3 p1,
                                          float3 dp0,
                                          float3 dp1,
                                          float eta,
                                          float thickness,
                                          int max_iter,
                                          float& toc)
{
    float3 mov = (dp0 + dp1) * 0.5f;
    dp1 -= mov;
    dp0 -= mov;
    
    float maxDispMag = luisa::length(dp0) + luisa::length(dp1);
    
    if(maxDispMag == 0.0f)
    {
        return false;
    }

    float dist2_cur;
    int flag = point_point_distance_flag(p0, p1);
    point_point_distance2(flag, p0, p1, dist2_cur);
    float dist_cur = luisa::sqrt(dist2_cur);
    float gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    float toc_prev = toc;
    toc = 0.0f;
    
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        float tocLowerBound = (1.0f - eta) * (dist2_cur - thickness * thickness)
                              / ((dist_cur + thickness) * maxDispMag);

        p0 += tocLowerBound * dp0;
        p1 += tocLowerBound * dp1;
        flag = point_point_distance_flag(p0, p1);
        point_point_distance2(flag, p0, p1, dist2_cur);
        dist_cur = luisa::sqrt(dist2_cur);
        if(toc > 0.0f && (dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

// point_point_ccd for double
LC_GPU_CALLABLE inline bool point_point_ccd(double3 p0,
                                          double3 p1,
                                          double3 dp0,
                                          double3 dp1,
                                          double eta,
                                          double thickness,
                                          int max_iter,
                                          double& toc)
{
    double3 mov = (dp0 + dp1) * 0.5;
    dp1 -= mov;
    dp0 -= mov;
    
    double maxDispMag = luisa::length(dp0) + luisa::length(dp1);
    
    if(maxDispMag == 0.0)
    {
        return false;
    }

    double dist2_cur;
    int flag = point_point_distance_flag(p0, p1);
    point_point_distance2(flag, p0, p1, dist2_cur);
    double dist_cur = luisa::sqrt(dist2_cur);
    double gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    double toc_prev = toc;
    toc = 0.0;
    
    while(true)
    {
        if(max_iter >= 0)
        {
            if(--max_iter < 0)
                return true;
        }

        double tocLowerBound = (1.0 - eta) * (dist2_cur - thickness * thickness)
                               / ((dist_cur + thickness) * maxDispMag);

        p0 += tocLowerBound * dp0;
        p1 += tocLowerBound * dp1;
        flag = point_point_distance_flag(p0, p1);
        point_point_distance2(flag, p0, p1, dist2_cur);
        dist_cur = luisa::sqrt(dist2_cur);
        if(toc > 0.0 && (dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)
        {
            break;
        }

        toc += tocLowerBound;
        if(toc > toc_prev)
        {
            return false;
        }
    }

    return true;
}

}  // namespace uipc::backend::luisa::distance

#pragma once

namespace uipc::backend::luisa::distance
{
LC_GPU_CALLABLE inline void point_edge_distance2(const Vector3& p,
                                                 const Vector3& e0,
                                                 const Vector3& e1,
                                                 Float& dist2)
{
    Vector3 e = e1 - e0;
    Vector3 w0 = p - e0;
    
    Float e2 = luisa::dot(e, e);
    if(e2 <= 0.0f)
    {
        // Degenerate edge
        Vector3 diff = p - e0;
        dist2 = luisa::dot(diff, diff);
        return;
    }
    
    Float t = luisa::dot(w0, e) / e2;
    t = luisa::clamp(t, 0.0f, 1.0f);
    
    Vector3 closest = e0 + t * e;
    Vector3 diff = p - closest;
    dist2 = luisa::dot(diff, diff);
}

namespace details
{
// Generated gradient for point-edge distance (PE)
LC_GPU_CALLABLE inline void g_PE3D(Float p0, Float p1, Float p2,
                                   Float e00, Float e01, Float e02,
                                   Float e10, Float e11, Float e12,
                                   Float g[9])
{
    Float t2 = e00 - e10;
    Float t3 = e01 - e11;
    Float t4 = e02 - e12;
    Float t5 = p0 - e00;
    Float t6 = p1 - e01;
    Float t7 = p2 - e02;
    Float t8 = t2 * t2;
    Float t9 = t3 * t3;
    Float t10 = t4 * t4;
    Float t11 = t8 + t9 + t10;
    Float t12 = 1.0f / t11;
    Float t13 = t2 * t5;
    Float t14 = t3 * t6;
    Float t15 = t4 * t7;
    Float t16 = t13 + t14 + t15;
    Float t17 = t12 * t16;
    Float t18 = -t17;
    Float t19 = t17 + 1.0f;
    Float t20 = (t11 == 0.0f);
    Float t21 = t17 * t17;
    Float t22 = t18 * t19 * 2.0f;
    Float t23 = t20 ? 0.0f : t22;
    Float t24 = t8 * t12;
    Float t25 = t20 ? 1.0f : t24;
    Float t26 = t9 * t12;
    Float t27 = t20 ? 1.0f : t26;
    Float t28 = t10 * t12;
    Float t29 = t20 ? 1.0f : t28;
    Float t30 = t25 + t27 + t29 - 1.0f;
    Float t31 = t2 * t3 * t23;
    Float t32 = t2 * t4 * t23;
    Float t33 = t3 * t4 * t23;
    
    g[0] = t30 * 2.0f;
    g[1] = t31;
    g[2] = t32;
    g[3] = t31;
    g[4] = t27 * t30 * 2.0f;
    g[5] = t33;
    g[6] = t32;
    g[7] = t33;
    g[8] = t29 * t30 * 2.0f;
}

// Generated Hessian for point-edge distance (PE)
LC_GPU_CALLABLE inline void H_PE3D(Float p0, Float p1, Float p2,
                                   Float e00, Float e01, Float e02,
                                   Float e10, Float e11, Float e12,
                                   Float H[81])
{
    // Simplified Hessian computation
    // For full symbolic Hessian, we would use the generated code
    // This is a placeholder for the full implementation
    for(int i = 0; i < 81; ++i)
        H[i] = 0.0f;
        
    // TODO: Add full symbolic Hessian from original CUDA implementation
}
}  // namespace details

LC_GPU_CALLABLE inline void point_edge_distance2_gradient(const Vector3& p,
                                                          const Vector3& e0,
                                                          const Vector3& e1,
                                                          std::array<Float, 9>& grad)
{
    Float g[9];
    details::g_PE3D(p.x, p.y, p.z, e0.x, e0.y, e0.z, e1.x, e1.y, e1.z, g);
    for(int i = 0; i < 9; ++i)
        grad[i] = g[i];
}

LC_GPU_CALLABLE inline void point_edge_distance2_hessian(const Vector3& p,
                                                         const Vector3& e0,
                                                         const Vector3& e1,
                                                         std::array<Float, 81>& Hessian)
{
    Float H[81];
    details::H_PE3D(p.x, p.y, p.z, e0.x, e0.y, e0.z, e1.x, e1.y, e1.z, H);
    for(int i = 0; i < 81; ++i)
        Hessian[i] = H[i];
}
}  // namespace uipc::backend::luisa::distance

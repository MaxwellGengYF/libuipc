#pragma once
#include "type_define.h"
#include "point_point.h"
#include "point_edge.h"
#include "point_triangle.h"
#include "edge_edge.h"
#include <luisa/core/mathematics.h>

#include "details/distance_flagged.inl"

namespace uipc::backend::luisa::distance
{
namespace detail
{
    template <int N>
    LC_GPU_CALLABLE inline IndexT active_count(const Vector<IndexT, N>& flag)
    {
        IndexT count = 0;
#pragma unroll
        for(IndexT i = 0; i < N; ++i)
            count += flag[i];
        return count;
    }

    LC_GPU_CALLABLE inline Vector<IndexT, 2> pp_from_pe(const Vector<IndexT, 3>& flag)
    {
        assert(detail::active_count(flag) == 2 && "active count mismatch");

        Vector<IndexT, 2> offsets;
        if(flag[0] == 0)
        {
            offsets = Vector<IndexT, 2>{1, 2};
        }
        else if(flag[1] == 0)
        {
            offsets = Vector<IndexT, 2>{0, 2};
        }
        else if(flag[2] == 0)
        {
            offsets = Vector<IndexT, 2>{0, 1};
        }
        else
        {
            // Invalid flag - return default
            offsets = Vector<IndexT, 2>{0, 1};
        }
        return offsets;
    }

    LC_GPU_CALLABLE inline Vector<IndexT, 3> pe_from_pt(const Vector<IndexT, 4>& flag)
    {
        assert(detail::active_count(flag) == 3 && "active count mismatch");

        Vector<IndexT, 3> offsets;
        if(flag[0] == 0)
        {
            offsets = Vector<IndexT, 3>{1, 2, 3};
        }
        else if(flag[1] == 0)
        {
            offsets = Vector<IndexT, 3>{0, 2, 3};
        }
        else if(flag[2] == 0)
        {
            offsets = Vector<IndexT, 3>{0, 1, 3};
        }
        else if(flag[3] == 0)
        {
            offsets = Vector<IndexT, 3>{0, 1, 2};
        }
        else
        {
            // Invalid flag - return default
            offsets = Vector<IndexT, 3>{0, 1, 2};
        }
        return offsets;
    }

    LC_GPU_CALLABLE inline Vector<IndexT, 2> pp_from_pt(const Vector<IndexT, 4>& flag)
    {
        assert(detail::active_count(flag) == 2 && "active count mismatch");

        Vector<IndexT, 2> offsets;
        constexpr IndexT  N = 4;
        constexpr IndexT  M = 2;

        IndexT iM = 0;
#pragma unroll
        for(IndexT iN = 0; iN < N; ++iN)
        {
            if(flag[iN])
            {
                assert(iM < M && "active mismatch");
                offsets[iM] = iN;
                ++iM;
            }
        }
        return offsets;
    }

    LC_GPU_CALLABLE inline Vector<IndexT, 3> pe_from_ee(const Vector<IndexT, 4>& flag)
    {
        assert(detail::active_count(flag) == 3 && "active count mismatch");

        Vector<IndexT, 3> offsets;  // [P, E0, E1]
        if(flag[0] == 0)
        {
            offsets = Vector<IndexT, 3>{1, 2, 3};
        }
        else if(flag[1] == 0)
        {
            offsets = Vector<IndexT, 3>{0, 2, 3};
        }
        else if(flag[2] == 0)
        {
            offsets = Vector<IndexT, 3>{3, 0, 1};
        }
        else if(flag[3] == 0)
        {
            offsets = Vector<IndexT, 3>{2, 0, 1};
        }
        return offsets;
    }

    LC_GPU_CALLABLE inline Vector<IndexT, 2> pp_from_ee(const Vector<IndexT, 4>& flag)
    {
        assert(detail::active_count(flag) == 2 && "active count mismatch");

        Vector<IndexT, 2> offsets;
        constexpr IndexT  N = 4;
        constexpr IndexT  M = 2;

        IndexT iM = 0;
#pragma unroll
        for(IndexT iN = 0; iN < N; ++iN)
        {
            if(flag[iN])
            {
                assert(iM < M && "active mismatch");
                offsets[iM] = iN;
                ++iM;
            }
        }
        return offsets;
    }
}  // namespace detail


LC_GPU_CALLABLE inline IndexT degenerate_point_triangle(const Vector<IndexT, 4>& flag,
                                                        Vector<IndexT, 4>& offsets)
{
    // collect active indices
    IndexT dim = detail::active_count(flag);
    if(dim == 2)
    {
        offsets[0] = detail::pp_from_pt(flag)[0];
        offsets[1] = detail::pp_from_pt(flag)[1];
    }
    else if(dim == 3)
    {
        auto pe = detail::pe_from_pt(flag);
        offsets[0] = pe[0];
        offsets[1] = pe[1];
        offsets[2] = pe[2];
    }
    else if(dim == 4)
    {
        offsets = Vector<IndexT, 4>{0, 1, 2, 3};
    }
    return dim;
}

LC_GPU_CALLABLE inline IndexT degenerate_edge_edge(const Vector<IndexT, 4>& flag,
                                                   Vector<IndexT, 4>& offsets)
{
    // collect active indices
    IndexT dim = detail::active_count(flag);
    if(dim == 2)
    {
        offsets[0] = detail::pp_from_ee(flag)[0];
        offsets[1] = detail::pp_from_ee(flag)[1];
    }
    else if(dim == 3)
    {
        auto pe = detail::pe_from_ee(flag);
        offsets[0] = pe[0];
        offsets[1] = pe[1];
        offsets[2] = pe[2];
    }
    else if(dim == 4)
    {
        offsets = Vector<IndexT, 4>{0, 1, 2, 3};
    }
    return dim;
}

LC_GPU_CALLABLE inline IndexT degenerate_point_edge(const Vector<IndexT, 3>& flag,
                                                    Vector<IndexT, 3>& offsets)
{
    // collect active indices
    IndexT dim = detail::active_count(flag);
    if(dim == 2)
    {
        auto pp = detail::pp_from_pe(flag);
        offsets[0] = pp[0];
        offsets[1] = pp[1];
    }
    else if(dim == 3)
    {
        offsets = Vector<IndexT, 3>{0, 1, 2};
    }
    return dim;
}

LC_GPU_CALLABLE inline Vector<IndexT, 2> point_point_distance_flag(const Vector3& p0,
                                                                   const Vector3& p1)
{
    return Vector<IndexT, 2>{1, 1};
}

LC_GPU_CALLABLE inline Vector<IndexT, 3> point_edge_distance_flag(const Vector3& p,
                                                                  const Vector3& e0,
                                                                  const Vector3& e1)
{
    Vector<IndexT, 3> F;
    F[0] = 1;

    Vector3 e = e1 - e0;
    Float e2 = luisa::dot(e, e);
    if(e2 <= 0.0f)
    {
        // Degenerate edge -> fallback to point-point against the nearest endpoint.
        Vector3 d0_vec = p - e0;
        Vector3 d1_vec = p - e1;
        Float d0 = luisa::dot(d0_vec, d0_vec);
        Float d1 = luisa::dot(d1_vec, d1_vec);
        F[1] = (d0 <= d1) ? 1 : 0;
        F[2] = (d0 <= d1) ? 0 : 1;
        return F;
    }
    Float ratio = luisa::dot(e, p - e0) / e2;

    F[1] = ratio < 1.0f ? 1 : 0;
    F[2] = ratio > 0.0f ? 1 : 0;

    return F;
}

LC_GPU_CALLABLE inline Vector4i point_triangle_distance_flag(const Vector3& p,
                                                             const Vector3& t0,
                                                             const Vector3& t1,
                                                             const Vector3& t2)
{
    Vector4i F;
    F[0] = 1;

    // clear flags
    F[1] = 0;
    F[2] = 0;
    F[3] = 0;

    // Compute basis vectors
    Vector3 basis_row0 = t1 - t0;
    Vector3 basis_row1 = t2 - t0;

    Vector3 nVec = luisa::cross(basis_row0, basis_row1);

    Vector3 basis_perp_row1 = luisa::cross(basis_row0, nVec);
    
    // 2x2 matrix basis * basis^T
    float basis_basisT_00 = luisa::dot(basis_row0, basis_row0);
    float basis_basisT_01 = luisa::dot(basis_row0, basis_perp_row1);
    float basis_basisT_11 = luisa::dot(basis_perp_row1, basis_perp_row1);
    
    luisa::float2x2 basis_basisT = luisa::make_float2x2(basis_basisT_00, basis_basisT_01, 
                                                         basis_basisT_01, basis_basisT_11);
    auto invBasis = luisa::inverse(basis_basisT);
    
    Vector3 p_minus_t0 = p - t0;
    float dot0 = luisa::dot(basis_row0, p_minus_t0);
    float dot1 = luisa::dot(basis_perp_row1, p_minus_t0);
    luisa::float2 param0 = invBasis * luisa::float2{dot0, dot1};

    if(param0.x > 0.0f && param0.x < 1.0f && param0.y >= 0.0f)
    {
        // PE t0t1
        F[1] = 1;
        F[2] = 1;
    }
    else
    {
        Vector3 row0_1 = t2 - t1;
        Vector3 row1_1 = luisa::cross(row0_1, nVec);
        
        float basisT_00_1 = luisa::dot(row0_1, row0_1);
        float basisT_01_1 = luisa::dot(row0_1, row1_1);
        float basisT_11_1 = luisa::dot(row1_1, row1_1);
        
        luisa::float2x2 basis_basisT_1 = luisa::make_float2x2(basisT_00_1, basisT_01_1,
                                                              basisT_01_1, basisT_11_1);
        auto invBasis_1 = luisa::inverse(basis_basisT_1);
        
        Vector3 p_minus_t1 = p - t1;
        float dot0_1 = luisa::dot(row0_1, p_minus_t1);
        float dot1_1 = luisa::dot(row1_1, p_minus_t1);
        luisa::float2 param1 = invBasis_1 * luisa::float2{dot0_1, dot1_1};

        if(param1.x > 0.0f && param1.x < 1.0f && param1.y >= 0.0f)
        {
            // PE t1t2
            F[2] = 1;
            F[3] = 1;
        }
        else
        {
            Vector3 row0_2 = t0 - t2;
            Vector3 row1_2 = luisa::cross(row0_2, nVec);
            
            float basisT_00_2 = luisa::dot(row0_2, row0_2);
            float basisT_01_2 = luisa::dot(row0_2, row1_2);
            float basisT_11_2 = luisa::dot(row1_2, row1_2);
            
            luisa::float2x2 basis_basisT_2 = luisa::make_float2x2(basisT_00_2, basisT_01_2,
                                                                  basisT_01_2, basisT_11_2);
            auto invBasis_2 = luisa::inverse(basis_basisT_2);
            
            Vector3 p_minus_t2 = p - t2;
            float dot0_2 = luisa::dot(row0_2, p_minus_t2);
            float dot1_2 = luisa::dot(row1_2, p_minus_t2);
            luisa::float2 param2 = invBasis_2 * luisa::float2{dot0_2, dot1_2};

            if(param2.x > 0.0f && param2.x < 1.0f && param2.y >= 0.0f)
            {
                // PE t2t0
                F[3] = 1;
                F[1] = 1;
            }
            else
            {
                if(param0.x <= 0.0f && param2.x >= 1.0f)
                {
                    // PP t0
                    F[1] = 1;
                }
                else if(param1.x <= 0.0f && param0.x >= 1.0f)
                {
                    // PP t1
                    F[2] = 1;
                }
                else if(param2.x <= 0.0f && param1.x >= 1.0f)
                {
                    // PP t2
                    F[3] = 1;
                }
                else
                {  // PT
                    F[1] = 1;
                    F[2] = 1;
                    F[3] = 1;
                }
            }
        }
    }

    return F;
}

namespace detail
{
LC_GPU_CALLABLE inline void update_ee_near_parallel_candidate(const Vector3& p,
                                                              const Vector3& s0,
                                                              const Vector3& s1,
                                                              Float& minD,
                                                              Vector4i& F,
                                                              const Vector4i& F_pe,
                                                              const Vector4i& F_pp0,
                                                              const Vector4i& F_pp1,
                                                              bool is_initial)
{
    auto pe_flag = point_edge_distance_flag(p, s0, s1);
    Float d;
    if(pe_flag[1] && pe_flag[2])
        point_edge_distance2(p, s0, s1, d);
    else if(pe_flag[1])
        point_point_distance2(p, s0, d);
    else
        point_point_distance2(p, s1, d);
    if(is_initial || d < minD)
    {
        minD = d;
        if(pe_flag[1] && pe_flag[2])
            F = F_pe;
        else if(pe_flag[1])
            F = F_pp0;
        else
            F = F_pp1;
    }
}
}  // namespace detail

LC_GPU_CALLABLE inline Vector4i edge_edge_distance_flag(const Vector3& ea0,
                                                        const Vector3& ea1,
                                                        const Vector3& eb0,
                                                        const Vector3& eb1)
{
    Vector4i F = {1, 1, 1, 1};  // default EE
    constexpr Float kEeParallelRelTol = 1e-12f;

    Vector3 u = ea1 - ea0;
    Vector3 v = eb1 - eb0;
    Vector3 w = ea0 - eb0;
    Float a = luisa::dot(u, u);  // always >= 0
    Float b = luisa::dot(u, v);
    Float c = luisa::dot(v, v);  // always >= 0
    Float d = luisa::dot(u, w);
    Float e = luisa::dot(v, w);
    Float D = a * c - b * b;  // always >= 0
    Float tD = D;  // tc = tN / tD, default tD = D >= 0
    Float sN, tN;
    Float uxv2 = luisa::dot(luisa::cross(u, v), luisa::cross(u, v));
    bool near_parallel = (uxv2 <= kEeParallelRelTol * a * c);

    // For near-parallel/collinear edges, avoid the dim==4 EE branch (line-line
    // distance). Evaluate typed point-edge candidates (PP/PE) and pick the
    // minimum typed case, following the distance-type idea in C-IPC.
    if(near_parallel)
    {
        Float minD = 0.0f;
        detail::update_ee_near_parallel_candidate(ea0, eb0, eb1, minD, F,
                                                    Vector4i{1, 0, 1, 1},
                                                    Vector4i{1, 0, 1, 0},
                                                    Vector4i{1, 0, 0, 1},
                                                    true);
        detail::update_ee_near_parallel_candidate(ea1, eb0, eb1, minD, F,
                                                    Vector4i{0, 1, 1, 1},
                                                    Vector4i{0, 1, 1, 0},
                                                    Vector4i{0, 1, 0, 1},
                                                    false);
        detail::update_ee_near_parallel_candidate(eb0, ea0, ea1, minD, F,
                                                    Vector4i{1, 1, 1, 0},
                                                    Vector4i{1, 0, 1, 0},
                                                    Vector4i{0, 1, 1, 0},
                                                    false);
        detail::update_ee_near_parallel_candidate(eb1, ea0, ea1, minD, F,
                                                    Vector4i{1, 1, 0, 1},
                                                    Vector4i{1, 0, 0, 1},
                                                    Vector4i{0, 1, 0, 1},
                                                    false);
        return F;
    }

    // compute the line parameters of the two closest points
    sN = (b * e - c * d);
    if(sN <= 0.0f)
    {  // sc < 0 => the s=0 edge is visible
        tN = e;
        tD = c;

        // PE: Ea0Eb0Eb1

        F[0] = 1;  // Ea0
        F[1] = 0;
        F[2] = 1;  // Eb0
        F[3] = 1;  // Eb1
    }
    else if(sN >= D)
    {  // sc > 1  => the s=1 edge is visible
        tN = e + b;
        tD = c;
        // PE: Ea1Eb0Eb1

        F[0] = 0;
        F[1] = 1;  // Ea1
        F[2] = 1;  // Eb0
        F[3] = 1;  // Eb1
    }
    else
    {
        tN = (a * e - b * d);
        Vector3 u_cross_v = luisa::cross(u, v);
        if(tN > 0.0f && tN < tD
           && (luisa::dot(u_cross_v, w) == 0.0f || luisa::dot(u_cross_v, u_cross_v) < 1.0e-20f * a * c))
        {
            // avoid coplanar or nearly parallel EE
            if(sN < D / 2)
            {
                tN = e;
                tD = c;
                // PE: Ea0Eb0Eb1
                F[0] = 1;  // Ea0
                F[1] = 0;
                F[2] = 1;  // Eb0
                F[3] = 1;  // Eb1
            }
            else
            {
                tN = e + b;
                tD = c;
                // PE: Ea1Eb0Eb1
                F[0] = 0;
                F[1] = 1;  // Ea1
                F[2] = 1;  // Eb0
                F[3] = 1;  // Eb1
            }
        }
        // else defaultCase stays as EE
    }

    if(tN <= 0.0f)
    {  // tc < 0 => the t=0 edge is visible
        // recompute sc for this edge
        if(-d <= 0.0f)
        {
            // PP: Ea0Eb0
            F[0] = 1;  // Ea0
            F[1] = 0;
            F[2] = 1;  // Eb0
            F[3] = 0;
        }
        else if(-d >= a)
        {
            // PP: Ea1Eb0
            F[0] = 0;
            F[1] = 1;  // Ea1
            F[2] = 1;  // Eb0
            F[3] = 0;
        }
        else
        {
            // PE: Eb0Ea0Ea1
            F[0] = 1;  // Ea0
            F[1] = 1;  // Ea1
            F[2] = 1;  // Eb0
            F[3] = 0;
        }
    }
    else if(tN >= tD)
    {  // tc > 1  => the t=1 edge is visible
        // recompute sc for this edge
        if((-d + b) <= 0.0f)
        {
            // PP: Ea0Eb1
            F[0] = 1;  // Ea0
            F[1] = 0;
            F[2] = 0;
            F[3] = 1;  // Eb1
        }
        else if((-d + b) >= a)
        {
            // PP: Ea1Eb1
            F[0] = 0;
            F[1] = 1;  // Ea1
            F[2] = 0;
            F[3] = 1;  // Eb1
        }
        else
        {
            // PE: Eb1Ea0Ea1
            F[0] = 1;  // Ea0
            F[1] = 1;  // Ea1
            F[2] = 0;
            F[3] = 1;  // Eb1
        }
    }

    return F;
}

LC_GPU_CALLABLE inline void point_point_distance2(const Vector2i& flag,
                                                  const Vector3& a,
                                                  const Vector3& b,
                                                  Float& D)
{
    point_point_distance2(a, b, D);
}

LC_GPU_CALLABLE inline void point_edge_distance2(const Vector<IndexT, 3>& flag,
                                                 const Vector3& p,
                                                 const Vector3& e0,
                                                 const Vector3& e1,
                                                 Float& D)
{
    IndexT dim = detail::active_count(flag);
    Vector3 P[3] = {p, e0, e1};

    if(dim == 2)
    {
        Vector2i offsets = detail::pp_from_pe(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];

        point_point_distance2(P0, P1, D);
    }
    else if(dim == 3)
    {
        point_edge_distance2(p, e0, e1, D);
    }
    else
    {
        assert(false && "Invalid flag");
    }
}

LC_GPU_CALLABLE inline void point_triangle_distance2(const Vector4i& flag,
                                                     const Vector3& p,
                                                     const Vector3& t0,
                                                     const Vector3& t1,
                                                     const Vector3& t2,
                                                     Float& D)
{
    IndexT dim = detail::active_count(flag);
    Vector3 P[4] = {p, t0, t1, t2};

    if(dim == 2)
    {
        Vector2i offsets = detail::pp_from_pt(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];

        point_point_distance2(P0, P1, D);
    }
    else if(dim == 3)
    {
        Vector3i offsets = detail::pe_from_pt(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];
        Vector3& P2 = P[offsets[2]];

        point_edge_distance2(P0, P1, P2, D);
    }
    else if(dim == 4)
    {
        point_triangle_distance2(p, t0, t1, t2, D);
    }
    else
    {
        assert(false && "Invalid flag");
    }
}

LC_GPU_CALLABLE inline void edge_edge_distance2(const Vector4i& flag,
                                                const Vector3& ea0,
                                                const Vector3& ea1,
                                                const Vector3& eb0,
                                                const Vector3& eb1,
                                                Float& D)
{
    IndexT dim = detail::active_count(flag);
    Vector3 P[4] = {ea0, ea1, eb0, eb1};

    if(dim == 2)
    {
        Vector2i offsets = detail::pp_from_ee(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];

        point_point_distance2(P0, P1, D);
    }
    else if(dim == 3)
    {
        Vector3i offsets = detail::pe_from_ee(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];
        Vector3& P2 = P[offsets[2]];

        point_edge_distance2(P0, P1, P2, D);
    }
    else if(dim == 4)
    {
        edge_edge_distance2(ea0, ea1, eb0, eb1, D);
    }
    else
    {
        assert(false && "Invalid flag");
    }
}

// Include implementation details
#include "details/distance_flagged.inl"

}  // namespace uipc::backend::luisa::distance

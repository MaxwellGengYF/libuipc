#pragma once

namespace uipc::backend::luisa::distance
{
LC_GPU_CALLABLE inline void point_point_distance2_gradient(const Vector2i& flag,
                                                           const Vector3& a,
                                                           const Vector3& b,
                                                           std::array<Float, 6>& G)
{
    point_point_distance2_gradient(a, b, G);
}

LC_GPU_CALLABLE inline void point_edge_distance2_gradient(const Vector<IndexT, 3>& flag,
                                                          const Vector3& p,
                                                          const Vector3& e0,
                                                          const Vector3& e1,
                                                          std::array<Float, 9>& G)
{
    // Initialize to zero
    for(int i = 0; i < 9; ++i)
        G[i] = 0.0f;

    IndexT dim = detail::active_count(flag);
    Vector3 P[3] = {p, e0, e1};

    if(dim == 2)
    {
        Vector2i offsets = detail::pp_from_pe(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];

        std::array<Float, 6> G6;
        point_point_distance2_gradient(P0, P1, G6);

#pragma unroll
        for(int i = 0; i < 2; ++i)
        {
            int offset = offsets[i] * 3;
            int src_offset = i * 3;
            G[offset] = G6[src_offset];
            G[offset + 1] = G6[src_offset + 1];
            G[offset + 2] = G6[src_offset + 2];
        }
    }
    else if(dim == 3)
    {
        std::array<Float, 9> G9;
        point_edge_distance2_gradient(p, e0, e1, G9);
        for(int i = 0; i < 9; ++i)
            G[i] = G9[i];
    }
    else
    {
        assert(false && "Invalid flag");
    }
}

LC_GPU_CALLABLE inline void point_triangle_distance2_gradient(const Vector4i& flag,
                                                              const Vector3& p,
                                                              const Vector3& t0,
                                                              const Vector3& t1,
                                                              const Vector3& t2,
                                                              std::array<Float, 12>& G)
{
    // Initialize to zero
    for(int i = 0; i < 12; ++i)
        G[i] = 0.0f;

    IndexT dim = detail::active_count(flag);
    Vector3 P[4] = {p, t0, t1, t2};

    if(dim == 2)
    {
        Vector2i offsets = detail::pp_from_pt(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];

        std::array<Float, 6> G6;
        point_point_distance2_gradient(P0, P1, G6);

#pragma unroll
        for(int i = 0; i < 2; ++i)
        {
            int offset = offsets[i] * 3;
            int src_offset = i * 3;
            G[offset] = G6[src_offset];
            G[offset + 1] = G6[src_offset + 1];
            G[offset + 2] = G6[src_offset + 2];
        }
    }
    else if(dim == 3)
    {
        Vector3i offsets = detail::pe_from_pt(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];
        Vector3& P2 = P[offsets[2]];

        std::array<Float, 9> G9;
        point_edge_distance2_gradient(P0, P1, P2, G9);

#pragma unroll
        for(int i = 0; i < 3; ++i)
        {
            int offset = offsets[i] * 3;
            int src_offset = i * 3;
            G[offset] = G9[src_offset];
            G[offset + 1] = G9[src_offset + 1];
            G[offset + 2] = G9[src_offset + 2];
        }
    }
    else if(dim == 4)
    {
        std::array<Float, 12> G12;
        point_triangle_distance2_gradient(p, t0, t1, t2, G12);
        for(int i = 0; i < 12; ++i)
            G[i] = G12[i];
    }
    else
    {
        assert(false && "Invalid flag");
    }
}

LC_GPU_CALLABLE inline void edge_edge_distance2_gradient(const Vector4i& flag,
                                                         const Vector3& ea0,
                                                         const Vector3& ea1,
                                                         const Vector3& eb0,
                                                         const Vector3& eb1,
                                                         std::array<Float, 12>& G)
{
    // Initialize to zero
    for(int i = 0; i < 12; ++i)
        G[i] = 0.0f;

    IndexT dim = detail::active_count(flag);
    Vector3 P[4] = {ea0, ea1, eb0, eb1};

    if(dim == 2)
    {
        Vector2i offsets = detail::pp_from_ee(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];

        std::array<Float, 6> G6;
        point_point_distance2_gradient(P0, P1, G6);

#pragma unroll
        for(int i = 0; i < 2; ++i)
        {
            int offset = offsets[i] * 3;
            int src_offset = i * 3;
            G[offset] = G6[src_offset];
            G[offset + 1] = G6[src_offset + 1];
            G[offset + 2] = G6[src_offset + 2];
        }
    }
    else if(dim == 3)
    {
        Vector3i offsets = detail::pe_from_ee(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];
        Vector3& P2 = P[offsets[2]];

        std::array<Float, 9> G9;
        point_edge_distance2_gradient(P0, P1, P2, G9);

#pragma unroll
        for(int i = 0; i < 3; ++i)
        {
            int offset = offsets[i] * 3;
            int src_offset = i * 3;
            G[offset] = G9[src_offset];
            G[offset + 1] = G9[src_offset + 1];
            G[offset + 2] = G9[src_offset + 2];
        }
    }
    else if(dim == 4)
    {
        std::array<Float, 12> G12;
        edge_edge_distance2_gradient(ea0, ea1, eb0, eb1, G12);
        for(int i = 0; i < 12; ++i)
            G[i] = G12[i];
    }
    else
    {
        assert(false && "Invalid flag");
    }
}

LC_GPU_CALLABLE inline void point_point_distance2_hessian(const Vector2i& flag,
                                                          const Vector3& a,
                                                          const Vector3& b,
                                                          std::array<Float, 36>& H)
{
    point_point_distance2_hessian(a, b, H);
}

LC_GPU_CALLABLE inline void point_edge_distance2_hessian(const Vector<IndexT, 3>& flag,
                                                         const Vector3& p,
                                                         const Vector3& e0,
                                                         const Vector3& e1,
                                                         std::array<Float, 81>& H)
{
    // Initialize to zero
    for(int i = 0; i < 81; ++i)
        H[i] = 0.0f;

    IndexT dim = detail::active_count(flag);
    Vector3 P[3] = {p, e0, e1};

    if(dim == 2)
    {
        Vector2i offsets = detail::pp_from_pe(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];

        std::array<Float, 36> H6;
        point_point_distance2_hessian(P0, P1, H6);

#pragma unroll
        for(int i = 0; i < 2; ++i)
        {
            int offset_i = offsets[i] * 3;
            for(int j = 0; j < 2; ++j)
            {
                int offset_j = offsets[j] * 3;
                int src_i = i * 3;
                int src_j = j * 3;
                H[(offset_i) * 9 + (offset_j)] = H6[src_i * 6 + src_j];
                H[(offset_i) * 9 + (offset_j + 1)] = H6[src_i * 6 + src_j + 1];
                H[(offset_i) * 9 + (offset_j + 2)] = H6[src_i * 6 + src_j + 2];
                H[(offset_i + 1) * 9 + (offset_j)] = H6[(src_i + 1) * 6 + src_j];
                H[(offset_i + 1) * 9 + (offset_j + 1)] = H6[(src_i + 1) * 6 + src_j + 1];
                H[(offset_i + 1) * 9 + (offset_j + 2)] = H6[(src_i + 1) * 6 + src_j + 2];
                H[(offset_i + 2) * 9 + (offset_j)] = H6[(src_i + 2) * 6 + src_j];
                H[(offset_i + 2) * 9 + (offset_j + 1)] = H6[(src_i + 2) * 6 + src_j + 1];
                H[(offset_i + 2) * 9 + (offset_j + 2)] = H6[(src_i + 2) * 6 + src_j + 2];
            }
        }
    }
    else if(dim == 3)
    {
        std::array<Float, 81> H9;
        point_edge_distance2_hessian(p, e0, e1, H9);
        for(int i = 0; i < 81; ++i)
            H[i] = H9[i];
    }
    else
    {
        assert(false && "Invalid flag");
    }
}

LC_GPU_CALLABLE inline void point_triangle_distance2_hessian(const Vector4i& flag,
                                                             const Vector3& p,
                                                             const Vector3& t0,
                                                             const Vector3& t1,
                                                             const Vector3& t2,
                                                             std::array<Float, 144>& H)
{
    // Initialize to zero
    for(int i = 0; i < 144; ++i)
        H[i] = 0.0f;

    IndexT dim = detail::active_count(flag);
    Vector3 P[4] = {p, t0, t1, t2};

    if(dim == 2)
    {
        Vector2i offsets = detail::pp_from_pt(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];

        std::array<Float, 36> H6;
        point_point_distance2_hessian(P0, P1, H6);

#pragma unroll
        for(int i = 0; i < 2; ++i)
        {
            int offset_i = offsets[i] * 3;
            for(int j = 0; j < 2; ++j)
            {
                int offset_j = offsets[j] * 3;
                int src_i = i * 3;
                int src_j = j * 3;
                H[(offset_i) * 12 + (offset_j)] = H6[src_i * 6 + src_j];
                H[(offset_i) * 12 + (offset_j + 1)] = H6[src_i * 6 + src_j + 1];
                H[(offset_i) * 12 + (offset_j + 2)] = H6[src_i * 6 + src_j + 2];
                H[(offset_i + 1) * 12 + (offset_j)] = H6[(src_i + 1) * 6 + src_j];
                H[(offset_i + 1) * 12 + (offset_j + 1)] = H6[(src_i + 1) * 6 + src_j + 1];
                H[(offset_i + 1) * 12 + (offset_j + 2)] = H6[(src_i + 1) * 6 + src_j + 2];
                H[(offset_i + 2) * 12 + (offset_j)] = H6[(src_i + 2) * 6 + src_j];
                H[(offset_i + 2) * 12 + (offset_j + 1)] = H6[(src_i + 2) * 6 + src_j + 1];
                H[(offset_i + 2) * 12 + (offset_j + 2)] = H6[(src_i + 2) * 6 + src_j + 2];
            }
        }
    }
    else if(dim == 3)
    {
        Vector3i offsets = detail::pe_from_pt(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];
        Vector3& P2 = P[offsets[2]];

        std::array<Float, 81> H9;
        point_edge_distance2_hessian(P0, P1, P2, H9);

#pragma unroll
        for(int i = 0; i < 3; ++i)
        {
            int offset_i = offsets[i] * 3;
            for(int j = 0; j < 3; ++j)
            {
                int offset_j = offsets[j] * 3;
                int src_i = i * 3;
                int src_j = j * 3;
                H[(offset_i) * 12 + (offset_j)] = H9[src_i * 9 + src_j];
                H[(offset_i) * 12 + (offset_j + 1)] = H9[src_i * 9 + src_j + 1];
                H[(offset_i) * 12 + (offset_j + 2)] = H9[src_i * 9 + src_j + 2];
                H[(offset_i + 1) * 12 + (offset_j)] = H9[(src_i + 1) * 9 + src_j];
                H[(offset_i + 1) * 12 + (offset_j + 1)] = H9[(src_i + 1) * 9 + src_j + 1];
                H[(offset_i + 1) * 12 + (offset_j + 2)] = H9[(src_i + 1) * 9 + src_j + 2];
                H[(offset_i + 2) * 12 + (offset_j)] = H9[(src_i + 2) * 9 + src_j];
                H[(offset_i + 2) * 12 + (offset_j + 1)] = H9[(src_i + 2) * 9 + src_j + 1];
                H[(offset_i + 2) * 12 + (offset_j + 2)] = H9[(src_i + 2) * 9 + src_j + 2];
            }
        }
    }
    else if(dim == 4)
    {
        std::array<Float, 144> H12;
        point_triangle_distance2_hessian(p, t0, t1, t2, H12);
        for(int i = 0; i < 144; ++i)
            H[i] = H12[i];
    }
    else
    {
        assert(false && "Invalid flag");
    }
}

LC_GPU_CALLABLE inline void edge_edge_distance2_hessian(const Vector4i& flag,
                                                        const Vector3& ea0,
                                                        const Vector3& ea1,
                                                        const Vector3& eb0,
                                                        const Vector3& eb1,
                                                        std::array<Float, 144>& H)
{
    // Initialize to zero
    for(int i = 0; i < 144; ++i)
        H[i] = 0.0f;

    IndexT dim = detail::active_count(flag);
    Vector3 P[4] = {ea0, ea1, eb0, eb1};

    if(dim == 2)
    {
        Vector2i offsets = detail::pp_from_ee(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];

        std::array<Float, 36> H6;
        point_point_distance2_hessian(P0, P1, H6);

#pragma unroll
        for(int i = 0; i < 2; ++i)
        {
            int offset_i = offsets[i] * 3;
            for(int j = 0; j < 2; ++j)
            {
                int offset_j = offsets[j] * 3;
                int src_i = i * 3;
                int src_j = j * 3;
                H[(offset_i) * 12 + (offset_j)] = H6[src_i * 6 + src_j];
                H[(offset_i) * 12 + (offset_j + 1)] = H6[src_i * 6 + src_j + 1];
                H[(offset_i) * 12 + (offset_j + 2)] = H6[src_i * 6 + src_j + 2];
                H[(offset_i + 1) * 12 + (offset_j)] = H6[(src_i + 1) * 6 + src_j];
                H[(offset_i + 1) * 12 + (offset_j + 1)] = H6[(src_i + 1) * 6 + src_j + 1];
                H[(offset_i + 1) * 12 + (offset_j + 2)] = H6[(src_i + 1) * 6 + src_j + 2];
                H[(offset_i + 2) * 12 + (offset_j)] = H6[(src_i + 2) * 6 + src_j];
                H[(offset_i + 2) * 12 + (offset_j + 1)] = H6[(src_i + 2) * 6 + src_j + 1];
                H[(offset_i + 2) * 12 + (offset_j + 2)] = H6[(src_i + 2) * 6 + src_j + 2];
            }
        }
    }
    else if(dim == 3)
    {
        Vector3i offsets = detail::pe_from_ee(flag);
        Vector3& P0 = P[offsets[0]];
        Vector3& P1 = P[offsets[1]];
        Vector3& P2 = P[offsets[2]];

        std::array<Float, 81> H9;
        point_edge_distance2_hessian(P0, P1, P2, H9);

#pragma unroll
        for(int i = 0; i < 3; ++i)
        {
            int offset_i = offsets[i] * 3;
            for(int j = 0; j < 3; ++j)
            {
                int offset_j = offsets[j] * 3;
                int src_i = i * 3;
                int src_j = j * 3;
                H[(offset_i) * 12 + (offset_j)] = H9[src_i * 9 + src_j];
                H[(offset_i) * 12 + (offset_j + 1)] = H9[src_i * 9 + src_j + 1];
                H[(offset_i) * 12 + (offset_j + 2)] = H9[src_i * 9 + src_j + 2];
                H[(offset_i + 1) * 12 + (offset_j)] = H9[(src_i + 1) * 9 + src_j];
                H[(offset_i + 1) * 12 + (offset_j + 1)] = H9[(src_i + 1) * 9 + src_j + 1];
                H[(offset_i + 1) * 12 + (offset_j + 2)] = H9[(src_i + 1) * 9 + src_j + 2];
                H[(offset_i + 2) * 12 + (offset_j)] = H9[(src_i + 2) * 9 + src_j];
                H[(offset_i + 2) * 12 + (offset_j + 1)] = H9[(src_i + 2) * 9 + src_j + 1];
                H[(offset_i + 2) * 12 + (offset_j + 2)] = H9[(src_i + 2) * 9 + src_j + 2];
            }
        }
    }
    else if(dim == 4)
    {
        std::array<Float, 144> H12;
        edge_edge_distance2_hessian(ea0, ea1, eb0, eb1, H12);
        for(int i = 0; i < 144; ++i)
            H[i] = H12[i];
    }
    else
    {
        assert(false && "Invalid flag");
    }
}
}  // namespace uipc::backend::luisa::distance

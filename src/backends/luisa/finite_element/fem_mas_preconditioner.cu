#include <type_define.h>
#include <Eigen/Dense>
#include <linear_system/local_preconditioner.h>
#include <finite_element/finite_element_method.h>
#include <linear_system/global_linear_system.h>
#include <finite_element/fem_linear_subsystem.h>
#include <global_geometry/global_vertex_manager.h>
#include <finite_element/mas_preconditioner_engine.h>
#include <uipc/geometry/simplicial_complex.h>
#include <uipc/common/log.h>
#include <set>

namespace uipc::backend::luisa
{
// Free function: Fill identity indices
void fill_identity_indices(luisa::compute::Buffer<uint32_t>& buf, int count, SimEngine* engine)
{
    using namespace luisa::compute;
    buf = engine->device().create_buffer<uint32_t>(count);

    Kernel1D fill_kernel = [&](BufferVar<uint32_t> indices) noexcept
    {
        auto i = dispatch_x();
        indices.write(i, cast<uint32_t>(i));
    };

    auto shader = engine->device().compile(fill_kernel);
    engine->stream() << shader(buf).dispatch(count);
}

void assemble_diag_inv_for_unpartitioned(
    luisa::compute::Buffer<Matrix3x3>&    diag_inv,
    const luisa::compute::Buffer<int>&    unpart_flags,
    luisa::compute::BufferView<Matrix3x3> A_triplets,
    luisa::compute::BufferView<IndexT>    A_rows,
    luisa::compute::BufferView<IndexT>    A_cols,
    int                                   fem_block_offset,
    int                                   fem_block_count,
    SizeT                                 num_verts,
    SimEngine*                            engine)
{
    using namespace luisa::compute;

    diag_inv = engine->device().create_buffer<Matrix3x3>(num_verts);
    
    // Fill with identity
    Kernel1D fill_identity_kernel = [&](BufferVar<Matrix3x3> diag) noexcept
    {
        auto i = dispatch_x();
        diag.write(i, make_float3x3(1.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f,
                                    0.0f, 0.0f, 1.0f));
    };
    
    auto fill_shader = engine->device().compile(fill_identity_kernel);
    engine->stream() << fill_shader(diag_inv).dispatch(num_verts);

    // Assemble diagonal inverse
    auto triplet_count = A_triplets.size();
    
    Kernel1D assemble_diag_kernel = [&](BufferVar<Matrix3x3> diag_buffer,
                                        BufferVar<int> unpart_buffer,
                                        BufferVar<Matrix3x3> triplet_buffer,
                                        BufferVar<IndexT> row_buffer,
                                        BufferVar<IndexT> col_buffer,
                                        Var<int> fem_offset,
                                        Var<int> fem_count) noexcept
    {
        auto I = dispatch_x();
        $if(I < triplet_count)
        {
            auto g_i = row_buffer.read(I);
            auto g_j = col_buffer.read(I);
            auto H3x3 = triplet_buffer.read(I);

            auto i = cast<int>(g_i) - fem_offset;
            auto j = cast<int>(g_j) - fem_offset;

            $if((i >= 0) && (i < fem_count) && (j >= 0) && (j < fem_count))
            {
                $if((i == j) && (unpart_buffer.read(i) == 1))
                {
                    // Compute inverse of H3x3
                    auto inv = inverse(H3x3);
                    diag_buffer.write(i, inv);
                };
            };
        };
    };

    auto shader = engine->device().compile(assemble_diag_kernel);
    engine->stream() << shader(diag_inv, unpart_flags, A_triplets, A_rows, A_cols,
                               fem_block_offset, fem_block_count).dispatch(triplet_count);
}

void apply_diag_inv_for_unpartitioned(
    const luisa::compute::Buffer<Matrix3x3>& diag_inv,
    const luisa::compute::Buffer<int>&       unpart_flags,
    luisa::compute::BufferView<Float>        z,
    luisa::compute::BufferView<Float>        r,
    luisa::compute::BufferView<IndexT>       converged,
    SizeT                                    num_verts,
    SimEngine*                               engine)
{
    using namespace luisa::compute;

    Kernel1D apply_diag_kernel = [&](BufferVar<Float> r_buffer,
                                     BufferVar<Float> z_buffer,
                                     BufferVar<Matrix3x3> diag_buffer,
                                     BufferVar<int> unpart_buffer,
                                     BufferVar<IndexT> converged_buffer) noexcept
    {
        auto i = dispatch_x();
        $if(converged_buffer.read(0) == 0)
        {
            $if(unpart_buffer.read(i) == 1)
            {
                auto idx = i * 3;
                auto r0 = r_buffer.read(idx);
                auto r1 = r_buffer.read(idx + 1);
                auto r2 = r_buffer.read(idx + 2);
                auto r_vec = make_float3(r0, r1, r2);

                auto diag = diag_buffer.read(i);
                auto z_vec = diag * r_vec;

                z_buffer.write(idx, z_vec.x);
                z_buffer.write(idx + 1, z_vec.y);
                z_buffer.write(idx + 2, z_vec.z);
            };
        };
    };

    auto shader = engine->device().compile(apply_diag_kernel);
    engine->stream() << shader(r, z, diag_inv, unpart_flags, converged).dispatch(num_verts);
}

/**
 * @brief FEM MAS (Multiplicative Additive Schwarz) Preconditioner
 *
 * A multi-level domain-decomposition preconditioner that replaces the
 * simpler diagonal (Jacobi) preconditioner for much better convergence
 * on stiff problems. Based on the StiffGIPC paper.
 *
 * Prerequisites:
 *   The user must call `mesh_partition(sc, 16)` on their SimplicialComplex
 *   BEFORE `world.init()` to create a "mesh_part" vertex attribute.
 *   If this attribute is absent, the system will not be built and the
 *   default FEMDiagPreconditioner will be used instead.
 */
class FEMMASPreconditioner : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    static constexpr int BANKSIZE = MASPreconditionerEngine::BANKSIZE;

  private:
    FiniteElementMethod* finite_element_method = nullptr;
    GlobalLinearSystem*  global_linear_system  = nullptr;
    FEMLinearSubsystem*  fem_linear_subsystem  = nullptr;

    MASPreconditionerEngine engine;
    bool                    m_has_partition       = false;
    bool                    m_has_unpartitioned   = false;
    bool                    m_contact_aware       = false;
    luisa::compute::Buffer<uint32_t>  sorted_indices;

    // Diagonal fallback for unpartitioned vertices
    luisa::compute::Buffer<Matrix3x3> diag_inv;
    luisa::compute::Buffer<int>       unpartitioned_flags;  // 1 = unpartitioned, 0 = partitioned

    virtual void do_build(BuildInfo& info) override
    {
        finite_element_method       = &require<FiniteElementMethod>();
        global_linear_system        = &require<GlobalLinearSystem>();
        fem_linear_subsystem        = &require<FEMLinearSubsystem>();
        auto& global_vertex_manager = require<GlobalVertexManager>();

        // MAS activates if ANY FEM geometry has mesh_part attribute.
        // Unpartitioned meshes get diagonal (block-Jacobi) fallback internally.
        auto geo_slots       = world().scene().geometries();
        bool found_any_part  = false;
        for(SizeT i = 0; i < geo_slots.size(); i++)
        {
            auto& geo = geo_slots[i]->geometry();
            auto* sc  = geo.as<geometry::SimplicialComplex>();
            if(sc && sc->dim() >= 1)
            {
                auto mesh_part = sc->vertices().find<IndexT>("mesh_part");
                if(mesh_part)
                {
                    found_any_part = true;
                    break;
                }
            }
        }

        if(!found_any_part)
        {
            throw SimSystemException(
                "FEMMASPreconditioner: No 'mesh_part' attribute found on any geometry.");
        }

        // Contact-aware MAS: inject BCOO off-diagonal coupling into hierarchy
        auto ca_attr = world().scene().config().find<IndexT>(
            "linear_system/precond/mas/contact_aware");
        m_contact_aware = ca_attr ? (ca_attr->view()[0] != 0) : true;

        info.connect(fem_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override
    {
        auto& fem = finite_element_method->m_impl;

        SizeT vert_num = fem.xs.size();
        if(vert_num == 0)
            return;

        // ---- 1. Build vertex adjacency from element connectivity ----

        std::vector<std::set<unsigned int>> vert_neighbors(vert_num);

        auto add_edge = [&](IndexT a, IndexT b)
        {
            if(a != b && a >= 0 && b >= 0
               && a < static_cast<IndexT>(vert_num)
               && b < static_cast<IndexT>(vert_num))
            {
                vert_neighbors[a].insert(static_cast<unsigned int>(b));
                vert_neighbors[b].insert(static_cast<unsigned int>(a));
            }
        };

        for(auto& tet : fem.h_tets)
            for(int i = 0; i < 4; i++)
                for(int j = i + 1; j < 4; j++)
                    add_edge(tet[i], tet[j]);

        for(auto& tri : fem.h_codim_2ds)
            for(int i = 0; i < 3; i++)
                for(int j = i + 1; j < 3; j++)
                    add_edge(tri[i], tri[j]);

        for(auto& edge : fem.h_codim_1ds)
            add_edge(edge[0], edge[1]);

        // ---- 2. Build CSR neighbor arrays ----

        std::vector<unsigned int> h_neighbor_list;
        std::vector<unsigned int> h_neighbor_start(vert_num, 0);
        std::vector<unsigned int> h_neighbor_num(vert_num, 0);

        for(SizeT i = 0; i < vert_num; i++)
        {
            h_neighbor_start[i] = static_cast<unsigned int>(h_neighbor_list.size());
            h_neighbor_num[i]   = static_cast<unsigned int>(vert_neighbors[i].size());
            for(auto n : vert_neighbors[i])
                h_neighbor_list.push_back(n);
        }

        // ---- 3. Read mesh_part attribute and build partition mappings ----

        std::vector<IndexT> part_ids(vert_num, -1);
        bool                has_parts = false;

        // Partition IDs are mesh-local (each mesh starts at 0).
        // We add a running offset so that IDs are globally unique
        // across all geometries in the FEM system.
        IndexT partition_offset = 0;

        auto geo_slots = world().scene().geometries();
        for(auto& geo_info : fem.geo_infos)
        {
            auto& geo_slot = geo_slots[geo_info.geo_slot_index];
            auto& geo      = geo_slot->geometry();
            auto* sc       = geo.as<geometry::SimplicialComplex>();
            if(!sc) continue;

            auto mesh_part = sc->vertices().find<IndexT>("mesh_part");
            if(!mesh_part) continue;

            has_parts      = true;
            auto part_view = mesh_part->view();

            IndexT local_max = 0;
            for(SizeT v = 0; v < geo_info.vertex_count; v++)
            {
                IndexT local_pid = part_view[v];
                part_ids[geo_info.vertex_offset + v] = local_pid + partition_offset;
                local_max = std::max(local_max, local_pid);
            }
            partition_offset += local_max + 1;  // next mesh starts after this mesh's max
        }

        if(!has_parts)
        {
            m_has_partition = false;
            return;
        }

        // Check if any vertices are unpartitioned (part_ids[v] == -1)
        {
            std::vector<int> h_unpart_flags(vert_num, 0);
            bool any_unpartitioned = false;
            for(SizeT i = 0; i < vert_num; i++)
            {
                if(part_ids[i] < 0)
                {
                    h_unpart_flags[i] = 1;
                    any_unpartitioned = true;
                }
            }
            m_has_unpartitioned = any_unpartitioned;
            unpartitioned_flags = engine->device().create_buffer<int>(vert_num);
            unpartitioned_flags.view().copy_from(h_unpart_flags.data());
        }

        // ---- 4. Build partition-ordered index mappings ----

        IndexT max_part_id = 0;
        for(auto pid : part_ids)
            if(pid > max_part_id) max_part_id = pid;

        std::vector<std::vector<int>> part_blocks(max_part_id + 1);
        for(SizeT i = 0; i < vert_num; i++)
            if(part_ids[i] >= 0)
                part_blocks[part_ids[i]].push_back(static_cast<int>(i));

        // Validate: no partition block should exceed BANKSIZE.
        // If this fires, partition IDs from different meshes are colliding.
        for(SizeT b = 0; b < part_blocks.size(); b++)
        {
            UIPC_ASSERT(static_cast<int>(part_blocks[b].size()) <= BANKSIZE,
                         "MAS: partition {} has {} vertices (max {}). "
                         "Partition IDs from different meshes may be colliding — "
                         "need global offset.",
                         b,
                         part_blocks[b].size(),
                         BANKSIZE);
        }

        // Each partition block is padded to BANKSIZE alignment
        int part_map_size = 0;
        for(auto& block : part_blocks)
        {
            int padded = (static_cast<int>(block.size()) + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
            part_map_size += padded;
        }

        std::vector<int> h_part_to_real(part_map_size, -1);
        std::vector<int> h_real_to_part(vert_num, -1);

        int offset = 0;
        for(auto& block : part_blocks)
        {
            for(SizeT i = 0; i < block.size(); i++)
            {
                int real_idx                  = block[i];
                h_part_to_real[offset + (int)i] = real_idx;
                h_real_to_part[real_idx]        = offset + static_cast<int>(i);
            }
            int padded = (static_cast<int>(block.size()) + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
            offset += padded;
        }

        // Validate mappings to avoid out-of-range indices in MAS kernels.
        for(SizeT i = 0; i < vert_num; ++i)
        {
            auto pid = part_ids[i];
            if(pid < 0)
            {
                UIPC_ASSERT(h_real_to_part[i] == -1,
                            "MAS: unpartitioned vertex {} must map to -1, got {}.",
                            i,
                            h_real_to_part[i]);
            }
            else
            {
                UIPC_ASSERT(h_real_to_part[i] >= 0
                                && h_real_to_part[i] < part_map_size,
                            "MAS: real_to_part[{}]={} out of range [0, {}).",
                            i,
                            h_real_to_part[i],
                            part_map_size);
            }
        }

        for(int i = 0; i < part_map_size; ++i)
        {
            int rid = h_part_to_real[i];
            if(rid >= 0)
            {
                UIPC_ASSERT(rid < static_cast<int>(vert_num),
                            "MAS: part_to_real[{}]={} out of range [0, {}).",
                            i,
                            rid,
                            vert_num);
            }
        }

        // ---- 5. Initialize the engine ----

        engine.init_neighbor(static_cast<int>(vert_num),
                             static_cast<int>(h_neighbor_list.size()),
                             part_map_size,
                             h_neighbor_list,
                             h_neighbor_start,
                             h_neighbor_num,
                             h_part_to_real,
                             h_real_to_part);

        engine.init_matrix();
        m_has_partition = true;
    }

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        if(!m_has_partition || !engine.is_initialized())
            return;

        using namespace luisa::compute;

        auto A          = info.A();
        int  dof_offset = static_cast<int>(info.dof_offset());

        auto triplet_count = A.triplet_count();
        auto values_view   = A.values();
        auto row_view      = A.row_indices();
        auto col_view      = A.col_indices();

        auto* values  = reinterpret_cast<const Eigen::Matrix3d*>(values_view.data());
        auto* row_ids = reinterpret_cast<const int*>(row_view.data());
        auto* col_ids = reinterpret_cast<const int*>(col_view.data());

        // Contact-aware: pass BCOO row/col indices for off-diagonal coupling
        if(m_contact_aware)
        {
            engine.set_hessian_coupling(row_ids, col_ids,
                                        static_cast<int>(triplet_count),
                                        dof_offset / 3);
        }
        else
        {
            engine.set_hessian_coupling(nullptr, nullptr, 0, 0);
        }

        // MAS assembly for partitioned vertices
        fill_identity_indices(sorted_indices, triplet_count, engine);
        engine.set_preconditioner(values,
                                  row_ids,
                                  col_ids,
                                  reinterpret_cast<const uint32_t*>(sorted_indices.data()),
                                  dof_offset / 3,
                                  static_cast<int>(triplet_count),
                                  0);

        // Diagonal fallback assembly for unpartitioned vertices
        if(m_has_unpartitioned)
        {
            SizeT num_verts      = finite_element_method->xs().size();
            int fem_block_offset = dof_offset / 3;
            int fem_block_count  = static_cast<int>(info.dof_count()) / 3;

            assemble_diag_inv_for_unpartitioned(
                diag_inv, unpartitioned_flags, values_view, row_view, col_view,
                fem_block_offset, fem_block_count, num_verts, engine);
        }
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        if(!m_has_partition || !engine.is_initialized())
            return;

        using namespace luisa::compute;
        auto converged = info.converged();

        // MAS for partitioned vertices
        engine.apply(info.r(), info.z(), converged);

        // Diagonal fallback for unpartitioned vertices
        if(m_has_unpartitioned)
        {
            apply_diag_inv_for_unpartitioned(
                diag_inv, unpartitioned_flags, info.z(), info.r(), converged,
                diag_inv.size(), engine);
        }
    }
};

REGISTER_SIM_SYSTEM(FEMMASPreconditioner);
}  // namespace uipc::backend::luisa

#include <finite_element/fem_mas_preconditioner.h>
#include <finite_element/fem_mas_preconditioner_function.h>
#include <linear_system/global_linear_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <frame_stream.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/geometry/simplicial_complex.h>
#include <uipc/geometry/utils/mesh_partition.h>
#include <uipc/common/log.h>
#include <backends/common/backend_path_tool.h>
#include <sim_engine.h>
#include <set>
#include <memory>

/*************************************************************************************************
 * T6.4c — FEMMASPreconditioner (port of
 * src/backends/cuda/finite_element/fem_mas_preconditioner.cu).
 *
 * Read the scope note in fem_mas_preconditioner.h first.
 *
 * ACTIVATION (the reference's own contract, transcribed): the scene config
 * `linear_system/fem_preconditioner == "mas"`.  Anything else throws
 * SimSystemException, i.e. this preconditioner is *explicitly not in play* — the same
 * idiom FEMDiagPreconditioner uses when the value *is* "mas".  Nothing falls back
 * silently: with "mas" the diagonal preconditioner defers (its own throw) and this one
 * builds; without it this one is invalidated and the diagonal one builds.
 *
 * Kernel mapping (the reference declares both fallback kernels in the .cu):
 *   assemble_diag_inv_for_unpartitioned_kernel -> mas_unpart_assemble_kernel
 *                                                 (+ ..._detail::unpart_diag_inv_row)
 *   apply_diag_inv_for_unpartitioned_kernel    -> mas_unpart_apply_kernel
 *                                                 (+ ..._detail::unpart_diag_apply_row)
 *
 * Reference-fidelity notes:
 *  * `engine.init_neighbor(...)`/`init_matrix()` are the reference's single
 *    initialization pair; the hierarchy is invariant for the engine's lifetime
 *    (`set_preconditioner` only re-assembles and re-inverts).
 *  * the partition block size is the *fixed* BANKSIZE = 16 (the engine's cluster
 *    size); a pre-existing `mesh_part` attribute is respected as-is, which is the
 *    reference's documented "custom C++ partitioning" path.
 *  * the three `engine` calls are recorded on the port's frame stream (the reference
 *    launches on the legacy default stream); the fallback kernels too.
 *************************************************************************************************/
namespace uipc::backend::luisa
{
// Must match REGISTER_CONSTITUTION_UIDS EmptyUID in src/constitution/empty.cpp
// (verbatim from the reference).
constexpr ::uipc::U64 kEmptyConstitutionUID = 0ull;

namespace
{
using TripletDeviceView = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using DenseDeviceView   = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
using CDenseDeviceView  = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
using DiagInvDeviceView = fem_mas_preconditioner_detail::DiagInvDeviceView;
using UnpartDeviceView  = fem_mas_preconditioner_detail::UnpartDeviceView;

/// `assemble_diag_inv_for_unpartitioned_kernel`.
auto& mas_unpart_assemble_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,  // A rows
                                                lcrt::Buffer<std::int32_t>,  // A cols
                                                lcrt::Buffer<double>,        // A values
                                                lcrt::Buffer<LMatrix3x3>,    // diag_inv
                                                lcrt::Buffer<std::int32_t>,  // unpartitioned flags
                                                std::int32_t,                // total block rows
                                                std::int32_t,                // total block cols
                                                std::int32_t,                // fem block offset
                                                std::int32_t,                // fem block count
                                                uint32_t>(                   // triplet count
        [](lcrt::BufferVar<std::int32_t> rows,
           lcrt::BufferVar<std::int32_t> cols,
           lcrt::BufferVar<double>       vals,
           lcrt::BufferVar<LMatrix3x3>   diag_inv,
           lcrt::BufferVar<std::int32_t> unpart,
           lcrt::Var<std::int32_t>       total_block_rows,
           lcrt::Var<std::int32_t>       total_block_cols,
           lcrt::Var<std::int32_t>       fem_block_offset,
           lcrt::Var<std::int32_t>       fem_block_count,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto I = lcrt::dispatch_x();
            $if(I < n)
            {
                auto A = TripletDeviceView::from_handles(
                    rows, cols, vals, total_block_rows, total_block_cols, n);
                auto U = UnpartDeviceView::from_handles(unpart, fem_block_count);

                auto row = fem_mas_preconditioner_detail::unpart_diag_inv_row(
                    A, U, lcrt::cast<std::int32_t>(I), fem_block_offset, fem_block_count);

                $if(row.in_window)
                {
                    $if(scalar_detail::pred_and(
                           row.is_unpart,
                           fem_mas_preconditioner_detail::unpart_flag(U, row.index)))
                    {
                        lcrt::Var<LMatrix3x3> out;
                        store_mat3x3(out, row.value);
                        diag_inv->write(row.index, out);
                    };
                };
            };
        });
    return k;
}

/// `apply_diag_inv_for_unpartitioned_kernel`.
auto& mas_unpart_apply_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>,       // r (this subsystem's window)
                                                lcrt::Buffer<double>,       // z (this subsystem's window)
                                                lcrt::Buffer<std::int32_t>,  // converged
                                                lcrt::Buffer<LMatrix3x3>,   // diag_inv
                                                lcrt::Buffer<std::int32_t>,  // unpartitioned flags
                                                uint32_t>(                   // block count
        [](lcrt::BufferVar<double>       r,
           lcrt::BufferVar<double>       z,
           lcrt::BufferVar<std::int32_t> converged,
           lcrt::BufferVar<LMatrix3x3>   diag_inv,
           lcrt::BufferVar<std::int32_t> unpart,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                // the reference's `if(*converged != 0) return;` -- one scalar read for
                // the whole dispatch, so it wraps the body
                $if(converged->read(0) == 0)
                {
                    auto Z = DenseDeviceView::from_handles(z, n * 3u);
                    auto R = CDenseDeviceView::from_handles(r, n * 3u);
                    auto D = DiagInvDeviceView::from_handles(diag_inv, n);
                    auto U = UnpartDeviceView::from_handles(unpart, n);
                    $if(fem_mas_preconditioner_detail::unpart_flag(
                           U, lcrt::cast<std::int32_t>(i)))
                    {
                        fem_mas_preconditioner_detail::unpart_diag_apply_row(
                            Z, R, D, U, lcrt::cast<std::int32_t>(i));
                    };
                };
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FEMMASPreconditioner);

void FEMMASPreconditioner::do_build(BuildInfo& info)
{
    finite_element_method = &require<FiniteElementMethod>();
    global_linear_system  = &require<GlobalLinearSystem>();
    fem_linear_subsystem  = &require<FEMLinearSubsystem>();
    // the reference's fourth dependency edge: `require`d for the build order (the
    // preconditioner reads the method's buffers), never dereferenced.
    (void)require<GlobalVertexManager>();

    auto precond = world().scene().config().find<std::string>("linear_system/fem_preconditioner");
    if(!precond || precond->view()[0] != "mas")
    {
        throw SimSystemException(
            "FEMMASPreconditioner: linear_system/fem_preconditioner != \"mas\", disabled.");
    }

    info.connect(fem_linear_subsystem);
}

void FEMMASPreconditioner::do_init(InitInfo& info)
{
    (void)info;
    auto& fem = finite_element_method->m_impl;

    const SizeT vert_num = fem.xs.size();
    if(vert_num == 0)
        return;

    // ---- 1. Build the vertex adjacency from the element connectivity ----
    std::vector<std::set<unsigned int>> vert_neighbors(vert_num);

    auto add_edge = [&](IndexT a, IndexT b)
    {
        if(a != b && a >= 0 && b >= 0 && a < static_cast<IndexT>(vert_num)
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

    // ---- 2. Build the CSR neighbor arrays ----
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

    // ---- 3. Partition every FEM geometry and build the partition mappings ----
    //
    // MAS is all-or-nothing: every non-Empty FEM SimplicialComplex is covered.  A
    // pre-existing mesh_part attribute (custom C++ partitioning) is respected as-is;
    // everything else is partitioned internally on a *private clone* with the fixed
    // cluster size, so the user's scene geometry is never mutated.
    std::vector<IndexT> part_ids(vert_num, -1);
    bool                has_parts = false;

    // Partition ids are mesh-local (each mesh starts at 0); a running offset makes
    // them globally unique across all geometries in the FEM system.
    IndexT partition_offset = 0;

    auto geo_slots = world().scene().geometries();
    for(auto& geo_info : fem.geo_infos)
    {
        auto& geo_slot = geo_slots[geo_info.geo_slot_index];
        auto& geo      = geo_slot->geometry();
        auto* sc       = geo.as<geometry::SimplicialComplex>();
        if(!sc || sc->dim() < 1 || geo_info.vertex_count == 0)
            continue;

        // Empty constitution (no material): skip, it stays on the diagonal fallback
        auto cuid = geo.meta().find<U64>(builtin::constitution_uid);
        if(!cuid || cuid->view()[0] == kEmptyConstitutionUID)
            continue;

        std::vector<IndexT> local_parts;
        if(auto mesh_part = sc->vertices().find<IndexT>("mesh_part"))
        {
            auto pv = mesh_part->view();
            local_parts.assign(pv.begin(), pv.end());
        }
        else
        {
            auto tmp_geo = std::static_pointer_cast<geometry::Geometry>(geo.clone());
            auto* tmp_sc = tmp_geo->as<geometry::SimplicialComplex>();
            geometry::mesh_partition(*tmp_sc, BANKSIZE);
            auto pv = tmp_sc->vertices().find<IndexT>("mesh_part")->view();
            local_parts.assign(pv.begin(), pv.end());
        }

        has_parts        = true;
        IndexT local_max = 0;
        for(SizeT v = 0; v < geo_info.vertex_count; v++)
        {
            IndexT local_pid = local_parts[v];
            part_ids[geo_info.vertex_offset + v] = local_pid + partition_offset;
            local_max                            = std::max(local_max, local_pid);
        }
        partition_offset += local_max + 1;  // the next mesh starts after this mesh's max
    }

    if(!has_parts)
    {
        m_has_partition = false;
        return;
    }

    // Tag the Empty (non-cloth) FEM vertices; they must not carry mesh_part when
    // using MAS.
    std::vector<uint8_t> h_non_cloth_fem(vert_num, 0);
    for(auto& geo_info : fem.geo_infos)
    {
        auto& geo_slot = geo_slots[geo_info.geo_slot_index];
        auto& geo      = geo_slot->geometry();
        auto  cuid     = geo.meta().find<U64>(builtin::constitution_uid);
        if(!cuid || cuid->view()[0] != kEmptyConstitutionUID)
            continue;
        for(SizeT v = 0; v < geo_info.vertex_count; ++v)
            h_non_cloth_fem[geo_info.vertex_offset + v] = 1;
    }
    for(SizeT v = 0; v < vert_num; ++v)
    {
        UIPC_ASSERT(!(h_non_cloth_fem[v] != 0 && part_ids[v] >= 0),
                    "MAS: Empty FEM vertex {} has mesh_part / partition id {}. "
                    "Strip mesh_part on Empty geometries.",
                    v,
                    part_ids[v]);
    }

    // The unpartitioned vertex flags (the diagonal fallback's input)
    {
        std::vector<IndexT> h_unpart_flags(vert_num, 0);
        bool                any_unpartitioned = false;
        for(SizeT i = 0; i < vert_num; i++)
        {
            if(part_ids[i] < 0)
            {
                h_unpart_flags[i] = 1;
                any_unpartitioned = true;
            }
        }
        m_has_unpartitioned = any_unpartitioned;
        unpartitioned_flags.resize(vert_num);
        if(vert_num != 0)
            unpartitioned_flags.view().copy_from(span<const IndexT>{h_unpart_flags});
    }

    // ---- 4. Build the partition-ordered index mappings ----
    IndexT max_part_id = 0;
    for(auto pid : part_ids)
        if(pid > max_part_id)
            max_part_id = pid;

    std::vector<std::vector<int>> part_blocks(max_part_id + 1);
    for(SizeT i = 0; i < vert_num; i++)
        if(part_ids[i] >= 0)
            part_blocks[part_ids[i]].push_back(static_cast<int>(i));

    // No partition block may exceed BANKSIZE (a collision of partition ids from
    // different meshes would show up here).
    for(SizeT b = 0; b < part_blocks.size(); b++)
    {
        UIPC_ASSERT(static_cast<int>(part_blocks[b].size()) <= BANKSIZE,
                    "MAS: partition {} has {} vertices (max {}). "
                    "Partition IDs from different meshes may be colliding -- "
                    "need global offset.",
                    b,
                    part_blocks[b].size(),
                    BANKSIZE);
    }

    // Each partition block is padded to a BANKSIZE boundary
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
            int real_idx                    = block[i];
            h_part_to_real[offset + (int)i] = real_idx;
            h_real_to_part[real_idx]        = offset + static_cast<int>(i);
        }
        int padded = (static_cast<int>(block.size()) + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
        offset += padded;
    }

    // Validate the mappings (out-of-range indices would corrupt the MAS kernels)
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
            UIPC_ASSERT(h_real_to_part[i] >= 0 && h_real_to_part[i] < part_map_size,
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
                         span<const unsigned int>{h_neighbor_list},
                         span<const unsigned int>{h_neighbor_start},
                         span<const unsigned int>{h_neighbor_num},
                         span<const int>{h_part_to_real},
                         span<const int>{h_real_to_part});

    engine.init_matrix();

    m_has_partition = true;
}

void FEMMASPreconditioner::do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info)
{
    if(!m_has_partition || !engine.is_initialized())
        return;

    auto A          = info.A();
    int  dof_offset = static_cast<int>(info.dof_offset());

    // MAS assembly for the partitioned vertices
    engine.set_preconditioner(A.values(), A.row_indices(), A.col_indices(), dof_offset / 3);

    auto dump_mas = world().scene().config().find<IndexT>("extras/debug/dump_mas_matrices");
    if(dump_mas && dump_mas->view()[0] != 0)
    {
        // NOTE: the member `engine` shadows the base class accessor, exactly as in
        // the reference (which spells the base out for the same reason)
        auto& sim_engine = static_cast<SimEngine&>(SimSystem::engine());
        backend::BackendPathTool path_tool{std::string{workspace()}};
        auto out_dir = path_tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
        this->engine.dump_cluster_matrices_debug(out_dir.string(),
                                                 sim_engine.frame(),
                                                 sim_engine.newton_iter());
    }

    // Diagonal fallback assembly for the unpartitioned vertices
    if(m_has_unpartitioned)
    {
        SizeT num_verts        = finite_element_method->xs().size();
        int   fem_block_offset = dof_offset / 3;
        int   fem_block_count  = static_cast<int>(info.dof_count()) / 3;
        // the reference's `diag_inv.fill(Matrix3x3::Identity())`
        Matrix3x3 identity_matrix = Matrix3x3::Identity();

        diag_inv.resize(num_verts);
        diag_inv.fill(to_luisa(identity_matrix));

        auto n = static_cast<uint32_t>(A.triplet_count());
        if(n > 0)
        {
            auto& k = mas_unpart_assemble_kernel();
            frame_stream().record(k(A.row_indices(),
                                    A.col_indices(),
                                    A.values(),
                                    diag_inv.view(),
                                    unpartitioned_flags.view(),
                                    static_cast<std::int32_t>(A.total_rows()),
                                    static_cast<std::int32_t>(A.total_cols()),
                                    fem_block_offset,
                                    fem_block_count,
                                    n)
                                      .dispatch(n));
        }
    }
}

void FEMMASPreconditioner::do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info)
{
    if(!m_has_partition || !engine.is_initialized())
    {
        // MAS is enabled but nothing is partitionable (e.g. every FEM geometry has an
        // Empty constitution): the identity fallback, so this subsystem's z is never
        // left stale (the reference's `BufferLaunch(stream).copy(z, r)`)
        info.z().buffer_view().copy_from(info.r().buffer_view(), info.stream());
        return;
    }

    auto converged = info.converged();

    // MAS for the partitioned vertices
    engine.apply(info.r(), info.z(), converged, info.stream());

    // Diagonal fallback for the unpartitioned vertices
    if(m_has_unpartitioned)
    {
        auto n = static_cast<uint32_t>(diag_inv.size());
        if(n > 0)
        {
            auto& k = mas_unpart_apply_kernel();
            frame_stream().record(k(info.r().buffer_view(),
                                    info.z().buffer_view(),
                                    converged,
                                    diag_inv.cview(),
                                    unpartitioned_flags.cview(),
                                    n)
                                      .dispatch(n));
        }
    }
}
}  // namespace uipc::backend::luisa

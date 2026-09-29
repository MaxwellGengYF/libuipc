#pragma once
/********************************************************************
 * @file mas_preconditioner_engine.h
 * @brief T6.4c — the MAS (MultiLevel Additive Schwarz) preconditioner engine
 *        (port of src/backends/cuda/finite_element/mas_preconditioner_engine.{h,cu},
 *        205 + 1726 lines).
 *
 * WHAT IT IS
 * ----------
 * The multi-level domain-decomposition preconditioner of the StiffGIPC paper: it
 * partitions the DoF graph into banks of BANKSIZE = 16 nodes, builds a hierarchy of
 * ever coarser "banks of banks", assembles a 48x48 (16 x 3) block system per bank at
 * every level (the Galerkin restriction of the fine Hessian), inverts each of them
 * with a 48x48 Gauss-Jordan and uses them as an additive-Schwarz smoother with
 * injection prolongation: `z = M^{-1} r`.
 *
 * The reference works on 3x3 *block* matrices (the BCOO assembly of the global
 * linear system), so the engine is driven by `set_preconditioner()` once per Newton
 * iteration (assemble + invert) and by `apply()` once per PCG iteration.
 *
 * PORT LAYOUT (deviations from the reference, all forced by the DSL)
 * -----------------------------------------------------------------
 *  * `ClusterMatrixSymT<Scalar>` (`Eigen::Matrix<Scalar,3,3> M[136]`, 8640 B of
 *    double / 4320 B of float) is *not* an LC buffer element type: a C-array member
 *    is not reflectable, and a custom-struct element has no `fetch_add`.  The port
 *    stores the same bytes as one **flat scalar buffer** (one scalar per entry,
 *    column-major inside each 3x3 block, `sym_index(br,bc) * 9 + 3*j + i`), which is
 *    exactly the reference's memory layout *and* makes the 9 scalar atomics of
 *    `cuda_tool::eigen::atomic_add` expressible (`DenseVectorView::atomic_add`).
 *  * `Int2`/`LevelTable` device elements are flattened likewise: `level_sizes` holds
 *    2 ints per level (`x`, then `y`) and `coarse_tables` holds MAX_LEVELS ints per
 *    real vertex (the reference's `LevelTable::index[]`; the layout is identical, so
 *    the entries the reference leaves uninitialised are *zero* here — a
 *    `LDeviceBuffer::resize` value-initialises the tail — and they are never read).
 *  * `Eigen::Vector3f`/`float3` multi-level buffers are flat `float` buffers with 3
 *    entries per node (`VEC3_PER_NODE`).
 *  * the reference's `if(tid >= n) return;` guards become `$if`-guarded bodies with a
 *    clamped index (`gid = min(gid, n - 1)`), i.e. the port dispatches whole blocks
 *    and predicates instead of returning early, because the kernels use
 *    warp/shared-memory collectives (`$shared` + `warp_read_lane`) and a divergent
 *    *return* would leave the collectives half-populated (cookbook §7.10 RULE M2).
 *    Stores/atomics are gated on the unclamped index, so the extra lanes are inert.
 *  * the reference's level-0/level-x kernels exchange masks through a per-block
 *    `__shared__` array written by one lane and read by another **without
 *    `__syncwarp`** (its own warp-synchronous assumption); the port transcribes that
 *    assumption verbatim (`$shared` + the same indices).  The unit test pins the
 *    *result* (the connect masks, the level table and the cluster inverses) on small
 *    meshes, which is what catches it if it ever stops holding.
 *
 * TEST SEAMS (documented deviations, the §7.20 rule 9 precedent)
 * -------------------------------------------------------------
 * The reference's state is private and its only reader is its own `.cu`.  The port
 * keeps the *whole* state and the hierarchy step list **public** so the unit test can
 * drive the real engine step by step and read the intermediate structures back
 * (`level_sizes`, `going_next`, `coarse_tables`, `cluster_hessians`,
 * `cluster_inverses`) — no behaviour change, and the `mas_preconditioner_engine.cpp`
 * kernels are unchanged.
 *********************************************************************/
#include <type_define.h>
#include <luisa_device/linear/linear.h>
#include <uipc/common/span.h>
#include <filesystem>
#include <string_view>

namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

class MASPreconditionerEngine
{
  public:
    static constexpr int BANKSIZE          = 16;
    static constexpr int DEFAULT_BLOCKSIZE = 256;
    static constexpr int DEFAULT_WARPNUM   = 16;
    static constexpr int MAX_LEVELS        = 6;

    /// Symmetric upper-triangle block count: BANKSIZE*(BANKSIZE+1)/2
    static constexpr int SYM_BLOCK_COUNT = BANKSIZE * (BANKSIZE + 1) / 2;  // 136
    /// One cluster matrix is BANKSIZE*3 = 48 wide.
    static constexpr int CLUSTER_MAT_DIM = BANKSIZE * 3;  // 48
    /// Scalars per cluster matrix in the port's flat layout.
    static constexpr int CLUSTER_ENTRIES = SYM_BLOCK_COUNT * 9;  // 1224
    /// The multi-level R/Z buffers are float, 3 per node.
    static constexpr int VEC3_PER_NODE = 3;

    /// The reference's host-side `Int2` (a *device* element there; flattened here).
    struct Int2
    {
        int x = 0;
        int y = 0;
    };

    MASPreconditionerEngine()  = default;
    ~MASPreconditionerEngine() = default;

    // ---- Phase 1: initialize neighbor structures (called once) ----

    void init_neighbor(int                            vert_num,
                       int                            total_neighbor_num,
                       int                            part_map_size,
                       span<const unsigned int>       h_neighbor_list,
                       span<const unsigned int>       h_neighbor_start,
                       span<const unsigned int>       h_neighbor_num,
                       span<const int>                h_part_to_real,
                       span<const int>                h_real_to_part);

    // ---- Phase 1b: allocate matrix-level buffers (called once) ----

    void init_matrix();

    // ---- Phase 2: assemble preconditioner (per Newton iteration) ----

    /// `cuda_tool::CBufferView<Eigen::Matrix3d> triplet_values` + the two index
    /// views of the *assembled* BCOO matrix, exactly the reference's signature.
    void set_preconditioner(lud::LCBufferView<Float>        triplet_values,
                            lud::LCBufferView<std::int32_t> row_ids,
                            lud::LCBufferView<std::int32_t> col_ids,
                            int                             dof_offset);

    // ---- Phase 3: apply preconditioning z = M^{-1} r (per PCG iteration) ----

    void apply(lla::LCDenseVectorView<Float> r,
               lla::LDenseVectorView<Float>  z,
               lud::LCBufferView<IndexT>     converged,
               lud::LStream*                 stream = nullptr);

    [[nodiscard]] bool is_initialized() const { return m_initialized; }

    /// See the reference: 1 = fine level only (block Jacobi), 0 = full hierarchy.
    void set_active_level_num(int n)
    {
        if(n <= 0 || n > m_level_num)
            m_active_level_num = m_level_num;
        else
            m_active_level_num = n;
    }

    /// No-op kept for API compatibility (the reference's coarse damping is unused
    /// since `collect_final_Z` uses pure injection prolongation).
    void set_coarse_damping(Float /*omega*/) {}

    /// Dump the cluster matrices (.mtx) + the partition metadata (.json) for debug.
    void dump_cluster_matrices_debug(std::string_view output_dir,
                                     SizeT            frame,
                                     SizeT            newton_iter);

    // ===========================================================================
    // The hierarchy/assemble/apply step list (public: the test seam, see the header
    // note).  The reference makes them public because NVCC needs extended __device__
    // lambdas in public methods; the port's kernels live in the `.cpp`, but the list
    // is kept public so the unit test can drive the engine one step at a time.
    // ===========================================================================
    void compute_num_levels(int vert_num);
    int  reorder_realtime();
    void build_connect_mask_L0();
    void prepare_prefix_sum_L0();
    void build_level1();
    void build_connect_mask_Lx(int level);
    void next_level_cluster(int level);
    void prefix_sum_Lx(int level);
    void compute_next_level(int level);
    void aggregation_kernel();

    void scatter_hessian_to_clusters(lud::LCBufferView<Float>        triplet_values,
                                     lud::LCBufferView<std::int32_t> row_ids,
                                     lud::LCBufferView<std::int32_t> col_ids,
                                     int                             dof_offset);
    void invert_cluster_matrices();

    void build_multi_level_R(lla::LCDenseVectorView<Float> R,
                             lud::LCBufferView<IndexT>     converged,
                             lud::LStream*                 stream = nullptr);
    void schwarz_local_solve(lud::LCBufferView<IndexT> converged, lud::LStream* stream = nullptr);
    void collect_final_Z(lla::LDenseVectorView<Float> Z,
                         lud::LCBufferView<IndexT>     converged,
                         lud::LStream*                 stream = nullptr);

    // ---- host-side read helpers (test seams; the reference reads the same two
    //      scalars back with `level_sizes.view(level, 1).copy_to(&m_h_level_size)`) --
    [[nodiscard]] Int2 level_size(int level) const;
    [[nodiscard]] int  level_count() const noexcept { return m_level_num; }
    [[nodiscard]] int  num_cluster_blocks() const noexcept
    {
        return m_total_num_clusters / BANKSIZE;
    }

    // ===========================================================================
    //  State (public test seam; the reference keeps it private).
    // ===========================================================================
    bool m_initialized     = false;
    int  m_total_nodes     = 0;
    int  m_total_map_nodes = 0;
    int  m_level_num       = 0;
    int  m_active_level_num = 0;  // levels used during apply(); 0 = use m_level_num
    int  m_total_num_clusters = 0;
    Int2 m_h_level_size;

    // ---- buffers: hierarchy ----
    lud::LDeviceBuffer<std::int32_t>  level_sizes;  // 2 per level: x = count, y = offset
    lud::LDeviceBuffer<std::int32_t>  coarse_space_tables;  // m_total_nodes * m_level_num
    lud::LDeviceBuffer<std::int32_t>  prefix_original;
    lud::LDeviceBuffer<std::int32_t>  prefix_sum_original;
    lud::LDeviceBuffer<std::int32_t>  going_next;
    lud::LDeviceBuffer<std::int32_t>  dense_level;  // dead in the reference too (resize only)
    lud::LDeviceBuffer<std::int32_t>  coarse_tables;  // m_total_nodes * MAX_LEVELS
    lud::LDeviceBuffer<std::uint32_t> fine_connect_masks;
    lud::LDeviceBuffer<std::uint32_t> next_connect_masks;
    lud::LDeviceBuffer<std::uint32_t> next_prefixes;
    lud::LDeviceBuffer<std::uint32_t> next_prefix_sums;

    // ---- buffers: neighbor graph ----
    int                               m_neighbor_list_size = 0;
    lud::LDeviceBuffer<std::uint32_t> neighbor_lists;
    lud::LDeviceBuffer<std::uint32_t> neighbor_starts;
    lud::LDeviceBuffer<std::uint32_t> neighbor_nums;
    lud::LDeviceBuffer<std::uint32_t> neighbor_lists_init;
    lud::LDeviceBuffer<std::uint32_t> neighbor_nums_init;

    // ---- buffers: partition mappings ----
    lud::LDeviceBuffer<std::int32_t> part_to_real;  // partition index -> real vertex
    lud::LDeviceBuffer<std::int32_t> real_to_part;  // real vertex -> partition index

    // ---- buffers: cluster matrices (flat scalar layouts) ----
    /// The assembled cluster Hessians: double, `num_blocks * CLUSTER_ENTRIES`.
    lud::LDeviceBuffer<Float> cluster_hessians;
    /// The inverted clusters: float, `num_blocks * CLUSTER_ENTRIES` (GIPC parity).
    lud::LDeviceBuffer<float> cluster_inverses;

    // ---- buffers: multi-level residual / solution (float, 3 per node) ----
    lud::LDeviceBuffer<float> multi_level_R;
    lud::LDeviceBuffer<float> multi_level_Z;

    /// The reference's block size for the 48x48 Gauss-Jordan (2 clusters per block).
    static constexpr int INVERT_BLOCK_SIZE = 32 * 3;  // 96
};
}  // namespace uipc::backend::luisa

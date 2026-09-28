#pragma once
/********************************************************************
 * @file global_vertex_manager.h
 * @brief Global vertex manager (port of
 *        src/backends/cuda/global_geometry/global_vertex_manager.h).
 *
 * The public surface is kept identical (class names, info structs, accessors,
 * Impl members) so that cross-system code ports by changing only the namespace /
 * include prefix. Device types map per cookbook §3 / §5:
 *   cuda_tool::DeviceBuffer<T>  -> lud::LDeviceBuffer<T>
 *   cuda_tool::DeviceVar<T>     -> lud::LDeviceVar<T>
 *   cuda_tool::BufferView<T>    -> lud::LBufferView<T>
 *   cuda_tool::CBufferView<T>   -> lud::LCBufferView<T>  (= LBufferView<T>)
 * `Vector3` (an Eigen type) is *not* a legal LC buffer element (see type_define.h),
 * so the vector-valued attributes use the byte-identical device element type
 * LVector3 and the accessors exchange data with `to_luisa()/to_eigen()`.
 *
 * Deliberate deviations from the reference (all reported in the T4.1 report):
 *  - `min_pos`/`max_pos` (DeviceVar<Vector3>) are replaced by the packed scalar
 *    buffer `min_max_components`; LC's device_reduce is scalar-typed, exactly the
 *    T3.1/T3.4 "split a struct buffer into scalar buffers" precedent.
 *  - The three optional cross-system slots (GlobalTrajectoryFilter,
 *    GlobalActiveSetManager, GlobalJointDofManager) are declared but always empty
 *    until those systems are ported (T4.3/T5.x); GlobalJointDofManager is now
 *    resolved (T4.2), so `require_discard_friction()` is a no-op only for the two
 *    systems that do not exist yet; the alternative (inventing the systems) would
 *    diverge from the reference.
 *********************************************************************/
#include <type_define.h>
#include <sim_system.h>
#include <functional>
#include <Eigen/Geometry>
#include <collision_detection/aabb.h>
#include <utils/dump_utils.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
class GlobalTrajectoryFilter;
class GlobalActiveSetManager;
class GlobalJointDofManager;
class VertexReporter;
class GlobalVertexManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class VertexCountInfo
    {
      public:
        void count(SizeT count) noexcept;
        void changeable(bool is_changable) noexcept;

      private:
        friend class GlobalVertexManager;
        friend class VertexReporter;

        SizeT m_count     = 0;
        bool  m_changable = false;
    };

    class VertexAttributeInfo
    {
      public:
        VertexAttributeInfo(Impl* impl, SizeT index, SizeT frame) noexcept;
        SizeT                               frame() const noexcept;
        lud::LBufferView<LVector3>          rest_positions() const noexcept;
        lud::LBufferView<Float>             thicknesses() const noexcept;
        lud::LBufferView<IndexT>            coindices() const noexcept;
        lud::LBufferView<IndexT>            dimensions() const noexcept;
        lud::LBufferView<LVector3>          positions() const noexcept;
        lud::LBufferView<IndexT>            contact_element_ids() const noexcept;
        lud::LBufferView<IndexT>            subscene_element_ids() const noexcept;
        lud::LBufferView<IndexT>            body_ids() const noexcept;
        // vert-wise d_hat
        lud::LBufferView<Float> d_hats() const noexcept;

        /**
         * @breif require discard friction, if this update will ruin the friction computation
         *
         */
        void require_discard_friction() const noexcept;

      private:
        friend class GlobalVertexManager;
        SizeT m_index;
        Impl* m_impl;
        SizeT m_frame;
    };

    class VertexDisplacementInfo
    {
      public:
        VertexDisplacementInfo(Impl* impl, SizeT index) noexcept;
        lud::LBufferView<LVector3>     displacements() const noexcept;
        lud::LCBufferView<IndexT>      coindices() const noexcept;

      private:
        friend class GlobalVertexManager;
        SizeT m_index;
        Impl* m_impl;
    };

    /**
     * @brief A mapping from the global vertex index to the coindices.
     *
     * The values of coindices is dependent on the reporters, which can be:
     * 1) the local index of the vertex
     * 2) or any other information that is needed to be stored.
     */
    lud::LCBufferView<IndexT> coindices() const noexcept;

    /**
     * @brief A mapping from the global vertex index to the body id.
     */
    lud::LCBufferView<IndexT> body_ids() const noexcept;

    /**
     * @brief The d_hat of the vertices.
     *
     * The d_hat is used to compute the penetration depth.
     */
    lud::LCBufferView<Float> d_hats() const noexcept;

    /**
     * @brief The current positions of the vertices.
     */
    lud::LCBufferView<LVector3> positions() const noexcept;

    /**
     * @brief The positions of the vertices at last time step.
     *
     * Used to compute the friction.
     */
    lud::LCBufferView<LVector3> prev_positions() const noexcept;

    /**
     * @brief The rest positions of the vertices.
     *
     * Can be used to retrieve some quantities at the rest state.
     */
    lud::LCBufferView<LVector3> rest_positions() const noexcept;

    /**
     * @brief The safe positions of the vertices in line search.
     *
     * Used as a start point to do the line search.
     */
    lud::LCBufferView<LVector3> safe_positions() const noexcept;

    /**
     * @brief Indicate the contact element id of the vertices.
     */
    lud::LCBufferView<IndexT> contact_element_ids() const noexcept;

    /**
     * @brief Indicate the contact element id of the vertices.
     */
    lud::LCBufferView<IndexT> subscene_element_ids() const noexcept;

    /**
     * @brief The displacements of the vertices (after solving the linear system).
     *
     * The displacements are not scaled by the alpha.
     */
    lud::LCBufferView<LVector3> displacements() const noexcept;

    /**
     * @brief The thicknesses of the vertices.
     *
     * The thicknesses are used to compute the penetration depth.
     */
    lud::LCBufferView<Float> thicknesses() const noexcept;

    /**
     * @brief The dimension of the vertices.
     * - 0: Codim 0D
     * - 1: Codim 1D
     * - 2: Codim 2D
     * - 3: 3D
     */
    lud::LCBufferView<IndexT> dimensions() const noexcept;

    /**
     * @brief the axis align bounding box of the all vertices.
     */
    AABB vertex_bounding_box() const noexcept;

    /**
     * @brief the diagonal length of the rest-position bounding box, computed
     * once at init. Used for scene-relative parameter adaptation (e.g.
     * contact/d_hat_relative, newton/velocity_tol_relative).
     */
    Float scene_diagonal() const noexcept;

  public:
    class Impl
    {
      public:
        Impl() = default;
        void init();
        void update_attributes(SizeT frame);
        void rebuild();

        void record_prev_positions();
        void record_start_point();
        void step_forward(Float alpha);

        void collect_vertex_displacements();

        void setup_ccd(lud::LCBufferView<LVector3> base_positions);
        void restore_ccd();
        void overwrite_positions(lud::LCBufferView<LVector3> src);

        Float compute_axis_max_displacement();
        AABB  compute_vertex_bounding_box();

        template <typename T>
        lud::LBufferView<T> subview(lud::LDeviceBuffer<T>& buffer,
                                    SizeT index) const noexcept;

        bool dump(DumpInfo& info);
        bool try_recover(RecoverInfo& info);
        void apply_recover(RecoverInfo& info);
        void clear_recover(RecoverInfo& info);

        Float default_d_hat = 0.01;
        Float d_hat_relative = 0.0;  // > 0: override default with relative * scene_diagonal (Stiff-GIPC convention)

        lud::LDeviceBuffer<IndexT>   coindices;
        lud::LDeviceBuffer<IndexT>   body_ids;
        lud::LDeviceBuffer<Float>    d_hats;
        lud::LDeviceBuffer<IndexT>   dimensions;
        lud::LDeviceBuffer<LVector3> positions;
        lud::LDeviceBuffer<LVector3> prev_positions;
        lud::LDeviceBuffer<LVector3> rest_positions;
        lud::LDeviceBuffer<LVector3> safe_positions;
        lud::LDeviceBuffer<Float>    thicknesses;
        lud::LDeviceBuffer<IndexT>   contact_element_ids;
        lud::LDeviceBuffer<IndexT>   subscene_element_ids;
        lud::LDeviceBuffer<LVector3> displacements;
        lud::LDeviceBuffer<Float>    displacement_norms;

        lud::LDeviceVar<Float> axis_max_disp;
        // [min.x, min.y, min.z, max.x, max.y, max.z]
        lud::LDeviceBuffer<Float> min_max_components;

        // T4.1: declared for surface parity, always empty until the owning
        // systems are ported (see the file header).
        SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;
        SimSystemSlot<GlobalActiveSetManager> global_active_set_manager;
        SimSystemSlot<GlobalJointDofManager>  global_joint_dof_manager;
        SimSystemSlotCollection<VertexReporter> vertex_reporters;

        OffsetCountCollection<IndexT> reporter_vertex_offsets_counts;

        AABB vertex_bounding_box;

        // diagonal length of the rest-position bounding box, set in init()
        Float scene_diagonal = 0.0;

        BufferDump dump_positions;
        BufferDump dump_prev_positions;
    };

  protected:
    virtual void do_build() override;
    virtual bool do_dump(DumpInfo& info) override;
    virtual bool do_try_recover(RecoverInfo& info) override;
    virtual void do_apply_recover(RecoverInfo& info) override;
    virtual void do_clear_recover(RecoverInfo& info) override;

  private:
    friend class SimEngine;
    friend class MaxTranslationChecker;
    friend class GlobalTrajectoryFilter;
    friend class GlobalActiveSetManager;

    // Initialize the global vertex manager
    // - Create the surface mesh
    // - Setup surface attributes
    void init();

    // Update the surface attributes
    void update_attributes();

    void  rebuild();
    void  record_prev_positions();
    void  collect_vertex_displacements();
    Float compute_axis_max_displacement();

    AABB compute_vertex_bounding_box();
    void step_forward(Float alpha);
    void record_start_point();

    void setup_ccd(lud::LCBufferView<LVector3> base_positions);
    void restore_ccd();
    void overwrite_positions(lud::LCBufferView<LVector3> src);

    friend class VertexReporter;
    void add_reporter(VertexReporter* reporter);

    Impl m_impl;
};
}  // namespace uipc::backend::luisa

#include "details/global_vertex_manager.inl"

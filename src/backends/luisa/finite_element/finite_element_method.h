#pragma once
/********************************************************************
 * @file finite_element_method.h
 * @brief T5.1 — the FEM SimSystem (port of
 * src/backends/cuda/finite_element/finite_element_method.h).
 *
 * Public surface, nested info classes, `do_build()`/`init()` split,
 * `add_reporter()`/`add_constitution()` registration and the `friend` set are
 * identical to the reference, so T5.2–T5.5 (and the coupling systems) port by
 * changing only the namespace/include prefix.
 *
 * Device-type mapping (cookbook §3.5/§7.13):
 *   cuda_tool::DeviceBuffer<T>     -> lud::LDeviceBuffer<T>
 *   cuda_tool::CBufferView<Vector3> -> lud::LCBufferView<LVector3>
 *   Eigen::Vector3 attributes (host data, e.g. h_positions) stay Eigen: only the
 *   *device* buffers change, and the H2D/D2H crossings go through
 *   `to_luisa()/to_eigen()` (type_define.h).
 * The FEM element attributes needed two new LC buffer element types
 * (LVector4i for `tets`, LMatrix3x3 for `Dm3x3_invs`), registered in
 * type_define.h.
 *********************************************************************/
#include <sim_system.h>
#include <uipc/geometry/simplicial_complex.h>
#include <global_geometry/global_vertex_manager.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class FiniteElementVertexReporter;
class FiniteElementSurfaceReporter;
class FiniteElementBodyReporter;

// Systems of the later T5.x tasks that this header only forward-declares (they
// appear in `friend` declarations and in `SimSystemSlot*` members, which do not
// need a complete type). They are ported by T5.3/T5.4/T5.5:
class FiniteElementKinetic;          // T5.3
class FiniteElementElastics;         // T5.3 (ported in T5.1 as an interface+plumbing, see its header)
class FiniteElementConstitution;     // T5.2 (interface ported in T5.1, see its header)
class FiniteElementExtraConstitution;  // T5.5
class FEM3DConstitution;             // T5.2
class Codim2DConstitution;           // T5.2
class Codim1DConstitution;           // T5.2
class Codim0DConstitution;           // T5.2
class FiniteElementDiffParmReporter;             // T5.5
class FiniteElementConstitutionDiffParmReporter;  // T5.5
class FiniteElementExtraConstitutionDiffParmReporter;  // T5.5
class FEM3DConstitutionDiffParmReporter;         // T5.5
class Codim2DConstitutionDiffParmReporter;       // T5.5
class Codim1DConstitutionDiffParmReporter;       // T5.5
class Codim0DConstitutionDiffParmReporter;       // T5.5
class FiniteElementDiffDofReporter;              // T5.5
class FEMExternalForceManager;                   // T5.3
class FiniteElementAnimator;                     // T5.3
class FEMLinearSubsystem;                        // T5.4
class FEMLineSearchReporter;                     // T5.5
class FEMDiagPreconditioner;                     // T5.4
class FEMMASPreconditioner;                      // T5.4
class FEMTimeIntegrator;                         // T5.3
class FEMAdjointMethodReplayer;                  // T5.5
class FiniteElementKineticDiffParmReporter;      // T5.5

class FiniteElementMethod final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class DimUID
    {
      public:
        IndexT dim = -1;
        U64    uid = ~0ull;
    };

    class GeoInfo
    {
      public:
        IndexT geo_slot_index = -1;  // slot index in the scene

        DimUID dim_uid;

        SizeT vertex_offset = ~0ull;
        SizeT vertex_count  = 0ull;

        SizeT primitive_offset = ~0ull;
        SizeT primitive_count  = 0ull;
    };

    class ConstitutionInfo
    {
      public:
        SizeT geo_info_offset = ~0ull;
        SizeT geo_info_count  = 0ull;

        SizeT vertex_offset = ~0ull;
        SizeT vertex_count  = 0ull;

        SizeT primitive_offset = ~0ull;
        SizeT primitive_count  = 0ull;
    };

    class DimInfo
    {
      public:
        SizeT geo_info_offset  = ~0ull;
        SizeT geo_info_count   = 0ull;
        SizeT primitive_offset = ~0ull;
        SizeT primitive_count  = 0ull;
        SizeT vertex_offset    = ~0ull;
        SizeT vertex_count     = 0ull;
    };

    class ForEachInfo
    {
      public:
        SizeT          global_index() const noexcept { return m_global_index; }
        SizeT          local_index() const noexcept { return m_local_index; }
        const GeoInfo& geo_info() const noexcept { return *m_geo_info; }

      private:
        friend class FiniteElementMethod;
        SizeT          m_global_index = 0;
        SizeT          m_local_index  = 0;
        const GeoInfo* m_geo_info     = nullptr;
    };

    class FilteredInfo
    {
      public:
        FilteredInfo(Impl* impl, IndexT dim, SizeT index_in_dim) noexcept
            : m_impl(impl)
            , m_dim(dim)
            , m_index_in_dim(index_in_dim)
        {
        }

        span<const GeoInfo> geo_infos() const noexcept;

        const ConstitutionInfo& constitution_info() const noexcept;

        size_t vertex_count() const noexcept;

        size_t primitive_count() const noexcept;

        /**
         * @brief For each primitive or vertex in the filtered info
         *
         * @code
         *  vector<Float> lambdas(info.vertex_count());
         *
         *  info.for_each(geo_slots,
         *  [](SimplicialComplex& sc)
         *  {
         *      return sc.vertices().find<Float>("lambda").view();
         *  },
         *  [&](const ForEachInfo& I, Float lambda)
         *  {
         *      lambdas[I.global_index()] = lambda;
         *  });
         * @endcode
         */
        template <typename ForEach, typename ViewGetter>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ViewGetter&&                    view_getter,
                      ForEach&&                       for_each_action) const;

        template <typename ForEachGeometry>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ForEachGeometry&&               for_each) const;

      protected:
        friend class FiniteElementMethod;
        Impl*  m_impl         = nullptr;
        SizeT  m_index_in_dim = ~0ull;
        IndexT m_dim          = -1;
    };

    class Impl
    {
      public:
        void init(WorldVisitor& world);
        void _init_dof_info();
        void _classify_base_constitutions();

        void _build_geo_infos(WorldVisitor& world);
        void _build_base_constitution_infos();
        void _build_on_host(WorldVisitor& world);
        void _build_on_device();
        void _download_geometry_to_host();
        void _init_base_constitutions();
        void _init_extra_constitutions();
        void _init_diff_reporters();

        void write_scene(WorldVisitor& world);
        bool dump(DumpInfo& info);
        bool try_recover(RecoverInfo& info);
        void apply_recover(RecoverInfo& info);
        void clear_recover(RecoverInfo& info);

        // Forward Simulation:

        SimSystemSlot<GlobalVertexManager> global_vertex_manager;

        SimSystemSlot<FiniteElementElastics>               elastics;
        SimSystemSlotCollection<FiniteElementConstitution> constitutions;
        SimSystemSlotCollection<FiniteElementExtraConstitution> extra_constitutions;

        // Differentiable Simulation Systems (T5.5; the reference's slots exist
        // already, the reporters are ported later):

        SimSystemSlot<FiniteElementDiffParmReporter> kinetic_diff_parm_reporter;
        SimSystemSlotCollection<FiniteElementConstitutionDiffParmReporter>
            constitution_diff_parm_reporters;
        SimSystemSlotCollection<FiniteElementExtraConstitutionDiffParmReporter>
            extra_constitution_diff_parm_reporters;


        // Core Invariant Data:

        vector<GeoInfo> geo_infos;


        // Related Data:

        std::array<DimInfo, 4> dim_infos;

        unordered_map<U64, SizeT>    codim_0d_uid_to_index;
        vector<Codim0DConstitution*> codim_0d_constitutions;
        vector<ConstitutionInfo>     codim_0d_constitution_infos;

        unordered_map<U64, SizeT>    codim_1d_uid_to_index;
        vector<Codim1DConstitution*> codim_1d_constitutions;
        vector<ConstitutionInfo>     codim_1d_constitution_infos;

        unordered_map<U64, SizeT>    codim_2d_uid_to_index;
        vector<Codim2DConstitution*> codim_2d_constitutions;
        vector<ConstitutionInfo>     codim_2d_constitution_infos;

        unordered_map<U64, SizeT>  fem_3d_uid_to_index;
        vector<FEM3DConstitution*> fem_3d_constitutions;
        vector<ConstitutionInfo>   fem_3d_constitution_infos;

        unordered_map<U64, SizeT> extra_constitution_uid_to_index;


        // Simulation Data:

        vector<IndexT> h_vertex_contact_element_ids;
        vector<IndexT> h_vertex_subscene_contact_element_ids;
        vector<Float>  h_vertex_d_hat;

        vector<IndexT>  h_vertex_is_fixed;
        vector<IndexT>  h_vertex_is_dynamic;
        vector<IndexT>  h_vertex_body_id;
        vector<Vector3> h_gravities;

        vector<Vector3> h_positions;
        vector<Vector3> h_rest_positions;
        vector<Vector3> h_velocities;
        vector<Float>   h_thicknesses;
        vector<IndexT>  h_dimensions;
        vector<Float>   h_masses;

        vector<IndexT>   h_codim_0ds;
        vector<Vector2i> h_codim_1ds;
        vector<Vector3i> h_codim_2ds;
        vector<Vector4i> h_tets;

        vector<IndexT> h_body_self_collision;

        Vector3 default_gravity;
        Float   default_d_hat;


        // Element Attributes:

        lud::LDeviceBuffer<IndexT> codim_0ds;

        lud::LDeviceBuffer<LVector2i> codim_1ds;
        lud::LDeviceBuffer<Float>     rest_lengths;

        lud::LDeviceBuffer<LVector3i> codim_2ds;
        lud::LDeviceBuffer<Float>     rest_areas;

        lud::LDeviceBuffer<LVector4i> tets;
        lud::LDeviceBuffer<Float>     rest_volumes;


        // Vertex Attributes:

        lud::LDeviceBuffer<IndexT>   is_fixed;    // Vertex IsFixed
        lud::LDeviceBuffer<IndexT>   is_dynamic;  // Vertex IsDynamic
        lud::LDeviceBuffer<LVector3> gravities;   // Vertex Gravity

        lud::LDeviceBuffer<LVector3> x_bars;   // Rest Positions
        lud::LDeviceBuffer<LVector3> xs;       // Positions
        lud::LDeviceBuffer<LVector3> dxs;      // Displacements
        lud::LDeviceBuffer<LVector3> x_temps;  // Safe Positions for line search
        lud::LDeviceBuffer<LVector3> vs;       // Velocities
        lud::LDeviceBuffer<LVector3> x_tildes;  // Predicted Positions
        lud::LDeviceBuffer<LVector3> x_prevs;   // Positions at last frame
        lud::LDeviceBuffer<Float>    masses;    // Mass
        lud::LDeviceBuffer<Float>    thicknesses;  // Thickness

        lud::LDeviceBuffer<LVector3> vertex_external_forces;  // per-vertex F_ext
        lud::LDeviceBuffer<LVector3> vertex_external_force_accs;  // per-vertex a_ext = F/m


        //tex:
        // FEM3D Material Basis
        //$$
        // \begin{bmatrix}
        // \bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_0 & \bar{\mathbf{x}}_2 - \bar{\mathbf{x}}_0 & \bar{\mathbf{x}}_3 - \bar{\mathbf{x}}_0 \\
        // \end{bmatrix}^{-1}
        // $$
        lud::LDeviceBuffer<LMatrix3x3> Dm3x3_invs;


        // Dump:

        BufferDump dump_xs;       // Positions
        BufferDump dump_x_prevs;  // Positions at last frame
        BufferDump dump_vs;       // Velocities

        // Dof Info:

        void set_dof_info(SizeT frame, IndexT dof_offset, IndexT dof_count);  // only called by FEMLinearSubsystem
        IndexT dof_offset(SizeT frame) const noexcept;
        IndexT dof_count(SizeT frame) const noexcept;

      private:
        vector<IndexT> frame_to_dof_offset;
        vector<IndexT> frame_to_dof_count;
    };


  public:
    auto codim_0ds() const noexcept { return m_impl.codim_0ds.cview(); }
    auto codim_1ds() const noexcept { return m_impl.codim_1ds.cview(); }
    auto codim_2ds() const noexcept { return m_impl.codim_2ds.cview(); }
    auto tets() const noexcept { return m_impl.tets.cview(); }

    auto is_fixed() const noexcept { return m_impl.is_fixed.cview(); }
    auto is_dynamic() const noexcept { return m_impl.is_dynamic.cview(); }
    auto x_bars() const noexcept { return m_impl.x_bars.cview(); }
    auto xs() const noexcept { return m_impl.xs.cview(); }

    /**
     * @brief overwrite the vertex positions
     */
    void overwrite_xs(lud::LCBufferView<LVector3> xs);
    auto dxs() const noexcept { return m_impl.dxs.cview(); }
    auto x_temps() const noexcept { return m_impl.x_temps.cview(); }
    auto vs() const noexcept { return m_impl.vs.cview(); }
    auto x_tildes() const noexcept { return m_impl.x_tildes.cview(); }
    auto x_prevs() const noexcept { return m_impl.x_prevs.cview(); }
    auto masses() const noexcept { return m_impl.masses.cview(); }
    auto thicknesses() const noexcept { return m_impl.thicknesses.cview(); }
    auto rest_volumes() const noexcept { return m_impl.rest_volumes.cview(); }
    auto rest_areas() const noexcept { return m_impl.rest_areas.cview(); }
    auto rest_lengths() const noexcept { return m_impl.rest_lengths.cview(); }
    auto Dm3x3_invs() const noexcept { return m_impl.Dm3x3_invs.cview(); }
    auto gravities() const noexcept { return m_impl.gravities.cview(); }
    auto vertex_external_forces() const noexcept
    {
        return m_impl.vertex_external_forces.cview();
    }
    auto vertex_external_force_accs() const noexcept
    {
        return m_impl.vertex_external_force_accs.cview();
    }

    /**
     * @brief return the frame-local dof offset of FEM for the given frame
     */
    IndexT dof_offset(SizeT frame) const noexcept;
    /**
     * @brief return the frame-local dof count of FEM for the given frame
     */
    IndexT dof_count(SizeT frame) const noexcept;

    /**
     * @brief For each primitive
     *
     * @code
     *  vector<Float> lambdas(info.vertex_count());
     *
     *  for_each(geo_slots,
     *  [](SimplicialComplex& sc)
     *  {
     *      return sc.vertices().find<Float>("lambda").view();
     *  },
     *  [&](const ForEachInfo& I, Float lambda)
     *  {
     *      auto vI = I.globl_index();
     *      lambdas[vI] = lambda;
     *  });
     * @endcode
     */
    template <typename ForEach, typename ViewGetter>
    void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                  ViewGetter&&                    view_getter,
                  ForEach&&                       for_each_action);

    /**
     * @brief For each geometry
     *
     * @code
     *  for_each(geo_slots,
     *  [](const ForEachInfo& I,SimplicialComplex& sc)
     *  {
     *      auto geoI = I.global_index();
     *      ...
     *  });
     * @endcode
     */
    template <typename ForEachGeometry>
    void for_each(span<S<geometry::GeometrySlot>> geo_slots, ForEachGeometry&& for_each);

  private:
    friend class FiniteElementVertexReporter;
    friend class FiniteElementSurfaceReporter;
    friend class FiniteElementBodyReporter;

    friend class FEMLinearSubsystem;  // T5.4
    friend class FEMLineSearchReporter;  // T5.5


    friend class FiniteElementAnimator;  // T5.3
    friend class FEMDiagPreconditioner;  // T5.4
    friend class FEMMASPreconditioner;  // T5.4

    friend class FiniteElementDiffDofReporter;  // T5.5
    friend class FiniteElementKineticDiffParmReporter;  // T5.5
    friend class FEMAdjointMethodReplayer;  // T5.5
    friend class FEMTimeIntegrator;  // T5.3

    friend class FiniteElementStateAccessorFeatureOverrider;
    friend class FEMExternalForceManager;  // T5.3

    friend class SimEngine;
    void init();  // only be called by SimEngine

    friend class FiniteElementElastics;  // T5.3
    void add_elastics(FiniteElementElastics* elastics);
    friend class FiniteElementConstitution;  // T5.2
    void add_constitution(FiniteElementConstitution* constitution);  // only called by FiniteElementConstitution
    friend class FiniteElementExtraConstitution;  // T5.5
    void add_constitution(FiniteElementExtraConstitution* constitution);  // only called by FiniteElementExtraConstitution


    friend class FiniteElementConstitutionDiffParmReporter;  // T5.5
    void add_reporter(FiniteElementConstitutionDiffParmReporter* reporter);  // only called by FiniteElementConstitutionDiffParmReporter
    friend class FiniteElementExtraConstitutionDiffParmReporter;  // T5.5
    void add_reporter(FiniteElementExtraConstitutionDiffParmReporter* reporter);  // only called by FiniteElementExtraConstitutionDiffParmReporter

    friend class FiniteElementDiffParmReporter;  // T5.5
    void add_kinetic_reporter(FiniteElementDiffParmReporter* reporter);  // only called by FiniteElementKineticDiffParmReporter


    // Internal:

    template <typename ForEach, typename ViewGetter>
    static void _for_each(span<const GeoInfo>             geo_infos,
                          span<S<geometry::GeometrySlot>> geo_slots,
                          ViewGetter&&                    view_getter,
                          ForEach&&                       for_each_action);

    template <typename ForEachGeometry>
    static void _for_each(span<const GeoInfo>             geo_infos,
                          span<S<geometry::GeometrySlot>> geo_slots,
                          ForEachGeometry&&               for_each);

    virtual void do_build() override;

    virtual bool do_dump(DumpInfo& info) override;
    virtual bool do_try_recover(RecoverInfo& info) override;
    virtual void do_apply_recover(RecoverInfo& info) override;
    virtual void do_clear_recover(RecoverInfo& info) override;

    Impl m_impl;
};
}  // namespace uipc::backend::luisa

#include "details/finite_element_method.inl"

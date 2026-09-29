#pragma once
/********************************************************************
 * @file affine_body/affine_body_dynamics.h
 * @brief T6.1a — the ABD SimSystem (port of
 * src/backends/cuda/affine_body/affine_body_dynamics.h).
 *
 * Public surface, nested info classes, the `do_build()`/`init()` split,
 * `add_constitution()`/`add_kinetic()`/`add_reporter()` registration and the
 * `friend` set are identical to the reference, so T6.1b–T6.4 (and the coupling
 * systems) port by changing only the namespace/include prefix (cookbook §7.16
 * rule 9).
 *
 * Device-type mapping (cookbook §3.5/§7.13):
 *   cuda_tool::DeviceBuffer<T>       -> lud::LDeviceBuffer<T>
 *   cuda_tool::CBufferView<Vector12> -> lud::LCBufferView<LVector12>
 *   Eigen-typed *host* data (h_body_id_to_q, ...) stays Eigen: only the device
 *   buffers change element type, and every H2D/D2H crossing goes through
 *   `to_luisa()/to_eigen()` (affine_body/type_define.h).
 * The ABD element types needed three new LC buffer element types
 * (LVector12, LMatrix12x12 and the two aggregates ABDJacobi/ABDJacobiDyadicMass,
 * registered in affine_body/type_define.h).
 *
 * The reference includes <sim_engine.h> here for the SimSystemCreator's
 * `dynamic_cast<SimEngine&>`; the port includes the *qualified* path from the .cpp
 * instead (`<backends/luisa/sim_engine.h>`), because the test target's include
 * path puts src/backends/none first (cookbook §7.15 rule 7).
 *********************************************************************/
#include <sim_system.h>
#include <affine_body/abd_jacobi_matrix.h>
#include <uipc/geometry/simplicial_complex.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
// The reference's forward declarations.  The ones marked (T6.x) are not ported in
// T6.1a: they appear only in `friend` declarations and in `SimSystemSlot*`
// members, which do not need a complete type.
class AffineBodyConstitution;
class AffineBodyDiffParmReporter;           // M9 (diff_sim)
class AffineBodyDiffDofReporter;            // M9 (diff_sim)
class AffineBodyKineticDiffParmReporter;    // M9 (diff_sim)
class AffineBodyBodyReporter;
class AffineBodyKinetic;                    // T6.1a carries the interface, T6.2 the bdf bodies
class ABDLinearSubsystem;                   // T6.1b
class ABDLineSearchReporter;                // T6.4
class ABDDiagPreconditioner;                // T6.4
class AffineBodyAnimator;                   // T6.1b (declaration pulled forward here)
class AffineBodyStateAccessorFeatureOverrider;
class AffineBodyExternalForceManager;
class AffineBodyVertexReporter;
class AffinebodySurfaceReporter;
class InterAffineBodyAnimator;              // T6.1b
class InterAffineBodyConstitutionManager;   // T6.1b
class ABDGradientHessianComputer;           // not ported (unused in the reference's .cu)
class ABDAdjointMethodReplayer;             // M9
class ABDDiffSimManager;                    // M9
class ABDTimeIntegrator;                    // T6.1a carries the interface, T6.2 the bdf bodies

class AffineBodyDynamics : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class GeoInfo
    {
      public:
        IndexT geo_slot_index     = -1;
        IndexT geo_id             = -1;
        U64    constitution_uid   = 0;
        U64    constitution_index = 0;

        SizeT vertex_offset = 0;
        SizeT vertex_count  = 0;

        SizeT body_offset = 0;
        SizeT body_count  = 0;
    };

    class ForEachInfo
    {
      public:
        SizeT          global_index() const noexcept { return m_global_index; }
        SizeT          local_index() const noexcept { return m_local_index; }
        const GeoInfo& geo_info() const noexcept { return *m_geo_info; }

      private:
        friend class AffineBodyDynamics;
        SizeT          m_global_index = 0;
        SizeT          m_local_index  = 0;
        const GeoInfo* m_geo_info     = nullptr;
    };

    class ConstitutionInfo
    {
      public:
        SizeT geo_offset = 0;
        SizeT geo_count  = 0;

        SizeT body_offset = 0;
        SizeT body_count  = 0;

        SizeT vertex_offset = 0;
        SizeT vertex_count  = 0;
    };

    class FilteredInfo
    {
      public:
        FilteredInfo(Impl* impl, SizeT constitution_index) noexcept;

        /**
         * @brief Short-cut to traverse all bodies of the current constitution.
         *
         * @code
         *  for_each(geo_slots,
         *  [](SimplicialComplex& sc)
         *  {
         *      return sc.transforms().view();
         *  },
         *  [](const ForEachInfo& I, const Matrix4x4& transforms)
         *  {
         *      auto bodyI = I.global_index();
         *      ...
         *  })
         * @endcode
         */
        template <typename ViewGetterF, typename ForEachF>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ViewGetterF&&                   getter,
                      ForEachF&&                      for_each) const;

        /**
         * @brief Short-cut to traverse all geometries of the current constitution.
         */
        template <typename ForEachGeometry>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ForEachGeometry&&               for_every_geometry) const;

        span<const GeoInfo> geo_infos() const noexcept;

        const ConstitutionInfo& constitution_info() const noexcept;

        SizeT body_count() const noexcept;

        SizeT vertex_count() const noexcept;

      private:
        friend class Impl;
        SizeT m_constitution_index = ~0ull;
        Impl* m_impl               = nullptr;
    };

  protected:
    virtual void do_build() override;

    virtual bool do_dump(DumpInfo& info) override;
    virtual bool do_try_recover(RecoverInfo& info) override;
    virtual void do_apply_recover(RecoverInfo& info) override;
    virtual void do_clear_recover(RecoverInfo& info) override;

  public:
    class Impl
    {
      public:
        void init(WorldVisitor& world);

        void _build_constitutions(WorldVisitor& world);
        void _build_geo_infos(WorldVisitor& world);

        void _setup_geometry_attributes(WorldVisitor& world);
        void _build_geometry_on_host(WorldVisitor& world);
        void _build_geometry_on_device(WorldVisitor& world);
        void _distribute_geo_infos();
        void _download_geometry_to_host();

        void _init_dof_info();
        void _init_diff_reporters();

        void write_scene(WorldVisitor& world);

        bool dump(DumpInfo& info);
        bool try_recover(RecoverInfo& info);
        void apply_recover(RecoverInfo& info);
        void clear_recover(RecoverInfo& info);

        template <typename ViewGetterF, typename ForEachF>
        static void _for_each(span<S<geometry::GeometrySlot>> geo_slots,
                              span<const GeoInfo>             geo_infos,
                              ViewGetterF&&                   getter,
                              ForEachF&&                      for_each);

        /*
         * @brief Short-cut to traverse all bodies of current constitution.
         */
        template <typename ViewGetterF, typename ForEachF>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ViewGetterF&&                   getter,
                      ForEachF&&                      for_each);

        template <typename ForEachGeometry>
        static void _for_each(span<S<geometry::GeometrySlot>> geo_slots,
                              span<const GeoInfo>             geo_infos,
                              ForEachGeometry&&               for_every_geometry);

        template <typename ForEachGeometry>
        void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                      ForEachGeometry&&               for_every_geometry);

        SizeT geo_count() const noexcept { return abd_geo_count; }
        SizeT body_count() const noexcept { return abd_body_count; }
        SizeT vertex_count() const noexcept { return abd_vertex_count; }

        SimSystemSlotCollection<AffineBodyConstitution> constitutions;
        SimSystemSlot<AffineBodyKinetic> kinetic;  // ABD Kinetic, only one allowed

        SizeT abd_geo_count    = 0;
        SizeT abd_body_count   = 0;
        SizeT abd_vertex_count = 0;

        // core invariant data
        vector<GeoInfo> geo_infos;
        unordered_map<U64, IndexT> geo_id_to_geo_info_index;  // geo_id -> geo_info index
        unordered_map<U64, IndexT> constitution_uid_to_index;
        vector<ConstitutionInfo>   constitution_infos;

        // diff reporters (M9): the reference's slot exists, the reporters are not
        // ported (affine_body_diff_*_reporter needs <diff_sim/...>); the same
        // treatment as FiniteElementMethod's diff slots (T5.1 rule 13).
        SimSystemSlotCollection<AffineBodyDiffParmReporter> diff_parm_reporter;


        /******************************************************************************
        *                        host simulation data
        *******************************************************************************/
        Float             default_d_hat = 0.01;
        vector<ABDJacobi> h_vertex_id_to_J;
        vector<Float>     h_vertex_id_to_d_hat;
        vector<IndexT>    h_vertex_id_to_body_id;
        vector<IndexT>    h_vertex_id_to_contact_element_id;
        vector<IndexT>    h_vertex_id_to_subscene_contact_element_id;

        vector<Vector12> h_body_id_to_q;
        vector<Vector12> h_body_id_to_q_v;
        vector<IndexT>   h_body_id_to_dim;             // 2 or 3
        vector<IndexT>   h_body_id_to_self_collision;  // 0 or 1

        vector<ABDJacobiDyadicMass> h_body_id_to_abd_mass;
        vector<Matrix12x12>         h_body_id_to_abd_mass_inv;
        vector<Float>               h_body_id_to_volume;
        vector<Vector12>            h_body_id_to_abd_gravity;
        vector<IndexT>              h_body_id_to_is_fixed;
        vector<IndexT>              h_body_id_to_is_dynamic;
        vector<IndexT>              h_body_id_to_external_kinetic;

        /******************************************************************************
        *                        abd vertex attributes
        *******************************************************************************/
        ///tex: $$ \mathbf{J}(\bar{\mathbf{x}}) $$
        lud::LDeviceBuffer<ABDJacobi> vertex_id_to_J;
        lud::LDeviceBuffer<IndexT>    vertex_id_to_body_id;


        /******************************************************************************
        *                           abd body attributes
        *******************************************************************************/
        // dim = 2 or 3, 2 for affine body from trimesh, 3 for affine body from tetmesh
        lud::LDeviceBuffer<IndexT> body_id_to_dim;

        ///tex: $$ \mathbf{M}_i = \sum_{j \in \mathcal{B}_i} \mathbf{m}_j $$
        lud::LDeviceBuffer<ABDJacobiDyadicMass> body_id_to_abd_mass;
        ///tex: $$ \mathbf{M}_i^{-1} $$
        lud::LDeviceBuffer<LMatrix12x12> body_id_to_abd_mass_inv;

        ///tex: $$ V_{\perp}(m_q) = \kappa v_b \|AA^T - I_3\|_F^2 $$
        lud::LDeviceBuffer<Float> body_id_to_volume;

        ///tex: $$ \mathbf{m_q} = [\mathbf{p}; \mathbf{a}_1; \mathbf{a}_2; \mathbf{a}_3] $$
        lud::LDeviceBuffer<LVector12> body_id_to_q;

        ///tex: temp $$ \mathbf{q} $$ for line search rollback
        lud::LDeviceBuffer<LVector12> body_id_to_q_temp;

        ///tex: $$ \tilde{\mathbf{q}}_b = \mathbf{q}_b^t + \Delta t \dot{\mathbf{q}}_b^t
        ///      + \Delta t^2 \mathbf{M}^{-1} \mathbf{f}_b^{t+1} $$ (predicted q)
        lud::LDeviceBuffer<LVector12> body_id_to_q_tilde;
        ///tex: $$ \mathbf{q}^t_b $$
        lud::LDeviceBuffer<LVector12> body_id_to_q_prev;

        ///tex: $$ \dot{\mathbf{q}} $$
        lud::LDeviceBuffer<LVector12> body_id_to_q_v;

        ///tex: $$ \Delta\mathbf{q} $$ (the solver's unknown)
        lud::LDeviceBuffer<LVector12> body_id_to_dq;

        ///tex: $$ \mathbf{F}_i = \sum_{j \in \mathcal{B}_i} \mathbf{f}_j $$
        lud::LDeviceBuffer<LVector12> body_id_to_abd_force;

        lud::LDeviceBuffer<LVector12> body_id_to_abd_gravity;

        ///tex: $$ \mathbf{F}_{ext,i} $$ per body, set by constraints
        lud::LDeviceBuffer<LVector12> body_id_to_external_force;

        ///tex: $$ \mathbf{a}_{ext,i} = \mathbf{M}^{-1}\mathbf{F}_{ext,i} $$, by the reporter
        lud::LDeviceBuffer<LVector12> body_id_to_external_force_acc;

        lud::LDeviceBuffer<IndexT> body_id_to_is_fixed;          // Body IsFixed
        lud::LDeviceBuffer<IndexT> body_id_to_is_dynamic;        // Body IsKinematic
        lud::LDeviceBuffer<IndexT> body_id_to_external_kinetic;  // Body IsExternalKinetic

        template <typename T>
        lud::LBufferView<T> subview(lud::LDeviceBuffer<T>& body_id_to_values,
                                    SizeT                   constitution_index) const noexcept;

        template <typename T>
        span<T> subview(vector<T>& body_id_to_values, SizeT constitution_index) const noexcept;

        // Dump:

        BufferDump dump_q;
        BufferDump dump_q_v;
        BufferDump dump_q_prev;

        // Dof Info:
        void set_dof_info(SizeT frame, IndexT dof_offset, IndexT dof_count);

        /**
         * @brief dof offset in the global linear system at the given frame.
         */
        IndexT dof_offset(SizeT frame) const;

        /**
         * @brief dof count in the global linear system at the given frame.
         */
        IndexT dof_count(SizeT frame) const;

      private:
        vector<IndexT> frame_to_dof_offset;
        vector<IndexT> frame_to_dof_count;
    };

  public:
    /**
     * @brief affine body local vertex id -> ABD Jacobi matrix
     */
    auto Js() const noexcept { return m_impl.vertex_id_to_J.cview(); }

    /**
     * @brief affine body local vertex id -> body id
     */
    auto v2b() const noexcept { return m_impl.vertex_id_to_body_id.cview(); }

    /**
     * @brief return the dof `q` of the body
     */
    auto qs() const noexcept { return m_impl.body_id_to_q.cview(); }

    /**
     * @brief overwrite the dof `q` of the body
     */
    void overwrite_qs(lud::LCBufferView<LVector12> qs);

    /**
     * @brief return the delta dof `dq` of the body
     */
    auto dqs() const noexcept { return m_impl.body_id_to_dq.cview(); }

    /**
     * @brief return the predicted dof `q_tilde` of the body
     */
    auto q_tildes() const noexcept { return m_impl.body_id_to_q_tilde.cview(); }

    /**
     * @brief return the previous dof `q_prev` of the body
     */
    auto q_prevs() const noexcept { return m_impl.body_id_to_q_prev.cview(); }

    /**
     * @brief return the velocity `q_v` of the body
     */
    auto q_vs() const noexcept { return m_impl.body_id_to_q_v.cview(); }

    /**
     * @brief return the volume of the body
     */
    auto body_volumes() const noexcept { return m_impl.body_id_to_volume.cview(); }

    /**
     * @brief return the mass matrix of the body
     */
    auto body_masses() const noexcept { return m_impl.body_id_to_abd_mass.cview(); }

    /**
     * @brief return the inverse mass matrix of the body
     */
    auto body_mass_invs() const noexcept { return m_impl.body_id_to_abd_mass_inv.cview(); }

    /**
     * @brief return the gravity acceleration applied on the body
     */
    auto body_gravities() const noexcept { return m_impl.body_id_to_abd_gravity.cview(); }

    /**
     * @brief return the external force applied on the body
     */
    auto body_external_forces() const noexcept
    {
        return m_impl.body_id_to_external_force.cview();
    }

    /**
     * @brief return the external acceleration applied on the body
     */
    auto body_external_force_accs() const noexcept
    {
        return m_impl.body_id_to_external_force_acc.cview();
    }

    /**
     * @brief return whether the body is fixed
     */
    auto body_is_fixed() const noexcept { return m_impl.body_id_to_is_fixed.cview(); }

    /**
     * @brief return whether the body is dynamic, consider velocity
     */
    auto body_is_dynamic() const noexcept { return m_impl.body_id_to_is_dynamic.cview(); }

    /**
     * @brief return whether the body is controlled by external kinetic data
     */
    auto body_external_kinetic() const noexcept
    {
        return m_impl.body_id_to_external_kinetic.cview();
    }

    /**
     * @brief return the body dimension of ABD at the given body.
     *
     * - dim = 2: affine body represented by trimesh (still 3D entity)
     * - dim = 3: affine body represented by tetmesh
     */
    auto body_dim() const noexcept { return m_impl.body_id_to_dim.cview(); }

    /**
     * @brief return the frame-local dof offset of ABD at the given frame.
     */
    IndexT dof_offset(SizeT frame) const;

    /**
     * @brief return the frame-local dof count of ABD at the given frame.
     */
    IndexT dof_count(SizeT frame) const;

    /*
     * @brief Short-cut to traverse all bodies of current constitution.
     */
    template <typename ViewGetterF, typename ForEachF>
    void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                  ViewGetterF&&                   getter,
                  ForEachF&&                      for_each);
    template <typename ForEachGeometry>
    void for_each(span<S<geometry::GeometrySlot>> geo_slots,
                  ForEachGeometry&&               for_every_geometry);

  private:
    friend class AffineBodyConstitution;
    void add_constitution(AffineBodyConstitution* constitution);  // only by AffineBodyConstitution

    friend class AffineBodyKinetic;
    void add_kinetic(AffineBodyKinetic* kinetic);  // only by AffineBodyKinetic

    friend class InterAffineBodyAnimator;             // T6.1b
    friend class AffineBodyVertexReporter;
    friend class AffinebodySurfaceReporter;
    friend class AffineBodyBodyReporter;
    friend class InterAffineBodyConstitutionManager;  // T6.1b

    friend class ABDLinearSubsystem;      // T6.1b
    friend class ABDLineSearchReporter;   // T6.4
    friend class ABDDiagPreconditioner;   // T6.4

    friend class AffineBodyConstitution;
    friend class ABDGradientHessianComputer;  // not ported (see the class list above)
    friend class AffineBodyAnimator;          // T6.1b

    friend class AffineBodyDiffParmReporter;  // M9
    friend class AffineBodyDiffDofReporter;   // M9
    friend class ABDAdjointMethodReplayer;    // M9
    friend class ABDDiffSimManager;           // M9

    friend class AffineBodyKineticDiffParmReporter;  // M9
    friend class ABDTimeIntegrator;
    friend class AffineBodyStateAccessorFeatureOverrider;
    friend class AffineBodyExternalForceManager;

    void add_reporter(AffineBodyKineticDiffParmReporter* reporter);

    friend class SimEngine;
    void init();  // only be called by SimEngine

    Impl m_impl;
};
}  // namespace uipc::backend::luisa

#include "details/affine_body_dynamics.inl"

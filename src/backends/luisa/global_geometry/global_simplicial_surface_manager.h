#pragma once
/********************************************************************
 * @file global_simplicial_surface_manager.h
 * @brief Global simplicial surface manager (port of
 *        src/backends/cuda/global_geometry/global_simplicial_surface_manager.h).
 *
 * Public surface identical. The one non-mechanical kernel-side change is
 * `_collect_codim_vertices()`: the reference partitions with
 * `DeviceSelect().Flagged(surf_vertices, flags, codim_vertices, count, n)`, which
 * no longer exists in LC, so the port uses T3.2's `select_if` (its
 * `keep_when_zero == false` flavour is exactly a flagged partition) read back
 * through the standard mid-frame read-back. The device element types are
 * LVector2i/LVector3i (byte-identical to Vector2i/Vector3i, see type_define.h).
 *********************************************************************/
#include <type_define.h>
#include <sim_system.h>
#include <global_geometry/global_vertex_manager.h>

namespace uipc::backend::luisa
{
class SimplicialSurfaceReporter;
class GlobalSimplicialSurfaceManager : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class ReporterInfo
    {
      public:
        SizeT surf_vertex_offset   = 0;
        SizeT surf_vertex_count    = 0;
        SizeT surf_edge_offset     = 0;
        SizeT surf_edge_count      = 0;
        SizeT surf_triangle_offset = 0;
        SizeT surf_triangle_count  = 0;
    };

    class SurfaceCountInfo
    {
      public:
        void surf_vertex_count(SizeT count) noexcept;
        void surf_edge_count(SizeT count) noexcept;
        void surf_triangle_count(SizeT count) noexcept;
        void changable(bool value) noexcept;

      private:
        friend class Impl;
        SizeT m_surf_vertex_count   = 0;
        SizeT m_surf_edge_count     = 0;
        SizeT m_surf_triangle_count = 0;
        bool  m_changable           = false;
    };

    class SurfaceAttributeInfo
    {
      public:
        SurfaceAttributeInfo(Impl* impl, SizeT index)
            : m_impl(impl)
            , m_index(index)
        {
        }

        lud::LBufferView<IndexT>    surf_vertices() noexcept;
        lud::LBufferView<LVector2i> surf_edges() noexcept;
        lud::LBufferView<LVector3i> surf_triangles() noexcept;
        const ReporterInfo&         reporter_info() const noexcept;

      private:
        friend class Impl;
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
    };

    class SurfaceInitInfo
    {
      public:
    };


    class Impl
    {
      public:
        void init();
        void _collect_codim_vertices();

        // core invariant data
        vector<ReporterInfo> reporter_infos;
        // related data
        SizeT total_surf_vertex_count   = 0;
        SizeT total_surf_edge_count     = 0;
        SizeT total_surf_triangle_count = 0;

        lud::LDeviceBuffer<IndexT>    codim_vertices;
        lud::LDeviceBuffer<IndexT>    surf_vertices;
        lud::LDeviceBuffer<IndexT>    codim_vertex_flags;
        lud::LDeviceBuffer<LVector2i> surf_edges;
        lud::LDeviceBuffer<LVector3i> surf_triangles;

        GlobalVertexManager*                                  global_vertex_manager = nullptr;
        SimSystemSlotCollection<SimplicialSurfaceReporter>    reporters;
        // T4.1: uint32 rather than int -- T3.2's select_if writes its count as
        // uint32_t; the value is identical for any n < 2^31 (see the port note above).
        lud::LDeviceVar<uint32_t> selected_codim_0d_count;
    };

    lud::LCBufferView<IndexT>    codim_vertices() const noexcept;
    lud::LCBufferView<IndexT>    surf_vertices() const noexcept;
    lud::LCBufferView<LVector2i> surf_edges() const noexcept;
    lud::LCBufferView<LVector3i> surf_triangles() const noexcept;

  protected:
    virtual void do_build() override;

  private:
    friend class SimEngine;
    Impl m_impl;
    void init();     // only called by SimEngine
    void rebuild();  // only called by SimEngine

    friend class SimplicialSurfaceReporter;
    void add_reporter(SimplicialSurfaceReporter* reporter) noexcept;  // only called by SimplicialSurfaceReporter
};
}  // namespace uipc::backend::luisa

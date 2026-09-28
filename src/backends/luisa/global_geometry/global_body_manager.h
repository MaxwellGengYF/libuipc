#pragma once
/********************************************************************
 * @file global_body_manager.h
 * @brief Global body manager (port of
 *        src/backends/cuda/global_geometry/global_body_manager.h).
 *
 * Public surface identical; device types map to the luisa_device layer (see
 * global_vertex_manager.h for the element-type note).
 *********************************************************************/
#include <type_define.h>
#include <sim_system.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::luisa
{
class BodyReporter;
class GlobalBodyManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class BodyCountInfo
    {
      public:
        void count(SizeT count) noexcept;
        void changeable(bool is_changable) noexcept;

      private:
        friend class GlobalBodyManager;
        friend class BodyReporter;
        SizeT m_count     = 0;
        bool  m_changable = false;
    };

    class BodyAttributeInfo
    {
      public:
        lud::LBufferView<IndexT> coindices() const noexcept;
        lud::LBufferView<IndexT> self_collision() const noexcept;

      private:
        BodyAttributeInfo(Impl* impl, SizeT index) noexcept;
        friend class GlobalBodyManager;
        SizeT m_index;
        Impl* m_impl;
    };

    /**
     * @brief A mapping from the global body index to the coindices.
     *
     * The values of coindices is dependent on the reporters, which can be:
     * 1) the local index of the body
     * 2) or any other information that is needed to be stored.
     */
    lud::LCBufferView<IndexT> coindices() const noexcept;


    /*
     * @breif Tells if the body needs self-collision detection.
     *
     * Indexed by body index, 1 for self-collision, 0 for no self-collision.
     */
    lud::LCBufferView<IndexT> self_collision() const noexcept;

  public:
    class Impl
    {
      public:
        void init();
        void rebuild();

        template <typename T>
        lud::LBufferView<T> subview(lud::LDeviceBuffer<T>& buffer,
                                    SizeT index) const noexcept;

        lud::LDeviceBuffer<IndexT> coindices;
        lud::LDeviceBuffer<IndexT> self_collision;

        OffsetCountCollection<IndexT> reporter_body_offsets_counts;

        SimSystemSlotCollection<BodyReporter> body_reporters;
    };

  private:
    virtual void do_build() final override;

    friend class SimEngine;
    void init();  // only be called by SimEngine

    friend class BodyReporter;
    void add_reporter(BodyReporter* reporter);  // only be called by BodyReporter

    Impl m_impl;
};
}  // namespace uipc::backend::luisa

#include "details/global_body_manager.inl"

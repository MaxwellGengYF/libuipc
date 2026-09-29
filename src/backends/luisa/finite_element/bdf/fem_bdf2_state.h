#pragma once
/********************************************************************
 * @file fem_bdf2_state.h
 * @brief T5.3 — the BDF2 history state (port of
 * src/backends/cuda/finite_element/bdf/fem_bdf2_state.h).
 *
 * `x^{n-1}` and `\dot{x}^{n-1}` as two device buffers plus their dump/recover
 * files.  The dump/recover bodies follow T5.1's `FiniteElementMethod` port
 * exactly (same `BufferDump` usage, same file names, same
 * `dump → load → apply_to → clean_up` sequence) so the two backends' dump sets
 * stay readable by each other.
 *
 * Device mapping: `cuda_tool::DeviceBuffer<Vector3>` -> `lud::LDeviceBuffer<LVector3>`,
 * `cuda_tool::CBufferView<Vector3>` -> `lud::LCBufferView<LVector3>` (both views
 * are byte-compatible with `uipc::Vector3`, see type_define.h).
 *********************************************************************/
#include <sim_system.h>
#include <finite_element/finite_element_method.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class FEMBDF2State final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl
    {
      public:
        bool dump(DumpInfo& info);
        bool try_recover(RecoverInfo& info);
        void apply_recover(RecoverInfo& info);
        void clear_recover(RecoverInfo& info);

        // x^{n-1}
        lud::LDeviceBuffer<LVector3> x_n_1s;
        BufferDump                   dump_x_n_1s;
        // \dot{x}^{n-1}
        lud::LDeviceBuffer<LVector3> v_n_1s;
        BufferDump                   dump_v_n_1s;
    };

    lud::LCBufferView<LVector3> x_n_1s() const
    {
        return m_impl.x_n_1s.cview();
    }
    lud::LCBufferView<LVector3> v_n_1s() const
    {
        return m_impl.v_n_1s.cview();
    }

  private:
    void resize(SizeT size);

    bool do_dump(DumpInfo& info) override;
    bool do_try_recover(RecoverInfo& info) override;
    void do_apply_recover(RecoverInfo& info) override;
    void do_clear_recover(RecoverInfo& info) override;

    friend class FEMBDF2Integrator;

    lud::LBufferView<LVector3> x_n_1s() { return m_impl.x_n_1s.view(); }
    lud::LBufferView<LVector3> v_n_1s() { return m_impl.v_n_1s.view(); }

    Impl m_impl;

    void do_build() override;
};
}  // namespace uipc::backend::luisa

#pragma once
/********************************************************************
 * @file affine_body/bdf/abd_bdf2_state.h
 * @brief T6.2 — the BDF2 ABD history state (port of
 * src/backends/cuda/affine_body/bdf/abd_bdf2_state.h).
 *
 * `q^{n-1}` and `\dot{q}^{n-1}` as two device buffers plus their dump/recover files.
 * Structurally the FEM twin (`finite_element/bdf/fem_bdf2_state.h`, T5.3): the same
 * nested `Impl`, the same friend-set (the *integrator* holds the mutable views), the
 * same `BufferDump` usage, the same file names (`q_n_1.<frame>`, `qv_n_1.<frame>`)
 * so the two backends' dump sets stay readable by each other.
 *
 * Device mapping: `cuda_tool::DeviceBuffer<Vector12>` ->
 * `lud::LDeviceBuffer<LVector12>`, `cuda_tool::CBufferView<Vector12>` ->
 * `lud::LCBufferView<LVector12>` (byte-compatible with `uipc::Vector12`, see
 * affine_body/type_define.h).
 *********************************************************************/
#include <sim_system.h>
#include <affine_body/affine_body_dynamics.h>
#include <utils/dump_utils.h>

namespace uipc::backend::luisa
{
class ABDBDF2State final : public SimSystem
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

        // q^{n-1}
        lud::LDeviceBuffer<LVector12> q_n_1s;
        BufferDump                    dump_q_n_1s;
        // \dot{q}^{n-1}
        lud::LDeviceBuffer<LVector12> q_v_n_1s;
        BufferDump                    dump_q_v_n_1s;
    };

    lud::LCBufferView<LVector12> q_n_1s() const
    {
        return m_impl.q_n_1s.cview();
    }
    lud::LCBufferView<LVector12> q_v_n_1s() const
    {
        return m_impl.q_v_n_1s.cview();
    }

  private:
    void resize(SizeT size);

    bool do_dump(DumpInfo& info) override;
    bool do_try_recover(RecoverInfo& info) override;
    void do_apply_recover(RecoverInfo& info) override;
    void do_clear_recover(RecoverInfo& info) override;

    friend class ABDBDF2Integrator;

    lud::LBufferView<LVector12> q_n_1s() { return m_impl.q_n_1s.view(); }
    lud::LBufferView<LVector12> q_v_n_1s() { return m_impl.q_v_n_1s.view(); }

    Impl m_impl;

    void do_build() override;
};
}  // namespace uipc::backend::luisa

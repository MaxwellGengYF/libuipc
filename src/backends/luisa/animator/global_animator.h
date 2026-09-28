#pragma once
/********************************************************************
 * @file global_animator.h
 * @brief Port of src/backends/cuda/animator/global_animator.h (T4.2).
 *
 * Identical shape. This is the system the engine drives once per frame (init
 * during scene initialization, step inside the PredictMotion phase); it owns the
 * frontend animation update and the per-Newton-iteration substep ratio that the
 * animation-aware constraints consume.
 *********************************************************************/
#include <sim_system.h>

namespace uipc::backend::luisa
{
class Animator;
class GlobalAnimator final : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    /**
     * @brief aim_pos_this_iter = aim_position * alpha + prev_position * (1 - alpha)
     */
    Float substep_ratio() noexcept;

  private:
    friend class SimEngine;
    void init();                                 // only be called by SimEngine
    void step();                                 // only be called by SimEngine
    void compute_substep_ratio(SizeT newton_iter);  // only be called by SimEngine
    friend class Animator;
    void register_animator(Animator* animator);  // only be called by Animator


    virtual void do_build() override;

    SimSystemSlotCollection<Animator> m_animators;

    Float m_substep_ratio = 1.0;
};
}  // namespace uipc::backend::luisa

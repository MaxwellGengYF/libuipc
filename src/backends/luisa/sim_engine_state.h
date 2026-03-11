#pragma once

namespace uipc::backend::luisa
{
enum class SimEngineState
{
    None = 0,
    BuildSystems,
    InitScene,
    RebuildScene,
    PredictMotion,
    ComputeDyTopoEffect,
    // ComputeGradientHessian,
    SolveGlobalLinearSystem,
    LineSearch,
    UpdateVelocity,
};
}  // namespace uipc::backend::luisa

#pragma once
/********************************************************************
 * @file sim_engine_state.h
 * @brief Engine pipeline state machine (mirrors the CUDA backend's states).
 *********************************************************************/
namespace uipc::backend::luisa
{
enum class SimEngineState
{
    None = 0,
    BuildSystems,
    InitScene,
    RebuildScene,
    PredictMotion,             // Common
    ComputeDyTopoEffect,       // Common
    SolveGlobalLinearSystem,   // Common
    LineSearch,                // Common
    UpdateVelocity,            // Common
    AdvanceNonPenetrate,       // AL-IPC
    RecoverNonPenetrate,       // AL-IPC
};
}// namespace uipc::backend::luisa

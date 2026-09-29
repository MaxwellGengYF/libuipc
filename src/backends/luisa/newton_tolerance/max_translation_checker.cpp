#include <newton_tolerance/max_translation_checker.h>
#include <uipc/geometry/attribute_slot.h>
#include <fmt/format.h>

/*************************************************************************************************
* T6.4a — MaxTranslationChecker (port of
* src/backends/cuda/newton_tolerance/max_translation_checker.cu).
*
* The five bodies are the reference's, statement for statement:
*  * `do_build`: `require<GlobalVertexManager>()`, the scene config's `dt` (with the
*    reference's own UIPC_ASSERT) and the `newton/velocity_tol` knob; the *relative*
*    knob is optional in the reference (`rel_attr ? rel_attr->view()[0] : 0.0`) and
*    stays optional here -- `newton/velocity_tol_relative` exists in the default
*    scene config (T4.1's Stiff-GIPC work), but a scene may delete it.
*  * `do_pre_newton`: resets `res0` and records the frame.
*  * `do_check`: the abs_tol comparison quoted in the header; `res0` is captured on
*    the **first** iteration (`newton_iter == 0`) exactly as the reference does.
*  * `do_report`: `fmt::format("Residual/AbsTol: {}/{}", res, abs_tol)`.
*
* Mapping note (cookbook §3.7 / §7.13): the reference's
* `config.find<Float>("dt")` is the port's `world().scene().config().find<Float>(...)`
* and `AttributeSlot::view()[0]` is the ported attribute-slot read -- both are
* *host* reads, so no DSL is involved in this file.
*************************************************************************************************/
namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(MaxTranslationChecker);

void MaxTranslationChecker::do_build(BuildInfo& info)
{
    (void)info;
    vertex_manager = require<GlobalVertexManager>();
    auto& config   = world().scene().config();
    dt_attr        = config.find<Float>("dt");
    UIPC_ASSERT(dt_attr, "Scene config must have a 'dt' attribute.");
    auto newton_velocity_tol_attr = config.find<Float>("newton/velocity_tol");
    newton_velocity_tol           = newton_velocity_tol_attr->view()[0];
    auto rel_attr         = config.find<Float>("newton/velocity_tol_relative");
    velocity_tol_relative = rel_attr ? rel_attr->view()[0] : 0.0;
}

void MaxTranslationChecker::do_pre_newton(PreNewtonInfo& info)
{
    res0  = 0.0;
    frame = info.frame();
}

void MaxTranslationChecker::do_check(CheckResultInfo& info)
{
    // scene-relative mode (Stiff-GIPC convention):
    // abs_tol = velocity_tol_relative * scene_diagonal * dt
    Float vel_tol = newton_velocity_tol;
    if(velocity_tol_relative > 0.0)
        vel_tol = velocity_tol_relative * vertex_manager->scene_diagonal();
    abs_tol          = vel_tol * dt_attr->view()[0];
    res              = vertex_manager->compute_axis_max_displacement();
    auto newton_iter = info.newton_iter();
    if(newton_iter == 0)
        res0 = res;  // record the initial residual

    auto converged = res <= abs_tol;

    info.converged(converged);
}

std::string MaxTranslationChecker::do_report()
{
    std::string report = fmt::format("Residual/AbsTol: {}/{}", res, abs_tol);
    return report;
}
}  // namespace uipc::backend::luisa

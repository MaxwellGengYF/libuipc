#include <contact_system/adaptive_strategies/gipc_adaptive_parameter_strategy.h>
#include <contact_system/global_contact_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <linear_system/global_linear_system.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>
#include <sim_engine.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/zip.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(GIPCAdaptiveParameterStrategy);

void GIPCAdaptiveParameterStrategy::do_build(BuildInfo& info)
{
    auto scene = world().scene();
    {
        auto kappas = scene.contact_tabular().contact_models().find<Float>("resistance");
        auto kappa_view = kappas->view();
        auto found_min_kappa = std::ranges::min(kappa_view);
        if(found_min_kappa >= 0.0)
            throw SimSystemException{"No Adaptive Kappa is detected"};
    }

    contact_manager = require<GlobalContactManager>();
    vertex_manager = require<GlobalVertexManager>();
    surface_manager = require<GlobalSimplicialSurfaceManager>();
    linear_system = require<GlobalLinearSystem>();
    dytopo_effect_manager = require<GlobalDyTopoEffectManager>();

    min_kappa = scene.config().find<Float>("contact/adaptive/min_kappa")->view()[0];
    init_kappa = scene.config().find<Float>("contact/adaptive/init_kappa")->view()[0];
    max_kappa = scene.config().find<Float>("contact/adaptive/max_kappa")->view()[0];

    on_write_scene([this] { write_scene(); });
}

void GIPCAdaptiveParameterStrategy::do_init(InitInfo& info)
{
    auto scene = world().scene();
    auto N = scene.contact_tabular().element_count();
    auto contact_models = scene.contact_tabular().contact_models();
    auto kappas = contact_models.find<Float>("resistance");
    auto topos = contact_models.find<Vector2i>("topo");

    auto kappa_view = kappas->view();
    auto topo_view = topos->view();

    std::vector<Vector2i> h_adaptive_topos;
    h_adaptive_topos.reserve(topos->size());
    h_adaptive_kappa_index.reserve(topos->size());

    // copy if a contact model is adaptive
    for(auto [i, topo] : uipc::enumerate(topo_view))
    {
        if(kappa_view[i] < 0.0)
        {
            h_adaptive_topos.push_back(topo);
            h_adaptive_kappa_index.push_back(i);
        }
    }

    auto& engine = this->engine();
    auto& device = engine.luisa_device();
    auto& stream = engine.compute_stream();

    // Copy adaptive topos to device buffer
    adaptive_topos = device.create_buffer<Vector2i>(h_adaptive_topos.size());
    stream << adaptive_topos.copy_from(h_adaptive_topos.data());

    // Resize gradient buffers
    auto dof_count = linear_system->dof_count();
    contact_gradient = device.create_buffer<Float>(dof_count);
    non_contact_gradient = device.create_buffer<Float>(dof_count);

    // Initialize test contact tabular
    // non-adaptive kappa to 0.0 (don't contribute)
    // adaptive kappa to 1.0 (contribute)
    test_contact_tabular = make_shared<Buffer<ContactCoeff>>(device.create_buffer<ContactCoeff>(N * N));
    stream << test_contact_tabular->fill(ContactCoeff{0.0f, 0.0f});

    // Kernel to set adaptive kappas to 1.0
    if(!h_adaptive_topos.empty())
    {
        auto adaptive_count = h_adaptive_topos.size();
        auto adaptive_topos_view = adaptive_topos.view();
        auto test_tabular_view = test_contact_tabular->view();

        Kernel1D setup_adaptive_kernel = [&](BufferVar<Vector2i> adaptive_topos,
                                              BufferVar<ContactCoeff> contact_tabular,
                                              UInt N,
                                              UInt adaptive_count) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < adaptive_count)
            {
                Vector2i topo = adaptive_topos.read(i);
                auto idxL = topo.x() * N + topo.y();
                auto idxR = topo.y() * N + topo.x();
                
                auto coefL = contact_tabular.read(idxL);
                auto coefR = contact_tabular.read(idxR);
                coefL.kappa = 1.0f;
                coefR.kappa = 1.0f;
                contact_tabular.write(idxL, coefL);
                contact_tabular.write(idxR, coefR);
            };
        };

        auto shader = device.compile(setup_adaptive_kernel);
        stream << shader(adaptive_topos_view, test_tabular_view, N, adaptive_count)
                      .dispatch(adaptive_count);
    }
}

void GIPCAdaptiveParameterStrategy::compute_gradient(AdaptiveParameterInfo& info)
{
    auto& engine = this->engine();
    auto& device = engine.luisa_device();
    auto& stream = engine.compute_stream();

    // set a test contact tabular
    auto original_contact_tabular = info.exchange_contact_tabular(test_contact_tabular);

    auto _compute_gradient = [this, &device, &stream](EnergyComponentFlags flags,
                                                       BufferView<Float> gradient)
    {
        // 1. Compute dytopo effect
        GlobalDyTopoEffectManager::ComputeDyTopoEffectInfo dytopo_effect_info;
        dytopo_effect_info.component_flags(flags);
        dytopo_effect_info.gradient_only(true);
        dytopo_effect_manager->compute_dytopo_effect(dytopo_effect_info);

        // 2. Compute the gradient
        GlobalLinearSystem::ComputeGradientInfo grad_info;
        grad_info.buffer_view(gradient);
        grad_info.flags(flags);
        linear_system->compute_gradient(grad_info);
    };

    // Resize gradient buffers
    auto dof_count = linear_system->dof_count();
    contact_gradient = device.create_buffer<Float>(dof_count);
    non_contact_gradient = device.create_buffer<Float>(dof_count);

    // compute contact gradient
    _compute_gradient(EnergyComponentFlags::Contact, contact_gradient.view());
    // compute non-contact gradient
    _compute_gradient(EnergyComponentFlags::Complement, non_contact_gradient.view());

    // recover original contact tabular
    info.exchange_contact_tabular(original_contact_tabular);
}

void GIPCAdaptiveParameterStrategy::do_compute_parameters(AdaptiveParameterInfo& info)
{
    auto& engine = this->engine();
    auto& device = engine.luisa_device();
    auto& stream = engine.compute_stream();

    compute_gradient(info);

    // Compute dot products: contact2 = dot(contact_grad, contact_grad)
    //                       proj = dot(contact_grad, non_contact_grad)
    auto dof_count = linear_system->dof_count();

    // Create buffers for reduction results
    auto contact2_buffer = device.create_buffer<Float>(1);
    auto proj_buffer = device.create_buffer<Float>(1);

    // Initialize output buffers
    stream << contact2_buffer.fill(0.0f)
           << proj_buffer.fill(0.0f);

    // Kernel to compute dot products using atomic operations
    Kernel1D compute_dots_kernel = [&](BufferVar<Float> contact_grad,
                                        BufferVar<Float> non_contact_grad,
                                        BufferVar<Float> out_contact2,
                                        BufferVar<Float> out_proj,
                                        UInt size) noexcept
    {
        auto tid = dispatch_id().x;
        
        $if(tid < size)
        {
            auto cg = contact_grad.read(tid);
            auto ncg = non_contact_grad.read(tid);
            auto local_contact2 = cg * cg;
            auto local_proj = cg * ncg;
            
            out_contact2.atomic(0).fetch_add(local_contact2);
            out_proj.atomic(0).fetch_add(local_proj);
        };
    };

    auto dots_shader = device.compile(compute_dots_kernel);
    
    // Compute dot products
    stream << dots_shader(contact_gradient.view(),
                           non_contact_gradient.view(),
                           contact2_buffer.view(),
                           proj_buffer.view(),
                           dof_count)
                  .dispatch(dof_count);

    // Copy results to host
    Float h_contact2 = 0.0f;
    Float h_proj = 0.0f;
    stream << contact2_buffer.copy_to(&h_contact2)
           << proj_buffer.copy_to(&h_proj)
           << synchronize();

    // 1. No contact contribution -> keep initial kappa
    Float proj_kappa = (h_contact2 == 0.0f) ? init_kappa : -h_proj / h_contact2;

    // 2. Clamp to [min_kappa, max_kappa]
    new_kappa = std::clamp(proj_kappa, min_kappa, max_kappa);

    logger::info(R"(Adaptive Contact Parameter: > Kappa = {:e}
* ProjKappa = {:e}, Contact^2 = {:e}, Contact.NonContact = {:e}
* MinKappa = {:e}, InitKappa={:e}, MaxKappa = {:e})",
                 new_kappa,
                 proj_kappa,
                 h_contact2,
                 h_proj,
                 min_kappa,
                 init_kappa,
                 max_kappa);

    // Update contact tabular with new kappa values
    auto contact_tabular_view = info.contact_tabular();
    auto N = static_cast<SizeT>(std::sqrt(contact_tabular_view.size()));
    auto adaptive_count = adaptive_topos.size();

    if(adaptive_count > 0)
    {
        Kernel1D update_kappa_kernel = [&](BufferVar<Vector2i> adaptive_topos,
                                            BufferVar<ContactCoeff> contact_tabular,
                                            Float new_kappa,
                                            UInt N,
                                            UInt adaptive_count) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < adaptive_count)
            {
                Vector2i topo = adaptive_topos.read(i);
                auto idxL = topo.x() * N + topo.y();
                auto idxR = topo.y() * N + topo.x();
                
                auto coefL = contact_tabular.read(idxL);
                auto coefR = contact_tabular.read(idxR);
                coefL.kappa = new_kappa;
                coefR.kappa = new_kappa;
                contact_tabular.write(idxL, coefL);
                contact_tabular.write(idxR, coefR);
            };
        };

        auto update_shader = device.compile(update_kappa_kernel);
        stream << update_shader(adaptive_topos.view(),
                                 contact_tabular_view,
                                 new_kappa,
                                 N,
                                 adaptive_count)
                      .dispatch(adaptive_count);
    }
}

void GIPCAdaptiveParameterStrategy::write_scene()
{
    auto scene = world().scene();
    auto kappa = scene.contact_tabular().contact_models().find<Float>("resistance");
    auto kappa_view = view(*kappa);
    for(auto index : h_adaptive_kappa_index)
        kappa_view[index] = new_kappa;
}
}  // namespace uipc::backend::luisa

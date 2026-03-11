#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/affine_body_constitution.h>
#include <luisa/luisa-compute.h>
#include <kernel_cout.h>
#include <affine_body/abd_line_search_subreporter.h>
#include <affine_body/affine_body_kinetic.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(ABDLineSearchReporter);

void ABDLineSearchReporter::do_build(LineSearchReporter::BuildInfo& info)
{
    m_impl.affine_body_dynamics = require<AffineBodyDynamics>();
}

void ABDLineSearchReporter::Impl::init(LineSearchReporter::InitInfo& info)
{
    auto reporter_view = reporters.view();
    for(auto&& [i, R] : enumerate(reporter_view))
        R->m_index = i;  // Assign index for each reporter
    for(auto&& [i, R] : enumerate(reporter_view))
        R->init();

    reporter_energy_offsets_counts.resize(reporter_view.size());
}

void ABDLineSearchReporter::Impl::record_start_point(LineSearcher::RecordInfo& info)
{
    auto& device = info.device();
    auto& stream = info.stream();

    // Copy q to q_temp using buffer copy
    stream << abd().body_id_to_q_temp.copy_from(abd().body_id_to_q.view())
           << luisa::compute::synchronize();
}

void ABDLineSearchReporter::Impl::step_forward(LineSearcher::StepInfo& info)
{
    auto& device = info.device();
    auto& stream = info.stream();

    // Define kernel for stepping forward: qs = q_temps + alpha * dqs
    luisa::compute::Kernel1D step_forward_kernel = [&](
        luisa::compute::BufferVar<Vector12> qs,
        luisa::compute::BufferVar<Vector12> q_temps,
        luisa::compute::BufferVar<Vector12> dqs,
        luisa::compute::BufferVar<bool> is_fixed,
        luisa::compute::Var<float> alpha) noexcept
    {
        auto i = luisa::compute::dispatch_x();
        luisa::compute::Var<bool> fixed = is_fixed.read(i);
        luisa::compute::$if(!fixed)
        {
            luisa::compute::Var<Vector12> q_temp = q_temps.read(i);
            luisa::compute::Var<Vector12> dq = dqs.read(i);
            qs.write(i, q_temp + alpha * dq);
        };
    };

    auto shader = device.compile(step_forward_kernel);
    stream << shader(abd().body_id_to_q.view(),
                     abd().body_id_to_q_temp.view(),
                     abd().body_id_to_dq.view(),
                     abd().body_id_to_is_fixed.view(),
                     info.alpha)
                .dispatch(abd().abd_body_count)
           << luisa::compute::synchronize();
}

void ABDLineSearchReporter::Impl::compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    auto& device = info.device();
    auto& stream = info.stream();

    auto body_count = abd().body_count();

    // Compute kinetic energy
    {
        body_id_to_kinetic_energy.resize(body_count);

        ABDLineSearchReporter::ComputeEnergyInfo this_info;
        this_info.m_energies = body_id_to_kinetic_energy.view();
        this_info.m_dt       = info.dt();

        abd().kinetic->compute_energy(this_info);

        // Zero out the kinetic energy of fixed bodies and bodies with external kinetic
        luisa::compute::Kernel1D zero_fixed_energy_kernel = [&](
            luisa::compute::BufferVar<float> kinetic_energy,
            luisa::compute::BufferVar<bool> is_fixed,
            luisa::compute::BufferVar<bool> external_kinetic) noexcept
        {
            auto i = luisa::compute::dispatch_x();
            luisa::compute::Var<bool> fixed = is_fixed.read(i);
            luisa::compute::Var<bool> external = external_kinetic.read(i);
            luisa::compute::$if(fixed || external)
            {
                kinetic_energy.write(i, 0.0f);
            };
        };

        auto zero_shader = device.compile(zero_fixed_energy_kernel);
        stream << zero_shader(body_id_to_kinetic_energy.view(),
                              abd().body_id_to_is_fixed.view(),
                              abd().body_id_to_external_kinetic.view())
                    .dispatch(abd().abd_body_count)
               << luisa::compute::synchronize();

        // Sum up the kinetic energy using reduction
        // Create a buffer for the result
        auto result_buffer = device.create_buffer<float>(1);
        
        // Use LuisaCompute's built-in reduction or a custom kernel
        // For now, we copy back to host and sum (can be optimized with parallel reduction kernel)
        std::vector<float> host_energies(body_count);
        stream << body_id_to_kinetic_energy.copy_to(host_energies.data())
               << luisa::compute::synchronize();
        
        float K = 0.0f;
        for(float e : host_energies)
            K += e;
        
        // Upload back to device
        stream << result_buffer.copy_from(&K)
               << luisa::compute::synchronize();
        
        // Store result (using the first element as the sum)
        abd_kinetic_energy = K;
    }

    // Compute shape energy
    {
        body_id_to_shape_energy.resize(body_count);

        // Distribute the computation of shape energy to each constitution
        for(auto&& [i, cst] : enumerate(abd().constitutions.view()))
        {
            auto shape_energy = abd().subview(body_id_to_shape_energy, cst->m_index);

            ABDLineSearchReporter::ComputeEnergyInfo this_info;
            this_info.m_energies = shape_energy;
            this_info.m_dt       = info.dt();
            cst->compute_energy(this_info);
        }

        // Sum up the shape energy
        std::vector<float> host_energies(body_count);
        stream << body_id_to_shape_energy.copy_to(host_energies.data())
               << luisa::compute::synchronize();
        
        float shape_E = 0.0f;
        for(float e : host_energies)
            shape_E += e;
        
        abd_shape_energy = shape_E;
    }

    // Collect the energy from other reporters
    {
        auto         reporter_view = reporters.view();
        span<IndexT> counts        = reporter_energy_offsets_counts.counts();
        for(auto&& [i, R] : enumerate(reporter_view))
        {
            ReportExtentInfo this_info;
            R->report_extent(this_info);
            counts[i] = this_info.m_energy_count;
        }

        reporter_energy_offsets_counts.scan();
        reporter_energies.resize(reporter_energy_offsets_counts.total_count());

        for(auto&& [i, R] : enumerate(reporter_view))
        {
            ComputeEnergyInfo this_info;
            auto [offset, count] = reporter_energy_offsets_counts[i];
            this_info.m_energies = reporter_energies.view(offset, count);
            this_info.m_dt       = info.dt();
            R->compute_energy(this_info);
        }

        // Compute the total energy from all reporters
        std::vector<float> host_energies(reporter_energies.size());
        if(!host_energies.empty())
        {
            stream << reporter_energies.copy_to(host_energies.data())
                   << luisa::compute::synchronize();
            
            float other_E = 0.0f;
            for(float e : host_energies)
                other_E += e;
            
            total_reporter_energy = other_E;
        }
        else
        {
            total_reporter_energy = 0.0f;
        }
    }

    // Sum up energies
    Float K       = abd_kinetic_energy;
    Float shape_E = abd_shape_energy;
    Float other_E = total_reporter_energy;

    Float E = K + shape_E + other_E;

    info.energy(E);
}

void ABDLineSearchReporter::do_init(LineSearchReporter::InitInfo& info)
{
    m_impl.init(info);
}

void ABDLineSearchReporter::do_record_start_point(LineSearcher::RecordInfo& info)
{
    m_impl.record_start_point(info);
}

void ABDLineSearchReporter::do_step_forward(LineSearcher::StepInfo& info)
{
    m_impl.step_forward(info);
}

void ABDLineSearchReporter::do_compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    m_impl.compute_energy(info);
}

void ABDLineSearchReporter::add_reporter(ABDLineSearchSubreporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter is null");
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    m_impl.reporters.register_sim_system(*reporter);
}
}  // namespace uipc::backend::luisa

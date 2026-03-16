#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/affine_body_constitution.h>
#include <luisa/dsl/dsl.h>
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

    reporter_energy_offsets_counts.resize(device(), reporter_view.size());
}

void ABDLineSearchReporter::Impl::record_start_point(LineSearcher::RecordInfo& info)
{
    // Copy q to q_temp using buffer copy
    stream() << abd().body_id_to_q_temp.copy_from(abd().body_id_to_q.view());
}

void ABDLineSearchReporter::Impl::step_forward(LineSearcher::StepInfo& info)
{
    using namespace luisa::compute;

    // Kernel for stepping forward: qs = q_temps + alpha * dqs
    Kernel1D step_kernel = [&](BufferVar<const IndexT> is_fixed,
                                BufferVar<const Vector12> q_temps,
                                BufferVar<Vector12> qs,
                                BufferVar<const Vector12> dqs,
                                Float alpha) noexcept
    {
        auto i = dispatch_id().x;
        $if(i < is_fixed.size())
        {
            $if(is_fixed.read(i) == 0)
            {
                Vector12 q_temp = q_temps.read(i);
                Vector12 dq = dqs.read(i);
                Vector12 result;
                for(int k = 0; k < 12; ++k)
                    result[k] = q_temp[k] + alpha * dq[k];
                qs.write(i, result);
            };
        };
    };

    auto shader = device().compile(step_kernel);
    stream() << shader(abd().body_id_to_is_fixed.view(),
                       abd().body_id_to_q_temp.view(),
                       abd().body_id_to_q.view(),
                       abd().body_id_to_dq.view(),
                       info.alpha).dispatch(abd().abd_body_count);
}

void ABDLineSearchReporter::Impl::compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    using namespace luisa::compute;

    auto body_count = abd().body_count();

    // Compute kinetic energy
    {
        body_id_to_kinetic_energy.resize(body_count);

        ABDLineSearchReporter::ComputeEnergyInfo this_info;
        this_info.m_energies = body_id_to_kinetic_energy.view();
        this_info.m_dt       = info.dt();

        abd().kinetic->compute_energy(this_info);

        // Zero out the kinetic energy of fixed bodies and bodies with external kinetic
        Kernel1D zero_fixed_kernel = [&](BufferVar<Float> kinetic_energy,
                                          BufferVar<const IndexT> is_fixed,
                                          BufferVar<const IndexT> external_kinetic) noexcept
        {
            auto i = dispatch_id().x;
            $if(i < is_fixed.size())
            {
                $if(is_fixed.read(i) != 0 || external_kinetic.read(i) != 0)
                {
                    kinetic_energy.write(i, 0.0f);
                };
            };
        };

        auto zero_shader = device().compile(zero_fixed_kernel);
        stream() << zero_shader(body_id_to_kinetic_energy.view(),
                                abd().body_id_to_is_fixed.view(),
                                abd().body_id_to_external_kinetic.view()).dispatch(abd().abd_body_count);

        // Sum up the kinetic energy using reduction
        // TODO: Use GPU parallel reduction for better performance
        // For now, copy to host and sum (similar to CUDA version's approach)
        std::vector<Float> host_energies(body_count);
        stream() << body_id_to_kinetic_energy.copy_to(host_energies.data())
                 << synchronize();
        
        Float K = 0.0;
        for(auto e : host_energies)
            K += e;
        
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
        std::vector<Float> host_energies(body_count);
        stream() << body_id_to_shape_energy.copy_to(host_energies.data())
                 << synchronize();
        
        Float shape_E = 0.0;
        for(auto e : host_energies)
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

        reporter_energy_offsets_counts.scan(stream());
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
        std::vector<Float> host_energies(reporter_energies.size());
        if(!host_energies.empty())
        {
            stream() << reporter_energies.copy_to(host_energies.data())
                     << synchronize();
            
            Float other_E = 0.0;
            for(auto e : host_energies)
                other_E += e;
            
            total_reporter_energy = other_E;
        }
        else
        {
            total_reporter_energy = 0.0;
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

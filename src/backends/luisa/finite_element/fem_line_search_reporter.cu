#include <finite_element/fem_line_search_reporter.h>
#include <finite_element/fem_line_search_subreporter.h>
#include <finite_element/finite_element_kinetic.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <kernel_cout.h>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(FEMLineSearchReporter);

void FEMLineSearchReporter::do_init(InitInfo& info)
{
    m_impl.init(info);
}

void FEMLineSearchReporter::do_build(LineSearchReporter::BuildInfo& info)
{
    m_impl.finite_element_method = require<FiniteElementMethod>();
}

void FEMLineSearchReporter::do_record_start_point(LineSearcher::RecordInfo& info)
{
    m_impl.record_start_point(info);
}

void FEMLineSearchReporter::do_step_forward(LineSearcher::StepInfo& info)
{
    m_impl.step_forward(info);
}

void FEMLineSearchReporter::do_compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    m_impl.compute_energy(info);
}

void FEMLineSearchReporter::Impl::record_start_point(LineSearcher::RecordInfo& info)
{
    using namespace luisa::compute;

    fem().x_temps = fem().xs;
}

void FEMLineSearchReporter::Impl::step_forward(LineSearcher::StepInfo& info)
{
    using namespace luisa::compute;

    auto is_fixed = fem().is_fixed.view();
    auto x_temps = fem().x_temps.view();
    auto xs = fem().xs.view();
    auto dxs = fem().dxs.view();
    auto alpha = info.alpha();
    auto vertex_count = fem().xs.size();

    Kernel1D step_forward_kernel = [&](BufferVar<IndexT> is_fixed_buffer,
                                       BufferVar<Vector3> x_temps_buffer,
                                       BufferVar<Vector3> xs_buffer,
                                       BufferVar<Vector3> dxs_buffer,
                                       Var<Float> alpha_val) noexcept
    {
        auto i = dispatch_x();
        $if(is_fixed_buffer.read(i) == 0)
        {
            auto x_temp = x_temps_buffer.read(i);
            auto dx = dxs_buffer.read(i);
            xs_buffer.write(i, x_temp + alpha_val * dx);
        };
    };

    auto shader = sim_engine->device().compile(step_forward_kernel);
    sim_engine->stream() << shader(is_fixed, x_temps, xs, dxs, alpha).dispatch(vertex_count);
}

void FEMLineSearchReporter::Impl::compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    using namespace luisa::compute;

    // Compute Kinetic (special)
    {
        auto vertex_count = fem().xs.size();
        kinetic_energies.resize(vertex_count);
        auto kinetic_info = ComputeEnergyInfo{kinetic_energies.view(), info.dt()};
        finite_element_kinetic->compute_energy(kinetic_info);

        // Reduction using LuisaCompute's built-in reduction
        total_kinetic_energy = sim_engine->device().create_buffer<Float>(1);
        
        // Use a simple parallel reduction kernel
        auto kinetic_data = kinetic_energies.view();
        auto kinetic_result = total_kinetic_energy.view();
        auto count = kinetic_energies.size();

        Kernel1D reduce_kernel = [&](BufferVar<Float> input,
                                     BufferVar<Float> output,
                                     Var<IndexT> n) noexcept
        {
            Shared<Float> shared{256};
            auto tid = thread_x();
            auto gid = dispatch_x();

            // Load data into shared memory
            $if(gid < n)
            {
                shared[tid] = input.read(gid);
            }
            $else
            {
                shared[tid] = 0.0f;
            };

            // Parallel reduction in shared memory
            for(auto s = 128u; s > 0u; s >>= 1)
            {
                $if(tid < s)
                {
                    shared[tid] += shared[tid + s];
                };
            };

            // Write result
            $if(tid == 0)
            {
                output.atomic(0).fetch_add(shared[0]);
            };
        };

        auto shader = sim_engine->device().compile(reduce_kernel);
        sim_engine->stream() << kinetic_result.fill(0.0f)
                             << shader(kinetic_data, kinetic_result, count)
                                    .dispatch((count + 255) / 256, 256);
    }

    // Collect the energy from all reporters
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
            auto [offset, count] = reporter_energy_offsets_counts[i];
            auto this_info =
                ComputeEnergyInfo{reporter_energies.view(offset, count), info.dt()};
            R->compute_energy(this_info);
        }

        // Reduction using LuisaCompute's built-in reduction
        total_reporter_energy = sim_engine->device().create_buffer<Float>(1);
        
        auto reporter_data = reporter_energies.view();
        auto reporter_result = total_reporter_energy.view();
        auto count = reporter_energies.size();

        Kernel1D reduce_reporter_kernel = [&](BufferVar<Float> input,
                                              BufferVar<Float> output,
                                              Var<IndexT> n) noexcept
        {
            Shared<Float> shared{256};
            auto tid = thread_x();
            auto gid = dispatch_x();

            // Load data into shared memory
            $if(gid < n)
            {
                shared[tid] = input.read(gid);
            }
            $else
            {
                shared[tid] = 0.0f;
            };

            // Parallel reduction in shared memory
            for(auto s = 128u; s > 0u; s >>= 1)
            {
                $if(tid < s)
                {
                    shared[tid] += shared[tid + s];
                };
            };

            // Write result
            $if(tid == 0)
            {
                output.atomic(0).fetch_add(shared[0]);
            };
        };

        auto shader = sim_engine->device().compile(reduce_reporter_kernel);
        sim_engine->stream() << reporter_result.fill(0.0f)
                             << shader(reporter_data, reporter_result, count)
                                    .dispatch((count + 255) / 256, 256);
    }

    // Copy results to host
    luisa::vector<Float> kinetic_host(1);
    luisa::vector<Float> reporter_host(1);
    sim_engine->stream() << total_kinetic_energy.copy_to(kinetic_host.data())
                         << total_reporter_energy.copy_to(reporter_host.data())
                         << synchronize();

    Float K       = kinetic_host[0];
    Float other_E = reporter_host[0];
    Float total_E = K + other_E;

    info.energy(total_E);
}

void FEMLineSearchReporter::Impl::init(LineSearchReporter::InitInfo& info)
{
    kinetic_energies.resize(fem().xs.size());

    auto reporter_view = reporters.view();
    for(auto&& [i, R] : enumerate(reporter_view))
        R->m_index = i;
    for(auto&& [i, R] : enumerate(reporter_view))
        R->init();

    reporter_energy_offsets_counts.resize(reporter_view.size());
}

void FEMLineSearchReporter::add_reporter(FEMLineSearchSubreporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter is null");
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    m_impl.reporters.register_sim_system(*reporter);
}

void FEMLineSearchReporter::add_kinetic(FiniteElementKinetic* kinetic)
{
    UIPC_ASSERT(kinetic, "kinetic is null");
    check_state(SimEngineState::BuildSystems, "add_kinetic()");
    m_impl.finite_element_kinetic.register_sim_system(*kinetic);
}
}  // namespace uipc::backend::luisa

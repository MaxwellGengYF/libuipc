#include <finite_element/fem_line_search_reporter.h>
#include <finite_element/fem_line_search_subreporter.h>
#include <finite_element/finite_element_kinetic.h>
#include <frame_stream.h>
#include <luisa_device/linear/linear.h>

/***************************************************************
 * T5.3 (pulled forward from T5.5) — the FEM line-search reporter
 * (reference: src/backends/cuda/finite_element/fem_line_search_reporter.cu).
 *
 * This is the fan-out that the Phase-8 line search drives: it steps every vertex
 * forward along `dxs`, sums the kinetic energy (special-cased through
 * `FiniteElementKinetic`, exactly like the reference) and the reporters' energies,
 * and combines them with the reference's one-thread kernel.
 *
 * Mapping notes (cookbook §7.13 rule 3 / §7.15 rule 6):
 *  * `cuda_tool::DeviceReduce().Sum(in, out, n)` -> `lla::device_reduce_sum` on the
 *    shared frame stream; the reduction is batched with the kernel dispatches of
 *    the same frame.  `n == 0` writes the identity in both (the reference's CUB
 *    reduce does the same), so the unguarded call is kept verbatim in the shape
 *    the reference has it -- except that a zero-length CUB reduce must not be
 *    launched at all in the port (`device_reduce_sum` with n == 0 writes the
 *    identity, which is what we want).
 *  * `fem().x_temps = fem().xs;` is a *deep* copy in cuda_tool; an LDeviceBuffer
 *    assignment only shares the allocation, so `copy_device` is required
 *    (cookbook §7.16 rule 4b).
 *  * `resize_discard` -> `loose_resize` (the T4.3 dytopo reporter's helper).
 ***************************************************************/
namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

namespace
{
/// cuda_tool's `DeviceBuffer::operator=` deep-copies; an LDeviceBuffer assignment
/// does not (cookbook §7.16 rule 4b).
template <typename T>
void copy_device(lud::LDeviceBuffer<T>& dst, const lud::LDeviceBuffer<T>& src)
{
    dst.resize(src.size());
    if(src.size() != 0u)
    {
        dst.view().copy_from(src.cview());
    }
}

/// `cuda_tool::DeviceReduce().Sum(in.data(), out.data(), n)` on the shared frame
/// stream (T3.2, cookbook §7.15 rule 6).  A zero-length reduction writes the
/// reduce identity in the reference's CUB path too, but an empty `LBufferView`
/// carries a *null* handle that must not become a kernel argument, so that case
/// is written with a one-element copy recorded on the same stream -- the exact
/// shape T4.3's dytopo line-search subreporter uses.
void reduce_sum_onto(const lud::LDeviceBuffer<Float>& in, lud::LDeviceVar<Float>& out)
{
    if(in.size() > 0u)
    {
        lla::device_reduce_sum(frame_stream(), in.cview(), out.view(), in.size());
    }
    else
    {
        const Float zero = 0.0;
        frame_stream().record(out.view().native().copy_from(lcrt::span<const Float>{&zero, 1u}));
    }
}

/// `FEMLineSearchReporter_step_forward_kernel`
auto& fem_line_search_step_forward_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<LVector3>,  // x_temps
                                                lcrt::Buffer<LVector3>,  // xs
                                                lcrt::Buffer<LVector3>,  // dxs
                                                double,                  // alpha
                                                uint32_t>(               // n
        [](lcrt::BufferVar<LVector3> x_temps,
           lcrt::BufferVar<LVector3> xs,
           lcrt::BufferVar<LVector3> dxs,
           lcrt::Var<double>         alpha,
           lcrt::Var<uint32_t>       n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                auto x = load_vec3(x_temps->read(i)) + alpha * load_vec3(dxs->read(i));
                lcrt::Var<LVector3> out;
                store_vec3(out, x);
                xs->write(i, out);
            };
        });
    return k;
}

/// `FEMLineSearchReporter_combine_energy_kernel`: one thread of the whole grid
/// (`blockIdx.x == 0 && threadIdx.x == 0`), dispatched with a single thread here.
auto& fem_line_search_combine_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>,  // kinetic_energy
                                   lcrt::Buffer<double>,  // reporter_energy
                                   lcrt::Buffer<double>>(  // total_energy
            [](lcrt::BufferVar<double> kinetic_energy,
               lcrt::BufferVar<double> reporter_energy,
               lcrt::BufferVar<double> total_energy) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i == 0u)
                {
                    total_energy->write(i, kinetic_energy->read(i) + reporter_energy->read(i));
                };
            });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FEMLineSearchReporter);

void FEMLineSearchReporter::do_init(InitInfo& info)
{
    m_impl.init(info);
}

void FEMLineSearchReporter::do_build(LineSearchReporter::BuildInfo& info)
{
    (void)info;
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
    (void)info;

    copy_device(fem().x_temps, fem().xs);
}

void FEMLineSearchReporter::Impl::step_forward(LineSearcher::StepInfo& info)
{
    auto n = static_cast<uint32_t>(fem().xs.size());
    if(n > 0)
    {
        auto& k = fem_line_search_step_forward_kernel();
        frame_stream().record(k(fem().x_temps.cview(), fem().xs.view(), fem().dxs.cview(), info.alpha, n)
                                  .dispatch(n));
    }
}

void FEMLineSearchReporter::Impl::compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    // Compute Kinetic (special)
    {
        auto vertex_count = fem().xs.size();
        loose_resize(kinetic_energies, vertex_count);
        auto kinetic_info = ComputeEnergyInfo{kinetic_energies.view(), info.dt()};
        finite_element_kinetic->compute_energy(kinetic_info);

        reduce_sum_onto(kinetic_energies, total_kinetic_energy);
    }

    // Collect the energy from all reporters
    {
        auto          reporter_view = reporters.view();
        span<IndexT>  counts        = reporter_energy_offsets_counts.counts();
        for(auto&& [i, R] : enumerate(reporter_view))
        {
            ReportExtentInfo this_info;
            R->report_extent(this_info);
            counts[i] = this_info.m_energy_count;
        }

        reporter_energy_offsets_counts.scan();
        loose_resize(reporter_energies, reporter_energy_offsets_counts.total_count());

        for(auto&& [i, R] : enumerate(reporter_view))
        {
            auto [offset, count] = reporter_energy_offsets_counts[i];
            auto this_info =
                ComputeEnergyInfo{reporter_energies.view(offset, count), info.dt()};
            R->compute_energy(this_info);
        }

        reduce_sum_onto(reporter_energies, total_reporter_energy);
    }

    auto& k = fem_line_search_combine_energy_kernel();
    frame_stream().record(k(total_kinetic_energy.cview(),
                            total_reporter_energy.cview(),
                            info.energy())
                            .dispatch(1u));
}

void FEMLineSearchReporter::Impl::init(LineSearchReporter::InitInfo& info)
{
    (void)info;

    loose_resize(kinetic_energies, fem().xs.size());

    auto reporter_view = reporters.view();
    for(auto&& [i, R] : enumerate(reporter_view))
        R->m_index = static_cast<IndexT>(i);
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

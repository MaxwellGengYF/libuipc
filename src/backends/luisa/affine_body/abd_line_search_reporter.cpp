#include <affine_body/abd_line_search_reporter.h>
#include <affine_body/abd_line_search_subreporter.h>
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/affine_body_constitution.h>
#include <frame_stream.h>
#include <luisa_device/linear/linear.h>

/*************************************************************************************************
* T6.4a — the ABD line-search reporter, completed (port of
* src/backends/cuda/affine_body/abd_line_search_reporter.cu, 215 lines).
*
* T6.1b pulled this file's `add_reporter` forward because the ABD animator's and
* the inter-affine-body manager's line-search subreporters derive from
* `ABDLineSearchSubreporter` and `do_build()` calls it -- a vtable needs it.  T6.4a
* adds everything else its SCOPE NOTE (T6.4) listed and deletes that note:
*  * `Impl::{init, record_start_point, step_forward, compute_energy}` (the three
*    reference kernels + the kinetic/constitution/reporter energy fan-out),
*  * the five `do_*` hooks and `REGISTER_SIM_SYSTEM`.
*
* Mapping notes (the FEM sibling, fem_line_search_reporter.cpp, is the model):
*  * `cuda_tool::DeviceReduce().Sum(in, out, n)` -> `lla::device_reduce_sum` on the
*    shared frame stream; a zero-length reduce writes the identity in both, but an
*    empty `LBufferView` carries a *null* handle that must not become a kernel
*    argument, so the `n == 0` case is a one-element copy on the same stream.
*  * `resize_discard(n)` -> `LDeviceBuffer::resize(n)`: the port's `LDeviceBuffer`
*    *is* cuda_tool's resize shape (amortized capacity growth + a value-initialised
*    tail, buffer.h), and every element of the three energy buffers is written by
*    the fan-out below before it is read, so the "discard" (leave the new range
*    uninitialised) part is unobservable.  `loose_resize` (the FEM/T4.3 helper) is
*    therefore not needed here.
*  * `q_temps = qs` (a *deep* copy in cuda_tool) -> `copy_device` below, because an
*    `LDeviceBuffer` assignment only shares the allocation (cookbook §7.16 rule 4b).
*  * `abd().kinetic->compute_energy(...)` / `cst->compute_energy(...)` /
*    `R->report_extent/compute_energy(...)` are the reference's fan-outs, reachable
*    because this class is the `friend` of all three (abd_line_search_reporter.h's
*    `friend class ABDLineSearchReporter;` declarations, T6.1a/T6.1b).
*
* The reference's `abd_line_search_reporter_step_forward_kernel` and the two small
* kernels are transcribed with the port's `select`/`$if` rules:
*  * `if(is_fixed(i)) return;` -> a store guarded by `$if(!is_fixed(i))` (the value
*    is a device predicate, RULE M2);
*  * `if(is_fixed(i) || external_kinetic(i)) kinetic_energy(i) = 0.0;` -> a
*    `$if`-guarded store (the zero is a constant, so no `select` is needed and the
*    untouched entries keep whatever the kinetic wrote -- exactly the reference);
*  * the one-thread combine kernel -> `dispatch(1u)` + `$if(i == 0u)` (the FEM
*    sibling's spelling of `if(blockIdx.x != 0 || threadIdx.x != 0) return;`).
*************************************************************************************************/
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
/// stream (T3.2).  A zero-length reduction writes the reduce identity in the
/// reference's CUB path too, but an empty `LBufferView` carries a *null* handle
/// that must not become a kernel argument, so that case is a one-element copy
/// recorded on the same stream.
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

/// `abd_line_search_reporter_step_forward_kernel`:
/// `if(is_fixed(i)) return; qs(i) = q_temps(i) + alpha * dqs(i);`
auto& abd_line_search_step_forward_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,  // is_fixed
                                                lcrt::Buffer<LVector12>,     // q_temps
                                                lcrt::Buffer<LVector12>,     // qs
                                                lcrt::Buffer<LVector12>,     // dqs
                                                double,                      // alpha
                                                uint32_t>(                   // n
        [](lcrt::BufferVar<std::int32_t> is_fixed,
           lcrt::BufferVar<LVector12>    q_temps,
           lcrt::BufferVar<LVector12>    qs,
           lcrt::BufferVar<LVector12>    dqs,
           lcrt::Var<double>             alpha,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                $if(is_fixed->read(i) == 0)
                {
                    auto q = load_vec12(q_temps->read(i)) + alpha * load_vec12(dqs->read(i));
                    lcrt::Var<LVector12> out;
                    store_vec12(out, q);
                    qs->write(i, out);
                };
            };
        });
    return k;
}

/// `abd_line_search_reporter_compute_energy_kernel`:
/// `if(is_fixed(i) || external_kinetic(i)) kinetic_energy(i) = 0.0;`
auto& abd_line_search_zero_kinetic_energy_kernel()
{
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,  // is_fixed
                                                lcrt::Buffer<std::int32_t>,  // external_kinetic
                                                lcrt::Buffer<double>,        // kinetic_energy
                                                uint32_t>(                   // n
        [](lcrt::BufferVar<std::int32_t> is_fixed,
           lcrt::BufferVar<std::int32_t> external_kinetic,
           lcrt::BufferVar<double>       kinetic_energy,
           lcrt::Var<uint32_t>           n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                $if((is_fixed->read(i) != 0) | (external_kinetic->read(i) != 0))
                {
                    kinetic_energy->write(i, 0.0);
                };
            };
        });
    return k;
}

/// `abd_line_search_reporter_combine_energy_kernel`: one thread of the whole grid
/// (`blockIdx.x == 0 && threadIdx.x == 0`), dispatched with a single thread here.
auto& abd_line_search_combine_energy_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<double>,  // kinetic_energy
                                   lcrt::Buffer<double>,  // shape_energy
                                   lcrt::Buffer<double>,  // reporter_energy
                                   lcrt::Buffer<double>>(  // total_energy
            [](lcrt::BufferVar<double> kinetic_energy,
               lcrt::BufferVar<double> shape_energy,
               lcrt::BufferVar<double> reporter_energy,
               lcrt::BufferVar<double> total_energy) noexcept
            {
                auto i = lcrt::dispatch_x();
                $if(i == 0u)
                {
                    total_energy->write(i,
                                        kinetic_energy->read(i) + shape_energy->read(i)
                                            + reporter_energy->read(i));
                };
            });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(ABDLineSearchReporter);

void ABDLineSearchReporter::do_build(LineSearchReporter::BuildInfo& info)
{
    (void)info;
    m_impl.affine_body_dynamics = require<AffineBodyDynamics>();
}

void ABDLineSearchReporter::Impl::init(LineSearchReporter::InitInfo& info)
{
    (void)info;

    auto reporter_view = reporters.view();
    for(auto&& [i, R] : enumerate(reporter_view))
        R->m_index = static_cast<IndexT>(i);  // Assign index for each reporter
    for(auto&& [i, R] : enumerate(reporter_view))
        R->init();

    reporter_energy_offsets_counts.resize(reporter_view.size());
}

void ABDLineSearchReporter::Impl::record_start_point(LineSearcher::RecordInfo& info)
{
    (void)info;

    copy_device(abd().body_id_to_q_temp, abd().body_id_to_q);
}

void ABDLineSearchReporter::Impl::step_forward(LineSearcher::StepInfo& info)
{
    auto n = static_cast<uint32_t>(abd().abd_body_count);
    if(n > 0)
    {
        auto& k = abd_line_search_step_forward_kernel();
        frame_stream().record(k(abd().body_id_to_is_fixed.cview(),
                                abd().body_id_to_q_temp.cview(),
                                abd().body_id_to_q.view(),
                                abd().body_id_to_dq.cview(),
                                info.alpha,
                                n)
                                  .dispatch(n));
    }
}

void ABDLineSearchReporter::Impl::compute_energy(LineSearcher::ComputeEnergyInfo& info)
{
    auto body_count = abd().body_count();

    // Compute kinetic energy
    {
        body_id_to_kinetic_energy.resize(body_count);

        ABDLineSearchReporter::ComputeEnergyInfo this_info;
        this_info.m_energies = body_id_to_kinetic_energy.view();
        this_info.m_dt       = info.dt();

        abd().kinetic->compute_energy(this_info);

        // Zero out the kinetic energy of fixed bodies and bodies with external kinetic
        auto n = static_cast<uint32_t>(abd().abd_body_count);
        if(n > 0)
        {
            auto& k = abd_line_search_zero_kinetic_energy_kernel();
            frame_stream().record(k(abd().body_id_to_is_fixed.cview(),
                                    abd().body_id_to_external_kinetic.cview(),
                                    body_id_to_kinetic_energy.view(),
                                    n)
                                      .dispatch(n));
        }

        // Sum up the kinetic energy
        reduce_sum_onto(body_id_to_kinetic_energy, abd_kinetic_energy);
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
        reduce_sum_onto(body_id_to_shape_energy, abd_shape_energy);
    }

    // Collect the energy from other reporters
    {
        auto         reporter_view = reporters.view();
        span<IndexT> counts        = reporter_energy_offsets_counts.counts();
        for(auto&& [i, R] : enumerate(reporter_view))
        {
            ReportExtentInfo this_info;
            R->report_extent(this_info);
            counts[i] = static_cast<IndexT>(this_info.m_energy_count);
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
        reduce_sum_onto(reporter_energies, total_reporter_energy);
    }

    auto& k = abd_line_search_combine_energy_kernel();
    frame_stream().record(k(abd_kinetic_energy.cview(),
                            abd_shape_energy.cview(),
                            total_reporter_energy.cview(),
                            info.energy())
                            .dispatch(1u));
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

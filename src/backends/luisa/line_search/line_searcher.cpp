#include <line_search/line_searcher.h>

#include <line_search/line_search_reporter.h>

#include <frame_stream.h>

#include <luisa_device/linear/algorithms.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/zip.h>
#include <uipc/common/timer.h>

#include <span>

namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

namespace
{
    /// `cuda_tool::VarView<Float>{p}.fill(0.0)` needs a command that is *ordered*
    /// inside the caller's frame: this one-thread kernel is recorded on the frame
    /// command list (cookbook §7.11 rule 7).
    auto& fill_one_float_kernel()
    {
        static auto& k = lla::detail::cached_kernel<lcrt::Buffer<double>, double>(
            [](lcrt::BufferVar<double> v, lcrt::Var<double> value) noexcept {
                v->write(lcrt::dispatch_x(), value);
            });
        return k;
    }

    /// cuda_tool parity (`m_device_energy_values.copy_to(...)` synchronises): the
    /// work recorded in the frame command list must be committed and completed
    /// before the copy is ordered after it (cookbook §7.13 rule 3).
    void read_back(lud::LStream& stream, const lud::LBufferView<Float>& src, std::span<Float> dst)
    {
        const bool was_open = stream.frame_open();
        stream.synchronize();
        if(!dst.empty())
        {
            stream.record(src.native().copy_to(dst));
            stream.synchronize();
        }
        if(was_open)
        {
            stream.begin();
        }
    }
}// namespace

REGISTER_SIM_SYSTEM(LineSearcher);

void LineSearcher::do_build() {}

void LineSearcher::init()
{
    auto scene = world().scene();

    auto report_enery_attr = scene.config().find<IndexT>("line_search/report_energy");
    m_report_energy        = report_enery_attr->view()[0] != 0;

    auto max_iter_attr = scene.config().find<IndexT>("line_search/max_iter");
    m_max_iter         = max_iter_attr->view()[0];

    m_dt_attr = scene.config().find<Float>("dt");
    UIPC_ASSERT(m_dt_attr, "Scene config must have a 'dt' attribute.");

    const auto energy_count = m_reporters.view().size() + 1;
    m_device_energy_values.resize(energy_count);
    m_energy_values.resize(energy_count, 0);

    auto reporter_view = m_reporters.view();

    for(auto&& [i, R] : enumerate(reporter_view))
        R->m_index = i;

    for(auto&& [i, R] : enumerate(reporter_view))
        R->init();
}

void LineSearcher::record_start_point()
{
    for(auto&& R : m_reporters.view())
    {
        RecordInfo info;
        R->record_start_point(info);
    }
}

void LineSearcher::step_forward(Float alpha)
{
    for(auto&& R : m_reporters.view())
    {
        StepInfo info;
        info.alpha = alpha;
        R->step_forward(info);
    }
}

Float LineSearcher::compute_energy(bool is_initial)
{
    Timer timer{"Compute Energy"};

    auto& stream = frame_stream();

    auto reporter_view = m_reporters.view();

    for(auto&& [i, R] : enumerate(reporter_view))
    {
        ComputeEnergyInfo info{this, m_device_energy_values.view(i, 1u)};
        info.m_is_initial = is_initial;
        R->compute_energy(info);
        UIPC_ASSERT(info.m_energy_set,
                    "Energy[{}] not set by reporter, did you forget to call energy()?",
                    R->name());
    }

    const auto reporter_count = reporter_view.size();
    if(reporter_count > 0)
    {
        // cuda_tool::DeviceReduce().Sum(values, total_energy_device, reporter_count)
        lla::device_reduce_sum(stream,
                               m_device_energy_values.cview(0u, reporter_count),
                               m_device_energy_values.view(reporter_count, 1u),
                               reporter_count);
    }
    else
    {
        // cuda_tool::VarView<Float>{total_energy_device}.fill(0.0)
        stream.record(fill_one_float_kernel()(m_device_energy_values.view(reporter_count, 1u), 0.0)
                          .dispatch(1u));
    }

    read_back(stream, m_device_energy_values.cview(), span<Float>{m_energy_values});

    auto reporter_energies = span{m_energy_values}.first(reporter_count);
    for(auto&& [energy, reporter] : zip(reporter_energies, reporter_view))
    {
        UIPC_ASSERT(!std::isnan(energy) && std::isfinite(energy),
                    "Energy [{}] is {}",
                    reporter->name(),
                    energy);
    }

    Float total_energy = m_energy_values.back();
    UIPC_ASSERT(!std::isnan(total_energy) && std::isfinite(total_energy),
                "Total line-search energy is {}",
                total_energy);

    if(m_report_energy)
    {
        m_report_stream << R"(
-------------------------------------------------------------------------------
* Compute Energy *
-------------------------------------------------------------------------------
)";
        m_report_stream << "Total:" << total_energy << "\n";
        for(auto&& [R, value] : zip(reporter_view, reporter_energies))
        {
            m_report_stream << "  > " << R->name() << "=" << value << "\n";
        }
        m_report_stream << "-------------------------------------------------------------------------------";
        logger::info(m_report_stream.str());
        m_report_stream.str("");
    }

    return total_energy;
}

void LineSearcher::add_reporter(LineSearchReporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter is nullptr");
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    m_reporters.register_sim_system(*reporter);
}

LineSearcher::ComputeEnergyInfo::ComputeEnergyInfo(LineSearcher* impl,
                                                   lud::LBufferView<Float> energy) noexcept
    : m_impl(impl)
    , m_energy(energy)
{
}

Float LineSearcher::ComputeEnergyInfo::dt() noexcept
{
    return m_impl->m_dt_attr->view()[0];
}

lud::LBufferView<Float> LineSearcher::ComputeEnergyInfo::energy() noexcept
{
    m_energy_set = true;
    return m_energy;
}

bool LineSearcher::ComputeEnergyInfo::is_initial() noexcept
{
    return m_is_initial;
}

SizeT LineSearcher::max_iter() const noexcept
{
    return m_max_iter;
}
}// namespace uipc::backend::luisa

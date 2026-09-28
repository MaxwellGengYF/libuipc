#include <implicit_geometry/half_plane_body_reporter.h>
#include <frame_stream.h>
#include <luisa_device/linear/algorithms.h>

namespace uipc::backend::luisa
{
namespace lla = luisa_device::linalg;

namespace
{
/// The reference's `HalfPlaneBodyReporter_report_attributes_kernel`: a plain iota
/// over the reporter's body slice ("One position and one normal per half plane
/// body, so the body coindices are just the local index").
auto& half_plane_body_coindices_kernel()
{
    // Compiled once and deliberately leaked: a compiled Shader held by a
    // function-local static would be destroyed after Runtime::shutdown()
    // (cookbook §7.10 trap 1 / §7.13 rule 6).
    static auto& k = lla::detail::cached_kernel<lcrt::Buffer<IndexT>, uint32_t>(
        [](lcrt::BufferVar<IndexT> coindices, lcrt::Var<uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n)
            {
                coindices->write(i, lcrt::cast<IndexT>(i));
            };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(HalfPlaneBodyReporter);

void HalfPlaneBodyReporter::do_build(BuildInfo& info)
{
    m_impl.half_plane = &require<HalfPlane>();
}

void HalfPlaneBodyReporter::do_init(InitInfo& info) {}

void HalfPlaneBodyReporter::Impl::report_count(BodyCountInfo& info)
{
    // One position and one normal per half plane body
    // so body_count is equal to the position count.
    auto body_count = half_plane->m_impl.h_positions.size();
    info.count(body_count);
}

void HalfPlaneBodyReporter::Impl::report_attributes(BodyAttributeInfo& info)
{
    auto* k = &half_plane_body_coindices_kernel();
    auto  n = static_cast<uint32_t>(info.coindices().size());
    if(n > 0)
    {
        frame_stream().record((*k)(info.coindices(), n).dispatch(n));
    }

    // HalfPlane does not have self-collision, so we fill it with zeros.
    info.self_collision().fill(0, &frame_stream());
}

void HalfPlaneBodyReporter::do_report_count(BodyCountInfo& info)
{
    m_impl.report_count(info);
}

void HalfPlaneBodyReporter::do_report_attributes(BodyAttributeInfo& info)
{
    m_impl.report_attributes(info);
}
}  // namespace uipc::backend::luisa

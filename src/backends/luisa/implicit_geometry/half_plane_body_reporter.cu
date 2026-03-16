#include <implicit_geometry/half_plane_body_reporter.h>
#include <implicit_geometry/half_plane.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
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
    using namespace luisa::compute;

    // Create a kernel to fill coindices with iota values (0, 1, 2, ...)
    Kernel1D iota_kernel = [&](BufferVar<IndexT> coindices) noexcept
    {
        auto i = dispatch_id().x;
        $if(i < coindices.size())
        {
            coindices.write(i, cast<IndexT>(i));
        };
    };

    auto& stream = half_plane->engine().stream();
    auto  coindices_buffer = info.coindices();

    // Compile and dispatch the kernel
    auto shader = half_plane->engine().device().compile(iota_kernel);
    stream << shader(coindices_buffer).dispatch(coindices_buffer.size());

    // HalfPlane does not have self-collision, so we fill it with zeros.
    stream << info.self_collision().fill(0);
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
